"""Thin HTTP layer: validate input, retrieve evidence, generate, return traceable output."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.generation import GenerationError, generate
from app.knowledge import digest
from app.methods import RetrievalMethod, build_engines
from app.retrieval import ROOT, ChineseBM25
from app.telemetry import JsonlTraceSink, RequestTrace, annotate, code_signature, emit_safely, phase, trace_context

logger = logging.getLogger("rag.api")


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=1500)
    top_k: int = Field(default=5, ge=1, le=20)
    source_id: str | None = Field(default=None, min_length=1, max_length=100)
    method: RetrievalMethod = "bm25"

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class ComparisonRequest(QueryRequest):
    method: Literal["hybrid_rerank"] = "hybrid_rerank"
    top_k: int = Field(default=5, ge=2, le=20)
    recover: bool = False


class Hit(BaseModel):
    id: str
    title: str
    contents: str
    metadata: dict
    score: float
    rank: int


class RetrievalResponse(BaseModel):
    request_id: str
    method: str
    knowledge_version: str
    hits: list[Hit]
    retrieval_ms: float


class AnswerResponse(BaseModel):
    request_id: str
    method: str
    knowledge_version: str
    answer: str
    citations: list[str]
    insufficient_evidence: bool
    evidence: list[dict]
    model_called: bool
    model: str | None = None
    citation_check: str | None = None
    usage: dict = Field(default_factory=dict)
    excluded_listing_ids: list[str] = Field(default_factory=list)
    retrieval_ms: float


def create_app(retrievers: dict | None = None, generator=generate, knowledge=None,
               comparison_generator=None, trace_sink=None) -> FastAPI:
    sink = trace_sink if trace_sink is not None else JsonlTraceSink(ROOT / '.cache/request-telemetry.jsonl')
    implementation_version = code_signature(ROOT / 'app')
    @asynccontextmanager
    async def lifespan(app):
        # Injected legacy retrieval tests need no disk-backed knowledge service.
        service = knowledge
        if service is None and retrievers is None:
            from app.knowledge import KnowledgeService
            service = KnowledgeService(ROOT / "artifacts/knowledge")
        app.state.knowledge = service
        engines = dict(retrievers or {})
        try:
            if service is not None:
                service.start()
            if not engines and (service is None or service.pin() is None):
                bm25 = ChineseBM25()
                bm25.load()
                dense = None
                from app.dense import DENSE_INDEX, DenseRetriever
                if (DENSE_INDEX / "manifest.json").exists():
                    dense = DenseRetriever()
                    dense.load()
                engines.update(build_engines(bm25, dense))
            app.state.engines = engines
            yield
        finally:
            if service is not None:
                service.close()

    app = FastAPI(title="中文招生知识问答与检索评测", lifespan=lifespan)

    # The proxy delays default-store creation until application startup.
    class KnowledgeProxy:
        def __getattr__(self, name):
            service = app.state.knowledge
            if service is None:
                raise HTTPException(503, "Knowledge management is disabled for this app instance.")
            return getattr(service, name)

    from app.knowledge_routes import management_router
    app.include_router(management_router(KnowledgeProxy()))

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(ROOT / "app/static/index.html")

    @app.get("/manage", include_in_schema=False)
    def manage():
        return FileResponse(ROOT / "app/static/manage.html")

    @app.middleware("http")
    async def trace(request: Request, call_next):
        request.state.request_id = str(uuid4())
        endpoint = request.url.path if request.url.path in {'/retrieve', '/ask', '/ask-comparison', '/health'} else 'other'
        ledger = RequestTrace(request.state.request_id, request.headers.get('X-Task-ID'), endpoint)
        ledger.update(implementation_version=implementation_version)
        status, error_type = 500, None
        with trace_context(ledger):
            try:
                response = await call_next(request)
                status = response.status_code
                response.headers['X-Request-ID'] = ledger.request_id
                response.headers['X-Task-ID'] = ledger.task_id
                if status >= 400:
                    annotate(outcome='http_failure')
                    error_type = 'HTTPError'
                return response
            except BaseException as error:
                error_type = type(error).__name__
                annotate(outcome='unhandled_error')
                raise
            finally:
                record = ledger.finish(status, error_type)
                emit_safely(sink, record)
                logger.info('request_id=%s status=%d elapsed_ms=%.1f', ledger.request_id, status, record['elapsed_ms'])

    def resolve_engine(body: QueryRequest):
        service = app.state.knowledge
        from app.knowledge import KnowledgeError
        try:
            snapshot = service.pin() if service else None
        except (KnowledgeError, ValueError, OSError, KeyError):
            raise HTTPException(503, "Active knowledge version could not be loaded.") from None
        version = snapshot.version if snapshot else "baseline-static"
        annotate(method=body.method, knowledge_version=version)
        engine = (snapshot.engines if snapshot else app.state.engines).get(body.method)
        if engine is None:
            raise HTTPException(503, "Requested retrieval index is not available; build it first.")
        underlying = getattr(engine, 'retriever', engine)
        annotate(candidate_k=getattr(engine, 'candidate_k', None), final_top_k=body.top_k,
                 rrf_constant=getattr(underlying, 'rrf_constant', None))
        return engine, version

    def retrieve(body: QueryRequest):
        engine, version = resolve_engine(body)
        start = time.perf_counter()
        try:
            with phase('retrieval'):
                hits = engine.search(body.question, body.top_k, body.source_id)
        except Exception as exc:
            from app.embedding_api import EmbeddingError
            if isinstance(exc, EmbeddingError):
                raise HTTPException(502, str(exc)) from None
            raise
        # Normalize IDs at the boundary; corpus data remains unchanged.
        hits = [{**hit, "id": str(hit["id"]), "metadata": {
            **hit["metadata"], "knowledge_version": version,
            "content_sha256": digest(hit["contents"])}} for hit in hits]
        annotate(evidence_ids=[hit['id'] for hit in hits], evidence_sha256=[hit['metadata']['content_sha256'] for hit in hits])
        return hits, (time.perf_counter() - start) * 1000, version

    @app.get("/health")
    def health():
        service = app.state.knowledge
        snapshot = service.pin() if service else None
        return {"status": "ok", "retrieval_methods": sorted(snapshot.engines if snapshot else app.state.engines),
                "knowledge_version": snapshot.version if snapshot else "baseline-static",
                "knowledge": service.status() if service else None}

    @app.post("/retrieve", response_model=RetrievalResponse)
    def retrieve_route(body: QueryRequest, request: Request):
        hits, elapsed, version = retrieve(body)
        annotate(outcome='retrieved')
        return {"request_id": request.state.request_id, "method": body.method,
                "knowledge_version": version, "hits": hits, "retrieval_ms": elapsed}

    @app.post("/ask", response_model=AnswerResponse)
    def ask_route(body: QueryRequest, request: Request):
        hits, elapsed, version = retrieve(body)
        try:
            with phase('generation_and_validation'):
                answer = generator(body.question, hits)
        except GenerationError as exc:
            message = str(exc)
            status = 504 if "timed out" in message else 503 if "not configured" in message else 502
            raise HTTPException(status, message) from None
        for evidence in answer.get("evidence", []):
            evidence["knowledge_version"] = version
        insufficient = answer.get('insufficient_evidence') is True
        annotate(outcome='insufficient_evidence' if insufficient else 'answered_unreviewed',
                 refusal_reason='generator_reports_insufficient' if insufficient else None,
                 evidence_ids=[str(e['passage_id']) for e in answer.get('evidence', [])],
                 evidence_sha256=[digest(e['content']) for e in answer.get('evidence', [])])
        return {"request_id": request.state.request_id, "method": body.method,
                "retrieval_ms": elapsed, **answer, "knowledge_version": version}

    @app.post("/ask-comparison")
    def compare_route(body: ComparisonRequest, request: Request):
        from app.comparison import ComparisonRetriever
        from app.comparison_generation import bounded_answer, generate_comparison
        from app.embedding_api import EmbeddingError
        # Pin once. All subqueries, revision and generation use this same engine.
        engine, version = resolve_engine(body)
        if not hasattr(engine, "retriever") or not hasattr(engine.retriever, "bm25"):
            raise HTTPException(503, "Comparison requires the hybrid rerank engine.")
        started = time.perf_counter()
        retriever = ComparisonRetriever(engine.retriever, engine.reranker,
                                        engine.retriever.bm25.documents, engine.candidate_k)
        plan = retriever.planner.plan(body.question)
        try:
            if plan is None:
                with phase('retrieval'):
                    hits = engine.search(body.question, body.top_k, body.source_id)
                with phase('generation_and_validation'):
                    result = generator(body.question, hits)
                result["trace"] = {"policy": "standard-fallback",
                    "reason": "No unambiguous two-title comparison in the question.",
                    "final_ids": [str(h["id"]) for h in hits]}
            else:
                with phase('comparison_with_recovery'):
                    result = bounded_answer(retriever, plan, top_k=body.top_k,
                        source_id=body.source_id, recover=body.recover,
                        generator=comparison_generator or generate_comparison)
        except EmbeddingError as exc:
            raise HTTPException(502, str(exc)) from None
        except GenerationError as exc:
            message = str(exc)
            raise HTTPException(504 if "timed out" in message else 502, message) from None
        result["trace"].update({"request_id": request.state.request_id, "knowledge_version": version,
                                "total_ms": (time.perf_counter() - started) * 1000})
        for evidence in result.get("evidence", []):
            evidence["knowledge_version"] = version
        insufficient = result.get('insufficient_evidence') is True
        annotate(outcome='insufficient_evidence' if insufficient else 'answered_unreviewed',
                 refusal_reason='comparison_reports_insufficient' if insufficient else None,
                 evidence_ids=[str(e['passage_id']) for e in result.get('evidence', [])],
                 evidence_sha256=[digest(e['content']) for e in result.get('evidence', [])])
        logger.info("request_id=%s comparison_policy=%s version=%s recovery=%s elapsed_ms=%.1f",
            request.state.request_id, result["trace"]["policy"], version,
            result["trace"].get("recovery_triggered", False), result["trace"]["total_ms"])
        return {**result, "request_id": request.state.request_id, "knowledge_version": version}

    return app


app = create_app()
