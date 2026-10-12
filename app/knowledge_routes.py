"""Local demo management API. Bind to loopback; no user authentication/ACL here."""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.knowledge import KnowledgeError


class DocumentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str = Field(min_length=1, max_length=100, pattern=r"^[\w.-]+$")
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=200_000)
    url: str = Field(default="", max_length=2000)

    @field_validator("document_id", "title", "text")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class BuildInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    auto_publish: bool = True


class ActivateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = Field(min_length=1, max_length=81, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


def management_router(service):
    router = APIRouter(prefix="/knowledge", tags=["Knowledge lifecycle"])

    def call(operation, *args):
        try:
            return operation(*args)
        except KnowledgeError as exc:
            raise HTTPException(409, str(exc)) from None
        except (ValueError, OSError, KeyError):
            raise HTTPException(409, "Version validation failed; active version was kept.") from None

    @router.get("/status")
    def status():
        return service.status()

    @router.get("/documents")
    def documents(limit: int = Query(default=50, ge=1, le=200), offset: int = Query(default=0, ge=0)):
        return service.documents(limit, offset)

    @router.put("/documents")
    def upsert(body: DocumentInput):
        return call(service.upsert, body.document_id, body.title, body.text, body.url)

    @router.delete("/documents/{document_id}")
    def delete(document_id: str):
        return call(service.delete, document_id)

    @router.post("/jobs", status_code=202)
    def submit(body: BuildInput):
        try:
            return service.submit(body.auto_publish)
        except KnowledgeError as exc:
            raise HTTPException(429, str(exc)) from None

    @router.get("/jobs/{job_id}")
    def job(job_id: str):
        try:
            return service.job(job_id)
        except KnowledgeError as exc:
            raise HTTPException(404, str(exc)) from None

    @router.get("/versions")
    def versions():
        return service.versions()

    @router.post("/activate")
    def activate(body: ActivateInput):
        return call(service.activate, body.version)

    return router
