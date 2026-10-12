"""Create a local reviewable source/data/report bundle with an explicit allowlist."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval import ROOT

files = []
for folder in ("app", "scripts", "tests", "data", "artifacts/reports"):
    files.extend(path for path in (ROOT / folder).rglob("*") if path.is_file()
                 and "__pycache__" not in path.parts and path.suffix in {".py", ".html", ".json", ".jsonl", ".jpg"})
files.extend(ROOT.glob("*.md"))
files.extend(ROOT.glob("requirements*.txt"))
files.extend(ROOT / name for name in (".env.example", ".gitignore", "pytest.ini"))
for relative in ("extractive_qa/basic_qa.jsonl", "multi-doc_qa/multidoc_qa.jsonl",
                 "time-sensitive_qa/time_sensitive.jsonl", "noisy_qa/noisy_qa_ver3.jsonl"):
    files.append(ROOT / "DomainRAG/BCM/labeled_data" / relative)
files.append(ROOT / "DomainRAG/README.md")
files = sorted(set(files))
secrets = [value.encode() for key, value in dotenv_values(ROOT / ".env").items()
           if key.endswith("API_KEY") and value]
output = ROOT / "artifacts/deliverables/rag-domain-study-source.zip"
output.parent.mkdir(parents=True, exist_ok=True)
manifest = []
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        assert path.name != ".env" and ".venv" not in path.parts and ".git" not in path.parts
        data = path.read_bytes()
        if any(secret in data for secret in secrets):
            raise ValueError(f"Credential found in allowlisted file: {relative}")
        archive.writestr("rag-domain-study/" + relative, data)
        manifest.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    archive.writestr("rag-domain-study/BUNDLE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
report = {"path": str(output), "files": len(manifest), "bytes": output.stat().st_size,
          "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "credential_scan_passed": True,
          "excludes": [".env", ".venv", "indexes", "knowledge database/version files", "model weights", "upstream git checkouts"]}
(output.parent / "bundle_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
