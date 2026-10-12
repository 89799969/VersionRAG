"""Download data-only ONNX weights at a pinned Hub commit, no remote Python code."""
import hashlib
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/models/bge-small-zh-v1.5"
LOCK = ROOT / "models.lock.json"
REPO = "Xenova/bge-small-zh-v1.5"
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "onnx/model_quantized.onnx"]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    with httpx.Client(timeout=httpx.Timeout(90, connect=20), follow_redirects=True, trust_env=False) as client:
        if LOCK.exists():
            lock = json.loads(LOCK.read_text(encoding="utf-8"))
        else:
            response = client.get(f"https://huggingface.co/api/models/{REPO}", params={"blobs": "true"})
            response.raise_for_status()
            metadata = response.json()
            available = {item["rfilename"]: item for item in metadata["siblings"]}
            lock = {"repo": REPO, "revision": metadata["sha"],
                    "base_model": "BAAI/bge-small-zh-v1.5",
                    "base_model_license": "MIT", "format": "ONNX dynamic quantized",
                    "files": {name: {"sha256": available[name].get("lfs", {}).get("sha256")}
                              for name in FILES}}
        DEST.mkdir(parents=True, exist_ok=True)
        for name, expected in lock["files"].items():
            target = DEST / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and expected["sha256"] and digest(target) == expected["sha256"]:
                print(f"Verified cached {name}", flush=True)
                continue
            url = f"https://huggingface.co/{lock['repo']}/resolve/{lock['revision']}/{name}"
            partial = target.with_suffix(target.suffix + ".partial")
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for data in response.iter_bytes():
                        handle.write(data)
            actual = digest(partial)
            if expected["sha256"] and actual != expected["sha256"]:
                raise ValueError(f"Checksum mismatch: {name}")
            partial.replace(target)
            expected["sha256"] = actual
            expected["bytes"] = target.stat().st_size
            print(f"Downloaded {name}: {target.stat().st_size:,} bytes", flush=True)
        LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Pinned embedding model revision: {lock['revision']}")


if __name__ == "__main__":
    try:
        main()
    except (httpx.HTTPError, ValueError) as exc:
        print(f"Embedding download failed: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1)
