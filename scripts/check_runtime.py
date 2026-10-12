"""Report available packages without inspecting credentials or user configuration."""
import importlib.util
import json
import sys

print(json.dumps({
    "version": sys.version,
    "executable": sys.executable,
    "packages": {name: importlib.util.find_spec(name) is not None for name in (
        "pip", "venv", "ensurepip", "numpy", "scipy", "torch", "transformers",
        "jieba", "bm25s", "fastapi", "pytest", "sentence_transformers",
        "onnxruntime", "tokenizers", "huggingface_hub",
    )},
}, ensure_ascii=False, indent=2))
