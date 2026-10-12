"""Launch the existing local application; never install packages or rebuild data."""
from __future__ import annotations

import importlib.util
import json
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORT = 8018
BASE_URL = f"http://{HOST}:{PORT}"


def existing_service() -> bool:
    """Return whether the port is listening; reject unrelated or unready services."""
    with socket.socket() as connection:
        connection.settimeout(1)
        if connection.connect_ex((HOST, PORT)) != 0:
            return False
    # Avoid proxy configuration affecting a request to the user's own machine.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(BASE_URL + "/health", timeout=10) as response:
            health = json.load(response)
        with opener.open(BASE_URL + "/openapi.json", timeout=5) as response:
            schema = json.load(response)
        valid = (
            isinstance(health, dict)
            and health.get("status") == "ok"
            and bool(health.get("knowledge_version"))
            and "bm25" in health.get("retrieval_methods", [])
            and schema.get("info", {}).get("title") == "中文招生知识问答与检索评测"
        )
        if not valid:
            raise ValueError("unexpected service")
    except (OSError, urllib.error.URLError, ValueError, AttributeError, TypeError) as exc:
        raise RuntimeError(
            "8018端口已被占用，但无法确认是就绪的VersionRAG服务。"
            "请检查已有窗口；不要重复启动。"
        ) from exc
    print("VersionRAG已经运行，本次没有重复启动。", flush=True)
    print("停止服务请回到原来的启动窗口按 Ctrl+C。", flush=True)
    print(f"问答页面：{BASE_URL}/", flush=True)
    print(f"资料与版本：{BASE_URL}/manage", flush=True)
    return True


def main() -> int:
    try:
        if existing_service():
            return 0
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    required = ("uvicorn", "fastapi", "numpy", "scipy", "jieba", "bm25s", "dotenv")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        print("项目环境缺少依赖：" + "、".join(missing), file=sys.stderr)
        print("请按README恢复环境；启动脚本不会自动安装依赖。", file=sys.stderr)
        return 1
    if not (ROOT / ".env").is_file():
        print("项目目录缺少.env，请按README在本机配置模型服务。", file=sys.stderr)
        return 1

    # The application resolves .env, corpora and indexes relative to ROOT itself.
    sys.path.insert(0, str(ROOT))
    import uvicorn

    print("正在启动VersionRAG；现有依赖、资料和索引保持原样。", flush=True)
    print(f"启动完成后访问：{BASE_URL}/", flush=True)
    print(f"资料与版本：{BASE_URL}/manage", flush=True)
    print("保持这个终端打开；停止服务按 Ctrl+C。", flush=True)
    try:
        uvicorn.run("app.api:app", host=HOST, port=PORT)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
