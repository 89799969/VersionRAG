"""Add the missing local provider field, preserving all existing secret values."""
import json
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / ".env"
values = dotenv_values(path)
if "SILICONFLOW_API_KEY" not in values:
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\nSILICONFLOW_API_KEY=\n")
print(json.dumps({"siliconflow_key_configured": bool(values.get("SILICONFLOW_API_KEY")),
                  "deepseek_key_configured": bool(values.get("DEEPSEEK_API_KEY"))}))
