"""Log de auditoria: uma linha JSON por análise (entradas, resultado, fontes, texto da LLM)."""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def log_path() -> Path:
    return Path(os.getenv("AUDIT_LOG", ROOT / "logs" / "auditoria.jsonl"))


def log_analysis(record: dict) -> str:
    entry = {"id": str(uuid.uuid4()), "timestamp": datetime.now(timezone.utc).isoformat(), **record}
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return entry["id"]


def read_log() -> list[dict]:
    path = log_path()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in reversed(lines) if line.strip()]
