from datetime import datetime, timezone
from pathlib import Path

from app.config import settings


def append_audit_log(message: str) -> None:
    path = settings.data_dir / "system_actions.log"
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {message}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def append_aux_context_log(message: str) -> None:
    path = Path("IMPLEMENTATION_CONTEXT.md")
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"- {ts} - {message}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
