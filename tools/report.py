from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def save_report(report: str) -> dict:
    """Write a nonempty Markdown report and return its file path."""
    if not isinstance(report, str) or not report.strip():
        raise ValueError("Report cannot be empty")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS_DIR / f"research-{timestamp}-{uuid4().hex[:8]}.md"
    path.write_text(report.rstrip() + "\n", encoding="utf-8")
    return {"path": str(path)}
