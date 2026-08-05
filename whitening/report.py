"""Render a list of Checks as JSON and as a human-readable text report."""

from __future__ import annotations

import json
from pathlib import Path

from .checks import STATUS_MARK, Check, worst


def to_json(checks: list[Check], meta: dict) -> dict:
    return {
        "meta": meta,
        "overall_status": worst(checks),
        "checks": [c.to_dict() for c in checks],
    }


def to_text(checks: list[Check], meta: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("WHITENING PRE-FLIGHT REPORT")
    lines.append("=" * 78)
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    lines.append("-" * 78)

    for c in checks:
        lines.append(f"[{STATUS_MARK[c.status]:4s}] {c.name}")
        lines.append(f"       {c.message}")
        lines.append("")

    overall = worst(checks)
    lines.append("-" * 78)
    lines.append(f"OVERALL: {STATUS_MARK[overall]}")
    if overall in {"warn", "fail"}:
        flagged = [c.name for c in checks if c.status in {"warn", "fail"}]
        lines.append("Flagged checks: " + ", ".join(flagged))
    lines.append("=" * 78)
    return "\n".join(lines)


def write_report(checks: list[Check], meta: dict, out_dir: str | Path, stem: str = "whitening_report") -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    txt_path = out_dir / f"{stem}.txt"

    json_path.write_text(json.dumps(to_json(checks, meta), indent=2))
    txt_path.write_text(to_text(checks, meta))

    return {"json": json_path, "txt": txt_path}
