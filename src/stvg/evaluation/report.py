"""Human-readable reports: one run (``report.md``) and cross-run comparison tables."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from stvg.utils.io import read_json


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{100 * v:.2f}" if abs(v) <= 1.0 else f"{v:.3f}"
    return str(v)


def _table(header: list[str], rows: Iterable[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key + "."))
        else:
            flat[key] = v
    return flat


def render_metrics(metrics: dict[str, Any]) -> str:
    """Markdown report for a single run. Ratios are shown in percent; LGM values as-is."""
    meta = metrics.get("meta", {})
    parts = [f"# {meta.get('model', '?')} on {meta.get('benchmark', '?')}", ""]
    if metrics.get("summary"):
        s = metrics["summary"]
        parts += ["## Summary", "", _table(list(s), [list(s.values())]), ""]
    overall = metrics.get("overall")
    if overall:
        parts += ["## Overall", "", _table(["metric", "value"], _flatten(overall).items()), ""]
    for group in ("by_duration", "by_domain"):
        if metrics.get(group):
            flat = {name: _flatten(m) for name, m in metrics[group].items()}
            keys = [k for k in next(iter(flat.values())) if k != "num_samples"]
            parts += [f"## {group.replace('_', ' ').title()}", ""]
            parts.append(
                _table(
                    ["metric", *flat],
                    ([k, *(f.get(k, "") for f in flat.values())] for k in ["num_samples", *keys]),
                )
            )
            parts.append("")
    parts += ["## Config", "", "```", str(metrics.get("config", {})), "```", ""]
    return "\n".join(parts)


def compare_runs(run_dirs: Iterable[Path]) -> str:
    """Markdown table of the ``summary`` row of every run that has ``metrics.json``."""
    rows, header = [], ["run", "model", "benchmark"]
    for d in sorted(run_dirs):
        f = Path(d) / "metrics.json"
        if not f.exists():
            continue
        m = read_json(f)
        summary = m.get("summary", {})
        for k in summary:
            if k not in header:
                header.append(k)
        rows.append((m.get("meta", {}), summary))
    body = [
        [
            meta.get("run"),
            meta.get("model"),
            meta.get("benchmark"),
            *(s.get(k, "") for k in header[3:]),
        ]
        for meta, s in rows
    ]
    return _table(header, body) if body else "No runs with metrics.json found."
