"""V-STaR metrics (Cheng et al., 2025), aggregated as in the reference ``eval.py``.

Per sample we compute the VQA judge score, temporal IoU and spatial (per-frame) IoU for both
reasoning chains:

- Chain 1 (what -> when -> where): ``temporal`` then ``spatial`` given GT time span.
- Chain 2 (what -> where -> when): ``spatial_2`` without time, ``temporal_2`` given GT boxes.

Group metrics then follow the paper: Accuracy (judge score >= 2), R1@IoU, m_tIoU, AP@IoU
(fraction of annotated frames above the threshold), m_vIoU, AM / LGM per chain and their means
mAM / mLGM, plus the "combined" success rates used in the paper's analysis.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

TEMPORAL_THRESHOLDS = (0.3, 0.5, 0.7)
SPATIAL_THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)
# Thresholds for the "combined" success analysis in the reference implementation.
VQA_CORRECT, TIOU_OK, VIOU_OK = 2, 0.3, 0.1
_EPS = 1e-6


def duration_bucket(seconds: float) -> str:
    if seconds < 60:
        return "short"
    if seconds < 180:
        return "medium"
    return "long"


def _mean(values: Iterable[float]) -> float:
    arr = list(values)
    return float(np.mean(arr)) if arr else 0.0


def _rate(values: Iterable[float], thr: float) -> float:
    arr = list(values)
    return float(np.mean([v >= thr for v in arr])) if arr else 0.0


def _lgm(*values: float) -> float:
    """Modified log geometric mean: -mean(log(1 - x))."""
    return -sum(math.log(max(1.0 - v, _EPS)) for v in values) / len(values)


def aggregate(records: list[dict[str, Any]], has_vqa: bool) -> dict[str, Any]:
    """Aggregate per-sample records into the V-STaR metric suite."""
    n = len(records)
    out: dict[str, Any] = {"num_samples": n}
    if n == 0:
        return out

    if has_vqa:
        scores = [r["vqa_score"] for r in records]
        valid = [s for s in scores if s >= 0]
        out["vqa"] = {
            "accuracy": _mean(s >= VQA_CORRECT for s in scores),
            "avg_score": _mean(max(s, 0) for s in scores),
            "avg_valid_score": _mean(valid),
            "judge_parse_failures": n - len(valid),
        }

    for chain in (1, 2):
        tious = [r[f"tiou_{chain}"] for r in records]
        vious = [r[f"viou_{chain}"] for r in records]
        aps = {
            f"{t:g}": _mean(r[f"ap_{chain}"][f"{t:g}"] for r in records) for t in SPATIAL_THRESHOLDS
        }
        block: dict[str, Any] = {
            "temporal": {
                **{f"R1@{t:g}": _rate(tious, t) for t in TEMPORAL_THRESHOLDS},
                "m_tIoU": _mean(tious),
            },
            "spatial": {**{f"AP@{k}": v for k, v in aps.items()}, "m_vIoU": _mean(vious)},
        }
        t_ok = [r[f"tiou_{chain}"] >= TIOU_OK for r in records]
        s_ok = [r[f"viou_{chain}"] >= VIOU_OK for r in records]
        combined = {"temporal&spatial": _mean(a and b for a, b in zip(t_ok, s_ok, strict=True))}
        if has_vqa:
            acc = out["vqa"]["accuracy"]
            q_ok = [r["vqa_score"] >= VQA_CORRECT for r in records]
            m_t, m_v = block["temporal"]["m_tIoU"], block["spatial"]["m_vIoU"]
            block["AM"] = (acc + m_t + m_v) / 3
            block["LGM"] = _lgm(acc, m_t, m_v)
            combined.update(
                {
                    "vqa&temporal": _mean(a and b for a, b in zip(q_ok, t_ok, strict=True)),
                    "vqa&spatial": _mean(a and b for a, b in zip(q_ok, s_ok, strict=True)),
                    "vqa&temporal&spatial": _mean(
                        a and b and c for a, b, c in zip(q_ok, t_ok, s_ok, strict=True)
                    ),
                }
            )
        block["combined"] = combined
        out[f"chain_{chain}"] = block

    if has_vqa:
        out["mAM"] = (out["chain_1"]["AM"] + out["chain_2"]["AM"]) / 2
        out["mLGM"] = (out["chain_1"]["LGM"] + out["chain_2"]["LGM"]) / 2
    return out


def aggregate_groups(records: list[dict[str, Any]], has_vqa: bool) -> dict[str, Any]:
    """Overall metrics plus breakdowns by video duration and domain."""
    by_duration: dict[str, list] = defaultdict(list)
    by_domain: dict[str, list] = defaultdict(list)
    for r in records:
        by_duration[r["duration"]].append(r)
        by_domain[r["domain"]].append(r)
    return {
        "overall": aggregate(records, has_vqa),
        "by_duration": {
            k: aggregate(by_duration[k], has_vqa)
            for k in ("short", "medium", "long")
            if k in by_duration
        },
        "by_domain": {k: aggregate(v, has_vqa) for k, v in sorted(by_domain.items())},
    }
