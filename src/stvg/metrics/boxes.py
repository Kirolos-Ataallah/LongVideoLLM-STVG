"""Bounding-box conventions and conversions.

Video-LLMs disagree on how they express boxes. Every model declares one of the formats below in
its output ``meta["bbox_format"]``; scoring converts raw predictions to absolute ``xyxy`` pixel
coordinates of the original video before comparing with ground truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum


class BoxFormat(StrEnum):
    ABS_ORIGINAL = "abs_original"  # pixels of the original video frame
    ABS_INPUT = "abs_input"  # pixels of the resized frame fed to the model (Qwen2/2.5-VL)
    REL_1000 = "rel_1000"  # normalised to [0, 1000] (Qwen3-VL, Qwen3.5)
    REL_1 = "rel_1"  # normalised to [0, 1]


def to_pixel_box(
    box: Sequence[float],
    fmt: str | BoxFormat,
    width: int,
    height: int,
    input_hw: Sequence[int] | None = None,
) -> list[float]:
    """Convert an ``xyxy`` box in format ``fmt`` to original-frame pixels."""
    x1, y1, x2, y2 = (float(v) for v in box)
    fmt = BoxFormat(fmt)
    if fmt is BoxFormat.ABS_ORIGINAL:
        sx = sy = 1.0
    elif fmt is BoxFormat.ABS_INPUT:
        if input_hw is None:
            raise ValueError("abs_input boxes need the model input resolution (input_hw)")
        sx, sy = width / input_hw[1], height / input_hw[0]
    elif fmt is BoxFormat.REL_1000:
        sx, sy = width / 1000.0, height / 1000.0
    else:
        sx, sy = float(width), float(height)
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two ``xyxy`` boxes (no clipping, matches the V-STaR reference)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
