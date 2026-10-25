"""Benchmark-agnostic temporal and spatio-temporal grounding metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from stvg.metrics.boxes import box_iou

Box = Sequence[float]


def temporal_iou(gt: Sequence[float], pred: Sequence[float] | None) -> float:
    """IoU between two ``[start, end]`` segments in seconds. Invalid predictions score 0."""
    if not pred or len(pred) != 2:
        return 0.0
    gs, ge = gt
    ps, pe = pred
    inter = max(0.0, min(ge, pe) - max(gs, ps))
    union = max(ge, pe) - min(gs, ps)
    return inter / union if union > 0 else 0.0


def best_box_iou(gt: Box, preds: Sequence[Box]) -> float:
    """Max IoU over one or more predicted boxes for the same frame."""
    return max((box_iou(gt, p) for p in preds), default=0.0)


def per_frame_ious(
    gt_boxes: Mapping[str, Box], pred_boxes: Mapping[str, Sequence[Box]] | None
) -> list[float]:
    """IoU for every annotated frame key; frames without a prediction score 0.

    Keys identify frames (V-STaR uses whole seconds as strings). Predictions on frames that are
    not annotated are ignored, following the V-STaR protocol.
    """
    pred_boxes = pred_boxes or {}
    return [best_box_iou(gt, pred_boxes.get(key, [])) for key, gt in gt_boxes.items()]


def recall_at(values: Sequence[float], thresholds: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {f"{t:g}": 0.0 for t in thresholds}
    return {f"{t:g}": float((arr >= t).mean()) for t in thresholds}
