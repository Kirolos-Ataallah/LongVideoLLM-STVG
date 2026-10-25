"""Reference metric code copied from the official V-STaR ``eval.py``
(https://huggingface.co/datasets/V-STaR-Bench/V-STaR) and only trimmed of I/O.
Used exclusively as a parity oracle in tests; do not import from library code.
"""

import ast

import numpy as np


def calculate_temporal_iou(gt_range, pred_range):
    if not pred_range:
        return 0.0
    if isinstance(pred_range, str):
        try:
            pred_range = ast.literal_eval(pred_range)
        except (ValueError, SyntaxError):
            return 0.0
    if (
        not isinstance(pred_range, list | tuple)
        or len(pred_range) != 2
        or not all(isinstance(x, int | float) for x in pred_range)
    ):
        return 0.0
    gt_start, gt_end = gt_range
    pred_start, pred_end = pred_range
    intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
    union = max(gt_end, pred_end) - min(gt_start, pred_start)
    return intersection / union if union > 0 else 0.0


def compute_iou(gt_bbox, pred_bbox):
    if not isinstance(pred_bbox, list | tuple) or len(pred_bbox) != 4:
        return 0.0
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = (
        gt_bbox["xmin"],
        gt_bbox["ymin"],
        gt_bbox["xmax"],
        gt_bbox["ymax"],
    )
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_bbox
    x1, y1 = max(gt_xmin, pred_xmin), max(gt_ymin, pred_ymin)
    x2, y2 = min(gt_xmax, pred_xmax), min(gt_ymax, pred_ymax)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    gt_area = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    union = gt_area + pred_area - intersection
    return intersection / union if union > 0 else 0.0


def calculate_bbox_iou(gt_bbox, pred_bboxes):
    try:
        if not pred_bboxes:
            return 0.0
        if isinstance(pred_bboxes[0], int | float) and len(pred_bboxes) == 4:
            pred_bboxes = [pred_bboxes]
        return max([compute_iou(gt_bbox, pred_bbox) for pred_bbox in pred_bboxes])
    except Exception:
        return 0.0


def calculate_spatial_metrics(gt_bboxes, pred_bboxes):
    if not pred_bboxes:
        return [0.0] * 5, 0.0
    iou_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    ious = []
    for gt_bbox_data in gt_bboxes:
        frame_id = str(gt_bbox_data["timestamp"])
        if frame_id in pred_bboxes:
            gt_bbox = {k: gt_bbox_data[k] for k in ("xmin", "ymin", "xmax", "ymax")}
            ious.append(calculate_bbox_iou(gt_bbox, pred_bboxes[frame_id]))
        else:
            ious.append(0.0)
    miou = np.mean(ious) if ious else 0.0
    aps = [float(np.mean([iou >= t for iou in ious])) if ious else 0.0 for t in iou_thresholds]
    return aps, miou
