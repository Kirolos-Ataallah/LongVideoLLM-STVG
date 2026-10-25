import random

import pytest

from stvg.benchmarks.vstar.scoring import SPATIAL_THRESHOLDS, aggregate
from stvg.metrics import (
    BoxFormat,
    box_iou,
    per_frame_ious,
    recall_at,
    temporal_iou,
    to_pixel_box,
)
from tests import reference_vstar as ref


def test_temporal_iou_basic():
    assert temporal_iou([0, 10], [5, 15]) == pytest.approx(5 / 15)
    assert temporal_iou([0, 10], [0, 10]) == 1.0
    assert temporal_iou([0, 10], [20, 30]) == 0.0
    assert temporal_iou([0, 10], None) == 0.0
    assert temporal_iou([0, 10], [1]) == 0.0


def test_box_iou_basic():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


@pytest.mark.parametrize(
    ("fmt", "box", "input_hw", "expected"),
    [
        (BoxFormat.ABS_ORIGINAL, [10, 20, 30, 40], None, [10, 20, 30, 40]),
        (BoxFormat.REL_1000, [500, 500, 1000, 1000], None, [320, 180, 640, 360]),
        (BoxFormat.REL_1, [0.5, 0.5, 1, 1], None, [320, 180, 640, 360]),
        (BoxFormat.ABS_INPUT, [160, 90, 320, 180], (180, 320), [320, 180, 640, 360]),
    ],
)
def test_to_pixel_box(fmt, box, input_hw, expected):
    assert to_pixel_box(box, fmt, 640, 360, input_hw) == pytest.approx(expected)


def test_parity_with_reference_random():
    rng = random.Random(0)
    for _ in range(500):
        gt = sorted(rng.uniform(0, 100) for _ in range(2))
        pred = sorted(rng.uniform(0, 100) for _ in range(2))
        assert temporal_iou(gt, pred) == pytest.approx(ref.calculate_temporal_iou(gt, pred))

        def rand_box():
            x1, y1 = rng.uniform(0, 600), rng.uniform(0, 300)
            return [x1, y1, x1 + rng.uniform(1, 200), y1 + rng.uniform(1, 200)]

        gt_list = [
            dict(zip(("xmin", "ymin", "xmax", "ymax"), rand_box(), strict=True), timestamp=t)
            for t in range(rng.randint(1, 8))
        ]
        pred = {str(t): rand_box() for t in range(10) if rng.random() < 0.6}
        aps_ref, miou_ref = ref.calculate_spatial_metrics(gt_list, pred)

        gt = {str(b["timestamp"]): [b["xmin"], b["ymin"], b["xmax"], b["ymax"]] for b in gt_list}
        ious = per_frame_ious(gt, {k: [v] for k, v in pred.items()})
        miou = sum(ious) / len(ious) if pred else 0.0
        aps = recall_at(ious, SPATIAL_THRESHOLDS) if pred else dict.fromkeys(range(5), 0.0)
        assert miou == pytest.approx(miou_ref)
        assert list(aps.values()) == pytest.approx(aps_ref)


def _record(score, t1, v1, t2, v2):
    ap = dict.fromkeys(("0.1", "0.3", "0.5", "0.7", "0.9"), 0.0)
    return {
        "vqa_score": score,
        "tiou_1": t1,
        "viou_1": v1,
        "tiou_2": t2,
        "viou_2": v2,
        "ap_1": ap,
        "ap_2": ap,
    }


def test_aggregate_am_lgm():
    records = [_record(3, 0.5, 0.2, 0.4, 0.1), _record(1, 0.1, 0.0, 0.2, 0.3)]
    m = aggregate(records, has_vqa=True)
    assert m["vqa"]["accuracy"] == 0.5
    assert m["chain_1"]["temporal"]["m_tIoU"] == pytest.approx(0.3)
    assert m["chain_1"]["AM"] == pytest.approx((0.5 + 0.3 + 0.1) / 3)
    assert m["chain_1"]["combined"]["vqa&temporal&spatial"] == 0.5
    assert m["mAM"] == pytest.approx((m["chain_1"]["AM"] + m["chain_2"]["AM"]) / 2)
    assert m["chain_1"]["LGM"] > 0
    assert "vqa" not in aggregate(records, has_vqa=False)
