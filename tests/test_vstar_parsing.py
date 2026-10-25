import pytest

from stvg.benchmarks.vstar.parsing import parse_boxes, parse_judge_score, parse_timestamps


@pytest.mark.parametrize(
    ("text", "official", "robust"),
    [
        ("The event happens from 3.5 to 10 seconds.", [3.5, 10.0], [3.5, 10.0]),
        ("12.0 - 4.0", [12.0, 4.0], [4.0, 12.0]),
        ("00:05 - 00:12", None, [5.0, 12.0]),
        ("Between 2s and 7s in the 60.2 second video", None, [2.0, 7.0]),
        ("no idea", None, None),
        ("", None, None),
    ],
)
def test_parse_timestamps(text, official, robust):
    assert parse_timestamps(text, "official") == official
    assert parse_timestamps(text, "robust") == robust


def test_parse_boxes_markdown_and_shapes():
    text = (
        'Here:\n```json\n{"0": [1, 2, 3, 4], "1": [[5, 6, 7, 8]], "2": ["dog", [1, 1, 2, 2]]}\n```'
    )
    assert parse_boxes(text) == {
        "0": [[1, 2, 3, 4]],
        "1": [[5, 6, 7, 8]],
        "2": [[1, 1, 2, 2]],
    }


def test_parse_boxes_list_of_dicts_and_truncated():
    assert parse_boxes('[{"3": [1, 2, 3, 4]}, {"4": [5, 6, 7, 8]}]') == {
        "3": [[1, 2, 3, 4]],
        "4": [[5, 6, 7, 8]],
    }
    # An unclosed object defeats the reference extractor; the robust mode recovers it.
    truncated = '{"3": [1, 2, 3, 4], "4": [5, 6, 7, 8]'
    assert parse_boxes(truncated, "official") is None
    assert parse_boxes(truncated, "robust") == {"3": [[1, 2, 3, 4]], "4": [[5, 6, 7, 8]]}


def test_parse_boxes_robust_extras():
    assert parse_boxes('{"3.0": {"bbox_2d": [1, 2, 3, 4]}}', "robust") == {"3": [[1, 2, 3, 4]]}
    assert parse_boxes('{"3.0": {"bbox_2d": [1, 2, 3, 4]}}', "official") == {}
    assert parse_boxes("at 5s: [1, 2, 3, 4]", "official") is None
    assert parse_boxes('"5s": [1, 2, 3, 4], "6s": [2, 3, 4, 5]', "robust") == {
        "5": [[1, 2, 3, 4]],
        "6": [[2, 3, 4, 5]],
    }
    # Image-style grounding output has no time keys: kept as-is officially, never matches a frame.
    image_style = '[{"bbox_2d": [1, 2, 3, 4], "label": "dog"}]'
    assert parse_boxes(image_style, "official") == {"bbox_2d": [[1, 2, 3, 4]]}
    assert parse_boxes(image_style, "robust") == {}


def test_parse_judge_score():
    assert parse_judge_score("2") == 2
    assert parse_judge_score(" 3\n") == 3
    assert parse_judge_score("Rating: 2") == -1
    assert parse_judge_score("Rating: 2", "robust") == 2
    assert parse_judge_score(None) == -1
