"""Turn free-form V-STaR answers into structured predictions.

Two modes are provided:

- ``official``: a faithful port of the extraction in the V-STaR reference code
  (``inference_demo.py`` / ``eval.py``), for numbers comparable with the paper.
- ``robust``: a superset that additionally accepts clock times (``01:05``), ranges written as
  ``"3.5 - 10s"``, ``{"bbox_2d": ...}`` objects, non-integer second keys (``"3.0"``, ``"3s"``) and
  judge replies such as ``"Rating: 2"``. Use it to study how much of a model's score is lost to
  formatting rather than grounding.
"""

from __future__ import annotations

import json
import re
from typing import Any

NUMBER = r"\d+(?:\.\d+)?"
_OFFICIAL_NUMBER = re.compile(rf"\b{NUMBER}\b")
_CLOCK = re.compile(r"\b(?:(\d{1,2}):)?(\d{1,2}):(\d{2}(?:\.\d+)?)\b")
_RANGE = re.compile(
    rf"({NUMBER})\s*(?:s|secs?|seconds?)?\s*(?:-|–|—|~|to|and|until|till)\s*({NUMBER})",
    re.IGNORECASE,
)
_MARKDOWN_JSON = re.compile(r"```json\s*\n(\[.*?\]|\{.*?\})\s*\n```", re.DOTALL)
_ANY_JSON = re.compile(r"(\[[\s\S]*\]|\{[\s\S]*\})", re.DOTALL)
_KEY_BOX = re.compile(
    rf"[\"']?({NUMBER})\s*s?[\"']?\s*:\s*\[\s*\[?\s*(-?{NUMBER})\s*,\s*(-?{NUMBER})\s*,"
    rf"\s*(-?{NUMBER})\s*,\s*(-?{NUMBER})"
)

MODES = ("official", "robust")


# ---------------------------------------------------------------------------- temporal
def parse_timestamps(text: str | None, mode: str = "official") -> list[float] | None:
    """Extract a ``[start, end]`` segment in seconds, or ``None``."""
    if not text:
        return None
    if mode == "official":
        nums = _OFFICIAL_NUMBER.findall(text)
        return [float(nums[0]), float(nums[1])] if len(nums) == 2 else None

    text = _CLOCK.sub(
        lambda m: f"{int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + float(m.group(3)):g}",
        text,
    )
    match = _RANGE.search(text)
    if match:
        start, end = float(match.group(1)), float(match.group(2))
    else:
        nums = re.findall(NUMBER, text)
        if len(nums) < 2:
            return None
        start, end = float(nums[0]), float(nums[1])
    return [min(start, end), max(start, end)]


# ----------------------------------------------------------------------------- spatial
def _fix_incomplete_json(s: str) -> str:
    """Balance brackets (reference implementation)."""
    for open_c, close_c in (("[", "]"), ("{", "}")):
        n_open, n_close = s.count(open_c), s.count(close_c)
        if n_open > n_close:
            s += close_c * (n_open - n_close)
        elif n_close > n_open:
            s = open_c * (n_close - n_open) + s
    return s


def _load_box_json(text: str) -> Any:
    match = _MARKDOWN_JSON.search(text) or _ANY_JSON.search(text)
    if not match:
        return None
    payload = match.group(1).strip().replace("'", '"')
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        try:
            return json.loads(_fix_incomplete_json(payload))
        except json.JSONDecodeError:
            return None
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        merged: dict[str, Any] = {}
        for x in obj:
            merged.update(x)
        obj = merged
    if isinstance(obj, list):
        try:
            obj = {str(item[0]): item[1] for item in obj}
        except (IndexError, KeyError, TypeError):
            return None
    return obj


def _is_box(v: Any) -> bool:
    return (
        isinstance(v, list | tuple)
        and len(v) == 4
        and all(isinstance(x, int | float) and not isinstance(x, bool) for x in v)
    )


def coerce_boxes(value: Any, mode: str = "official") -> list[list[float]]:
    """Normalise one frame's predicted value into a list of ``xyxy`` boxes."""
    if _is_box(value):
        return [[float(x) for x in value]]
    if mode == "robust" and isinstance(value, dict):
        for key in ("bbox_2d", "bbox", "box"):
            if key in value:
                return coerce_boxes(value[key], mode)
        return []
    if isinstance(value, list | tuple):
        if len(value) == 1:
            return coerce_boxes(value[0], mode)
        if len(value) == 2 and isinstance(value[1], list | tuple):  # [label, box]
            return coerce_boxes(value[1], mode)
        if value and all(_is_box(v) for v in value):
            return [[float(x) for x in v] for v in value]
    return []


def _normalise_key(key: Any) -> str | None:
    try:
        return str(round(float(str(key).strip().rstrip("sS"))))
    except ValueError:
        return None


def parse_boxes(text: str | None, mode: str = "official") -> dict[str, list[list[float]]] | None:
    """Extract ``{second: [box, ...]}`` in the model's raw coordinate convention."""
    if not text:
        return None
    obj = _load_box_json(text)
    result: dict[str, list[list[float]]] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            k = _normalise_key(key) if mode == "robust" else str(key)
            boxes = coerce_boxes(value, mode)
            if k is not None and boxes:
                result.setdefault(k, []).extend(boxes)
    if mode == "robust" and not result:
        for m in _KEY_BOX.finditer(text):
            k = _normalise_key(m.group(1))
            if k is not None:
                result.setdefault(k, []).append([float(m.group(i)) for i in range(2, 6)])
    return result or (None if obj is None else {})


# ------------------------------------------------------------------------------- judge
def parse_judge_score(text: str | None, mode: str = "official") -> int:
    """0-3 rating from the judge, ``-1`` if unparseable (reference behaviour)."""
    if text is None:
        return -1
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        if mode == "robust":
            m = re.search(r"\b([0-3])\b", text)
            if m:
                return int(m.group(1))
        return -1
