"""Data containers shared by models, benchmarks and the evaluation runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Sample:
    """One benchmark item: a video plus its benchmark-specific annotation."""

    sample_id: str
    video_path: str
    annotation: dict[str, Any]


@dataclass
class VideoQuery:
    """A single model call: one video, one prompt.

    ``qid`` is ``"<sample_id>/<task>"`` and is the key under which predictions are stored.
    ``video_kwargs`` lets a benchmark override the model's default frame sampling for this query
    (e.g. a clip range); models are free to ignore keys they do not support.
    """

    qid: str
    sample_id: str
    task: str
    video_path: str
    prompt: str
    system_prompt: str | None = None
    video_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutput:
    """What a model returns for a query.

    ``text`` is the final answer with any reasoning trace removed, ``raw_text`` the full decoded
    generation. ``meta`` carries everything scoring may need to interpret the answer without
    re-loading the model, most importantly the box coordinate convention (``bbox_format``)
    and the resolution of the frames the model actually saw (``input_hw``).
    """

    text: str
    raw_text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
