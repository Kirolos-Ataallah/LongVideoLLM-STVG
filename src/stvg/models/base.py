"""The interface every evaluated Video-LLM implements (baselines and our own models)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from omegaconf import DictConfig

from stvg.metrics.boxes import BoxFormat
from stvg.types import ModelOutput, VideoQuery


class VideoLLM(ABC):
    """A model that answers a text prompt about a video.

    Subclasses implement :meth:`generate`. Everything model-specific (chat template, frame
    sampling, coordinate convention, reasoning traces) stays inside the subclass; benchmarks only
    see :class:`VideoQuery` in and :class:`ModelOutput` out.

    Config keys used by every model (see ``configs/models/*.yaml``):

    - ``name``: unique identifier, used in run names and reports.
    - ``type``: registry key of the implementation.
    - ``bbox_format``: coordinate convention of boxes the model emits (:class:`BoxFormat`).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.bbox_format = BoxFormat(cfg.get("bbox_format", BoxFormat.ABS_ORIGINAL))

    @property
    def name(self) -> str:
        return self.cfg.name

    @abstractmethod
    def generate(self, queries: Sequence[VideoQuery]) -> list[ModelOutput]:
        """Answer a batch of queries. Must return one output per query, in order."""

    def describe(self) -> dict[str, Any]:
        """Metadata stored alongside predictions."""
        return {"name": self.name, "bbox_format": str(self.bbox_format)}

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release accelerator memory (called before judging)."""
