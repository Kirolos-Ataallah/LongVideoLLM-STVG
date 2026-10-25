"""The interface every evaluation benchmark implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from stvg.types import Sample, VideoQuery

if TYPE_CHECKING:
    from stvg.judges.base import LLMJudge


class Benchmark(ABC):
    """A benchmark turns annotations into model queries and model answers into metrics.

    Lifecycle::

        bench.prepare()                      # download / extract (idempotent)
        samples = bench.load_samples()
        queries = [q for s in samples for q in bench.build_queries(s)]
        ...                                  # runner calls the model, stores raw predictions
        metrics = bench.evaluate(samples, predictions, judge, out_dir)

    ``predictions`` maps ``qid`` to the stored record (``text``, ``raw_text``, ``meta``, ``error``)
    so that scoring is decoupled from inference and can be re-run without the model.
    """

    #: Task names this benchmark emits, in the order they are queried for a sample.
    tasks: tuple[str, ...] = ()
    #: Whether :meth:`evaluate` needs an LLM judge.
    requires_judge: bool = False

    def __init__(self, cfg: DictConfig, data_root: str | Path) -> None:
        self.cfg = cfg
        self.root = Path(data_root) / cfg.get("data_dir", cfg.name)

    @property
    def name(self) -> str:
        return self.cfg.name

    def prepare(self, force: bool = False) -> None:  # noqa: B027 - optional hook
        """Download and unpack the benchmark under ``self.root``."""

    @abstractmethod
    def load_samples(self) -> list[Sample]: ...

    @abstractmethod
    def build_queries(self, sample: Sample) -> list[VideoQuery]: ...

    @abstractmethod
    def evaluate(
        self,
        samples: list[Sample],
        predictions: dict[str, dict[str, Any]],
        judge: LLMJudge | None,
        out_dir: Path,
    ) -> dict[str, Any]:
        """Score predictions, write per-sample details under ``out_dir``, return metrics."""

    def summary_row(self, metrics: dict[str, Any]) -> dict[str, float]:
        """Headline numbers for cross-run comparison tables (``stvg report``)."""
        return {}

    @staticmethod
    def qid(sample_id: str, task: str) -> str:
        return f"{sample_id}/{task}"
