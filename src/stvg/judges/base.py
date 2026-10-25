"""Text-only LLMs used to grade open-ended answers.

The judge is deliberately generic: it maps chat messages to a reply. The grading rubric (system
prompt, template, score parsing) belongs to the benchmark that uses it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from omegaconf import DictConfig

Messages = list[dict[str, str]]


class LLMJudge(ABC):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    @property
    def name(self) -> str:
        return self.cfg.name

    @abstractmethod
    def complete(self, batch: Sequence[Messages]) -> list[str]:
        """Return one reply per conversation."""

    def close(self) -> None:  # noqa: B027 - optional hook
        pass
