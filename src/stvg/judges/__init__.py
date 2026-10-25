"""LLM judges. Importing this package registers all built-in judges."""

from stvg.judges import hf, openai_compat  # noqa: F401
from stvg.judges.base import LLMJudge

__all__ = ["LLMJudge"]
