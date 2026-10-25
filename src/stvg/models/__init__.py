"""Video-LLM wrappers. Importing this package registers all built-in models."""

from stvg.models import qwen_vl, qwen_vl_vllm  # noqa: F401
from stvg.models.base import VideoLLM

__all__ = ["VideoLLM"]
