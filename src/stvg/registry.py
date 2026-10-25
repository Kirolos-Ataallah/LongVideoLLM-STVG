"""Lightweight component registries.

Every pluggable component (model, benchmark, judge) registers itself under a short key.
Configs then select an implementation with ``type: <key>``::

    @MODELS.register("qwen_vl")
    class QwenVL(VideoLLM): ...

    model = MODELS.build(cfg.model)  # cfg.model.type == "qwen_vl"

Implementation modules must keep heavy imports (torch, transformers, ...) inside functions or
``__init__`` so that importing the registry stays cheap.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, type] = {}

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            if key in self._entries and self._entries[key] is not cls:
                raise KeyError(f"{self.kind} '{key}' is already registered by {self._entries[key]}")
            self._entries[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> type:
        if key not in self._entries:
            raise KeyError(f"Unknown {self.kind} type '{key}'. Registered: {sorted(self._entries)}")
        return self._entries[key]

    def build(self, cfg: Any, **kwargs: Any) -> Any:
        """Instantiate ``cfg.type`` with the full config node as first argument."""
        return self.get(cfg.type)(cfg, **kwargs)

    def keys(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries


MODELS = Registry("model")
BENCHMARKS = Registry("benchmark")
JUDGES = Registry("judge")


def import_builtin_components() -> None:
    """Import the packages whose modules register built-in components."""
    import stvg.benchmarks
    import stvg.judges
    import stvg.models  # noqa: F401
