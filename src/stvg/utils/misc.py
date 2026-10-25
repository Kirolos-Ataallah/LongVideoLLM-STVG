from __future__ import annotations

import random
import re
from collections.abc import Iterator, Sequence

import numpy as np

_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def chunked[T](items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def strip_reasoning(text: str) -> str:
    """Drop a ``<think>...</think>`` trace and return the final answer."""
    parts = _THINK_CLOSE.split(text)
    return parts[-1].strip() if len(parts) > 1 else text.strip()


def parse_shard(spec: str | None) -> tuple[int, int]:
    """``"2/8"`` -> ``(2, 8)``; ``None`` -> ``(0, 1)``."""
    if not spec:
        return 0, 1
    index, total = (int(x) for x in str(spec).split("/"))
    if not 0 <= index < total:
        raise ValueError(f"Invalid shard '{spec}'")
    return index, total
