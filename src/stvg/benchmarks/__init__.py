"""Evaluation benchmarks. Importing this package registers all built-in benchmarks."""

from stvg.benchmarks import vstar  # noqa: F401
from stvg.benchmarks.base import Benchmark

__all__ = ["Benchmark"]
