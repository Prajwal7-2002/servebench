"""servebench: load-test any OpenAI-compatible LLM endpoint."""

__version__ = "1.0.1"

from .api import benchmark, check_regression, compare
from .check import CheckReport, CheckThresholds
from .results import LevelResult, RequestResult, RunResult, Stats, Summary

__all__ = [
    "benchmark",
    "check_regression",
    "compare",
    "CheckThresholds",
    "CheckReport",
    "RunResult",
    "LevelResult",
    "RequestResult",
    "Summary",
    "Stats",
    "__version__",
]
