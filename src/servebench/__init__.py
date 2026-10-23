"""servebench: load-test any OpenAI-compatible LLM endpoint."""

__version__ = "1.1.1"

from .api import benchmark, check_regression, compare
from .check import CheckReport, CheckThresholds
from .results import LevelResult, RequestResult, RunResult, Stats, Summary
from .roofline import RooflineAnalysis, RooflinePoint, compute_roofline, render_ascii_roofline
from .top import TopState, run_top_monitor
from .tune import TuningRecommendation, TuningReport, evaluate_tuning

__all__ = [
    "benchmark",
    "check_regression",
    "compare",
    "compute_roofline",
    "evaluate_tuning",
    "render_ascii_roofline",
    "run_top_monitor",
    "CheckThresholds",
    "CheckReport",
    "RooflineAnalysis",
    "RooflinePoint",
    "TuningRecommendation",
    "TuningReport",
    "TopState",
    "RunResult",
    "LevelResult",
    "RequestResult",
    "Summary",
    "Stats",
    "__version__",
]
