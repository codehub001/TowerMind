"""TowerMind experiment management, metrics, manifest recording, and comparison runner."""

from towermind.experiments.manifest import (
    EvaluationMetrics,
    ExperimentConfig,
    ExperimentManifest,
    SegmentRecord,
)
from towermind.experiments.report import (
    generate_markdown_report,
    save_report,
)
from towermind.experiments.runner import (
    run_comparison_benchmark,
)

__all__ = [
    "EvaluationMetrics",
    "ExperimentConfig",
    "ExperimentManifest",
    "SegmentRecord",
    "generate_markdown_report",
    "run_comparison_benchmark",
    "save_report",
]
