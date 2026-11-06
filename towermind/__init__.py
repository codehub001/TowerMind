"""TowerMind — Two-Timescale LLM Meta-Controller & Reinforcement Learning Suite.

Provides modular components for LLM-assisted RL:
  - Reward profiles & registries: bounded, versioned reward shaping
  - Safe LLM gateway: budget-capped, timeout-resilient advisor calls
  - Meta-controller: periodic policy evaluation and promotion gating
  - Strategy adapter: strategic heuristics & benchmark agents
  - Curriculum manager: multi-map topologies & held-out transfer benchmarks
  - Episode analyzer: counterfactual diagnostics & bottleneck analysis
  - Interactive UI: HTML5 canvas visualizer & dashboard
"""

from __future__ import annotations

from towermind.analysis.counterfactual import EpisodeAnalyzer, EpisodeDiagnostics
from towermind.curriculum.maps import (
    CURRICULUM_TIERS,
    DEFAULT_HELDOUT_MAPS,
    DEFAULT_TRAINING_MAPS,
    MAP_FACTORIES,
    CurriculumManager,
    get_map_config,
)
from towermind.experiments.generalization import run_generalization_benchmark
from towermind.experiments.runner import run_comparison_benchmark
from towermind.llm.base import LLMError, LLMProvider, LLMResponse, LLMUsage
from towermind.llm.gateway import LLMGateway
from towermind.llm.mock_provider import MockLLMProvider
from towermind.llm.providers import create_advisor_provider
from towermind.meta_controller.advisor import build_advisor_prompt
from towermind.meta_controller.evaluator import evaluate_policy
from towermind.meta_controller.loop import run_segmented_experiment
from towermind.meta_controller.promotion import PromotionGate
from towermind.rewards.profile import REWARD_BOUNDS, RewardProfile
from towermind.rewards.registry import RewardProfileRegistry
from towermind.rewards.validator import validate_profile_dict
from towermind.strategy.adapter import StrategicAgent, StrategyAdapter
from towermind.strategy.profile import StrategyProfile, get_default_strategy_profile
from towermind.ui.generator import build_dashboard_html, export_dashboard

__version__ = "0.2.0"

__all__ = [
    "CURRICULUM_TIERS",
    "DEFAULT_HELDOUT_MAPS",
    "DEFAULT_TRAINING_MAPS",
    "MAP_FACTORIES",
    "REWARD_BOUNDS",
    "CurriculumManager",
    "EpisodeAnalyzer",
    "EpisodeDiagnostics",
    "LLMError",
    "LLMGateway",
    "LLMProvider",
    "LLMResponse",
    "LLMUsage",
    "MockLLMProvider",
    "PromotionGate",
    "RewardProfile",
    "RewardProfileRegistry",
    "StrategicAgent",
    "StrategyAdapter",
    "StrategyProfile",
    "__version__",
    "build_advisor_prompt",
    "build_dashboard_html",
    "create_advisor_provider",
    "evaluate_policy",
    "export_dashboard",
    "get_default_strategy_profile",
    "get_map_config",
    "run_comparison_benchmark",
    "run_dashboard_server",
    "run_generalization_benchmark",
    "run_segmented_experiment",
    "validate_profile_dict",
]


def __getattr__(name: str):
    if name == "run_dashboard_server":
        from towermind.ui.server import run_dashboard_server

        return run_dashboard_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
