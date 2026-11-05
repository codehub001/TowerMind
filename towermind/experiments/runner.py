"""Orchestrator for controlled benchmark experiments comparing Baseline PPO vs LLM-Guided PPO."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from towermind.experiments.manifest import ExperimentConfig
from towermind.experiments.report import generate_markdown_report, save_report
from towermind.llm.base import LLMProvider
from towermind.llm.gateway import GatewayConfig, LLMGateway
from towermind.llm.mock_provider import MockLLMProvider
from towermind.meta_controller.loop import run_segmented_experiment
from towermind.rewards.profile import get_baseline_profile
from towermind.rewards.registry import RewardProfileRegistry

logger = logging.getLogger(__name__)


def run_comparison_benchmark(
    seed: int = 42,
    total_timesteps: int = 5000,
    segment_timesteps: int = 2500,
    eval_episodes: int = 5,
    num_steps: int = 256,
    output_dir: str = "benchmark_results",
    provider: Optional[LLMProvider] = None,
    gateway_config: Optional[GatewayConfig] = None,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """Run a controlled benchmark comparing Baseline PPO with LLM-Guided PPO across matched seeds.

    Args:
        seed: Random seed for training and evaluation.
        total_timesteps: Total environment timesteps per condition.
        segment_timesteps: Timesteps between LLM meta-controller review boundaries.
        eval_episodes: Evaluation episodes for measuring aggregate performance.
        num_steps: PPO rollout buffer steps.
        output_dir: Directory where manifests, reports, and checkpoints are stored.
        provider: LLMProvider to use (defaults to MockLLMProvider for offline execution).
        gateway_config: Optional configuration for LLM gateway.
        device_str: Torch device ("cpu" or "cuda").

    Returns:
        Dict containing baseline manifest, guided manifest, report text, and output paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base_ckpt_dir = out_path / "baseline_checkpoints"
    guided_ckpt_dir = out_path / "guided_checkpoints"

    # ---- 1. Condition A: Baseline PPO (Fixed baseline-v1 profile) ----
    logger.info("Starting Condition A: Baseline PPO run...")
    baseline_cfg = ExperimentConfig(
        experiment_id=f"baseline_seed_{seed}",
        seed=seed,
        total_timesteps=total_timesteps,
        segment_timesteps=segment_timesteps,
        eval_episodes=eval_episodes,
        num_steps=num_steps,
        reward_profile_id="baseline-v1",
    )

    # In baseline run, gateway always falls back to baseline-v1 without LLM changes
    mock_base_provider = MockLLMProvider(default_payload=get_baseline_profile("baseline-v1").to_dict())
    baseline_gateway = LLMGateway(mock_base_provider, GatewayConfig(cache_enabled=False))
    baseline_reg = RewardProfileRegistry()

    baseline_manifest = run_segmented_experiment(
        config=baseline_cfg,
        gateway=baseline_gateway,
        registry=baseline_reg,
        checkpoint_dir=str(base_ckpt_dir),
        device_str=device_str,
    )
    baseline_manifest_file = out_path / "baseline_manifest.json"
    baseline_manifest.save_json(baseline_manifest_file)

    # ---- 2. Condition B: LLM-Guided PPO (Adaptive reward profiling) ----
    logger.info("Starting Condition B: LLM-Guided PPO run...")
    guided_cfg = ExperimentConfig(
        experiment_id=f"guided_seed_{seed}",
        seed=seed,
        total_timesteps=total_timesteps,
        segment_timesteps=segment_timesteps,
        eval_episodes=eval_episodes,
        num_steps=num_steps,
        reward_profile_id="baseline-v1",
    )

    active_provider = provider or MockLLMProvider()
    guided_gw_cfg = gateway_config or GatewayConfig(cache_enabled=True)
    guided_gateway = LLMGateway(active_provider, guided_gw_cfg)
    guided_reg = RewardProfileRegistry()

    guided_manifest = run_segmented_experiment(
        config=guided_cfg,
        gateway=guided_gateway,
        registry=guided_reg,
        checkpoint_dir=str(guided_ckpt_dir),
        device_str=device_str,
    )
    guided_manifest_file = out_path / "guided_manifest.json"
    guided_manifest.save_json(guided_manifest_file)

    # ---- 3. Markdown Report & Comparison Summary ----
    report_md = generate_markdown_report(
        baseline_manifest=baseline_manifest,
        guided_manifest=guided_manifest,
        title=f"TowerMind MVP Controlled Benchmark (Seed {seed})",
    )
    report_file = out_path / "comparison_report.md"
    save_report(report_md, report_file)

    summary_file = out_path / "comparison_summary.json"
    summary_data = {
        "seed": seed,
        "total_timesteps": total_timesteps,
        "baseline": baseline_manifest.to_dict(),
        "guided": guided_manifest.to_dict(),
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info("Benchmark complete. Report written to %s", report_file)

    return {
        "baseline_manifest": baseline_manifest,
        "guided_manifest": guided_manifest,
        "report_markdown": report_md,
        "report_path": str(report_file),
        "summary_path": str(summary_file),
    }
