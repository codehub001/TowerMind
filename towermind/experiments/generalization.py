"""Held-out map generalization benchmark comparing training vs unseen map performance."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from towermind.curriculum.maps import (
    DEFAULT_HELDOUT_MAPS,
    DEFAULT_TRAINING_MAPS,
    get_map_config,
)
from towermind.experiments.manifest import EvaluationMetrics
from towermind.meta_controller.evaluator import evaluate_policy
from towermind.rewards.profile import RewardProfile, get_baseline_profile

logger = logging.getLogger(__name__)


def run_generalization_benchmark(
    agent: Any,
    training_maps: Optional[List[str]] = None,
    heldout_maps: Optional[List[str]] = None,
    episodes_per_map: int = 3,
    base_seed: int = 42,
    reward_profile: Optional[RewardProfile] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate an agent across training maps and unseen held-out maps.

    Computes zero-shot transfer performance, transfer gaps, and per-map breakdowns.
    """
    train_maps = training_maps or DEFAULT_TRAINING_MAPS
    test_maps = heldout_maps or DEFAULT_HELDOUT_MAPS
    profile = reward_profile or get_baseline_profile()

    results_by_map: Dict[str, Dict[str, Any]] = {}

    def eval_map(m_name: str) -> EvaluationMetrics:
        cfg = get_map_config(m_name)
        return evaluate_policy(
            agent=agent,
            episodes=episodes_per_map,
            seed=base_seed,
            reward_profile=profile,
            game_config=cfg,
        )

    # 1. Evaluate training maps
    train_metrics: List[EvaluationMetrics] = []
    for m in train_maps:
        m_res = eval_map(m)
        train_metrics.append(m_res)
        results_by_map[m] = {"type": "training", "metrics": m_res.to_dict()}

    # 2. Evaluate held-out maps
    heldout_metrics: List[EvaluationMetrics] = []
    for m in test_maps:
        m_res = eval_map(m)
        heldout_metrics.append(m_res)
        results_by_map[m] = {"type": "heldout", "metrics": m_res.to_dict()}

    # Aggregate summaries
    mean_train_win = float(np.mean([m.win_rate for m in train_metrics]))
    mean_heldout_win = float(np.mean([m.win_rate for m in heldout_metrics]))
    mean_train_waves = float(np.mean([m.mean_waves for m in train_metrics]))
    mean_heldout_waves = float(np.mean([m.mean_waves for m in heldout_metrics]))

    summary = {
        "training_maps": train_maps,
        "heldout_maps": test_maps,
        "episodes_per_map": episodes_per_map,
        "train_mean_win_rate": round(mean_train_win, 4),
        "heldout_mean_win_rate": round(mean_heldout_win, 4),
        "win_rate_generalization_gap": round(mean_heldout_win - mean_train_win, 4),
        "train_mean_waves": round(mean_train_waves, 2),
        "heldout_mean_waves": round(mean_heldout_waves, 2),
        "waves_generalization_gap": round(mean_heldout_waves - mean_train_waves, 2),
        "per_map_results": results_by_map,
    }

    # Generate Markdown summary
    report_lines = [
        "# TowerMind Map Generalization Benchmark Report",
        "",
        f"**Evaluation Episodes per Map:** {episodes_per_map}  ",
        f"**Base Seed:** {base_seed}  ",
        "",
        "## 1. Summary Transfer Performance",
        "",
        "| Partition | Mean Win Rate | Mean Waves Survived |",
        "|---|---|---|",
        f"| **Training Maps** ({', '.join(train_maps)}) | {mean_train_win * 100:.1f}% | {mean_train_waves:.2f} |",
        f"| **Held-Out Maps** ({', '.join(test_maps)}) | {mean_heldout_win * 100:.1f}% | {mean_heldout_waves:.2f} |",
        f"| **Generalization Transfer Gap** | {(mean_heldout_win - mean_train_win) * 100:+.1f}% | {mean_heldout_waves - mean_train_waves:+.2f} waves |",
        "",
        "## 2. Per-Map Breakdown",
        "",
        "| Map | Type | Win Rate | Mean Waves | Legality Rate |",
        "|---|---|---|---|---|",
    ]

    for m_name, item in results_by_map.items():
        m = item["metrics"]
        report_lines.append(
            f"| `{m_name}` | {item['type']} | {m['win_rate'] * 100:.1f}% | {m['mean_waves']:.2f} | {m['legality_rate'] * 100:.1f}% |"
        )

    report_lines.extend(
        [
            "",
            "---",
            "",
            "> **Methodology:** Unseen held-out maps evaluate whether the agent learned generalized path-coverage heuristics rather than memorizing coordinate patterns on specific training geometries.",
        ]
    )

    report_md = "\n".join(report_lines)

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        with open(out_p / "generalization_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(out_p / "generalization_report.md", "w", encoding="utf-8") as f:
            f.write(report_md)

    return {
        "summary": summary,
        "report_markdown": report_md,
    }
