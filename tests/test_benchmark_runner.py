"""Unit and integration tests for benchmark runner and report generation."""

import json
import os

from towermind.experiments.manifest import ExperimentConfig, ExperimentManifest
from towermind.experiments.report import generate_markdown_report
from towermind.experiments.runner import run_comparison_benchmark
from towermind.llm.mock_provider import MockLLMProvider


def test_markdown_report_formatting():
    from towermind.experiments.manifest import EvaluationMetrics, SegmentRecord

    base_m = EvaluationMetrics(
        episodes=5,
        win_rate=0.2,
        mean_waves=4.5,
        mean_lives=8.0,
        mean_reward=15.0,
        mean_steps=80.0,
        wins=1,
        losses=4,
        legality_rate=0.99,
    )
    guid_m = EvaluationMetrics(
        episodes=5,
        win_rate=0.4,
        mean_waves=6.5,
        mean_lives=12.0,
        mean_reward=25.0,
        mean_steps=110.0,
        wins=2,
        losses=3,
        legality_rate=0.99,
    )

    base_man = ExperimentManifest(
        experiment_id="base_test",
        config=ExperimentConfig(experiment_id="base_test", total_timesteps=500),
        final_metrics=base_m,
    )
    guid_man = ExperimentManifest(
        experiment_id="guid_test",
        config=ExperimentConfig(experiment_id="guid_test", total_timesteps=500),
        final_metrics=guid_m,
        segments=[
            SegmentRecord(
                segment_index=1,
                timesteps_so_far=500,
                active_profile_id="cand_1",
                proposed_profile_id="cand_1",
                decision="promoted",
                advisor_rationale="Penalize leaks",
                advisor_hypothesis="Fewer leaks",
            )
        ],
        telemetry={"total_requests": 1, "tokens": {"total_tokens": 100, "estimated_cost_usd": 0.0002}},
    )

    md = generate_markdown_report(base_man, guid_man)
    assert "# TowerMind MVP Experiment Comparison Report" in md
    assert "Win Rate" in md
    assert "20.0%" in md
    assert "40.0%" in md
    assert "+20.0%" in md  # Delta
    assert "PROMOTED" in md
    assert "Penalize leaks" in md


def test_run_comparison_benchmark(tmp_path):
    out_dir = tmp_path / "bench_out"
    provider = MockLLMProvider()

    res = run_comparison_benchmark(
        seed=42,
        total_timesteps=512,  # Short for testing: 2 segments of 256
        segment_timesteps=256,
        eval_episodes=2,
        num_steps=128,
        output_dir=str(out_dir),
        provider=provider,
    )

    assert "baseline_manifest" in res
    assert "guided_manifest" in res
    assert os.path.exists(res["report_path"])
    assert os.path.exists(res["summary_path"])

    # Verify summary JSON content
    with open(res["summary_path"], encoding="utf-8") as f:
        data = json.load(f)
    assert data["seed"] == 42
    assert "baseline" in data
    assert "guided" in data

    # Verify markdown report
    with open(res["report_path"], encoding="utf-8") as f:
        md_text = f.read()
    assert "TowerMind MVP Controlled Benchmark" in md_text
    assert "Metric Performance Comparison" in md_text
