"""Unit tests for multi-map curriculum and generalization benchmark."""

from ai_gym_td.agents import GreedyAgent
from towermind.curriculum.maps import (
    CurriculumManager,
    get_map_config,
)
from towermind.experiments.generalization import run_generalization_benchmark


def test_map_registry_all_maps():
    map_names = ["s_curve", "straight", "zigzag", "chokepoint", "spiral"]
    for m in map_names:
        cfg = get_map_config(m)
        assert cfg.map_name == m
        grid = cfg.make_grid()
        assert len(grid) == cfg.height
        assert len(grid[0]) == cfg.width


def test_curriculum_advancement():
    mgr = CurriculumManager(initial_tier=0)
    assert mgr.current_tier_idx == 0
    assert "straight" in mgr.current_maps

    # Fails qualification
    assert not mgr.evaluate_advancement(win_rate=0.10, mean_waves=3.0)
    assert mgr.current_tier_idx == 0

    # Qualifies for next tier
    assert mgr.evaluate_advancement(win_rate=0.50, mean_waves=8.0)
    assert mgr.current_tier_idx == 1
    assert "zigzag" in mgr.current_maps


def test_generalization_benchmark(tmp_path):
    agent = GreedyAgent()
    res = run_generalization_benchmark(
        agent=agent,
        training_maps=["straight"],
        heldout_maps=["s_curve"],
        episodes_per_map=1,
        base_seed=42,
        output_dir=str(tmp_path),
    )

    assert "summary" in res
    assert "report_markdown" in res
    assert res["summary"]["train_mean_waves"] >= 0.0
    assert (tmp_path / "generalization_summary.json").exists()
    assert (tmp_path / "generalization_report.md").exists()
