"""Unit tests for counterfactual strategy and episode diagnostics."""

from ai_gym_td.env import TowerDefenseEnv
from towermind.analysis.counterfactual import EpisodeAnalyzer, EpisodeDiagnostics


def test_episode_analyzer():
    env = TowerDefenseEnv()
    obs, info = env.reset(seed=42)

    # Place an archer at a legal grass location
    legal = env.get_legal_actions()
    build_act = next((a for a in legal if a[0] == 1), None)
    assert build_act is not None

    obs, r, term, trunc, info = env.step(build_act)

    analyzer = EpisodeAnalyzer()
    diag = analyzer.analyze(env.game, events_accum=info.get("events", []))

    assert isinstance(diag, EpisodeDiagnostics)
    assert diag.total_path_cells > 0
    assert diag.covered_path_cells > 0
    assert 0.0 < diag.path_coverage_ratio <= 1.0
    assert diag.tower_counts["archer"] == 1
    assert diag.gold_invested_by_tower["archer"] == 25

    d_dict = diag.to_dict()
    assert "path_coverage_ratio" in d_dict
    assert "diagnostic_notes" in d_dict
