"""Unit tests for TowerMind dashboard generator and UI server."""

from towermind.ui.generator import build_dashboard_html, export_dashboard


def test_build_dashboard_html():
    sample_comp = {
        "baseline": {"final_metrics": {"win_rate": 0.2, "mean_waves": 4.5, "mean_reward": 15.0}},
        "guided": {
            "final_metrics": {"win_rate": 0.4, "mean_waves": 6.5, "mean_reward": 25.0},
            "segments": [
                {
                    "segment_index": 1,
                    "timesteps_so_far": 256,
                    "active_profile_id": "c1",
                    "decision": "promoted",
                }
            ],
        },
    }
    sample_gen = {
        "train_mean_win_rate": 0.3,
        "heldout_mean_win_rate": 0.25,
        "training_maps": ["s_curve"],
        "heldout_maps": ["spiral"],
    }

    html = build_dashboard_html(sample_comp, sample_gen)
    assert "<!DOCTYPE html>" in html
    assert "TowerMind" in html
    assert "gameCanvas" in html
    assert "Live Visualizer" in html
    assert "Benchmark Comparison" in html
    assert "Meta-Controller Review" in html


def test_export_dashboard(tmp_path):
    out_file = tmp_path / "dashboard_test.html"
    res = export_dashboard(output_path=out_file)
    assert res.exists()
    with open(res, encoding="utf-8") as f:
        content = f.read()
    assert '<canvas id="gameCanvas"' in content
