"""Unit tests for meta-controller advisor prompts, evaluation, and promotion gate."""

from ai_gym_td.agents import GreedyAgent
from towermind.experiments.manifest import EvaluationMetrics
from towermind.meta_controller.advisor import (
    build_advisor_prompt,
)
from towermind.meta_controller.evaluator import evaluate_policy
from towermind.meta_controller.promotion import PromotionGate
from towermind.rewards.profile import RewardProfile, get_baseline_profile


def test_build_advisor_prompt():
    profile = get_baseline_profile("baseline-v1")
    metrics = EvaluationMetrics(
        episodes=5,
        win_rate=0.4,
        mean_waves=7.2,
        mean_lives=12.0,
        mean_reward=28.5,
        mean_steps=120.0,
        wins=2,
        losses=3,
        legality_rate=0.98,
        repair_rate=0.02,
    )

    sys_prompt, payload, schema = build_advisor_prompt(
        current_profile=profile,
        metrics=metrics,
        segment_idx=1,
        total_segments=3,
    )

    assert "reinforcement-learning experiment advisor" in sys_prompt
    assert payload["training_progress"]["completed_segment"] == 1
    assert payload["training_progress"]["total_segments"] == 3
    assert payload["measured_evaluation_metrics"]["win_rate"] == 0.4
    assert schema["type"] == "object"
    assert "kill" in schema["properties"]
    assert "leak" in schema["properties"]


def test_promotion_gate_rules():
    gate = PromotionGate(min_legality_rate=0.90, max_allowed_win_rate_drop=0.10)
    current_p = get_baseline_profile("base")
    cand_p = RewardProfile(profile_id="cand-1", kill=2.0, leak=-15.0)

    healthy_metrics = EvaluationMetrics(
        episodes=5,
        win_rate=0.4,
        mean_waves=6.0,
        mean_lives=10.0,
        mean_reward=20.0,
        mean_steps=100.0,
        wins=2,
        losses=3,
        legality_rate=0.99,
    )

    # 1. Identical profile is not promoted (no-op)
    res_same = gate.evaluate(current_p, current_p, healthy_metrics)
    assert not res_same.promoted
    assert "identical" in res_same.reason

    # 2. Unhealthy legality rate rejects candidate to stabilize policy
    unhealthy_metrics = EvaluationMetrics(
        episodes=5,
        win_rate=0.4,
        mean_waves=6.0,
        mean_lives=10.0,
        mean_reward=20.0,
        mean_steps=100.0,
        wins=2,
        losses=3,
        legality_rate=0.75,
    )
    res_illegal = gate.evaluate(current_p, cand_p, unhealthy_metrics)
    assert not res_illegal.promoted
    assert "below safe threshold" in res_illegal.reason

    # 3. Pilot evaluation with catastrophic win drop is rejected
    pilot_bad = EvaluationMetrics(
        episodes=5,
        win_rate=0.1,
        mean_waves=3.0,
        mean_lives=2.0,
        mean_reward=5.0,
        mean_steps=50.0,
        wins=0,
        losses=5,
        legality_rate=0.99,
    )
    res_bad_pilot = gate.evaluate(current_p, cand_p, healthy_metrics, pilot_metrics=pilot_bad)
    assert not res_bad_pilot.promoted
    assert "dropped more than allowed tolerance" in res_bad_pilot.reason

    # 4. Pilot evaluation with improvement is promoted
    pilot_good = EvaluationMetrics(
        episodes=5,
        win_rate=0.6,
        mean_waves=8.0,
        mean_lives=15.0,
        mean_reward=35.0,
        mean_steps=140.0,
        wins=3,
        losses=2,
        legality_rate=0.99,
    )
    res_good_pilot = gate.evaluate(current_p, cand_p, healthy_metrics, pilot_metrics=pilot_good)
    assert res_good_pilot.promoted
    assert "showed improvement" in res_good_pilot.reason


def test_evaluator_with_greedy_agent():
    agent = GreedyAgent()
    metrics = evaluate_policy(agent, episodes=2, seed=42, max_steps=500)

    assert metrics.episodes == 2
    assert 0.0 <= metrics.win_rate <= 1.0
    assert metrics.mean_waves >= 0.0
    assert metrics.legality_rate == 1.0
    assert "legality_summary" in metrics.extra
