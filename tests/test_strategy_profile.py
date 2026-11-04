"""Unit tests for StrategyProfile and StrategyAdapter."""

from ai_gym_td.env import TowerDefenseEnv
from ai_gym_td.legality import is_action_legal
from towermind.strategy.adapter import StrategicAgent
from towermind.strategy.profile import (
    StrategyProfile,
    validate_strategy_profile_dict,
)


def test_valid_strategy_profile():
    data = {
        "profile_id": "strat_001",
        "tower_priority": ["ice", "archer", "cannon", "tesla"],
        "spending_targets": {"archer": 0.35, "cannon": 0.25, "ice": 0.25, "tesla": 0.15},
        "risk_mode": "conservative",
        "focus": "path_chokepoints",
        "coverage_priority": 0.8,
        "rationale": "Slow fast scouts at chokepoints.",
        "hypothesis": "Reduce leaks on wave 4.",
    }
    res = validate_strategy_profile_dict(data)
    assert res.is_valid
    assert res.profile is not None
    assert res.profile.risk_mode == "conservative"
    assert res.profile.focus == "path_chokepoints"


def test_invalid_spending_targets():
    # Sum exceeds 1.05
    data = {
        "profile_id": "strat_bad_sum",
        "spending_targets": {"archer": 0.8, "cannon": 0.8},
    }
    res = validate_strategy_profile_dict(data)
    assert not res.is_valid
    assert any("sum" in err for err in res.errors)


def test_invalid_tower_name():
    data = {
        "profile_id": "strat_bad_tower",
        "tower_priority": ["laser_cannon"],
    }
    res = validate_strategy_profile_dict(data)
    assert not res.is_valid
    assert any("Unknown tower" in err for err in res.errors)


def test_strategy_adapter_and_agent_legality():
    env = TowerDefenseEnv()
    obs, info = env.reset(seed=42)

    profile = StrategyProfile(
        profile_id="test_agent_strat",
        tower_priority=("ice", "archer", "cannon"),
        risk_mode="aggressive",
        focus="path_entrance",
    )
    agent = StrategicAgent(profile)

    # Agent acts for 10 steps, ensuring 100% legal actions
    for _ in range(10):
        action = agent.act(obs, info)
        assert is_action_legal(action, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()
