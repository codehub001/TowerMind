"""Unit tests for RewardProfile, validator, and registry."""

from dataclasses import FrozenInstanceError

import pytest

from ai_gym_td.config import RewardConfig
from towermind.rewards.profile import (
    RewardProfile,
    get_baseline_profile,
)
from towermind.rewards.registry import RewardProfileRegistry
from towermind.rewards.validator import (
    RewardValidationError,
    enforce_valid_profile,
    validate_profile,
    validate_profile_dict,
)


def test_baseline_profile_creation():
    profile = get_baseline_profile("baseline-test")
    assert profile.profile_id == "baseline-test"
    assert profile.kill == 1.0
    assert profile.leak == -10.0
    assert profile.win == 50.0
    assert profile.lose == -50.0
    assert profile.invalid_action == -0.1
    assert profile.schema_version == 1

    # Convert to RewardConfig
    rew_cfg = profile.to_reward_config()
    assert isinstance(rew_cfg, RewardConfig)
    assert rew_cfg.kill == 1.0
    assert rew_cfg.leak == -10.0
    assert rew_cfg.win == 50.0
    assert rew_cfg.lose == -50.0
    assert rew_cfg.invalid_action == -0.1


def test_profile_immutability():
    profile = get_baseline_profile()
    with pytest.raises(FrozenInstanceError):
        profile.kill = 5.0  # Frozen dataclass


def test_valid_profile_validation():
    profile = RewardProfile(
        profile_id="exp-001",
        kill=2.0,
        leak=-15.0,
        win=75.0,
        lose=-75.0,
        invalid_action=-0.5,
        rationale="Penalize leaks more heavily.",
        hypothesis="Should build more towers near the exit.",
        created_by="llm",
    )
    result = validate_profile(profile)
    assert result.is_valid
    assert len(result.errors) == 0
    assert result.profile == profile


def test_validator_rejects_unknown_fields():
    data = {
        "profile_id": "test-unknown",
        "kill": 1.0,
        "leak": -10.0,
        "arbitrary_code": "import os; os.system('echo bad')",
    }
    result = validate_profile_dict(data)
    assert not result.is_valid
    assert any("Disallowed extra fields" in err for err in result.errors)


def test_validator_rejects_out_of_bounds_weights():
    # Kill too large (> 10.0)
    data = {"profile_id": "test-bounds", "kill": 999.0}
    result = validate_profile_dict(data)
    assert not result.is_valid
    assert any("out of permitted range" in err for err in result.errors)

    # Win negative (< 0.0)
    data = {"profile_id": "test-bounds", "win": -10.0}
    result = validate_profile_dict(data)
    assert not result.is_valid
    assert any("win" in err and "out of permitted range" in err for err in result.errors)


def test_validator_rejects_non_finite_values():
    data = {"profile_id": "test-nan", "kill": float("nan")}
    result = validate_profile_dict(data)
    assert not result.is_valid
    assert any("must be finite" in err for err in result.errors)

    data = {"profile_id": "test-inf", "kill": float("inf")}
    result = validate_profile_dict(data)
    assert not result.is_valid
    assert any("must be finite" in err for err in result.errors)


def test_validator_rejects_invalid_profile_id():
    # Spaces or special characters
    data = {"profile_id": "invalid id with spaces!", "kill": 1.0}
    result = validate_profile_dict(data)
    assert not result.is_valid
    assert any("Invalid 'profile_id'" in err for err in result.errors)


def test_enforce_valid_profile_raises():
    with pytest.raises(RewardValidationError):
        enforce_valid_profile({"profile_id": "bad", "kill": 1000.0})


def test_registry_registration_and_retrieval(tmp_path):
    registry = RewardProfileRegistry()
    assert len(registry) == 1
    assert registry.active_profile_id == "baseline-v1"

    p2 = RewardProfile(
        profile_id="candidate-001",
        kill=2.0,
        leak=-12.0,
        parent_profile_id="baseline-v1",
    )
    registry.register(p2, set_active=True)
    assert len(registry) == 2
    assert registry.active_profile_id == "candidate-001"
    assert registry["candidate-001"] == p2

    # Switching active
    registry.set_active("baseline-v1")
    assert registry.active_profile_id == "baseline-v1"

    # History tracking
    history = registry.get_history()
    assert [p.profile_id for p in history] == ["baseline-v1", "candidate-001", "baseline-v1"]

    # Reject duplicate ID with different values
    p2_altered = RewardProfile(profile_id="candidate-001", kill=3.0)
    with pytest.raises(ValueError, match="already registered with different configuration"):
        registry.register(p2_altered)

    # Save and reload
    save_file = tmp_path / "profiles.json"
    registry.save_to_file(save_file)
    assert save_file.exists()

    loaded = RewardProfileRegistry.load_from_file(save_file)
    assert len(loaded) == 2
    assert loaded.active_profile_id == "baseline-v1"
    assert loaded["candidate-001"] == p2
