"""Immutable RewardProfile data model with schema versioning and bounds.

Every reward shaping profile is represented as a frozen dataclass.
It converts cleanly to/from dictionaries and to the engine's `RewardConfig`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from ai_gym_td.config import RewardConfig

# Authoritative bounds for each reward weight to prevent unbounded scaling
# and training instability.
REWARD_BOUNDS: Mapping[str, Tuple[float, float]] = {
    "kill": (0.0, 10.0),
    "leak": (-50.0, 0.0),
    "win": (0.0, 200.0),
    "lose": (-200.0, 0.0),
    "step": (-1.0, 1.0),
    "invalid_action": (-5.0, 0.0),
    "economy": (-1.0, 1.0),
    "scale": (0.01, 1.0),
}

CURRENT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RewardProfile:
    """Versioned, immutable reward shaping specification."""

    profile_id: str
    kill: float = 1.0
    leak: float = -10.0
    win: float = 50.0
    lose: float = -50.0
    step: float = 0.0
    invalid_action: float = -0.1
    economy: float = 0.0
    scale: float = 0.1
    rationale: str = ""
    hypothesis: str = ""
    created_by: str = "baseline"  # "baseline", "human", or "llm"
    parent_profile_id: Optional[str] = None
    schema_version: int = CURRENT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to serializable JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RewardProfile:
        """Create RewardProfile from dictionary.

        Note: Does not perform validation; use `validate_profile_dict` first.
        """
        valid_fields = {
            "profile_id",
            "kill",
            "leak",
            "win",
            "lose",
            "step",
            "invalid_action",
            "economy",
            "scale",
            "rationale",
            "hypothesis",
            "created_by",
            "parent_profile_id",
            "schema_version",
        }
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_reward_config(self) -> RewardConfig:
        """Export to ai_gym_td RewardConfig for the gym environment."""
        return RewardConfig(
            kill=float(self.kill),
            leak=float(self.leak),
            win=float(self.win),
            lose=float(self.lose),
            step=float(self.step),
            invalid_action=float(self.invalid_action),
            economy=float(self.economy),
            scale=float(self.scale),
        )

    @classmethod
    def from_reward_config(
        cls,
        cfg: RewardConfig,
        profile_id: str = "baseline-v1",
        rationale: str = "Default baseline environment reward configuration.",
        created_by: str = "baseline",
        parent_profile_id: Optional[str] = None,
    ) -> RewardProfile:
        """Construct a RewardProfile from an existing RewardConfig."""
        return cls(
            profile_id=profile_id,
            kill=cfg.kill,
            leak=cfg.leak,
            win=cfg.win,
            lose=cfg.lose,
            step=cfg.step,
            invalid_action=cfg.invalid_action,
            economy=cfg.economy,
            scale=cfg.scale,
            rationale=rationale,
            hypothesis="Baseline benchmark performance.",
            created_by=created_by,
            parent_profile_id=parent_profile_id,
        )


def get_baseline_profile(profile_id: str = "baseline-v1") -> RewardProfile:
    """Return the canonical baseline RewardProfile matching RewardConfig defaults."""
    return RewardProfile.from_reward_config(RewardConfig(), profile_id=profile_id)
