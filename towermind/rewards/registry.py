"""Versioned store and history for validated RewardProfile objects."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Dict, List, Optional

from towermind.rewards.profile import RewardProfile, get_baseline_profile
from towermind.rewards.validator import enforce_valid_profile


class RewardProfileRegistry:
    """Manages versioned, validated RewardProfile instances and selection history."""

    def __init__(self, baseline: Optional[RewardProfile] = None):
        self._profiles: Dict[str, RewardProfile] = {}
        self._history: List[str] = []

        # Always register initial baseline
        init_baseline = baseline or get_baseline_profile()
        self.register(init_baseline)
        self._active_id: str = init_baseline.profile_id

    @property
    def active_profile(self) -> RewardProfile:
        """The currently active profile."""
        return self._profiles[self._active_id]

    @property
    def active_profile_id(self) -> str:
        """ID of currently active profile."""
        return self._active_id

    def set_active(self, profile_id: str) -> RewardProfile:
        """Switch the active profile to an existing registered profile."""
        if profile_id not in self._profiles:
            raise KeyError(f"Profile '{profile_id}' not found in registry")
        self._active_id = profile_id
        if not self._history or self._history[-1] != profile_id:
            self._history.append(profile_id)
        return self._profiles[profile_id]

    def register(
        self,
        profile_or_dict: RewardProfile | Dict,
        set_active: bool = False,
    ) -> RewardProfile:
        """Validate and register a new profile.

        Raises:
            RewardValidationError: if validation fails.
            ValueError: if profile_id already exists with different parameters.
        """
        validated = enforce_valid_profile(profile_or_dict)
        pid = validated.profile_id

        if pid in self._profiles:
            existing = self._profiles[pid]
            if existing == validated:
                # Idempotent re-registration of exact same profile
                if set_active:
                    self.set_active(pid)
                return existing
            raise ValueError(
                f"Profile ID '{pid}' already registered with different configuration. "
                "Profiles are immutable; use a new profile_id."
            )

        self._profiles[pid] = validated
        self._history.append(pid)
        if set_active:
            self._active_id = pid
        return validated

    def get(self, profile_id: str) -> Optional[RewardProfile]:
        """Lookup profile by ID, returning None if not found."""
        return self._profiles.get(profile_id)

    def __getitem__(self, profile_id: str) -> RewardProfile:
        if profile_id not in self._profiles:
            raise KeyError(f"Profile '{profile_id}' not found in registry")
        return self._profiles[profile_id]

    def __contains__(self, profile_id: str) -> bool:
        return profile_id in self._profiles

    def __len__(self) -> int:
        return len(self._profiles)

    def __iter__(self) -> Iterator[RewardProfile]:
        return iter(self._profiles.values())

    def list_profiles(self) -> List[RewardProfile]:
        """Return list of all registered profiles in registration order."""
        return list(self._profiles.values())

    def get_history(self) -> List[RewardProfile]:
        """Return list of profiles in chronological activation order."""
        return [self._profiles[pid] for pid in self._history]

    def save_to_file(self, path: str | Path) -> None:
        """Export all registered profiles and active state to a JSON file."""
        data = {
            "active_profile_id": self._active_id,
            "history": self._history,
            "profiles": [p.to_dict() for p in self._profiles.values()],
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, path: str | Path) -> RewardProfileRegistry:
        """Load registry state from an exported JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        profiles_data = data.get("profiles", [])
        if not profiles_data:
            return cls()

        baseline = RewardProfile.from_dict(profiles_data[0])
        registry = cls(baseline=baseline)

        for p_data in profiles_data[1:]:
            profile = RewardProfile.from_dict(p_data)
            registry.register(profile)

        if "history" in data:
            registry._history = list(data["history"])

        active_id = data.get("active_profile_id")
        if active_id and active_id in registry:
            registry.set_active(active_id)

        return registry


_DEFAULT_REGISTRY: Optional[RewardProfileRegistry] = None


def get_default_registry() -> RewardProfileRegistry:
    """Get or initialize the global default RewardProfileRegistry."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = RewardProfileRegistry()
    return _DEFAULT_REGISTRY
