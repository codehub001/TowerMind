"""Curriculum manager and multi-map registry for training and generalization benchmarks."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from ai_gym_td.config import GameConfig

MAP_FACTORIES: Dict[str, Callable[[], GameConfig]] = {
    "s_curve": GameConfig.default_20x12,
    "straight": GameConfig.straight_path,
    "zigzag": GameConfig.zigzag_path,
    "chokepoint": GameConfig.chokepoint_path,
    "spiral": GameConfig.spiral_path,
}

# Curriculum tiers ordered by tactical difficulty
CURRICULUM_TIERS: List[List[str]] = [
    ["straight", "s_curve"],  # Tier 1: Open / standard paths
    ["zigzag"],  # Tier 2: Sharp corners
    ["chokepoint", "spiral"],  # Tier 3: Chokepoints and long winding paths
]

DEFAULT_TRAINING_MAPS: List[str] = ["s_curve", "zigzag"]
DEFAULT_HELDOUT_MAPS: List[str] = ["chokepoint", "spiral"]


def get_map_config(map_name: str) -> GameConfig:
    """Instantiate a fresh GameConfig for the named map."""
    factory = MAP_FACTORIES.get(map_name)
    if factory is None:
        raise KeyError(f"Unknown map '{map_name}'. Available maps: {sorted(MAP_FACTORIES.keys())}")
    return factory()


class CurriculumManager:
    """Manages progressive multi-map curriculum advancement and tier tracking."""

    def __init__(
        self,
        initial_tier: int = 0,
        promotion_win_rate: float = 0.40,
        promotion_waves: float = 7.0,
    ):
        self.current_tier_idx = min(initial_tier, len(CURRICULUM_TIERS) - 1)
        self.promotion_win_rate = promotion_win_rate
        self.promotion_waves = promotion_waves
        self._history: List[Dict[str, Any]] = []

    @property
    def current_maps(self) -> List[str]:
        """Maps in the currently unlocked curriculum tier."""
        return CURRICULUM_TIERS[self.current_tier_idx]

    @property
    def is_final_tier(self) -> bool:
        return self.current_tier_idx >= len(CURRICULUM_TIERS) - 1

    def evaluate_advancement(self, win_rate: float, mean_waves: float) -> bool:
        """Check if metrics qualify for advancing to the next curriculum tier."""
        qualified = win_rate >= self.promotion_win_rate or mean_waves >= self.promotion_waves
        if qualified and not self.is_final_tier:
            self.current_tier_idx += 1
            return True
        return False

    def get_training_and_heldout_sets(
        self,
    ) -> Tuple[List[str], List[str]]:
        """Return standardized training and held-out map sets for generalization."""
        return list(DEFAULT_TRAINING_MAPS), list(DEFAULT_HELDOUT_MAPS)
