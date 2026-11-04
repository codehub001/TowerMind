"""TowerMind curriculum learning and map management."""

from towermind.curriculum.maps import (
    CURRICULUM_TIERS,
    DEFAULT_HELDOUT_MAPS,
    DEFAULT_TRAINING_MAPS,
    MAP_FACTORIES,
    CurriculumManager,
    get_map_config,
)

__all__ = [
    "CURRICULUM_TIERS",
    "DEFAULT_HELDOUT_MAPS",
    "DEFAULT_TRAINING_MAPS",
    "MAP_FACTORIES",
    "CurriculumManager",
    "get_map_config",
]
