"""Deterministic strategy adapter and StrategicAgent guided by StrategyProfile."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ai_gym_td.agents import Agent
from ai_gym_td.config import TOWERS
from ai_gym_td.legality import extract_mask, get_legal_actions, is_action_legal, repair_action
from towermind.strategy.profile import StrategyProfile, get_default_strategy_profile

TOWER_NAME_TO_INDEX: Dict[str, int] = {
    "archer": 1,
    "cannon": 2,
    "ice": 3,
    "tesla": 4,
}
INDEX_TO_TOWER_NAME: Dict[int, str] = {v: k for k, v in TOWER_NAME_TO_INDEX.items()}


class StrategyAdapter:
    """Interprets a StrategyProfile to rank and score legal candidate placements."""

    def __init__(self, profile: Optional[StrategyProfile] = None):
        self.profile = profile or get_default_strategy_profile()

    def select_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> np.ndarray:
        """Select the highest-ranked legal placement action conforming to the StrategyProfile."""
        mask = extract_mask(obs)
        legal_actions = get_legal_actions(mask)

        # If only pass is available, pass
        if len(legal_actions) <= 1:
            return np.array([0, 0, 0], dtype=np.int64)

        gold = float(info.get("gold", 0))
        wave = int(info.get("wave", 0))

        # Check cash reserve based on risk mode
        cash_reserve = 0.0
        if self.profile.risk_mode == "conservative":
            cash_reserve = 30.0
        elif self.profile.risk_mode == "balanced" and wave <= 3:
            cash_reserve = 15.0

        # Current grid spatial features
        # Grid channels: 0=grass, 1=path, 2=spawn, 3=base, 4=blocked, 5=tower_type, 6=tower_cooldown, 7=path_dist
        grid = obs["grid"]
        path_dist = grid[..., 7]
        H, W = path_dist.shape

        # Evaluate candidate placements
        best_action: Optional[Tuple[int, int, int]] = None
        best_score = -float("inf")

        for action in legal_actions:
            t, y, x = action
            if t == 0:
                continue

            tower_name = INDEX_TO_TOWER_NAME.get(t)
            if not tower_name:
                continue

            spec = TOWERS.get(tower_name)
            if not spec:
                continue

            # Respect cash reserve
            if (gold - spec.cost) < cash_reserve:
                continue

            score = 0.0

            # 1. Tower priority weighting
            if tower_name in self.profile.tower_priority:
                priority_rank = self.profile.tower_priority.index(tower_name)
                # Lower index in priority list = higher score
                score += (len(self.profile.tower_priority) - priority_rank) * 20.0

            # 2. Spending target preference
            target_prop = self.profile.spending_targets.get(tower_name, 0.25)
            score += target_prop * 30.0

            # 3. Spatial Focus weighting
            cell_dist = path_dist[y, x]  # normalized distance to path [0, 1]
            # Closer to path is generally better for range
            dist_score = (1.0 - cell_dist) * 25.0
            score += dist_score

            if self.profile.focus == "path_entrance":
                # Prefer left/top cells near spawn (low x / low y)
                norm_progress = (x + y) / (W + H)
                score += (1.0 - norm_progress) * 20.0
            elif self.profile.focus == "path_exit":
                # Prefer right/bottom cells near base
                norm_progress = (x + y) / (W + H)
                score += norm_progress * 20.0
            elif self.profile.focus == "path_chokepoints":
                # Cells with very low path_dist (adjacent to multiple path tiles)
                if cell_dist < 0.15:
                    score += 25.0
            elif self.profile.focus == "balanced_coverage":
                score += 10.0

            if score > best_score:
                best_score = score
                best_action = (t, y, x)

        if best_action is not None and is_action_legal(best_action, mask):
            return np.array(best_action, dtype=np.int64)

        # Fallback to pass
        return np.array([0, 0, 0], dtype=np.int64)


class StrategicAgent(Agent):
    """Autonomous agent executing a deterministic StrategyProfile."""

    def __init__(self, profile: Optional[StrategyProfile] = None):
        self.adapter = StrategyAdapter(profile)

    def set_profile(self, profile: StrategyProfile) -> None:
        self.adapter = StrategyAdapter(profile)

    def act(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> np.ndarray:
        raw_act = self.adapter.select_action(obs, info)
        clean_act, _ = repair_action(raw_act, obs, fallback_strategy="pass")
        return clean_act
