"""Action legality validation, repair mechanisms, and audit tracking.

In Tower Defense, actions are discrete tuples: [tower_type, y, x].
- tower_type == 0: pass (always legal during non-terminal state)
- tower_type > 0: build tower of type tower_type at grid coordinate (x, y)

An action is only legal if:
1. Coordinates (y, x) are within grid bounds [0, H) and [0, W).
2. The tower type index is within [0, T].
3. For tower_type > 0, the player has sufficient gold to afford the tower.
4. The cell (y, x) is buildable terrain (TILE_GRASS) and is not already occupied.
5. In observation masks, mask[tower_type, y, x] == 1.
"""

from typing import Any, Dict, List, Tuple, Union

import numpy as np


def extract_mask(obs_or_mask: Union[Dict[str, np.ndarray], np.ndarray]) -> np.ndarray:
    """Extract action_mask array from either an observation dict or raw array."""
    if isinstance(obs_or_mask, dict):
        if "action_mask" not in obs_or_mask:
            raise KeyError("Observation dictionary must contain 'action_mask' key")
        mask = obs_or_mask["action_mask"]
    else:
        mask = obs_or_mask

    mask_arr = np.asarray(mask, dtype=np.int8)
    if mask_arr.ndim != 3:
        raise ValueError(f"action_mask must be 3-dimensional (T+1, H, W), got shape {mask_arr.shape}")
    return mask_arr


def is_action_legal(
    action: Any,
    obs_or_mask: Union[Dict[str, np.ndarray], np.ndarray],
) -> bool:
    """Check if action [tower_type, y, x] is strictly legal given obs or action_mask.

    Args:
        action: Array-like [tower_type, y, x].
        obs_or_mask: Either Gymnasium obs dict containing 'action_mask' or raw 3D mask.

    Returns:
        True if the action is legal, False otherwise.
    """
    try:
        mask = extract_mask(obs_or_mask)
    except Exception:
        return False

    try:
        action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
        if action_arr.shape[0] != 3:
            return False
        t, y, x = int(action_arr[0]), int(action_arr[1]), int(action_arr[2])
    except (ValueError, TypeError):
        return False

    T_plus_1, H, W = mask.shape
    if not (0 <= t < T_plus_1 and 0 <= y < H and 0 <= x < W):
        return False

    # Pass action (t=0) is always legal during gameplay
    if t == 0:
        return True

    return bool(mask[t, y, x] == 1)


def get_legal_actions(
    obs_or_mask: Union[Dict[str, np.ndarray], np.ndarray],
) -> List[Tuple[int, int, int]]:
    """Enumerate all legal (tower_type, y, x) action tuples in the current state.

    The pass action (0, 0, 0) is always included as the first element.
    """
    mask = extract_mask(obs_or_mask)
    legal_actions = [(0, 0, 0)]
    T_plus_1, H, W = mask.shape

    for t in range(1, T_plus_1):
        ys, xs = np.where(mask[t] == 1)
        for y, x in zip(ys, xs):
            legal_actions.append((int(t), int(y), int(x)))

    return legal_actions


def repair_action(
    action: Any,
    obs_or_mask: Union[Dict[str, np.ndarray], np.ndarray],
    fallback_strategy: str = "pass",
) -> Tuple[np.ndarray, bool]:
    """Validate and repair an action if it is illegal.

    Args:
        action: Array-like [tower_type, y, x].
        obs_or_mask: Observation dict or raw mask.
        fallback_strategy: 'pass' (default) returns [0, 0, 0].
                           'nearest_legal' finds the closest legal (y, x) for that tower type
                           if available, otherwise falls back to pass.

    Returns:
        (repaired_action, was_repaired)
    """
    mask = extract_mask(obs_or_mask)
    if is_action_legal(action, mask):
        return np.asarray(action, dtype=np.int64).reshape(-1), False

    action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
    if action_arr.shape[0] == 3 and fallback_strategy == "nearest_legal":
        t, y, x = int(action_arr[0]), int(action_arr[1]), int(action_arr[2])
        T_plus_1, H, W = mask.shape
        if 1 <= t < T_plus_1 and np.any(mask[t] == 1):
            ys, xs = np.where(mask[t] == 1)
            dists = (ys - y) ** 2 + (xs - x) ** 2
            best_idx = int(np.argmin(dists))
            return np.array([t, ys[best_idx], xs[best_idx]], dtype=np.int64), True

    # Default fallback: pass action
    return np.array([0, 0, 0], dtype=np.int64), True


class ActionLegalityTracker:
    """Tracks action legality statistics across an episode or benchmark evaluation."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.sampled: int = 0
        self.legal: int = 0
        self.illegal: int = 0
        self.repaired: int = 0
        self.passes: int = 0
        self.builds: int = 0
        self.reasons: Dict[str, int] = {
            "out_of_bounds": 0,
            "unaffordable_or_unbuildable": 0,
            "invalid_shape": 0,
        }

    def record(
        self,
        raw_action: Any,
        obs_or_mask: Union[Dict[str, np.ndarray], np.ndarray],
        repaired: bool = False,
    ) -> bool:
        """Record an action and return whether it was originally legal."""
        self.sampled += 1
        if repaired:
            self.repaired += 1

        mask = extract_mask(obs_or_mask)
        try:
            arr = np.asarray(raw_action, dtype=np.int64).reshape(-1)
            if arr.shape[0] != 3:
                self.illegal += 1
                self.reasons["invalid_shape"] += 1
                return False
            t, y, x = int(arr[0]), int(arr[1]), int(arr[2])
        except Exception:
            self.illegal += 1
            self.reasons["invalid_shape"] += 1
            return False

        T_plus_1, H, W = mask.shape
        if not (0 <= t < T_plus_1 and 0 <= y < H and 0 <= x < W):
            self.illegal += 1
            self.reasons["out_of_bounds"] += 1
            return False

        if t == 0:
            self.legal += 1
            self.passes += 1
            return True

        if mask[t, y, x] == 1:
            self.legal += 1
            self.builds += 1
            return True

        self.illegal += 1
        self.reasons["unaffordable_or_unbuildable"] += 1
        return False

    @property
    def legality_rate(self) -> float:
        return self.legal / self.sampled if self.sampled > 0 else 1.0

    @property
    def repair_rate(self) -> float:
        return self.repaired / self.sampled if self.sampled > 0 else 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "sampled": self.sampled,
            "legal": self.legal,
            "illegal": self.illegal,
            "repaired": self.repaired,
            "passes": self.passes,
            "builds": self.builds,
            "legality_rate": round(self.legality_rate, 4),
            "repair_rate": round(self.repair_rate, 4),
            "reasons": dict(self.reasons),
        }
