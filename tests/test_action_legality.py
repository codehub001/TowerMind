"""Tests for action legality, validation, and repair mechanisms."""

import numpy as np

from ai_gym_td.config import TILE_BASE, TILE_GRASS, TILE_PATH, TILE_SPAWN
from ai_gym_td.env import (
    ActionLegalityTracker,
    TowerDefenseEnv,
    get_legal_actions,
    is_action_legal,
    repair_action,
)


def test_pass_action_always_legal():
    env = TowerDefenseEnv()
    obs, info = env.reset(seed=42)

    # Pass action [0, y, x] is legal for any valid grid coordinates
    assert is_action_legal(np.array([0, 0, 0]), obs)
    assert is_action_legal([0, 2, 3], obs)
    assert env.is_action_legal([0, 0, 0])


def test_out_of_bounds_action_illegal():
    env = TowerDefenseEnv()
    obs, info = env.reset(seed=42)

    H, W = env.cfg.height, env.cfg.width
    T = len(env._tower_names)

    assert not is_action_legal([T + 1, 0, 0], obs)
    assert not is_action_legal([-1, 0, 0], obs)
    assert not is_action_legal([1, H, 0], obs)
    assert not is_action_legal([1, 0, W], obs)
    assert not is_action_legal([1, -1, 0], obs)
    assert not is_action_legal([1, 0, -1], obs)
    assert not is_action_legal([1, 0], obs)  # Wrong shape
    assert not is_action_legal("invalid", obs)


def test_cannot_build_on_path_spawn_or_base():
    env = TowerDefenseEnv()
    obs, info = env.reset(seed=42)

    # Locate path, spawn, base cells
    grid = np.array(env.game.grid)
    spawn_cells = np.argwhere(grid == TILE_SPAWN)
    path_cells = np.argwhere(grid == TILE_PATH)
    base_cells = np.argwhere(grid == TILE_BASE)

    assert len(spawn_cells) > 0
    assert len(path_cells) > 0
    assert len(base_cells) > 0

    sy, sx = spawn_cells[0]
    py, px = path_cells[0]
    by, bx = base_cells[0]

    for ti in range(1, len(env._tower_names) + 1):
        assert not is_action_legal([ti, sy, sx], obs)
        assert not is_action_legal([ti, py, px], obs)
        assert not is_action_legal([ti, by, bx], obs)


def test_unaffordable_tower_illegal():
    env = TowerDefenseEnv()
    obs, info = env.reset(seed=42)

    # Deplete gold to 0
    env.game._gold = 0
    obs = env._observe()

    for ti in range(1, len(env._tower_names) + 1):
        for y in range(env.cfg.height):
            for x in range(env.cfg.width):
                assert not is_action_legal([ti, y, x], obs)


def test_occupied_cell_illegal():
    env = TowerDefenseEnv()
    obs, info = env.reset(seed=42)

    # Find a legal grass cell and build an archer
    grid = np.array(env.game.grid)
    grass_cells = np.argwhere(grid == TILE_GRASS)
    assert len(grass_cells) > 0
    y, x = grass_cells[0]

    # Before build: archer is legal
    assert is_action_legal([1, y, x], obs)

    # Step action to place tower
    obs, r, term, trunc, info = env.step([1, y, x])
    assert info["placed_tower"]

    # After build: cell is occupied, another tower cannot be built there
    for ti in range(1, len(env._tower_names) + 1):
        assert not is_action_legal([ti, y, x], obs)


def test_get_legal_actions_enumeration():
    env = TowerDefenseEnv()
    obs, info = env.reset(seed=42)

    legal_actions = get_legal_actions(obs)
    assert len(legal_actions) > 0
    assert legal_actions[0] == (0, 0, 0)  # Pass is first

    for act in legal_actions:
        assert is_action_legal(act, obs)


def test_repair_action():
    env = TowerDefenseEnv()
    obs, info = env.reset(seed=42)

    # 1. Legal action is untouched
    legal_act = np.array([0, 0, 0], dtype=np.int64)
    repaired, was_repaired = repair_action(legal_act, obs)
    assert np.array_equal(repaired, legal_act)
    assert not was_repaired

    # 2. Illegal action with 'pass' fallback
    illegal_act = np.array([99, 99, 99], dtype=np.int64)
    repaired, was_repaired = repair_action(illegal_act, obs, fallback_strategy="pass")
    assert np.array_equal(repaired, [0, 0, 0])
    assert was_repaired

    # 3. Illegal action with 'nearest_legal' fallback
    # Find a path cell
    grid = np.array(env.game.grid)
    path_cells = np.argwhere(grid == TILE_PATH)
    py, px = path_cells[0]

    illegal_build = np.array([1, py, px], dtype=np.int64)
    repaired, was_repaired = repair_action(illegal_build, obs, fallback_strategy="nearest_legal")
    assert was_repaired
    assert is_action_legal(repaired, obs)
    assert repaired[0] == 1  # Kept archer type but moved to legal grass coordinate


def test_action_legality_tracker():
    env = TowerDefenseEnv()
    obs, info = env.reset(seed=42)

    tracker = ActionLegalityTracker()
    assert tracker.legality_rate == 1.0

    # Record legal pass
    assert tracker.record([0, 0, 0], obs)
    # Record illegal action
    assert not tracker.record([99, 99, 99], obs, repaired=True)

    summary = tracker.summary()
    assert summary["sampled"] == 2
    assert summary["legal"] == 1
    assert summary["illegal"] == 1
    assert summary["repaired"] == 1
    assert summary["legality_rate"] == 0.5


def test_engine_can_build_tower():
    env = TowerDefenseEnv()
    env.reset(seed=42)
    game = env.game

    # Grass tile can be built
    grid = np.array(game.grid)
    grass_cells = np.argwhere(grid == TILE_GRASS)
    y, x = grass_cells[0]
    assert game.can_build_tower(x, y, "archer")

    # Path tile cannot be built
    path_cells = np.argwhere(grid == TILE_PATH)
    py, px = path_cells[0]
    assert not game.can_build_tower(px, py, "archer")

    # Invalid tower name cannot be built
    assert not game.can_build_tower(x, y, "non_existent_tower")

    # Out of bounds cannot be built
    assert not game.can_build_tower(-1, 0, "archer")
    assert not game.can_build_tower(0, game.cfg.height, "archer")
