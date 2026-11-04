"""Policy evaluation utility producing EvaluationMetrics over fixed seeds."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import torch

from ai_gym_td.config import GameConfig
from ai_gym_td.env import TowerDefenseEnv
from ai_gym_td.legality import ActionLegalityTracker, repair_action
from ai_gym_td.ppo import TowerDefensePPOAgent, batch_obs
from towermind.experiments.manifest import EvaluationMetrics
from towermind.rewards.profile import RewardProfile


def evaluate_policy(
    agent: Any,
    episodes: int = 5,
    seed: int = 42,
    reward_profile: Optional[RewardProfile] = None,
    game_config: Optional[GameConfig] = None,
    max_steps: int = 3000,
) -> EvaluationMetrics:
    """Evaluate an agent across fixed seeds and return standardized EvaluationMetrics."""
    if game_config is not None:
        cfg = game_config
    else:
        cfg = GameConfig()

    if reward_profile is not None:
        cfg = GameConfig(
            width=cfg.width,
            height=cfg.height,
            path=cfg.path,
            blocked_cells=cfg.blocked_cells,
            map_name=cfg.map_name,
            rewards=reward_profile.to_reward_config(),
        )

    env = TowerDefenseEnv(config=cfg, max_env_steps=max_steps)
    tracker = ActionLegalityTracker()

    wins = 0
    losses = 0
    all_waves: List[float] = []
    all_lives: List[float] = []
    all_rewards: List[float] = []
    all_steps: List[int] = []

    is_torch_agent = isinstance(agent, (TowerDefensePPOAgent, torch.nn.Module))

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        if hasattr(agent, "reset"):
            agent.reset()
        ep_reward = 0.0
        steps = 0

        while True:
            if is_torch_agent:
                with torch.no_grad():
                    batched, masks = batch_obs([obs])
                    act, _, _, _ = agent.get_action_and_value(batched, action_mask=masks)
                raw_act = act[0].cpu().numpy()
            else:
                raw_act = agent.act(obs, info)

            # Repair action if needed and record in tracker
            clean_act, was_repaired = repair_action(raw_act, obs, fallback_strategy="pass")
            tracker.record(raw_act, obs, repaired=was_repaired)

            obs, reward, terminated, truncated, info = env.step(clean_act)
            ep_reward += reward
            steps += 1

            if terminated or truncated:
                break

        if env.game.phase == env.game.WON:
            wins += 1
        elif env.game.phase == env.game.LOST:
            losses += 1

        all_waves.append(float(env.game.wave_number))
        all_lives.append(float(env.game.lives))
        all_rewards.append(float(ep_reward))
        all_steps.append(steps)

    env.close()

    win_rate = wins / episodes if episodes > 0 else 0.0
    mean_waves = float(np.mean(all_waves)) if all_waves else 0.0
    mean_lives = float(np.mean(all_lives)) if all_lives else 0.0
    mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
    mean_steps = float(np.mean(all_steps)) if all_steps else 0.0

    return EvaluationMetrics(
        episodes=episodes,
        win_rate=win_rate,
        mean_waves=mean_waves,
        mean_lives=mean_lives,
        mean_reward=mean_reward,
        mean_steps=mean_steps,
        wins=wins,
        losses=losses,
        legality_rate=tracker.legality_rate,
        repair_rate=tracker.repair_rate,
        extra={
            "legality_summary": tracker.summary(),
        },
    )
