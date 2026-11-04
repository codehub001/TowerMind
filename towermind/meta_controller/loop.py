"""Segmented PPO training loop with LLM meta-controller review and checkpoint boundaries."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ai_gym_td.config import GameConfig
from ai_gym_td.env import TowerDefenseEnv
from ai_gym_td.legality import ActionLegalityTracker, repair_action
from ai_gym_td.ppo import TowerDefensePPOAgent, batch_obs
from towermind.experiments.manifest import (
    ExperimentConfig,
    ExperimentManifest,
    SegmentRecord,
)
from towermind.llm.gateway import LLMGateway
from towermind.meta_controller.advisor import build_advisor_prompt
from towermind.meta_controller.evaluator import evaluate_policy
from towermind.meta_controller.promotion import PromotionGate
from towermind.rewards.profile import get_baseline_profile
from towermind.rewards.registry import RewardProfileRegistry, get_default_registry

logger = logging.getLogger(__name__)


def train_ppo_segment(
    agent: TowerDefensePPOAgent,
    optimizer: optim.Optimizer,
    env: TowerDefenseEnv,
    timesteps: int,
    num_steps: int = 256,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    update_epochs: int = 4,
    num_minibatches: int = 4,
    device: Optional[torch.device] = None,
    repair_actions: bool = True,
) -> Dict[str, Any]:
    """Train the agent for a bounded number of environment timesteps."""
    dev = device if device is not None else torch.device("cpu")
    obs_space = env.observation_space
    grid_shape = obs_space["grid"].shape
    glob_shape = obs_space["global"].shape
    nvec = env.action_space.nvec

    obs_grid = torch.zeros((num_steps, *grid_shape), device=dev)
    obs_glob = torch.zeros((num_steps, *glob_shape), device=dev)
    actions = torch.zeros((num_steps, 3), dtype=torch.int64, device=dev)
    logprobs = torch.zeros(num_steps, device=dev)
    rewards = torch.zeros(num_steps, device=dev)
    dones = torch.zeros(num_steps, device=dev)
    values = torch.zeros(num_steps, device=dev)
    masks_t = torch.zeros((num_steps, nvec[0]), dtype=torch.int8, device=dev)
    masks_y = torch.zeros((num_steps, nvec[1]), dtype=torch.int8, device=dev)
    masks_x = torch.zeros((num_steps, nvec[2]), dtype=torch.int8, device=dev)

    tracker = ActionLegalityTracker()
    obs, info = env.reset()
    next_obs = obs
    next_done = False

    num_updates = max(1, timesteps // num_steps)
    total_steps_executed = 0

    for _update in range(1, num_updates + 1):
        for step in range(num_steps):
            total_steps_executed += 1
            batched_obs, batched_masks = batch_obs([next_obs])
            batched_obs = {k: v.to(dev) for k, v in batched_obs.items()}
            batched_masks = [m.to(dev) for m in batched_masks]

            with torch.no_grad():
                act, lg, _, val = agent.get_action_and_value(batched_obs, action_mask=batched_masks)
            raw_a = act[0].cpu().numpy()

            if repair_actions:
                a, was_repaired = repair_action(raw_a, next_obs, fallback_strategy="pass")
                tracker.record(raw_a, next_obs, repaired=was_repaired)
            else:
                a = raw_a
                tracker.record(raw_a, next_obs, repaired=False)

            obs_grid[step] = torch.from_numpy(next_obs["grid"])
            obs_glob[step] = torch.from_numpy(next_obs["global"])
            actions[step] = torch.from_numpy(raw_a)
            logprobs[step] = lg[0]
            values[step] = val[0]
            masks_t[step] = batched_masks[0][0]
            masks_y[step] = batched_masks[1][0]
            masks_x[step] = batched_masks[2][0]

            obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            rewards[step] = float(reward)
            dones[step] = float(done)
            next_obs = obs
            next_done = done

            if done:
                obs, info = env.reset()
                next_obs = obs

        # GAE Advantage Estimation
        with torch.no_grad():
            next_obs_batched, _ = batch_obs([next_obs])
            next_obs_batched = {k: v.to(dev) for k, v in next_obs_batched.items()}
            next_value = agent.get_value(next_obs_batched)
            advantages = torch.zeros_like(rewards, device=dev)
            lastgaelam = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - float(next_done)
                    nextvalues = next_value[0]
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        # PPO Minibatch Updates
        b_inds = np.arange(num_steps)
        batch_size = num_steps
        minibatch_size = batch_size // num_minibatches
        for _ in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                mb_obs = {"grid": obs_grid[mb_inds], "global": obs_glob[mb_inds]}
                mb_masks = [
                    masks_t[mb_inds],
                    masks_y[mb_inds],
                    masks_x[mb_inds],
                ]
                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    mb_obs,
                    action=actions[mb_inds],
                    action_mask=mb_masks,
                )
                logratio = new_logprob - logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * ((new_value - returns[mb_inds]) ** 2).mean()
                e_loss = entropy.mean()
                loss = pg_loss + vf_coef * v_loss - ent_coef * e_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

    return {
        "steps": total_steps_executed,
        "legality_rate": tracker.legality_rate,
        "repair_rate": tracker.repair_rate,
        "legality_summary": tracker.summary(),
    }


def run_segmented_experiment(
    config: ExperimentConfig,
    gateway: LLMGateway,
    registry: Optional[RewardProfileRegistry] = None,
    promotion_gate: Optional[PromotionGate] = None,
    checkpoint_dir: Optional[str] = None,
    device_str: str = "cpu",
) -> ExperimentManifest:
    """Execute end-to-end segmented PPO training with LLM review checkpoints."""
    reg = registry or get_default_registry()
    gate = promotion_gate or PromotionGate()
    device = torch.device(device_str)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    manifest = ExperimentManifest(
        experiment_id=config.experiment_id,
        config=config,
    )

    # Initialize environment and agent
    active_profile = reg.get(config.reward_profile_id) or get_baseline_profile()
    env = TowerDefenseEnv(config=GameConfig(rewards=active_profile.to_reward_config()))
    env.reset(seed=config.seed)

    agent = TowerDefensePPOAgent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    # Initial baseline evaluation
    baseline_metrics = evaluate_policy(
        agent=agent,
        episodes=config.eval_episodes,
        seed=config.seed + 1000,
        reward_profile=active_profile,
    )
    manifest.baseline_metrics = baseline_metrics

    num_segments = max(1, config.total_timesteps // config.segment_timesteps)
    current_metrics = baseline_metrics
    timesteps_so_far = 0

    for seg_idx in range(1, num_segments + 1):
        # 1. Ask LLM Advisor for guidance based on current metrics
        sys_prompt, user_payload, schema = build_advisor_prompt(
            current_profile=active_profile,
            metrics=current_metrics,
            segment_idx=seg_idx,
            total_segments=num_segments,
            history=[s.to_dict() for s in manifest.segments],
        )

        proposal_res = gateway.propose_reward_profile(
            system_prompt=sys_prompt,
            user_payload=user_payload,
            schema=schema,
            fallback_profile=active_profile,
        )

        decision = "continued"
        candidate_profile = proposal_res.profile
        proposed_id = candidate_profile.profile_id

        if proposal_res.is_fallback:
            decision = "fallback"
            logger.info(
                "Segment %d: Fallback to profile %s (%s)",
                seg_idx,
                active_profile.profile_id,
                proposal_res.fallback_reason,
            )
        else:
            # 2. Evaluate candidate with PromotionGate
            prom_res = gate.evaluate(
                current_profile=active_profile,
                candidate_profile=candidate_profile,
                current_metrics=current_metrics,
            )

            if prom_res.promoted:
                reg.register(candidate_profile, set_active=True)
                active_profile = candidate_profile
                decision = "promoted"
                # Re-create environment with promoted reward config
                env.close()
                env = TowerDefenseEnv(config=GameConfig(rewards=active_profile.to_reward_config()))
                logger.info("Segment %d: Promoted candidate %s", seg_idx, proposed_id)
            else:
                decision = "rejected"
                logger.info("Segment %d: Rejected candidate %s (%s)", seg_idx, proposed_id, prom_res.reason)

        # 3. Train segment
        train_ppo_segment(
            agent=agent,
            optimizer=optimizer,
            env=env,
            timesteps=config.segment_timesteps,
            num_steps=config.num_steps,
            gamma=config.gamma,
            device=device,
        )
        timesteps_so_far += config.segment_timesteps

        # 4. Save segment checkpoint
        ckpt_path = None
        if checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, f"ckpt_seg_{seg_idx}.pt")
            torch.save(agent.state_dict(), ckpt_path)
            manifest.checkpoint_paths.append(ckpt_path)

        # 5. Evaluate current policy
        current_metrics = evaluate_policy(
            agent=agent,
            episodes=config.eval_episodes,
            seed=config.seed + 1000 + seg_idx * 10,
            reward_profile=active_profile,
        )

        # 6. Record segment record
        record = SegmentRecord(
            segment_index=seg_idx,
            timesteps_so_far=timesteps_so_far,
            active_profile_id=active_profile.profile_id,
            proposed_profile_id=proposed_id,
            decision=decision,
            metrics=current_metrics,
            advisor_rationale=candidate_profile.rationale,
            advisor_hypothesis=candidate_profile.hypothesis,
        )
        manifest.segments.append(record)

    env.close()
    manifest.final_metrics = current_metrics
    manifest.telemetry = gateway.telemetry_summary()

    if checkpoint_dir:
        manifest.save_json(os.path.join(checkpoint_dir, "manifest.json"))

    return manifest
