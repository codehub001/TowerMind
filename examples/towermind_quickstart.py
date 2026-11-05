"""TowerMind Quickstart — Run StrategicAgent with Episode Diagnostics and RewardProfile."""

from __future__ import annotations

from ai_gym_td.env import TowerDefenseEnv
from towermind import (
    EpisodeAnalyzer,
    RewardProfile,
    RewardProfileRegistry,
    StrategicAgent,
    StrategyProfile,
)


def main():
    print("=" * 60)
    print("TowerMind Quickstart: StrategicAgent & Diagnostics")
    print("=" * 60)

    # 1. Setup custom RewardProfile
    registry = RewardProfileRegistry()
    profile = RewardProfile(
        profile_id="strategic-v1",
        kill=2.0,
        leak=-15.0,
        win=60.0,
        lose=-40.0,
        rationale="Reward aggressive clearing and prioritize leak prevention",
    )
    registry.register(profile, set_active=True)

    # 2. Instantiate environment
    env = TowerDefenseEnv()
    env.cfg.rewards = profile.to_reward_config()

    # 3. Instantiate StrategicAgent with a balanced coverage profile
    strategy = StrategyProfile(
        profile_id="balanced-v1",
        focus="balanced_coverage",
        risk_mode="balanced",
    )
    agent = StrategicAgent(strategy)

    # 4. Run an evaluation episode
    obs, info = env.reset(seed=42)
    total_reward = 0.0
    while True:
        action = agent.act(obs, info)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward

        if term or trunc:
            break

    # 5. Extract deterministic counterfactual diagnostics
    analyzer = EpisodeAnalyzer()
    diag = analyzer.analyze(env.game)

    print(f"Outcome: {diag.phase.upper()}")
    print(f"Waves Survived: {diag.final_wave}/{env.cfg.max_waves}")
    print(f"Final Gold: {diag.final_gold}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Path Coverage Ratio: {diag.path_coverage_ratio * 100:.1f}%")
    print(f"Towers Deployed: {sum(diag.tower_counts.values())} ({diag.tower_counts})")
    print(f"Diagnostic Notes: {diag.diagnostic_notes}")

    env.close()


if __name__ == "__main__":
    main()
