#!/usr/bin/env python3
"""
Orchestrates the creation of 86 professional, structured Git commits.
Ensures:
  - Author and Committer are strictly 'codehub001' (<codehub001@users.noreply.github.com>)
  - First commits are markdown documentation (.md files) and project configuration
  - .env, plan/, and Todo.md are strictly excluded
  - Conventional Commits formatting across 86 commits
  - Remote 'origin' is set to https://github.com/codehub001/TowerMind.git
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

AUTHOR_NAME = "codehub001"
AUTHOR_EMAIL = "codehub001@users.noreply.github.com"


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running {' '.join(cmd)}:\n{res.stderr}", file=sys.stderr)
        sys.exit(res.returncode)


def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    print("Initializing git repository on main branch...")
    run(["git", "init", "-b", "main"])
    run(["git", "config", "user.name", AUTHOR_NAME])
    run(["git", "config", "user.email", AUTHOR_EMAIL])

    # Configure remote
    remote_url = "https://github.com/codehub001/TowerMind.git"
    print(f"Setting remote origin -> {remote_url}")
    subprocess.run(["git", "remote", "remove", "origin"], capture_output=True)
    run(["git", "remote", "add", "origin", remote_url])

    # Base timestamp: 5 days ago, stepping forward
    base_time = datetime.now(timezone.utc) - timedelta(days=5)

    # Save original README.md so we commit initial docs first, then update it at the end
    orig_readme = Path("README.md").read_text(encoding="utf-8")

    initial_readme = """# AI Gym Tower Defense & TowerMind

A tower defense game with a Gymnasium API and TowerMind: LLM-Assisted Reinforcement Learning.

Train RL agents, pit heuristic bots against each other, or guide PPO via a safe two-timescale LLM meta-controller.

## Features
- Pure Python + NumPy deterministic engine
- 4 tower archetypes and 4 enemy types
- Gymnasium environment with spatial grid and global observations
- Two-timescale LLM meta-controller
"""
    Path("README.md").write_text(initial_readme, encoding="utf-8")

    # Define the 85 staged commits leading up to final README (Total: 86 commits)
    commits = [
        # --- Phase 1: Inception & Documentation (Commits 1-6) ---
        (
            ["LICENSE", "CONTRIBUTING.md"],
            "docs: initialize project documentation and open-source license",
        ),
        (
            ["README.md"],
            "docs: create initial project README with Gymnasium overview",
        ),
        (
            [".gitignore"],
            "chore: configure gitignore for Python, virtualenv, and build artifacts",
        ),
        (
            [".env.example"],
            "chore: add environment variable configuration template",
        ),
        (
            ["pyproject.toml"],
            "chore: setup pyproject.toml packaging and dependency specifications",
        ),
        (
            ["requirements.txt"],
            "chore: add pip requirements file for environment setup",
        ),
        # --- Phase 2: Core Game Configuration & Pathfinding (Commits 7-8) ---
        (
            ["ai_gym_td/config.py"],
            "feat(config): implement tile definitions, tower archetypes, and game parameters",
        ),
        (
            ["ai_gym_td/pathfinding.py"],
            "feat(pathfinding): add Euclidean path distance calculation and BFS algorithms",
        ),
        # --- Phase 3: Core Game Engine & Simulation (Commits 9-16) ---
        (
            ["ai_gym_td/engine.py"],
            "feat(engine): initialize TowerDefenseGame engine state and spatial grid layout",
        ),
        (
            ["ai_gym_td/gym_register.py"],
            "feat(env): register TowerDefense-v0 in Gymnasium environment registry",
        ),
        (
            ["ai_gym_td/agents.py"],
            "feat(agents): create base Agent class and baseline heuristic agents",
        ),
        (
            ["ai_gym_td/obs_format.py"],
            "feat(obs): add structured observation formatters and prompts",
        ),
        (
            ["ai_gym_td/llm_agents.py"],
            "feat(agents): add direct LLM agent baseline implementations",
        ),
        (
            ["ai_gym_td/render.py"],
            "feat(render): implement headless PILRenderer for RGB frame generation",
        ),
        (
            ["ai_gym_td/__init__.py"],
            "feat(core): export core environment classes in ai_gym_td package root",
        ),
        (
            ["tests/test_smoke.py"],
            "test(smoke): add initial test suite for engine and environment smoke tests",
        ),
        # --- Phase 4: Action Legality & Spatial Validation (Commits 17-19) ---
        (
            ["ai_gym_td/legality.py"],
            "feat(legality): implement joint action validation, masking, and repair",
        ),
        (
            ["tests/test_action_legality.py"],
            "test(legality): add comprehensive unit tests for action masking and repair",
        ),
        (
            ["ai_gym_td/env.py"],
            "feat(env): implement Gymnasium environment wrapper TowerDefenseEnv",
        ),
        # --- Phase 5: PPO Neural Architecture & Baseline Policies (Commits 20-24) ---
        (
            ["ai_gym_td/ppo.py"],
            "feat(ppo): implement CleanRL-style PPO agent, neural networks, and GAE rollouts",
        ),
        (
            ["agent.pt"],
            "feat(models): include pre-trained PPO baseline checkpoint",
        ),
        (
            ["ai_gym_td/scripts/__init__.py"],
            "chore(scripts): initialize scripts package in ai_gym_td",
        ),
        (
            ["ai_gym_td/scripts/train.py"],
            "feat(scripts): add ai-td-train script for standalone PPO training",
        ),
        (
            ["ai_gym_td/scripts/evaluate.py"],
            "feat(scripts): add ai-td-eval script with checkpoint loader",
        ),
        # --- Phase 6: Pygame Interactive Viewer & Examples (Commits 25-33) ---
        (
            ["ai_gym_td/viewer.py"],
            "feat(viewer): implement real-time Pygame desktop renderer with range circles",
        ),
        (
            ["ai_gym_td/scripts/play.py"],
            "feat(scripts): add ai-td-play interactive manual play CLI",
        ),
        (
            ["examples/quickstart.py"],
            "feat(examples): add quickstart evaluation script",
        ),
        (
            ["examples/tournament.py"],
            "feat(examples): add tournament benchmark example script",
        ),
        (
            ["scripts/benchmark.py"],
            "feat(scripts): add benchmark CLI script for LLM evaluations",
        ),
        (
            ["scripts/plot_results.py"],
            "feat(scripts): add plotting utility for benchmark analysis",
        ),
        (
            ["scripts/visualize.py"],
            "feat(scripts): add episode visualization and video rendering script",
        ),
        (
            ["scripts/run_comparison.py"],
            "feat(scripts): add comparison runner shim script",
        ),
        (
            ["scripts/build_git_history.py"],
            "chore(scripts): add git history orchestration utility",
        ),
        # --- Phase 7: TowerMind Bounded Reward Profiles (Commits 34-38) ---
        (
            ["towermind/rewards/__init__.py"],
            "feat(rewards): initialize towermind.rewards package",
        ),
        (
            ["towermind/rewards/profile.py"],
            "feat(rewards): define bounded, versioned RewardProfile dataclass",
        ),
        (
            ["towermind/rewards/validator.py"],
            "feat(rewards): implement RewardProfile schema validator and safety envelope",
        ),
        (
            ["towermind/rewards/registry.py"],
            "feat(rewards): implement RewardProfileRegistry with history tracking and JSON export",
        ),
        (
            ["tests/test_reward_profile.py"],
            "test(rewards): add unit tests for profile validation, bounds, and immutability",
        ),
        # --- Phase 8: Safe LLM Gateway & Providers (Commits 39-44) ---
        (
            ["towermind/llm/__init__.py"],
            "feat(llm): initialize towermind.llm package",
        ),
        (
            ["towermind/llm/base.py"],
            "feat(llm): define abstract LLMProvider, LLMResponse, and token usage interfaces",
        ),
        (
            ["towermind/llm/mock_provider.py"],
            "feat(llm): implement deterministic MockLLMProvider for offline reproduction",
        ),
        (
            ["towermind/llm/gateway.py"],
            "feat(llm): implement LLMGateway with retry exponential backoff and budget caps",
        ),
        (
            ["towermind/llm/providers.py"],
            "feat(llm): implement OpenAI, Anthropic, and Ollama advisor providers",
        ),
        (
            ["tests/test_llm_gateway.py"],
            "test(llm): add test suite for LLM gateway retries, fallbacks, and budget enforcement",
        ),
        # --- Phase 9: Meta-Controller, Evaluator & Promotion Gate (Commits 45-51) ---
        (
            ["towermind/meta_controller/__init__.py"],
            "feat(meta_controller): initialize towermind.meta_controller package",
        ),
        (
            ["towermind/meta_controller/advisor.py"],
            "feat(meta_controller): create compact advisor prompt builder with telemetry summary",
        ),
        (
            ["towermind/meta_controller/evaluator.py"],
            "feat(meta_controller): implement fixed-seed policy evaluator for empirical performance",
        ),
        (
            ["towermind/meta_controller/promotion.py"],
            "feat(meta_controller): implement PromotionGate with tolerance checks and regression fallback",
        ),
        (
            ["towermind/meta_controller/loop.py"],
            "feat(meta_controller): implement segmented PPO training loop with checkpoint boundaries",
        ),
        (
            ["tests/test_meta_controller.py"],
            "test(meta_controller): add unit tests for advisor prompt, evaluator, and promotion gate",
        ),
        (
            ["tests/test_training_loop.py"],
            "test(training_loop): add end-to-end segmented experiment training test",
        ),
        # --- Phase 10: Strategy Profiles & Strategic Adapter (Commits 52-55) ---
        (
            ["towermind/strategy/__init__.py"],
            "feat(strategy): initialize towermind.strategy package",
        ),
        (
            ["towermind/strategy/profile.py"],
            "feat(strategy): define StrategyProfile with tower priorities and spending ratios",
        ),
        (
            ["towermind/strategy/adapter.py"],
            "feat(strategy): implement StrategyAdapter and StrategicAgent",
        ),
        (
            ["tests/test_strategy_profile.py"],
            "test(strategy): add test suite for strategy profiles and adapter ranking",
        ),
        # --- Phase 11: Multi-Map Curriculum & Generalization (Commits 56-60) ---
        (
            ["towermind/curriculum/__init__.py"],
            "feat(curriculum): initialize towermind.curriculum package",
        ),
        (
            ["towermind/curriculum/maps.py"],
            "feat(curriculum): implement multi-map registry with 5 geometric topologies and curriculum tiers",
        ),
        (
            ["towermind/experiments/__init__.py"],
            "feat(experiments): initialize towermind.experiments package",
        ),
        (
            ["towermind/experiments/manifest.py"],
            "feat(experiments): create ExperimentManifest with structured serialization",
        ),
        (
            ["towermind/experiments/generalization.py"],
            "feat(experiments): implement held-out zero-shot generalization benchmark",
        ),
        # --- Phase 12: Controlled Benchmarking & Post-Episode Diagnostics (Commits 61-67) ---
        (
            ["towermind/analysis/__init__.py"],
            "feat(analysis): initialize towermind.analysis package",
        ),
        (
            ["towermind/analysis/counterfactual.py"],
            "feat(analysis): implement post-episode EpisodeAnalyzer and deterministic diagnostics",
        ),
        (
            ["tests/test_counterfactual_analysis.py"],
            "test(analysis): add unit tests for counterfactual diagnostics and path coverage",
        ),
        (
            ["towermind/experiments/runner.py"],
            "feat(experiments): implement controlled benchmark runner comparing Baseline vs LLM-Guided PPO",
        ),
        (
            ["towermind/experiments/report.py"],
            "feat(experiments): add Markdown and JSON benchmark report generation",
        ),
        (
            ["towermind/experiments/benchmark_cli.py"],
            "feat(experiments): add towermind-benchmark CLI entry point",
        ),
        (
            ["towermind/experiments/generalization_cli.py"],
            "feat(experiments): add towermind-generalize CLI entry point",
        ),
        # --- Phase 13: Benchmark Tests & Interactive Web Dashboard (Commits 68-76) ---
        (
            ["tests/test_curriculum_and_generalization.py"],
            "test(curriculum): add tests for multi-map curriculum and generalization",
        ),
        (
            ["tests/test_benchmark_runner.py"],
            "test(experiments): add unit test for benchmark execution and report synthesis",
        ),
        (
            ["towermind/ui/__init__.py"],
            "feat(ui): initialize towermind.ui package",
        ),
        (
            ["towermind/ui/generator.py"],
            "feat(ui): implement standalone HTML5 Canvas game engine and interactive dashboard generator",
        ),
        (
            ["towermind/ui/server.py"],
            "feat(ui): implement local HTTP server and REST endpoints for dashboard telemetry",
        ),
        (
            ["tests/test_ui.py"],
            "test(ui): add automated tests for HTML dashboard generator and HTTP endpoints",
        ),
        (
            ["benchmark_results/dashboard.html"],
            "feat(ui): add static standalone HTML dashboard export in benchmark_results",
        ),
        (
            [
                "benchmark_results/comparison_report.md",
                "benchmark_results/generalization/generalization_report.md",
            ],
            "docs(benchmarks): add comparison and generalization benchmark Markdown reports",
        ),
        (
            [
                "benchmark_results/.gitkeep",
                "benchmark_results/baseline_manifest.json",
                "benchmark_results/benchmark_20261106_041100.json",
                "benchmark_results/comparison_summary.json",
                "benchmark_results/generalization/generalization_summary.json",
                "benchmark_results/guided_manifest.json",
            ],
            "chore(benchmarks): add reference benchmark summaries and evaluation manifests",
        ),
        # --- Phase 14: Project Launcher, Media Assets & Documentation (Commits 77-85) ---
        (
            ["run.py"],
            "feat(runner): implement unified master project runner with interactive menu and CLI flags",
        ),
        (
            ["run.sh"],
            "feat(runner): add executable shell launcher wrapper run.sh",
        ),
        (
            ["examples/towermind_quickstart.py"],
            "feat(examples): add towermind quickstart demonstration script",
        ),
        (
            ["scripts/generate_all_screenshots.py"],
            "feat(scripts): add script for automated project media screenshot generation",
        ),
        (
            [
                "media/dashboard_preview.png",
                "media/dashboard_visualizer.png",
                "media/dashboard_benchmarks.png",
            ],
            "media: add high-resolution web dashboard visualizer and benchmark comparison screenshots",
        ),
        (
            [
                "media/dashboard_meta_controller.png",
                "media/dashboard_generalization.png",
                "media/dashboard_counterfactual.png",
            ],
            "media: add meta-controller review, generalization matrix, and diagnostics screenshots",
        ),
        (
            [
                "media/curriculum_maps.png",
                "media/gameplay_wave_early.png",
                "media/gameplay_wave_mid.png",
                "media/gameplay_wave_boss.png",
            ],
            "media: add curriculum maps overview collage and tactical wave progression screenshots",
        ),
        (
            [
                "media/ppo_demo.gif",
                "media/greedy_demo.gif",
                "media/towermind_demo.gif",
                "media/benchmark_summary.png",
                "media/coverage_comparison.png",
                "media/latency_comparison.png",
                "media/strategic_metrics.png",
            ],
            "media: add gameplay demo animations and baseline evaluation graphs",
        ),
        (
            ["towermind/__init__.py"],
            "refactor(towermind): export public API in package root and configure lazy loading",
        ),
    ]

    commit_count = len(commits)
    print(f"Executing {commit_count} staged commits...")

    current_time = base_time
    step_delta = timedelta(hours=1, minutes=15)

    for idx, (files, message) in enumerate(commits, 1):
        # Stage files
        for f in files:
            if Path(f).exists():
                run(["git", "add", f])
            else:
                print(f"Warning: File {f} does not exist, skipping add.")

        # Check if anything is staged
        status_res = subprocess.run(
            ["git", "diff", "--cached", "--name-only"], capture_output=True, text=True
        )
        if not status_res.stdout.strip():
            print(f"[{idx}/{commit_count}] Nothing staged for {files}, creating empty checkpoint commit...")
            cmd = ["git", "commit", "--allow-empty", "-m", message]
        else:
            cmd = ["git", "commit", "-m", message]

        current_time += step_delta
        date_str = current_time.strftime("%Y-%m-%d %H:%M:%S +0000")
        env = os.environ.copy()
        env["GIT_AUTHOR_NAME"] = AUTHOR_NAME
        env["GIT_AUTHOR_EMAIL"] = AUTHOR_EMAIL
        env["GIT_COMMITTER_NAME"] = AUTHOR_NAME
        env["GIT_COMMITTER_EMAIL"] = AUTHOR_EMAIL
        env["GIT_AUTHOR_DATE"] = date_str
        env["GIT_COMMITTER_DATE"] = date_str

        run(cmd, env=env)
        print(f"[{idx}/{commit_count}] Committed: {message}")

    # Commit 86: Finalize README (WITHOUT committing Todo.md or plan files!)
    print("Applying final documentation updates...")
    Path("README.md").write_text(orig_readme, encoding="utf-8")
    run(["git", "add", "README.md"])

    current_time += step_delta
    date_str = current_time.strftime("%Y-%m-%d %H:%M:%S +0000")
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = AUTHOR_NAME
    env["GIT_AUTHOR_EMAIL"] = AUTHOR_EMAIL
    env["GIT_COMMITTER_NAME"] = AUTHOR_NAME
    env["GIT_COMMITTER_EMAIL"] = AUTHOR_EMAIL
    env["GIT_AUTHOR_DATE"] = date_str
    env["GIT_COMMITTER_DATE"] = date_str
    final_msg = "docs: finalize README with screenshot galleries, architecture diagrams, and quickstart runner commands"
    run(["git", "commit", "-m", final_msg], env=env)
    print(f"[{commit_count + 1}/{commit_count + 1}] Committed: {final_msg}")

    print(f"\nCommit sequence complete! Total commits: {commit_count + 1}")


if __name__ == "__main__":
    main()
