"""CLI entry point for running map generalization benchmarks."""

from __future__ import annotations

import argparse

from ai_gym_td.agents import GreedyAgent, RuleBasedAgent
from towermind.experiments.generalization import run_generalization_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run TowerMind zero-shot map generalization benchmark.")
    parser.add_argument(
        "--agent",
        choices=["greedy", "rule"],
        default="greedy",
        help="Agent heuristic to evaluate on held-out maps (default: greedy)",
    )
    parser.add_argument(
        "--episodes-per-map",
        type=int,
        default=2,
        help="Episodes to run on each map (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results/generalization",
        help="Directory to save report and json summary (default: benchmark_results/generalization)",
    )
    args = parser.parse_args()

    agent = GreedyAgent() if args.agent == "greedy" else RuleBasedAgent()

    print("=" * 70)
    print("TowerMind — Zero-Shot Map Generalization Benchmark")
    print(f"Agent: {args.agent} | Episodes per map: {args.episodes_per_map}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 70)

    res = run_generalization_benchmark(
        agent=agent,
        episodes_per_map=args.episodes_per_map,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 70)
    print("Generalization Benchmark Complete!")
    print("=" * 70)
    print("\nReport Preview:\n")
    print(res["report_markdown"])


if __name__ == "__main__":
    main()
