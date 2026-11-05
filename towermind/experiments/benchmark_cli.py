"""CLI entry point for running the controlled TowerMind comparison benchmark."""

from __future__ import annotations

import argparse

from towermind.experiments.runner import run_comparison_benchmark
from towermind.llm.mock_provider import MockLLMProvider


def main():
    parser = argparse.ArgumentParser(
        description="Run controlled comparison benchmark between Baseline PPO and LLM-Guided PPO."
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for training & evaluation (default: 42)")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2000,
        help="Total training timesteps per condition (default: 2000)",
    )
    parser.add_argument(
        "--segment-timesteps",
        type=int,
        default=1000,
        help="Timesteps per meta-controller segment (default: 1000)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="Episodes per evaluation phase (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save report and artifacts (default: benchmark_results)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device ('cpu' or 'cuda')",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("TowerMind MVP — Controlled Benchmark Execution")
    print(
        f"Seed: {args.seed} | Timesteps: {args.total_timesteps:,} | Segments: {args.total_timesteps // args.segment_timesteps}"
    )
    print(f"Output Directory: {args.output_dir}")
    print("=" * 70)

    # Use deterministic offline mock provider for reproducibility
    provider = MockLLMProvider()

    res = run_comparison_benchmark(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        segment_timesteps=args.segment_timesteps,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        provider=provider,
        device_str=args.device,
    )

    print("\n" + "=" * 70)
    print("Benchmark Completed Successfully!")
    print(f"Report: {res['report_path']}")
    print(f"Summary: {res['summary_path']}")
    print("=" * 70)
    print("\nPreview of Report:\n")
    print(res["report_markdown"][:1200] + "\n...")


if __name__ == "__main__":
    main()
