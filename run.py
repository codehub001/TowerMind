#!/usr/bin/env python3
"""
TowerMind & AI Gym Tower Defense — Master Project Launcher.

Provides a unified, easy-to-use entry point to run any part of the project:
  - Web Visualizer & Analytics Dashboard
  - Manual Pygame play
  - Real-time AI agent visualizer
  - PPO vs LLM-Guided PPO benchmarks
  - Held-out map generalization testing
  - Full test suite
  - Complete automated project pipeline
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Theme Colors (Deep Charcoal + Orange Accent matching UI)
ORANGE = "\033[38;2;226;72;0m"
ORANGE_BOLD = "\033[1;38;2;226;72;0m"
WHITE_BOLD = "\033[1;37m"
GRAY = "\033[38;5;244m"
GREEN = "\033[38;5;71m"
RED = "\033[38;5;203m"
RESET = "\033[0m"


def print_banner() -> None:
    print(f"""{ORANGE_BOLD}========================================================================
TowerMind & AI Gym Tower Defense — Project Runner
========================================================================{RESET}""")


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def get_python_exe() -> str:
    root = get_project_root()
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.exists() and os.access(venv_python, os.X_OK):
        return str(venv_python)
    return sys.executable


def run_cmd(args: list[str], description: str = "") -> int:
    if description:
        print(f"\n{ORANGE_BOLD}>> {description}{RESET}")
    print(f"{GRAY}$ {' '.join(args)}{RESET}\n")
    try:
        res = subprocess.run(args, cwd=get_project_root())
        return res.returncode
    except KeyboardInterrupt:
        print(f"\n{ORANGE}Interrupted by user.{RESET}")
        return 130


# --- Command Actions ---


def action_ui(port: int = 8080, open_browser: bool = False) -> int:
    py = get_python_exe()
    cmd = [py, "-m", "towermind.ui.server", "--port", str(port)]
    if open_browser:
        cmd.append("--open")
    return run_cmd(cmd, f"Starting TowerMind Web Dashboard on http://127.0.0.1:{port}")


def action_export_ui() -> int:
    py = get_python_exe()
    cmd = [py, "-m", "towermind.ui.server", "--export-only"]
    return run_cmd(cmd, "Exporting standalone static dashboard HTML")


def action_play(map_name: str = "s_curve") -> int:
    py = get_python_exe()
    cmd = [py, "-m", "ai_gym_td.scripts.play", "--map", map_name]
    return run_cmd(cmd, f"Launching Pygame Play Mode (Map: {map_name})")


def action_watch(agent: str = "towermind", map_name: str = "s_curve") -> int:
    py = get_python_exe()
    cmd = [py, "-m", "ai_gym_td.scripts.evaluate", "--agent", agent, "--map", map_name, "--render"]
    return run_cmd(cmd, f"Watching AI Agent '{agent}' play live in Pygame (Map: {map_name})")


def action_benchmark(episodes: int = 5) -> int:
    py = get_python_exe()
    cmd = [py, "-m", "towermind.experiments.benchmark_cli", "--episodes", str(episodes)]
    return run_cmd(
        cmd, f"Running Controlled Benchmark (Baseline PPO vs TowerMind LLM-Guided, {episodes} eps)"
    )


def action_generalize(episodes: int = 3) -> int:
    py = get_python_exe()
    cmd = [py, "-m", "towermind.experiments.generalization_cli", "--episodes", str(episodes)]
    return run_cmd(cmd, f"Running Held-Out Map Generalization Test ({episodes} eps/map)")


def action_test() -> int:
    py = get_python_exe()
    cmd = [py, "-m", "pytest", "tests/"]
    return run_cmd(cmd, "Running full 57-test test suite")


def action_all(episodes: int = 3) -> int:
    print(f"\n{ORANGE_BOLD}Running Full TowerMind Pipeline...{RESET}")
    # 1. Run tests
    ret = action_test()
    if ret != 0:
        print(f"\n{RED}Tests failed. Aborting full pipeline.{RESET}")
        return ret

    # 2. Run benchmark
    ret = action_benchmark(episodes=episodes)
    if ret != 0:
        print(f"\n{RED}Benchmark failed.{RESET}")
        return ret

    # 3. Run generalization
    ret = action_generalize(episodes=episodes)
    if ret != 0:
        print(f"\n{RED}Generalization test failed.{RESET}")
        return ret

    # 4. Export static UI
    action_export_ui()

    # 5. Launch UI server
    print(f"\n{GREEN}Pipeline completed successfully. Launching dashboard...{RESET}")
    return action_ui(port=8080)


# --- Interactive Menu ---


def interactive_menu() -> None:
    while True:
        print_banner()
        print(f"""{WHITE_BOLD}Select an action to run:{RESET}

  {ORANGE_BOLD}[1]{RESET} {WHITE_BOLD}Launch Web Dashboard & Visualizer{RESET}    (http://127.0.0.1:8080)
  {ORANGE_BOLD}[2]{RESET} {WHITE_BOLD}Play Game Manually (Pygame){RESET}            (mouse/keyboard controls)
  {ORANGE_BOLD}[3]{RESET} {WHITE_BOLD}Watch Strategic AI Agent Play{RESET}          (live Pygame window)
  {ORANGE_BOLD}[4]{RESET} {WHITE_BOLD}Run Benchmark Comparison{RESET}               (Baseline PPO vs LLM-Guided)
  {ORANGE_BOLD}[5]{RESET} {WHITE_BOLD}Run Generalization Test{RESET}                (held-out maps evaluation)
  {ORANGE_BOLD}[6]{RESET} {WHITE_BOLD}Run Full Test Suite{RESET}                    (57 unit & integration tests)
  {ORANGE_BOLD}[7]{RESET} {WHITE_BOLD}Run Full Project Pipeline{RESET}              (test -> benchmark -> ui)
  {ORANGE_BOLD}[8]{RESET} {WHITE_BOLD}Export Static Dashboard HTML{RESET}           (no background server)
  
  {GRAY}[0] Exit{RESET}
""")
        try:
            choice = input(f"{WHITE_BOLD}Enter choice [0-8]: {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

        if choice == "1":
            action_ui(port=8080)
        elif choice == "2":
            action_play(map_name="s_curve")
        elif choice == "3":
            action_watch(agent="towermind", map_name="s_curve")
        elif choice == "4":
            action_benchmark(episodes=5)
        elif choice == "5":
            action_generalize(episodes=3)
        elif choice == "6":
            action_test()
        elif choice == "7":
            action_all(episodes=3)
        elif choice == "8":
            action_export_ui()
        elif choice in ("0", "q", "exit"):
            print("Goodbye!")
            sys.exit(0)
        else:
            print(f"{RED}Invalid choice '{choice}'. Please pick a number from 0 to 8.{RESET}")

        try:
            input(f"\n{GRAY}Press Enter to return to menu...{RESET}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)


# --- CLI Parser ---


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TowerMind & AI Gym Tower Defense Master Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  ./run.sh                  # Interactive menu
  ./run.sh ui               # Launch web dashboard at http://127.0.0.1:8080
  ./run.sh play             # Play manually in Pygame
  ./run.sh watch            # Watch AI agent play
  ./run.sh benchmark        # Run controlled comparison benchmark
  ./run.sh generalize       # Run map generalization test
  ./run.sh test             # Run pytest suite
  ./run.sh all              # Run end-to-end pipeline
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # UI
    p_ui = subparsers.add_parser("ui", help="Start web dashboard & visualizer server")
    p_ui.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    p_ui.add_argument("--open", action="store_true", help="Attempt to open browser automatically")

    # Export
    subparsers.add_parser("export", help="Export standalone static dashboard.html")

    # Play
    p_play = subparsers.add_parser("play", help="Play Tower Defense manually in Pygame")
    p_play.add_argument("--map", type=str, default="s_curve", help="Map name (default: s_curve)")

    # Watch / Eval
    p_watch = subparsers.add_parser("watch", help="Watch AI agent play with real-time Pygame renderer")
    p_watch.add_argument("--agent", type=str, default="towermind", help="Agent: towermind, heuristic, random")
    p_watch.add_argument("--map", type=str, default="s_curve", help="Map name (default: s_curve)")

    # Benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run Baseline vs LLM-Guided PPO comparison benchmark")
    p_bench.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes per condition (default: 5)"
    )

    # Generalization
    p_gen = subparsers.add_parser("generalize", help="Run held-out curriculum map generalization test")
    p_gen.add_argument("--episodes", type=int, default=3, help="Episodes per held-out map (default: 3)")

    # Test
    subparsers.add_parser("test", help="Run full 57-test pytest suite")

    # All / Pipeline
    p_all = subparsers.add_parser("all", help="Run test -> benchmark -> generalize -> launch UI")
    p_all.add_argument("--episodes", type=int, default=3, help="Episodes per condition (default: 3)")

    args = parser.parse_args()

    if not args.command:
        # If no arguments given, launch interactive menu
        interactive_menu()
        return

    code = 0
    if args.command == "ui":
        code = action_ui(port=args.port, open_browser=args.open)
    elif args.command == "export":
        code = action_export_ui()
    elif args.command == "play":
        code = action_play(map_name=args.map)
    elif args.command == "watch":
        code = action_watch(agent=args.agent, map_name=args.map)
    elif args.command == "benchmark":
        code = action_benchmark(episodes=args.episodes)
    elif args.command == "generalize":
        code = action_generalize(episodes=args.episodes)
    elif args.command == "test":
        code = action_test()
    elif args.command == "all":
        code = action_all(episodes=args.episodes)

    sys.exit(code)


if __name__ == "__main__":
    main()
