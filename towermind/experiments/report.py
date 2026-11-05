"""Markdown and JSON experiment comparison reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from towermind.experiments.manifest import ExperimentManifest


def generate_markdown_report(
    baseline_manifest: ExperimentManifest,
    guided_manifest: ExperimentManifest,
    title: str = "TowerMind MVP Experiment Comparison Report",
) -> str:
    """Generate a clean, publication-ready Markdown comparison report."""
    base_m = baseline_manifest.final_metrics
    guid_m = guided_manifest.final_metrics

    # Compute deltas
    def fmt_delta(val_g: Optional[float], val_b: Optional[float], pct: bool = False) -> str:
        if val_g is None or val_b is None:
            return "N/A"
        diff = val_g - val_b
        sign = "+" if diff > 0 else ""
        if pct:
            return f"{sign}{diff * 100:.1f}%"
        return f"{sign}{diff:.2f}"

    base_win = base_m.win_rate if base_m else 0.0
    guid_win = guid_m.win_rate if guid_m else 0.0
    base_waves = base_m.mean_waves if base_m else 0.0
    guid_waves = guid_m.mean_waves if guid_m else 0.0
    base_lives = base_m.mean_lives if base_m else 0.0
    guid_lives = guid_m.mean_lives if guid_m else 0.0
    base_rew = base_m.mean_reward if base_m else 0.0
    guid_rew = guid_m.mean_reward if guid_m else 0.0
    base_legal = base_m.legality_rate if base_m else 1.0
    guid_legal = guid_m.legality_rate if guid_m else 1.0

    lines = [
        f"# {title}",
        "",
        f"**Date:** {guided_manifest.timestamp}  ",
        f"**Baseline Experiment ID:** `{baseline_manifest.experiment_id}`  ",
        f"**Guided Experiment ID:** `{guided_manifest.experiment_id}`  ",
        f"**Random Seed:** {guided_manifest.config.seed}  ",
        f"**Training Timesteps:** {guided_manifest.config.total_timesteps:,}  ",
        f"**Evaluation Episodes:** {guided_manifest.config.eval_episodes}  ",
        "",
        "---",
        "",
        "## 1. Metric Performance Comparison",
        "",
        "| Metric | Baseline PPO | LLM-Guided PPO | Delta | Outcome |",
        "|---|---|---|---|---|",
        f"| **Win Rate** | {base_win * 100:.1f}% | {guid_win * 100:.1f}% | {fmt_delta(guid_win, base_win, pct=True)} | {'Improvement' if guid_win > base_win else ('Decline' if guid_win < base_win else 'Parity')} |",
        f"| **Mean Waves Survived** | {base_waves:.2f} | {guid_waves:.2f} | {fmt_delta(guid_waves, base_waves)} | {'Improvement' if guid_waves > base_waves else ('Decline' if guid_waves < base_waves else 'Parity')} |",
        f"| **Mean Final Lives** | {base_lives:.2f} | {guid_lives:.2f} | {fmt_delta(guid_lives, base_lives)} | {'Improvement' if guid_lives > base_lives else ('Decline' if guid_lives < base_lives else 'Parity')} |",
        f"| **Mean Episode Return** | {base_rew:.2f} | {guid_rew:.2f} | {fmt_delta(guid_rew, base_rew)} | {'Improvement' if guid_rew > base_rew else ('Decline' if guid_rew < base_rew else 'Parity')} |",
        f"| **Action Legality Rate** | {base_legal * 100:.1f}% | {guid_legal * 100:.1f}% | {fmt_delta(guid_legal, base_legal, pct=True)} | {'Healthy' if guid_legal >= 0.95 else 'Degraded'} |",
        "",
        "---",
        "",
        "## 2. Meta-Controller Segment Review & Decisions",
        "",
    ]

    if guided_manifest.segments:
        lines.extend(
            [
                "| Segment | Timesteps | Active Profile | Proposed Candidate | Decision | Rationale & Hypothesis |",
                "|---|---|---|---|---|---|",
            ]
        )
        for s in guided_manifest.segments:
            hyp = f"{s.advisor_rationale or ''} (Hypothesis: {s.advisor_hypothesis or ''})".strip()
            lines.append(
                f"| Seg {s.segment_index} | {s.timesteps_so_far:,} | `{s.active_profile_id}` | `{s.proposed_profile_id or 'None'}` | **{s.decision.upper()}** | {hyp} |"
            )
        lines.append("")
    else:
        lines.append("*No intermediate segments logged.* \n")

    lines.extend(
        [
            "---",
            "",
            "## 3. LLM Gateway Telemetry & Compute Budget",
            "",
            f"- **Total LLM Requests:** {guided_manifest.telemetry.get('total_requests', 0)}",
            f"- **Total Fallbacks Triggered:** {guided_manifest.telemetry.get('total_fallbacks', 0)}",
            f"- **Tokens Consumed:** {guided_manifest.telemetry.get('tokens', {}).get('total_tokens', 0)}",
            f"- **Estimated API Cost (USD):** ${guided_manifest.telemetry.get('tokens', {}).get('estimated_cost_usd', 0.0):.4f}",
            "",
            "---",
            "",
            "## 4. Scientific Findings & Honest Disclosure",
            "",
            "> **Note on Statistical Rigor:** This report reflects a preliminary, controlled MVP experiment run. "
            "Differences observed in a single matched-seed run serve as exploratory evidence of the meta-controller's "
            "workflow and safety guarantees, rather than definitive proof of universal policy superiority across all maps.",
            "",
        ]
    )

    return "\n".join(lines)


def save_report(content: str, path: str | Path) -> None:
    """Save markdown report to target path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
