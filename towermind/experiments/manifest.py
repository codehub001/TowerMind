"""Experiment configuration, metrics, and manifest schemas for reproducible runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics measured across fixed seeds."""

    episodes: int
    win_rate: float
    mean_waves: float
    mean_lives: float
    mean_reward: float
    mean_steps: float
    wins: int
    losses: int
    legality_rate: float = 1.0
    repair_rate: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episodes": self.episodes,
            "win_rate": round(self.win_rate, 4),
            "mean_waves": round(self.mean_waves, 2),
            "mean_lives": round(self.mean_lives, 2),
            "mean_reward": round(self.mean_reward, 2),
            "mean_steps": round(self.mean_steps, 1),
            "wins": self.wins,
            "losses": self.losses,
            "legality_rate": round(self.legality_rate, 4),
            "repair_rate": round(self.repair_rate, 4),
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationMetrics:
        return cls(
            episodes=data["episodes"],
            win_rate=float(data["win_rate"]),
            mean_waves=float(data["mean_waves"]),
            mean_lives=float(data.get("mean_lives", 0.0)),
            mean_reward=float(data["mean_reward"]),
            mean_steps=float(data.get("mean_steps", 0.0)),
            wins=int(data.get("wins", 0)),
            losses=int(data.get("losses", 0)),
            legality_rate=float(data.get("legality_rate", 1.0)),
            repair_rate=float(data.get("repair_rate", 0.0)),
            extra=data.get("extra", {}),
        )


@dataclass
class ExperimentConfig:
    """Hyperparameters and budget for an experiment."""

    experiment_id: str
    seed: int = 42
    total_timesteps: int = 10000
    segment_timesteps: int = 2500
    eval_episodes: int = 5
    learning_rate: float = 2.5e-4
    num_steps: int = 256
    gamma: float = 0.995
    reward_profile_id: str = "baseline-v1"
    map_name: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentConfig:
        return cls(**data)


@dataclass
class SegmentRecord:
    """Record of a single training segment boundary and LLM meta-controller review."""

    segment_index: int
    timesteps_so_far: int
    active_profile_id: str
    proposed_profile_id: Optional[str] = None
    decision: str = "continued"  # "promoted", "rejected", "fallback", "continued"
    metrics: Optional[EvaluationMetrics] = None
    advisor_rationale: Optional[str] = None
    advisor_hypothesis: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "timesteps_so_far": self.timesteps_so_far,
            "active_profile_id": self.active_profile_id,
            "proposed_profile_id": self.proposed_profile_id,
            "decision": self.decision,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "advisor_rationale": self.advisor_rationale,
            "advisor_hypothesis": self.advisor_hypothesis,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SegmentRecord:
        metrics = EvaluationMetrics.from_dict(data["metrics"]) if data.get("metrics") else None
        return cls(
            segment_index=data["segment_index"],
            timesteps_so_far=data["timesteps_so_far"],
            active_profile_id=data["active_profile_id"],
            proposed_profile_id=data.get("proposed_profile_id"),
            decision=data.get("decision", "continued"),
            metrics=metrics,
            advisor_rationale=data.get("advisor_rationale"),
            advisor_hypothesis=data.get("advisor_hypothesis"),
        )


@dataclass
class ExperimentManifest:
    """Full self-contained record of an experiment run for analysis and comparison."""

    experiment_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config: ExperimentConfig = field(default_factory=lambda: ExperimentConfig(experiment_id="default"))
    baseline_metrics: Optional[EvaluationMetrics] = None
    final_metrics: Optional[EvaluationMetrics] = None
    segments: List[SegmentRecord] = field(default_factory=list)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    checkpoint_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "config": self.config.to_dict(),
            "baseline_metrics": self.baseline_metrics.to_dict() if self.baseline_metrics else None,
            "final_metrics": self.final_metrics.to_dict() if self.final_metrics else None,
            "segments": [s.to_dict() for s in self.segments],
            "telemetry": self.telemetry,
            "checkpoint_paths": self.checkpoint_paths,
        }

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> ExperimentManifest:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        config = ExperimentConfig.from_dict(data["config"])
        baseline = (
            EvaluationMetrics.from_dict(data["baseline_metrics"]) if data.get("baseline_metrics") else None
        )
        final_m = EvaluationMetrics.from_dict(data["final_metrics"]) if data.get("final_metrics") else None
        segments = [SegmentRecord.from_dict(s) for s in data.get("segments", [])]

        return cls(
            experiment_id=data["experiment_id"],
            timestamp=data["timestamp"],
            config=config,
            baseline_metrics=baseline,
            final_metrics=final_m,
            segments=segments,
            telemetry=data.get("telemetry", {}),
            checkpoint_paths=data.get("checkpoint_paths", []),
        )
