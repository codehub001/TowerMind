"""Post-episode counterfactual strategy analyzer: path coverage, damage/gold efficiency, leaks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ai_gym_td.config import TOWERS
from ai_gym_td.engine import KillEvent, LeakEvent, TowerDefenseGame


@dataclass
class EpisodeDiagnostics:
    """Diagnostic telemetry extracted from an episode simulation."""

    total_path_cells: int
    covered_path_cells: int
    path_coverage_ratio: float
    uncovered_path_indices: List[int]
    tower_counts: Dict[str, int] = field(default_factory=dict)
    gold_invested_by_tower: Dict[str, int] = field(default_factory=dict)
    damage_per_gold: Dict[str, float] = field(default_factory=dict)
    kills_by_wave: Dict[int, int] = field(default_factory=dict)
    leaks_by_wave: Dict[int, int] = field(default_factory=dict)
    final_gold: int = 0
    final_wave: int = 0
    phase: str = "lost"
    diagnostic_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_path_cells": self.total_path_cells,
            "covered_path_cells": self.covered_path_cells,
            "path_coverage_ratio": round(self.path_coverage_ratio, 4),
            "tower_counts": self.tower_counts,
            "gold_invested_by_tower": self.gold_invested_by_tower,
            "damage_per_gold": {k: round(v, 2) for k, v in self.damage_per_gold.items()},
            "kills_by_wave": self.kills_by_wave,
            "leaks_by_wave": self.leaks_by_wave,
            "final_gold": self.final_gold,
            "final_wave": self.final_wave,
            "phase": self.phase,
            "diagnostic_notes": self.diagnostic_notes,
        }


class EpisodeAnalyzer:
    """Analyzes game state and accumulated events to identify strategic bottlenecks."""

    def analyze(
        self,
        game: TowerDefenseGame,
        events_accum: Optional[List[Any]] = None,
    ) -> EpisodeDiagnostics:
        events = events_accum or []
        path_coords = list(game.path)
        total_path = len(path_coords)

        # 1. Path coverage analysis
        covered_indices: Set[int] = set()
        towers = list(game.towers.values())

        for idx, (px, py) in enumerate(path_coords):
            for t in towers:
                dist = math.hypot(px - t.x, py - t.y)
                if dist <= t.spec.range:
                    covered_indices.add(idx)
                    break

        covered_count = len(covered_indices)
        cov_ratio = covered_count / total_path if total_path > 0 else 0.0
        uncovered = [i for i in range(total_path) if i not in covered_indices]

        # 2. Tower investments
        counts: Dict[str, int] = {name: 0 for name in TOWERS.keys()}
        invested: Dict[str, int] = {name: 0 for name in TOWERS.keys()}
        for t in towers:
            name = t.spec.name
            counts[name] = counts.get(name, 0) + 1
            invested[name] = invested.get(name, 0) + t.spec.cost

        # 3. Kills & leak accumulation
        kills_by_wave: Dict[int, int] = {}
        leaks_by_wave: Dict[int, int] = {}
        total_kills = 0

        for ev in events:
            if isinstance(ev, KillEvent):
                w = game.wave_number
                kills_by_wave[w] = kills_by_wave.get(w, 0) + 1
                total_kills += 1
            elif isinstance(ev, LeakEvent):
                w = game.wave_number
                leaks_by_wave[w] = leaks_by_wave.get(w, 0) + 1

        # 4. Gold efficiency estimation
        dpg: Dict[str, float] = {}
        for name, spent in invested.items():
            if spent > 0:
                # Estimate efficiency by kills per 100 gold
                dpg[name] = round((total_kills / spent) * 100.0, 2)
            else:
                dpg[name] = 0.0

        # 5. Diagnostic notes
        notes: List[str] = []
        if cov_ratio < 0.60:
            notes.append(f"Low path coverage ({cov_ratio * 100:.1f}%): enemies pass through blind spots.")
        if game.gold >= 50 and game.phase == game.LOST:
            notes.append(f"Unspent gold at defeat ({game.gold} gold): agent held cash instead of defending.")
        if counts.get("ice", 0) == 0 and sum(leaks_by_wave.values()) > 0:
            notes.append("No crowd-control (ice) towers placed to slow fast enemies.")
        if counts.get("archer", 0) == 0 and counts.get("cannon", 0) == 0:
            notes.append("Lacking early-game baseline damage dealers.")

        return EpisodeDiagnostics(
            total_path_cells=total_path,
            covered_path_cells=covered_count,
            path_coverage_ratio=cov_ratio,
            uncovered_path_indices=uncovered,
            tower_counts=counts,
            gold_invested_by_tower=invested,
            damage_per_gold=dpg,
            kills_by_wave=kills_by_wave,
            leaks_by_wave=leaks_by_wave,
            final_gold=game.gold,
            final_wave=game.wave_number,
            phase=game.phase,
            diagnostic_notes=notes,
        )
