"""Typed, versioned, and validated StrategyProfile data model."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

ALLOWED_TOWERS: Set[str] = {"archer", "cannon", "ice", "tesla"}
ALLOWED_RISK_MODES: Set[str] = {"balanced", "conservative", "aggressive"}
ALLOWED_FOCUS_MODES: Set[str] = {
    "path_chokepoints",
    "path_entrance",
    "path_exit",
    "balanced_coverage",
}
ALLOWED_CREATORS: Set[str] = {"baseline", "human", "llm"}
PROFILE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]{1,64}$")
CURRENT_STRATEGY_SCHEMA_VERSION = 1


class StrategyValidationError(ValueError):
    """Raised when a StrategyProfile fails validation."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or [message]


@dataclass(frozen=True)
class StrategyValidationResult:
    """Outcome of strategy profile validation."""

    is_valid: bool
    errors: List[str]
    profile: Optional[StrategyProfile] = None

    def raise_for_errors(self) -> None:
        if not self.is_valid:
            raise StrategyValidationError(
                f"Strategy validation failed: {'; '.join(self.errors)}",
                errors=self.errors,
            )


@dataclass(frozen=True)
class StrategyProfile:
    """High-level strategy guidance for tower placement and resource allocation."""

    profile_id: str
    tower_priority: Tuple[str, ...] = ("archer", "ice", "cannon", "tesla")
    spending_targets: Dict[str, float] = field(
        default_factory=lambda: {
            "archer": 0.40,
            "ice": 0.25,
            "cannon": 0.20,
            "tesla": 0.15,
        }
    )
    risk_mode: str = "balanced"
    focus: str = "balanced_coverage"
    coverage_priority: float = 0.5
    rationale: str = ""
    hypothesis: str = ""
    created_by: str = "baseline"
    schema_version: int = CURRENT_STRATEGY_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "tower_priority": list(self.tower_priority),
            "spending_targets": dict(self.spending_targets),
            "risk_mode": self.risk_mode,
            "focus": self.focus,
            "coverage_priority": self.coverage_priority,
            "rationale": self.rationale,
            "hypothesis": self.hypothesis,
            "created_by": self.created_by,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StrategyProfile:
        priority = tuple(data.get("tower_priority", ("archer", "ice", "cannon", "tesla")))
        targets = dict(
            data.get(
                "spending_targets",
                {
                    "archer": 0.40,
                    "ice": 0.25,
                    "cannon": 0.20,
                    "tesla": 0.15,
                },
            )
        )
        return cls(
            profile_id=str(data["profile_id"]),
            tower_priority=priority,
            spending_targets=targets,
            risk_mode=str(data.get("risk_mode", "balanced")),
            focus=str(data.get("focus", "balanced_coverage")),
            coverage_priority=float(data.get("coverage_priority", 0.5)),
            rationale=str(data.get("rationale", "")),
            hypothesis=str(data.get("hypothesis", "")),
            created_by=str(data.get("created_by", "llm")),
            schema_version=int(data.get("schema_version", CURRENT_STRATEGY_SCHEMA_VERSION)),
        )


def validate_strategy_profile_dict(data: Dict[str, Any]) -> StrategyValidationResult:
    """Validate a raw dictionary against the StrategyProfile schema."""
    errors: List[str] = []

    if not isinstance(data, dict):
        return StrategyValidationResult(is_valid=False, errors=["Data must be a dictionary"])

    # 1. Profile ID
    pid = data.get("profile_id")
    if not pid or not isinstance(pid, str) or not PROFILE_ID_PATTERN.match(pid):
        errors.append(f"Invalid 'profile_id': must be 1-64 alphanumeric chars/dashes, got {pid!r}")

    # 2. Schema version
    schema_ver = data.get("schema_version", CURRENT_STRATEGY_SCHEMA_VERSION)
    if schema_ver != CURRENT_STRATEGY_SCHEMA_VERSION:
        errors.append(f"Unsupported schema_version {schema_ver}; expected {CURRENT_STRATEGY_SCHEMA_VERSION}")

    # 3. Tower Priority
    priority = data.get("tower_priority")
    if not isinstance(priority, (list, tuple)) or len(priority) == 0:
        errors.append("'tower_priority' must be a non-empty list of tower names")
    else:
        for t in priority:
            if t not in ALLOWED_TOWERS:
                errors.append(f"Unknown tower '{t}' in tower_priority; must be in {sorted(ALLOWED_TOWERS)}")

    # 4. Spending Targets
    targets = data.get("spending_targets")
    if not isinstance(targets, dict) or len(targets) == 0:
        errors.append("'spending_targets' must be a non-empty mapping of tower names to proportions")
    else:
        target_sum = 0.0
        for name, prop in targets.items():
            if name not in ALLOWED_TOWERS:
                errors.append(f"Unknown tower '{name}' in spending_targets")
            if not isinstance(prop, (int, float)) or isinstance(prop, bool):
                errors.append(f"Spending target for '{name}' must be a float, got {prop!r}")
            else:
                fprop = float(prop)
                if not (0.0 <= fprop <= 1.0):
                    errors.append(f"Spending target for '{name}' out of range [0.0, 1.0]: {fprop}")
                target_sum += fprop
        if not (0.95 <= target_sum <= 1.05):
            errors.append(f"Spending targets must sum to ~1.0 (got {target_sum:.3f})")

    # 5. Risk mode & focus
    risk_mode = data.get("risk_mode", "balanced")
    if risk_mode not in ALLOWED_RISK_MODES:
        errors.append(f"Invalid risk_mode '{risk_mode}'; must be in {sorted(ALLOWED_RISK_MODES)}")

    focus = data.get("focus", "balanced_coverage")
    if focus not in ALLOWED_FOCUS_MODES:
        errors.append(f"Invalid focus '{focus}'; must be in {sorted(ALLOWED_FOCUS_MODES)}")

    # 6. Coverage priority
    cov_p = data.get("coverage_priority", 0.5)
    if not isinstance(cov_p, (int, float)) or isinstance(cov_p, bool):
        errors.append(f"'coverage_priority' must be a float, got {cov_p!r}")
    else:
        fcov = float(cov_p)
        if not (0.0 <= fcov <= 1.0):
            errors.append(f"'coverage_priority' must be between 0.0 and 1.0, got {fcov}")

    # 7. Text limits
    for field_name in ("rationale", "hypothesis"):
        val = data.get(field_name, "")
        if isinstance(val, str) and len(val) > 1000:
            errors.append(f"'{field_name}' exceeds 1000 characters ({len(val)})")

    if errors:
        return StrategyValidationResult(is_valid=False, errors=errors)

    profile = StrategyProfile.from_dict(data)
    return StrategyValidationResult(is_valid=True, errors=[], profile=profile)


def get_default_strategy_profile(profile_id: str = "strategy_default") -> StrategyProfile:
    """Return default balanced baseline StrategyProfile."""
    return StrategyProfile(profile_id=profile_id)
