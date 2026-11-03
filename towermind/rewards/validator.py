"""Strict validation for RewardProfile instances and candidate dictionaries.

Ensures:
1. All weights are finite floating-point numbers within bounded ranges.
2. No unknown keys are present in LLM proposals.
3. Profile IDs, text fields, and creator metadata meet length and character constraints.
4. Schema versions match supported versions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from towermind.rewards.profile import (
    CURRENT_SCHEMA_VERSION,
    REWARD_BOUNDS,
    RewardProfile,
)

PROFILE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]{1,64}$")
ALLOWED_CREATORS: Set[str] = {"baseline", "human", "llm"}
MAX_TEXT_LENGTH = 1000

KNOWN_FIELDS: Set[str] = {
    "profile_id",
    "kill",
    "leak",
    "win",
    "lose",
    "step",
    "invalid_action",
    "economy",
    "scale",
    "rationale",
    "hypothesis",
    "created_by",
    "parent_profile_id",
    "schema_version",
}


class RewardValidationError(ValueError):
    """Raised when a RewardProfile or reward candidate fails strict validation."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or [message]


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a validation check."""

    is_valid: bool
    errors: List[str]
    profile: Optional[RewardProfile] = None

    def raise_for_errors(self) -> None:
        if not self.is_valid:
            raise RewardValidationError(
                f"Reward profile validation failed: {'; '.join(self.errors)}",
                errors=self.errors,
            )


def validate_profile_dict(
    data: Dict[str, Any],
    require_all_fields: bool = False,
) -> ValidationResult:
    """Validate a raw dictionary (e.g. parsed from LLM JSON response) against RewardProfile schema."""
    errors: List[str] = []

    if not isinstance(data, dict):
        return ValidationResult(is_valid=False, errors=["Data must be a dictionary"])

    # 1. Check for unknown fields (no extraneous keys allowed from LLM)
    unknown_keys = set(data.keys()) - KNOWN_FIELDS
    if unknown_keys:
        errors.append(f"Disallowed extra fields: {sorted(unknown_keys)}")

    # 2. Check profile_id
    profile_id = data.get("profile_id")
    if profile_id is None:
        if require_all_fields:
            errors.append("Missing required field 'profile_id'")
    else:
        if not isinstance(profile_id, str) or not PROFILE_ID_PATTERN.match(profile_id):
            errors.append(
                f"Invalid 'profile_id': must be 1-64 alphanumeric characters, underscores, or hyphens, got {profile_id!r}"
            )

    # 3. Check schema_version
    schema_ver = data.get("schema_version", CURRENT_SCHEMA_VERSION)
    if schema_ver != CURRENT_SCHEMA_VERSION:
        errors.append(f"Unsupported schema_version {schema_ver}; expected {CURRENT_SCHEMA_VERSION}")

    # 4. Check created_by
    created_by = data.get("created_by", "llm")
    if created_by not in ALLOWED_CREATORS:
        errors.append(f"Invalid created_by '{created_by}'; must be one of {sorted(ALLOWED_CREATORS)}")

    # 5. Check text lengths
    for field_name in ("rationale", "hypothesis"):
        val = data.get(field_name, "")
        if val is not None and not isinstance(val, str):
            errors.append(f"Field '{field_name}' must be a string, got {type(val).__name__}")
        elif isinstance(val, str) and len(val) > MAX_TEXT_LENGTH:
            errors.append(
                f"Field '{field_name}' exceeds maximum length of {MAX_TEXT_LENGTH} chars ({len(val)})"
            )

    # 6. Check parent_profile_id
    parent_id = data.get("parent_profile_id")
    if parent_id is not None and not (isinstance(parent_id, str) and PROFILE_ID_PATTERN.match(parent_id)):
        errors.append(f"Invalid 'parent_profile_id': {parent_id!r}")

    # 7. Check numeric bounds and finiteness
    for weight_name, (low, high) in REWARD_BOUNDS.items():
        if weight_name in data:
            val = data[weight_name]
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                errors.append(f"Field '{weight_name}' must be a number, got {val!r}")
                continue
            fval = float(val)
            if not math.isfinite(fval):
                errors.append(f"Field '{weight_name}' must be finite, got {fval}")
                continue
            if not (low <= fval <= high):
                errors.append(f"Field '{weight_name}' value {fval} out of permitted range [{low}, {high}]")
        elif require_all_fields:
            errors.append(f"Missing required numeric field '{weight_name}'")

    if errors:
        return ValidationResult(is_valid=False, errors=errors)

    # Build validated profile if all checks passed
    try:
        profile = RewardProfile.from_dict(data)
        return ValidationResult(is_valid=True, errors=[], profile=profile)
    except Exception as e:
        return ValidationResult(is_valid=False, errors=[f"Instantiation error: {e}"])


def validate_profile(profile: RewardProfile) -> ValidationResult:
    """Validate an already instantiated RewardProfile."""
    if not isinstance(profile, RewardProfile):
        return ValidationResult(is_valid=False, errors=["Expected RewardProfile instance"])
    return validate_profile_dict(profile.to_dict(), require_all_fields=True)


def enforce_valid_profile(
    target: RewardProfile | Dict[str, Any],
) -> RewardProfile:
    """Validate target and return RewardProfile, or raise RewardValidationError."""
    if isinstance(target, RewardProfile):
        res = validate_profile(target)
    elif isinstance(target, dict):
        res = validate_profile_dict(target)
    else:
        raise RewardValidationError(f"Expected RewardProfile or dict, got {type(target).__name__}")

    res.raise_for_errors()
    assert res.profile is not None
    return res.profile
