"""Advisor prompt construction and JSON schema definitions for LLM meta-controllers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from towermind.experiments.manifest import EvaluationMetrics
from towermind.rewards.profile import (
    CURRENT_SCHEMA_VERSION,
    REWARD_BOUNDS,
    RewardProfile,
)

SYSTEM_ADVISOR_PROMPT = (
    "You are an expert reinforcement-learning experiment advisor for a Tower Defense game.\n"
    "Your objective is to propose a safe, bounded RewardProfile to guide the PPO policy.\n\n"
    "RULES & CONSTRAINTS:\n"
    "1. You cannot alter game rules, map layouts, physics, or action legality.\n"
    "2. You must output strictly valid JSON matching the provided schema.\n"
    "3. NEVER output Python code, scripts, or markdown outside JSON.\n"
    "4. All numeric reward weights must strictly observe the allowed bounds.\n"
    "5. Make targeted, incremental adjustments (e.g. modify at most 1 or 2 reward weights).\n"
    "6. Clearly articulate a testable hypothesis linking the reward change to observed metrics.\n"
    "7. If performance is already strong or evidence is inconclusive, retain the current weights."
)


def get_reward_profile_json_schema() -> Dict[str, Any]:
    """Return JSON schema defining the structured reward profile proposal."""
    properties: Dict[str, Any] = {
        "profile_id": {
            "type": "string",
            "description": "Unique identifier for this proposed candidate profile (e.g. 'llm_cand_seg1')",
            "maxLength": 64,
        },
        "rationale": {
            "type": "string",
            "description": "Concise plain-text diagnosis of current performance and reason for proposed weights.",
            "maxLength": 1000,
        },
        "hypothesis": {
            "type": "string",
            "description": "Specific measurable expectation (e.g., 'Increasing leak penalty from -10 to -15 will reduce leaks by 20%').",
            "maxLength": 1000,
        },
        "created_by": {
            "type": "string",
            "enum": ["llm"],
        },
        "schema_version": {
            "type": "integer",
            "enum": [CURRENT_SCHEMA_VERSION],
        },
    }

    # Add numeric properties with minimum and maximum bounds
    for weight_name, (low, high) in REWARD_BOUNDS.items():
        properties[weight_name] = {
            "type": "number",
            "minimum": low,
            "maximum": high,
            "description": f"Weight for {weight_name} in range [{low}, {high}]",
        }

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "RewardProfileProposal",
        "type": "object",
        "required": [
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
            "schema_version",
        ],
        "properties": properties,
        "additionalProperties": False,
    }


def build_advisor_prompt(
    current_profile: RewardProfile,
    metrics: EvaluationMetrics,
    segment_idx: int,
    total_segments: int,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Build the system prompt, factual user payload, and JSON schema for the LLM advisor.

    Returns:
        (system_prompt, user_payload, json_schema)
    """
    user_payload: Dict[str, Any] = {
        "training_progress": {
            "completed_segment": segment_idx,
            "total_segments": total_segments,
        },
        "current_reward_profile": current_profile.to_dict(),
        "measured_evaluation_metrics": metrics.to_dict(),
        "allowed_reward_bounds": {k: {"min": v[0], "max": v[1]} for k, v in REWARD_BOUNDS.items()},
        "recommendation_guidance": (
            "Propose one bounded candidate RewardProfile or confirm current weights. "
            "Explain what metric weakness is targeted."
        ),
    }

    if history:
        user_payload["segment_history"] = history[-3:]  # Include up to last 3 segments

    return SYSTEM_ADVISOR_PROMPT, user_payload, get_reward_profile_json_schema()
