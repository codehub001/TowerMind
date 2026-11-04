"""TowerMind meta-controller: advisor prompt builder, promotion gate, and training loop."""

from towermind.meta_controller.advisor import (
    SYSTEM_ADVISOR_PROMPT,
    build_advisor_prompt,
    get_reward_profile_json_schema,
)
from towermind.meta_controller.evaluator import evaluate_policy
from towermind.meta_controller.loop import (
    run_segmented_experiment,
    train_ppo_segment,
)
from towermind.meta_controller.promotion import (
    PromotionGate,
    PromotionResult,
)

__all__ = [
    "SYSTEM_ADVISOR_PROMPT",
    "PromotionGate",
    "PromotionResult",
    "build_advisor_prompt",
    "evaluate_policy",
    "get_reward_profile_json_schema",
    "run_segmented_experiment",
    "train_ppo_segment",
]
