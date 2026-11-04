"""Candidate evaluation gate and promotion rule for meta-controller proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from towermind.experiments.manifest import EvaluationMetrics
from towermind.rewards.profile import RewardProfile


@dataclass(frozen=True)
class PromotionResult:
    """Outcome of evaluating a candidate profile against guardrails."""

    promoted: bool
    reason: str
    active_profile: RewardProfile
    candidate_profile: Optional[RewardProfile] = None


class PromotionGate:
    """Evaluates proposed reward profile candidates against performance guardrails."""

    def __init__(
        self,
        min_legality_rate: float = 0.90,
        max_allowed_win_rate_drop: float = 0.05,
        max_allowed_waves_drop: float = 0.5,
    ):
        self.min_legality_rate = min_legality_rate
        self.max_allowed_win_rate_drop = max_allowed_win_rate_drop
        self.max_allowed_waves_drop = max_allowed_waves_drop

    def evaluate(
        self,
        current_profile: RewardProfile,
        candidate_profile: RewardProfile,
        current_metrics: EvaluationMetrics,
        pilot_metrics: Optional[EvaluationMetrics] = None,
    ) -> PromotionResult:
        """Decide whether to promote candidate_profile or retain current_profile.

        Args:
            current_profile: Currently active RewardProfile.
            candidate_profile: Newly proposed RewardProfile from advisor.
            current_metrics: Metrics under the current profile.
            pilot_metrics: Optional pilot evaluation metrics measured under candidate profile.
        """
        # 1. Parity check
        if candidate_profile == current_profile:
            return PromotionResult(
                promoted=False,
                reason="Candidate profile identical to active profile; retaining current.",
                active_profile=current_profile,
                candidate_profile=candidate_profile,
            )

        # 2. Legality guardrail check
        if current_metrics.legality_rate < self.min_legality_rate:
            return PromotionResult(
                promoted=False,
                reason=(
                    f"Policy legality rate ({current_metrics.legality_rate:.2f}) is below safe threshold "
                    f"({self.min_legality_rate:.2f}); stabilizing with current profile."
                ),
                active_profile=current_profile,
                candidate_profile=candidate_profile,
            )

        # 3. If pilot metrics are available, evaluate performance changes
        if pilot_metrics is not None:
            # Check for catastrophic win rate regression
            win_drop = current_metrics.win_rate - pilot_metrics.win_rate
            if win_drop > self.max_allowed_win_rate_drop:
                return PromotionResult(
                    promoted=False,
                    reason=(
                        f"Pilot win rate ({pilot_metrics.win_rate:.2f}) dropped more than allowed tolerance "
                        f"({win_drop:.2f} > {self.max_allowed_win_rate_drop:.2f})."
                    ),
                    active_profile=current_profile,
                    candidate_profile=candidate_profile,
                )

            # Check for catastrophic wave regression
            waves_drop = current_metrics.mean_waves - pilot_metrics.mean_waves
            if waves_drop > self.max_allowed_waves_drop:
                return PromotionResult(
                    promoted=False,
                    reason=(
                        f"Pilot waves survived ({pilot_metrics.mean_waves:.2f}) dropped significantly below current "
                        f"({current_metrics.mean_waves:.2f})."
                    ),
                    active_profile=current_profile,
                    candidate_profile=candidate_profile,
                )

            # Check for positive progress or acceptable exploration
            improved = (
                pilot_metrics.win_rate > current_metrics.win_rate
                or pilot_metrics.mean_waves > current_metrics.mean_waves
                or pilot_metrics.mean_reward >= current_metrics.mean_reward
            )
            if improved:
                return PromotionResult(
                    promoted=True,
                    reason=(
                        f"Pilot evaluation showed improvement or parity: "
                        f"waves={pilot_metrics.mean_waves:.2f} vs {current_metrics.mean_waves:.2f}, "
                        f"win_rate={pilot_metrics.win_rate:.2f} vs {current_metrics.win_rate:.2f}."
                    ),
                    active_profile=candidate_profile,
                    candidate_profile=candidate_profile,
                )
            else:
                return PromotionResult(
                    promoted=False,
                    reason="Pilot metrics did not meet promotion threshold; retaining current profile.",
                    active_profile=current_profile,
                    candidate_profile=candidate_profile,
                )

        # 4. Without pilot evaluation, accept candidate if well-formed and current policy is healthy
        return PromotionResult(
            promoted=True,
            reason="Candidate validated and policy legality healthy; promoted for next training segment.",
            active_profile=candidate_profile,
            candidate_profile=candidate_profile,
        )
