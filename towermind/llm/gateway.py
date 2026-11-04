"""Safe LLM gateway with timeouts, retries, budget enforcement, caching, and fallback."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from towermind.llm.base import (
    LLMAuthenticationError,
    LLMBudgetError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    LLMSchemaError,
    LLMTimeoutError,
    LLMUsage,
)
from towermind.rewards.profile import RewardProfile, get_baseline_profile
from towermind.rewards.validator import ValidationResult, validate_profile_dict

logger = logging.getLogger(__name__)


@dataclass
class GatewayConfig:
    """Configuration and guardrail limits for the LLM gateway."""

    timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_delay_seconds: float = 0.5
    retry_backoff_factor: float = 2.0
    budget_max_requests: int = 25
    budget_max_cost_usd: float = 2.00
    cache_enabled: bool = True


@dataclass
class GatewayProposalResult:
    """Outcome of an LLM-assisted reward profile proposal."""

    profile: RewardProfile
    is_fallback: bool
    fallback_reason: Optional[str] = None
    response: Optional[LLMResponse] = None
    validation_result: Optional[ValidationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
            "response": self.response.to_dict() if self.response else None,
            "validation_errors": self.validation_result.errors if self.validation_result else [],
        }


class LLMGateway:
    """Safe intermediary mediating between the RL training loop and external LLMs."""

    def __init__(self, provider: LLMProvider, config: Optional[GatewayConfig] = None):
        self.provider = provider
        self.config = config or GatewayConfig()

        # Telemetry & Budget
        self.total_requests: int = 0
        self.total_retries: int = 0
        self.total_fallbacks: int = 0
        self.total_usage = LLMUsage()
        self._cache: Dict[str, LLMResponse] = {}

    def _cache_key(
        self,
        system_prompt: str,
        user_payload: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> str:
        blob = json.dumps(
            {"p": system_prompt, "u": user_payload, "s": schema},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _check_budget(self) -> None:
        if self.total_requests >= self.config.budget_max_requests:
            raise LLMBudgetError(
                f"Request budget exceeded: {self.total_requests} >= {self.config.budget_max_requests}"
            )
        if self.total_usage.estimated_cost_usd >= self.config.budget_max_cost_usd:
            raise LLMBudgetError(
                f"Monetary budget exceeded: ${self.total_usage.estimated_cost_usd:.4f} >= ${self.config.budget_max_cost_usd:.4f}"
            )

    def execute_request(
        self,
        system_prompt: str,
        user_payload: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> LLMResponse:
        """Execute request with retries, timeout, budget checks, and caching."""
        self._check_budget()

        # Check cache
        cache_key = self._cache_key(system_prompt, user_payload, schema)
        if self.config.cache_enabled and cache_key in self._cache:
            cached_resp = self._cache[cache_key]
            return LLMResponse(
                raw_content=cached_resp.raw_content,
                structured_data=cached_resp.structured_data,
                provider_name=cached_resp.provider_name,
                model_name=cached_resp.model_name,
                latency_seconds=0.0,
                usage=LLMUsage(),
                cached=True,
            )

        attempt = 0
        delay = self.config.retry_delay_seconds
        last_exception: Optional[Exception] = None

        while attempt <= self.config.max_retries:
            attempt += 1
            self.total_requests += 1

            try:
                response = self.provider.generate_structured(
                    system_prompt=system_prompt,
                    user_payload=user_payload,
                    schema=schema,
                    timeout=self.config.timeout_seconds,
                )
                self.total_usage.add(response.usage)

                # Store in cache on success
                if self.config.cache_enabled:
                    self._cache[cache_key] = response

                return response

            except (LLMAuthenticationError, LLMSchemaError) as fatal_err:
                # Permanent failure — do not retry
                logger.error("Non-retryable LLM provider error: %s", fatal_err)
                raise fatal_err

            except (LLMTimeoutError, LLMRateLimitError, Exception) as transient_err:
                last_exception = transient_err
                if attempt <= self.config.max_retries:
                    self.total_retries += 1
                    logger.warning(
                        "LLM call attempt %d failed with %s; retrying in %.2fs...",
                        attempt,
                        transient_err,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= self.config.retry_backoff_factor
                else:
                    break

        raise last_exception or LLMError("LLM call failed after retries")

    def propose_reward_profile(
        self,
        system_prompt: str,
        user_payload: Dict[str, Any],
        schema: Dict[str, Any],
        fallback_profile: Optional[RewardProfile] = None,
    ) -> GatewayProposalResult:
        """Request candidate RewardProfile from LLM, safely falling back on any failure.

        Guarantees:
        1. Always returns a valid, bounded RewardProfile.
        2. Never throws an unhandled exception into the training loop.
        3. Records whether fallback was triggered and the exact reason.
        """
        fallback = fallback_profile or get_baseline_profile()

        try:
            response = self.execute_request(
                system_prompt=system_prompt,
                user_payload=user_payload,
                schema=schema,
            )
        except Exception as e:
            self.total_fallbacks += 1
            logger.warning("LLM call failed; falling back to profile %s: %s", fallback.profile_id, e)
            return GatewayProposalResult(
                profile=fallback,
                is_fallback=True,
                fallback_reason=f"LLM request error: {e}",
            )

        # Parse and validate response
        if response.error or not response.structured_data:
            self.total_fallbacks += 1
            reason = response.error or "Empty structured data in response"
            return GatewayProposalResult(
                profile=fallback,
                is_fallback=True,
                fallback_reason=reason,
                response=response,
            )

        # Schema & bounds validation
        val_result = validate_profile_dict(response.structured_data)
        if not val_result.is_valid:
            self.total_fallbacks += 1
            err_msg = "; ".join(val_result.errors)
            logger.warning("Proposed profile failed validation: %s", err_msg)
            return GatewayProposalResult(
                profile=fallback,
                is_fallback=True,
                fallback_reason=f"Validation failed: {err_msg}",
                response=response,
                validation_result=val_result,
            )

        assert val_result.profile is not None
        return GatewayProposalResult(
            profile=val_result.profile,
            is_fallback=False,
            response=response,
            validation_result=val_result,
        )

    def telemetry_summary(self) -> Dict[str, Any]:
        """Aggregate telemetry stats for reporting."""
        return {
            "total_requests": self.total_requests,
            "total_retries": self.total_retries,
            "total_fallbacks": self.total_fallbacks,
            "tokens": self.total_usage.to_dict(),
        }
