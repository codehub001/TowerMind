"""Deterministic mock LLM provider for offline development, tests, and CI."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional

from towermind.llm.base import (
    LLMAuthenticationError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    LLMUsage,
)


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider with configurable test scenarios."""

    def __init__(
        self,
        model_name: str = "mock-advisor-v1",
        default_payload: Optional[Dict[str, Any]] = None,
        simulate_timeout: bool = False,
        simulate_auth_error: bool = False,
        simulate_rate_limit: bool = False,
        simulate_malformed_json: bool = False,
        simulate_invalid_bounds: bool = False,
        simulate_unknown_fields: bool = False,
    ):
        self._provider_name = "mock"
        self._model_name = model_name
        self._call_count = 0
        self._history: List[Dict[str, Any]] = []

        # Fault injection toggles
        self.simulate_timeout = simulate_timeout
        self.simulate_auth_error = simulate_auth_error
        self.simulate_rate_limit = simulate_rate_limit
        self.simulate_malformed_json = simulate_malformed_json
        self.simulate_invalid_bounds = simulate_invalid_bounds
        self.simulate_unknown_fields = simulate_unknown_fields

        # Scripted responses or default valid response
        self._scripted_responses: List[Dict[str, Any]] = []
        self._response_generator: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

        self.default_payload = default_payload or {
            "profile_id": "llm-cand-001",
            "kill": 1.5,
            "leak": -15.0,
            "win": 60.0,
            "lose": -60.0,
            "step": 0.0,
            "invalid_action": -0.2,
            "economy": 0.0,
            "scale": 0.1,
            "rationale": "Penalize enemy leaks more heavily to encourage front-line defenses.",
            "hypothesis": "Higher leak penalty will reduce average leaks per wave.",
            "created_by": "llm",
            "schema_version": 1,
        }

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def history(self) -> List[Dict[str, Any]]:
        return self._history

    def set_scripted_responses(self, responses: List[Dict[str, Any]]) -> None:
        """Enqueue fixed sequential response dicts."""
        self._scripted_responses = list(responses)

    def set_generator(self, gen_fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """Set a dynamic callback that generates the structured response dict."""
        self._response_generator = gen_fn

    def generate_structured(
        self,
        system_prompt: str,
        user_payload: Dict[str, Any],
        schema: Dict[str, Any],
        timeout: float = 30.0,
    ) -> LLMResponse:
        self._call_count += 1
        start_time = time.time()

        self._history.append(
            {
                "system_prompt": system_prompt,
                "user_payload": user_payload,
                "schema": schema,
                "timeout": timeout,
            }
        )

        if self.simulate_timeout:
            raise LLMTimeoutError(f"Mock request timed out after {timeout}s")

        if self.simulate_auth_error:
            raise LLMAuthenticationError("Mock provider invalid API credentials")

        if self.simulate_rate_limit:
            raise LLMRateLimitError("Mock provider rate limit exceeded (429)")

        if self.simulate_malformed_json:
            return LLMResponse(
                raw_content="This is not valid JSON! {incomplete...",
                structured_data=None,
                provider_name=self.provider_name,
                model_name=self.model_name,
                latency_seconds=time.time() - start_time,
                error="JSONDecodeError: Unterminated object",
            )

        if self.simulate_invalid_bounds:
            payload = dict(self.default_payload)
            payload["kill"] = 9999.0  # violates bounds
            return LLMResponse(
                raw_content=json.dumps(payload),
                structured_data=payload,
                provider_name=self.provider_name,
                model_name=self.model_name,
                latency_seconds=time.time() - start_time,
                usage=LLMUsage(
                    prompt_tokens=120, completion_tokens=65, total_tokens=185, estimated_cost_usd=0.0003
                ),
            )

        if self.simulate_unknown_fields:
            payload = dict(self.default_payload)
            payload["prohibited_extra_key"] = "malicious_or_unknown"
            return LLMResponse(
                raw_content=json.dumps(payload),
                structured_data=payload,
                provider_name=self.provider_name,
                model_name=self.model_name,
                latency_seconds=time.time() - start_time,
                usage=LLMUsage(
                    prompt_tokens=120, completion_tokens=65, total_tokens=185, estimated_cost_usd=0.0003
                ),
            )

        # Normal response generation
        if self._scripted_responses:
            data = self._scripted_responses.pop(0)
        elif self._response_generator is not None:
            data = self._response_generator(user_payload)
        else:
            data = dict(self.default_payload)
            # Make profile_id unique per call if using default
            data["profile_id"] = f"{data['profile_id']}_{self._call_count}"

        raw_str = json.dumps(data)
        latency = time.time() - start_time

        return LLMResponse(
            raw_content=raw_str,
            structured_data=data,
            provider_name=self.provider_name,
            model_name=self.model_name,
            latency_seconds=max(0.001, latency),
            usage=LLMUsage(
                prompt_tokens=150,
                completion_tokens=80,
                total_tokens=230,
                estimated_cost_usd=0.0005,
            ),
        )
