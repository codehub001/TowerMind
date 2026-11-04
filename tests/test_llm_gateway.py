"""Unit tests for MockLLMProvider and LLMGateway."""

import pytest

from towermind.llm.base import (
    LLMAuthenticationError,
)
from towermind.llm.gateway import GatewayConfig, LLMGateway
from towermind.llm.mock_provider import MockLLMProvider
from towermind.rewards.profile import get_baseline_profile

SYSTEM_PROMPT = "You are an RL experiment advisor."
USER_PAYLOAD = {"win_rate": 0.2, "average_leaks": 5.4}
SCHEMA = {"type": "object"}


def test_mock_provider_normal_proposal():
    provider = MockLLMProvider()
    gateway = LLMGateway(provider, GatewayConfig(cache_enabled=False))
    fallback = get_baseline_profile("baseline-v1")

    res = gateway.propose_reward_profile(
        system_prompt=SYSTEM_PROMPT,
        user_payload=USER_PAYLOAD,
        schema=SCHEMA,
        fallback_profile=fallback,
    )

    assert not res.is_fallback
    assert res.fallback_reason is None
    assert res.profile.profile_id.startswith("llm-cand-001")
    assert res.profile.leak == -15.0
    assert res.profile.created_by == "llm"
    assert res.response is not None
    assert not res.response.cached


def test_gateway_caching():
    provider = MockLLMProvider()
    gateway = LLMGateway(provider, GatewayConfig(cache_enabled=True))
    fallback = get_baseline_profile()

    res1 = gateway.propose_reward_profile(SYSTEM_PROMPT, USER_PAYLOAD, SCHEMA, fallback)
    assert not res1.is_fallback
    assert not res1.response.cached

    # Second call with identical prompt and payload should hit cache
    res2 = gateway.propose_reward_profile(SYSTEM_PROMPT, USER_PAYLOAD, SCHEMA, fallback)
    assert not res2.is_fallback
    assert res2.response.cached
    assert provider.call_count == 1  # Provider called only once


def test_fallback_on_timeout():
    provider = MockLLMProvider(simulate_timeout=True)
    cfg = GatewayConfig(max_retries=1, retry_delay_seconds=0.01, cache_enabled=False)
    gateway = LLMGateway(provider, cfg)
    fallback = get_baseline_profile("fallback-target")

    res = gateway.propose_reward_profile(SYSTEM_PROMPT, USER_PAYLOAD, SCHEMA, fallback)
    assert res.is_fallback
    assert res.profile.profile_id == "fallback-target"
    assert "Mock request timed out" in str(res.fallback_reason)
    assert gateway.total_fallbacks == 1


def test_fallback_on_bounds_violation():
    provider = MockLLMProvider(simulate_invalid_bounds=True)
    gateway = LLMGateway(provider, GatewayConfig(cache_enabled=False))
    fallback = get_baseline_profile("fallback-bounds")

    res = gateway.propose_reward_profile(SYSTEM_PROMPT, USER_PAYLOAD, SCHEMA, fallback)
    assert res.is_fallback
    assert res.profile.profile_id == "fallback-bounds"
    assert "out of permitted range" in str(res.fallback_reason)


def test_fallback_on_unknown_fields():
    provider = MockLLMProvider(simulate_unknown_fields=True)
    gateway = LLMGateway(provider, GatewayConfig(cache_enabled=False))
    fallback = get_baseline_profile("fallback-fields")

    res = gateway.propose_reward_profile(SYSTEM_PROMPT, USER_PAYLOAD, SCHEMA, fallback)
    assert res.is_fallback
    assert res.profile.profile_id == "fallback-fields"
    assert "Disallowed extra fields" in str(res.fallback_reason)


def test_fallback_on_malformed_json():
    provider = MockLLMProvider(simulate_malformed_json=True)
    gateway = LLMGateway(provider, GatewayConfig(cache_enabled=False))
    fallback = get_baseline_profile("fallback-json")

    res = gateway.propose_reward_profile(SYSTEM_PROMPT, USER_PAYLOAD, SCHEMA, fallback)
    assert res.is_fallback
    assert res.profile.profile_id == "fallback-json"
    assert "JSONDecodeError" in str(res.fallback_reason)


def test_auth_error_is_fatal_no_retries():
    provider = MockLLMProvider(simulate_auth_error=True)
    cfg = GatewayConfig(max_retries=3, cache_enabled=False)
    gateway = LLMGateway(provider, cfg)
    fallback = get_baseline_profile()

    res = gateway.propose_reward_profile(SYSTEM_PROMPT, USER_PAYLOAD, SCHEMA, fallback)
    assert res.is_fallback
    assert "invalid API credentials" in str(res.fallback_reason)
    # Total requests should be 1 because auth errors are permanent and not retried
    assert gateway.total_requests == 1
    assert gateway.total_retries == 0


def test_budget_exhaustion():
    provider = MockLLMProvider()
    cfg = GatewayConfig(budget_max_requests=2, cache_enabled=False)
    gateway = LLMGateway(provider, cfg)
    fallback = get_baseline_profile()

    gateway.propose_reward_profile(SYSTEM_PROMPT, {"run": 1}, SCHEMA, fallback)
    gateway.propose_reward_profile(SYSTEM_PROMPT, {"run": 2}, SCHEMA, fallback)

    # 3rd request exceeds budget
    res = gateway.propose_reward_profile(SYSTEM_PROMPT, {"run": 3}, SCHEMA, fallback)
    assert res.is_fallback
    assert "Request budget exceeded" in str(res.fallback_reason)


def test_telemetry_summary():
    provider = MockLLMProvider()
    gateway = LLMGateway(provider, GatewayConfig(cache_enabled=False))
    fallback = get_baseline_profile()

    gateway.propose_reward_profile(SYSTEM_PROMPT, USER_PAYLOAD, SCHEMA, fallback)
    summary = gateway.telemetry_summary()

    assert summary["total_requests"] == 1
    assert summary["total_fallbacks"] == 0
    assert summary["tokens"]["total_tokens"] > 0


def test_provider_factory_and_real_adapters():
    from towermind.llm.providers import (
        OllamaAdvisorProvider,
        OpenAIAdvisorProvider,
        create_advisor_provider,
    )

    # 1. Mock creation
    mock_p = create_advisor_provider("mock", model_name="custom-mock")
    assert isinstance(mock_p, MockLLMProvider)
    assert mock_p.model_name == "custom-mock"

    # 2. OpenAI creation
    openai_p = create_advisor_provider("openai", api_key="sk-test12345")
    assert isinstance(openai_p, OpenAIAdvisorProvider)
    assert openai_p.provider_name == "openai"

    # 3. Ollama creation
    ollama_p = create_advisor_provider("ollama", model_name="llama3.2")
    assert isinstance(ollama_p, OllamaAdvisorProvider)
    assert ollama_p.provider_name == "ollama"

    # 4. Invalid provider
    with pytest.raises(ValueError, match="Unsupported provider"):
        create_advisor_provider("unknown_provider")

    # 5. Missing API key raises auth error
    no_key_provider = OpenAIAdvisorProvider(api_key="")
    with pytest.raises(LLMAuthenticationError, match="OPENAI_API_KEY is missing"):
        no_key_provider.generate_structured(SYSTEM_PROMPT, USER_PAYLOAD, SCHEMA)
