"""TowerMind LLM integration: provider abstractions, mock provider, and safe gateway."""

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
from towermind.llm.gateway import GatewayConfig, LLMGateway
from towermind.llm.mock_provider import MockLLMProvider
from towermind.llm.providers import (
    OllamaAdvisorProvider,
    OpenAIAdvisorProvider,
    create_advisor_provider,
)

__all__ = [
    "GatewayConfig",
    "LLMAuthenticationError",
    "LLMBudgetError",
    "LLMError",
    "LLMGateway",
    "LLMProvider",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMSchemaError",
    "LLMTimeoutError",
    "LLMUsage",
    "MockLLMProvider",
    "OllamaAdvisorProvider",
    "OpenAIAdvisorProvider",
    "create_advisor_provider",
]
