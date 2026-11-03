"""Base abstractions, protocols, usage schemas, and exceptions for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class LLMError(Exception):
    """Base exception for all LLM provider and gateway errors."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM provider call exceeds the timeout threshold."""


class LLMAuthenticationError(LLMError):
    """Raised when authentication with the provider fails (non-retryable)."""


class LLMSchemaError(LLMError):
    """Raised when the LLM response fails schema parsing or structural validation."""


class LLMRateLimitError(LLMError):
    """Raised when an LLM provider rate limit (429) is hit (retryable)."""


class LLMBudgetError(LLMError):
    """Raised when call count or monetary budget is exceeded for an experiment run."""


@dataclass
class LLMUsage:
    """Token counts and financial cost telemetry."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def add(self, other: LLMUsage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.estimated_cost_usd += other.estimated_cost_usd

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


@dataclass
class LLMResponse:
    """Standardized response container across all provider backends."""

    raw_content: str
    structured_data: Optional[Dict[str, Any]] = None
    provider_name: str = ""
    model_name: str = ""
    latency_seconds: float = 0.0
    usage: LLMUsage = field(default_factory=LLMUsage)
    error: Optional[str] = None
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "latency_seconds": round(self.latency_seconds, 4),
            "usage": self.usage.to_dict(),
            "cached": self.cached,
            "error": self.error,
            "structured_data": self.structured_data,
        }


class LLMProvider(ABC):
    """Abstract protocol for an LLM provider adapter."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider service (e.g. 'mock', 'openai', 'gemini')."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the specific model (e.g. 'mock-v1', 'gpt-4o-mini', 'gemini-1.5-flash')."""

    @abstractmethod
    def generate_structured(
        self,
        system_prompt: str,
        user_payload: Dict[str, Any],
        schema: Dict[str, Any],
        timeout: float = 30.0,
    ) -> LLMResponse:
        """Generate a structured JSON output satisfying the target schema.

        Args:
            system_prompt: System-level role and constraints.
            user_payload: Input metrics and context dict.
            schema: JSON schema dict required for output.
            timeout: Maximum seconds to wait.

        Returns:
            LLMResponse containing parsed dict in structured_data and usage stats.
        """
