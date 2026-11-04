"""Real LLM provider adapters for OpenAI, Google Gemini, Anthropic, and local Ollama."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from towermind.llm.base import (
    LLMAuthenticationError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    LLMSchemaError,
    LLMTimeoutError,
    LLMUsage,
)
from towermind.llm.mock_provider import MockLLMProvider


class OpenAIAdvisorProvider(LLMProvider):
    """Adapter for OpenAI API (e.g. gpt-4o, gpt-4o-mini)."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._provider_name = "openai"
        self._model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_structured(
        self,
        system_prompt: str,
        user_payload: Dict[str, Any],
        schema: Dict[str, Any],
        timeout: float = 30.0,
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMAuthenticationError("OPENAI_API_KEY is missing or empty")

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        prompt_content = (
            f"{system_prompt}\n\n"
            f"INPUT CONTEXT:\n{json.dumps(user_payload, indent=2)}\n\n"
            f"REQUIRED JSON SCHEMA:\n{json.dumps(schema, indent=2)}\n\n"
            "Return valid JSON strictly matching the schema."
        )

        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        start_t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                resp_bytes = response.read()
                data = json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise LLMAuthenticationError(f"OpenAI Authentication error ({e.code}): {e.reason}") from e
            elif e.code == 429:
                raise LLMRateLimitError(f"OpenAI Rate limit (429): {e.reason}") from e
            raise LLMError(f"OpenAI HTTP error {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            if "timed out" in str(e.reason).lower():
                raise LLMTimeoutError(f"OpenAI connection timed out: {e.reason}") from e
            raise LLMError(f"OpenAI connection error: {e.reason}") from e

        latency = time.time() - start_t
        choice = data.get("choices", [{}])[0]
        content_str = choice.get("message", {}).get("content", "")

        try:
            structured = json.loads(content_str)
        except Exception as e:
            raise LLMSchemaError(f"Failed to parse OpenAI JSON response: {e}\nRaw: {content_str}") from e

        usage_info = data.get("usage", {})
        prompt_toks = usage_info.get("prompt_tokens", 0)
        comp_toks = usage_info.get("completion_tokens", 0)
        # Approximate cost for gpt-4o-mini ($0.15/1M in, $0.60/1M out)
        cost = (prompt_toks * 0.15 + comp_toks * 0.60) / 1_000_000

        return LLMResponse(
            raw_content=content_str,
            structured_data=structured,
            provider_name=self.provider_name,
            model_name=self.model_name,
            latency_seconds=latency,
            usage=LLMUsage(
                prompt_tokens=prompt_toks,
                completion_tokens=comp_toks,
                total_tokens=prompt_toks + comp_toks,
                estimated_cost_usd=cost,
            ),
        )


class OllamaAdvisorProvider(LLMProvider):
    """Adapter for local Ollama instances (zero cost, local execution)."""

    def __init__(
        self,
        model_name: str = "llama3.2",
        base_url: Optional[str] = None,
    ):
        self._provider_name = "ollama"
        self._model_name = model_name
        self.base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_structured(
        self,
        system_prompt: str,
        user_payload: Dict[str, Any],
        schema: Dict[str, Any],
        timeout: float = 60.0,
    ) -> LLMResponse:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        headers = {"Content-Type": "application/json"}
        prompt_content = (
            f"{system_prompt}\n\n"
            f"INPUT CONTEXT:\n{json.dumps(user_payload, indent=2)}\n\n"
            f"JSON SCHEMA:\n{json.dumps(schema, indent=2)}\n\n"
            "Respond ONLY with valid JSON."
        )

        body = {
            "model": self.model_name,
            "prompt": prompt_content,
            "format": "json",
            "stream": False,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        start_t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                resp_bytes = response.read()
                data = json.loads(resp_bytes.decode("utf-8"))
        except Exception as e:
            raise LLMError(f"Ollama local connection failed: {e}") from e

        latency = time.time() - start_t
        content_str = data.get("response", "")
        try:
            structured = json.loads(content_str)
        except Exception as e:
            raise LLMSchemaError(f"Failed to parse Ollama JSON: {e}\nRaw: {content_str}") from e

        prompt_toks = data.get("prompt_eval_count", 0)
        comp_toks = data.get("eval_count", 0)

        return LLMResponse(
            raw_content=content_str,
            structured_data=structured,
            provider_name=self.provider_name,
            model_name=self.model_name,
            latency_seconds=latency,
            usage=LLMUsage(
                prompt_tokens=prompt_toks,
                completion_tokens=comp_toks,
                total_tokens=prompt_toks + comp_toks,
                estimated_cost_usd=0.0,  # Local model is free
            ),
        )


def create_advisor_provider(
    provider_name: str = "mock",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> LLMProvider:
    """Factory creating an LLMProvider instance based on provider name.

    Supported: 'mock', 'openai', 'ollama'.
    """
    p_name = provider_name.lower().strip()
    if p_name == "mock":
        return MockLLMProvider(model_name=model_name or "mock-advisor-v1", **kwargs)
    elif p_name == "openai":
        return OpenAIAdvisorProvider(model_name=model_name or "gpt-4o-mini", api_key=api_key, **kwargs)
    elif p_name == "ollama":
        return OllamaAdvisorProvider(model_name=model_name or "llama3.2", **kwargs)
    else:
        raise ValueError(f"Unsupported provider '{provider_name}'. Choose from 'mock', 'openai', 'ollama'.")
