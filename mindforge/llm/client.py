"""LLM client: HTTP interface to Ollama and OpenAI-compatible APIs.

Uses only stdlib (urllib) — no SDK dependencies required.
Supports:
- Ollama (default: http://localhost:11434)
- Any OpenAI-compatible API (OpenAI, Together, vLLM, LM Studio, etc.)
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""
    provider: str = "ollama"  # "ollama" or "openai"
    model: str = "llama3.2"
    base_url: str = ""  # Auto-set based on provider if empty
    api_key: str = ""  # Required for OpenAI provider
    temperature: float = 0.1  # Low temp for deterministic extraction
    max_tokens: int = 4096
    timeout: int = 120  # seconds

    def __post_init__(self) -> None:
        if not self.base_url:
            if self.provider == "ollama":
                self.base_url = "http://localhost:11434"
            elif self.provider == "openai":
                self.base_url = "https://api.openai.com"


@dataclass
class LLMResponse:
    """A response from the LLM."""
    content: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    success: bool = True
    error: str = ""


class LLMClient:
    """HTTP client for LLM inference.

    Supports Ollama's /api/generate and OpenAI's /v1/chat/completions.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        """Check if the LLM server is reachable."""
        if self._available is not None:
            return self._available
        self._available = self._check_health()
        return self._available

    def _check_health(self) -> bool:
        """Ping the LLM server to check availability."""
        try:
            if self.config.provider == "ollama":
                url = f"{self.config.base_url}/api/tags"
            else:
                url = f"{self.config.base_url}/v1/models"

            req = urllib.request.Request(url, method="GET")
            if self.config.api_key:
                req.add_header("Authorization", f"Bearer {self.config.api_key}")

            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        """Send a prompt to the LLM and return the response."""
        if self.config.provider == "ollama":
            return self._generate_ollama(prompt, system)
        else:
            return self._generate_openai(prompt, system)

    def _generate_ollama(self, prompt: str, system: str) -> LLMResponse:
        """Generate via Ollama's /api/generate endpoint."""
        url = f"{self.config.base_url}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        if system:
            payload["system"] = system

        return self._post_json(url, payload, self._parse_ollama_response)

    def _generate_openai(self, prompt: str, system: str) -> LLMResponse:
        """Generate via OpenAI-compatible /v1/chat/completions endpoint."""
        url = f"{self.config.base_url}/v1/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        return self._post_json(url, payload, self._parse_openai_response)

    def _post_json(self, url: str, payload: dict, parser) -> LLMResponse:
        """Send a JSON POST request and parse the response."""
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            if self.config.api_key:
                req.add_header("Authorization", f"Bearer {self.config.api_key}")

            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return parser(body)

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            logger.error("LLM HTTP error %d: %s", e.code, error_body[:500])
            return LLMResponse(
                content="", success=False,
                error=f"HTTP {e.code}: {error_body[:200]}",
            )
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            logger.error("LLM connection error: %s", e)
            return LLMResponse(
                content="", success=False,
                error=f"Connection error: {e}",
            )

    @staticmethod
    def _parse_ollama_response(body: dict) -> LLMResponse:
        return LLMResponse(
            content=body.get("response", ""),
            model=body.get("model", ""),
            prompt_tokens=body.get("prompt_eval_count", 0),
            completion_tokens=body.get("eval_count", 0),
        )

    @staticmethod
    def _parse_openai_response(body: dict) -> LLMResponse:
        choices = body.get("choices", [])
        content = ""
        if choices:
            content = choices[0].get("message", {}).get("content", "")
        usage = body.get("usage", {})
        return LLMResponse(
            content=content,
            model=body.get("model", ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
