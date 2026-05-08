"""Deterministic content-derivative LLM client for pipeline tests.

Produces wire-format JSON responses derived from chunk text via a fixed
rule. No network. Reproducible byte-for-byte across runs. Selected via
`config.llm.provider == "mock"` and the make_llm_client factory.
"""

from __future__ import annotations

from mindforge.llm.client import LLMClient, LLMConfig, LLMResponse


class MockLLMClient(LLMClient):
    """Deterministic mock that returns canned JSON without network calls."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        super().__init__(config)
        self._available = True

    def generate(self, prompt: str, system: str = "", response_format: str = "") -> LLMResponse:
        # Implementation in Task 3.
        return LLMResponse(content='{"concepts": []}', success=True)
