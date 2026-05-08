"""Tests for MockLLMClient and make_llm_client factory."""

from __future__ import annotations

import pytest

from mindforge.llm.client import LLMClient, LLMConfig, make_llm_client
from mindforge.llm.mock import MockLLMClient


class TestFactory:
    def test_factory_returns_mock_for_mock_provider(self) -> None:
        cfg = LLMConfig(provider="mock")
        client = make_llm_client(cfg)
        assert isinstance(client, MockLLMClient)

    def test_factory_returns_real_client_for_ollama(self) -> None:
        cfg = LLMConfig(provider="ollama")
        client = make_llm_client(cfg)
        assert isinstance(client, LLMClient)
        assert not isinstance(client, MockLLMClient)

    def test_factory_returns_real_client_for_openai(self) -> None:
        cfg = LLMConfig(provider="openai")
        client = make_llm_client(cfg)
        assert isinstance(client, LLMClient)
        assert not isinstance(client, MockLLMClient)

    def test_factory_rejects_unknown_provider(self) -> None:
        cfg = LLMConfig(provider="qwen")
        with pytest.raises(ValueError, match="unknown LLM provider: 'qwen'"):
            make_llm_client(cfg)


class TestMockClientAvailability:
    def test_mock_is_always_available(self) -> None:
        client = MockLLMClient(LLMConfig(provider="mock"))
        assert client.available is True
