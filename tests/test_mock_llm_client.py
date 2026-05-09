"""Tests for MockLLMClient and make_llm_client factory."""

from __future__ import annotations

import json

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


class TestMockGenerate:
    def setup_method(self) -> None:
        self.client = MockLLMClient(LLMConfig(provider="mock"))

    def test_extracts_titlecase_phrase_as_concept_name(self) -> None:
        prompt = "Some text mentioning KV Cache in passing."
        resp = self.client.generate(prompt, response_format="json")
        data = json.loads(resp.content)
        names = [c["name"] for c in data["concepts"]]
        assert "KV Cache" in names

    def test_deterministic_byte_for_byte(self) -> None:
        prompt = "KV Cache and Multi-Query Attention come up here."
        a = self.client.generate(prompt, response_format="json").content
        b = self.client.generate(prompt, response_format="json").content
        assert a == b

    def test_caps_at_three_concepts_per_call(self) -> None:
        # Five distinct titlecase phrases.
        prompt = "Alpha One and Beta Two and Gamma Three and Delta Four and Epsilon Five."
        resp = self.client.generate(prompt, response_format="json")
        data = json.loads(resp.content)
        assert len(data["concepts"]) == 3

    def test_dedupes_within_a_single_call(self) -> None:
        prompt = "KV Cache. Then more text. KV Cache again. KV Cache once more."
        resp = self.client.generate(prompt, response_format="json")
        data = json.loads(resp.content)
        names = [c["name"] for c in data["concepts"]]
        assert names.count("KV Cache") == 1

    def test_empty_text_returns_no_concepts(self) -> None:
        resp = self.client.generate("", response_format="json")
        data = json.loads(resp.content)
        assert data["concepts"] == []

    def test_lowercase_only_text_returns_no_concepts(self) -> None:
        resp = self.client.generate(
            "just lowercase words here, nothing capitalized.", response_format="json"
        )
        data = json.loads(resp.content)
        assert data["concepts"] == []

    def test_concepts_tagged_mock(self) -> None:
        prompt = "KV Cache exists."
        resp = self.client.generate(prompt, response_format="json")
        data = json.loads(resp.content)
        for c in data["concepts"]:
            assert c["tags"] == ["mock"]

    def test_definition_is_surrounding_sentence(self) -> None:
        prompt = "First sentence. KV Cache is a memory structure. Third sentence."
        resp = self.client.generate(prompt, response_format="json")
        data = json.loads(resp.content)
        kv = next(c for c in data["concepts"] if c["name"] == "KV Cache")
        assert "memory structure" in kv["definition"]
        assert "First sentence" not in kv["definition"]
        assert "Third sentence" not in kv["definition"]

    def test_short_phrases_under_3_chars_excluded(self) -> None:
        # "I" and "A" are titlecase but len < 3.
        prompt = "I think A is fine but Real Concept matters."
        resp = self.client.generate(prompt, response_format="json")
        data = json.loads(resp.content)
        names = [c["name"] for c in data["concepts"]]
        assert "Real Concept" in names
        assert "I" not in names
        assert "A" not in names

    def test_round_trips_through_real_parser(self) -> None:
        from mindforge.llm.extractor import _parse_llm_concepts

        prompt = "KV Cache and Multi-Query Attention are concepts."
        resp = self.client.generate(prompt, response_format="json")
        data = json.loads(resp.content)
        concepts = _parse_llm_concepts(
            data,
            source_chunks=["chunk-1"],
            source_files=["/tmp/x.md"],
        )
        names = {c.name for c in concepts}
        assert "KV Cache" in names
        assert "Multi-Query Attention" in names
        for c in concepts:
            assert (
                c.extraction_method == "llm"
            )  # parser stamps this; mock indistinguishable to consumer
            assert c.confidence == 0.9

    def test_strips_prompt_envelope_before_extracting(self) -> None:
        # The mock must operate on the chunk-text body of the LLM extraction
        # prompt, not the whole prompt template. Without the envelope strip,
        # boilerplate words ('Extract', 'TEXT', 'For', 'Respond') would shadow
        # real chunk concepts and consume the 3-concept-per-call cap.
        from mindforge.llm.extractor import EXTRACTION_USER_PROMPT

        prompt = EXTRACTION_USER_PROMPT.format(text="KV Cache exists.")
        resp = self.client.generate(prompt, response_format="json")
        data = json.loads(resp.content)
        names = [c["name"] for c in data["concepts"]]
        assert "KV Cache" in names
        # Boilerplate words from the prompt template must not appear as concepts.
        for boilerplate in ("Extract", "TEXT", "For", "Respond"):
            assert boilerplate not in names
