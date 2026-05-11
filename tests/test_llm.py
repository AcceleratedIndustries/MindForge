"""Tests for the LLM extraction system.

These tests use mocked HTTP responses so they don't require an actual
LLM server. They verify:
- JSON parsing from various LLM response formats
- Concept extraction from structured responses
- LLM-aware distillation (relationship/tag markers)
- Client configuration and error handling
- Pipeline integration with LLM fallback
"""

import json
from unittest.mock import MagicMock

from mindforge.distillation.raw import RawConcept
from mindforge.ingestion.chunker import Chunk
from mindforge.llm.client import LLMClient, LLMConfig, LLMResponse
from mindforge.llm.distiller import (
    _clean_markers,
    _extract_embedded_relationships,
    _extract_embedded_tags,
    distill_concept_smart,
    distill_llm_concept,
)
from mindforge.llm.extractor import (
    _batch_chunks,
    _extract_json_from_response,
    _name_in_text,
    _parse_llm_concepts,
    extract_concepts_llm,
)

# === Sample LLM responses for testing ===

VALID_LLM_RESPONSE = json.dumps(
    {
        "concepts": [
            {
                "name": "Vector Embeddings",
                "definition": "Dense numerical representations of data in a continuous vector space.",
                "explanation": "Each piece of data is mapped to a fixed-length array of floating-point numbers. Similar concepts end up close together in the vector space.",
                "insights": [
                    "Capture semantic meaning in numerical form",
                    "Enable similarity computation via cosine distance",
                ],
                "examples": ["Word2Vec maps words to 300-dimensional vectors"],
                "tags": ["ml", "nlp", "vectors"],
                "relationships": [
                    {"target": "Semantic Search", "type": "enables"},
                    {"target": "Neural Networks", "type": "depends_on"},
                ],
            },
            {
                "name": "Semantic Search",
                "definition": "Information retrieval that understands meaning rather than matching keywords.",
                "explanation": "Uses vector embeddings to find documents similar in meaning to a query.",
                "insights": ["Handles synonyms and paraphrases"],
                "examples": [],
                "tags": ["search", "nlp"],
                "relationships": [
                    {"target": "Vector Embeddings", "type": "uses"},
                ],
            },
        ]
    }
)

MARKDOWN_WRAPPED_RESPONSE = "```json\n" + VALID_LLM_RESPONSE + "\n```"


# === Tests for JSON parsing ===


class TestExtractJsonFromResponse:
    def test_plain_json(self):
        result = _extract_json_from_response(VALID_LLM_RESPONSE)
        assert result is not None
        assert len(result["concepts"]) == 2

    def test_markdown_fenced_json(self):
        result = _extract_json_from_response(MARKDOWN_WRAPPED_RESPONSE)
        assert result is not None
        assert len(result["concepts"]) == 2

    def test_json_with_surrounding_text(self):
        text = "Here are the concepts:\n" + VALID_LLM_RESPONSE + "\nDone."
        result = _extract_json_from_response(text)
        assert result is not None

    def test_invalid_json(self):
        result = _extract_json_from_response("This is not JSON at all.")
        assert result is None

    def test_empty_string(self):
        result = _extract_json_from_response("")
        assert result is None

    def test_partial_json(self):
        result = _extract_json_from_response('{"concepts": [')
        assert result is None


# === Tests for concept parsing ===


class TestParseLLMConcepts:
    def test_parse_valid_response(self):
        data = json.loads(VALID_LLM_RESPONSE)
        concepts = _parse_llm_concepts(data, ["chunk:0"], ["test.md"])
        assert len(concepts) == 2
        assert concepts[0].name == "Vector Embeddings"
        assert concepts[0].extraction_method == "llm"
        assert concepts[0].confidence == 0.9
        assert "test.md" in concepts[0].source_files

    def test_parse_includes_relationships_as_markers(self):
        data = json.loads(VALID_LLM_RESPONSE)
        concepts = _parse_llm_concepts(data, [], [])
        # The raw content should contain relationship markers
        assert "[[rel:enables:Semantic Search]]" in concepts[0].raw_content
        assert "[[rel:depends_on:Neural Networks]]" in concepts[0].raw_content

    def test_parse_includes_tag_markers(self):
        data = json.loads(VALID_LLM_RESPONSE)
        concepts = _parse_llm_concepts(data, [], [])
        assert "[[tags:ml,nlp,vectors]]" in concepts[0].raw_content

    def test_parse_empty_concepts(self):
        data = {"concepts": []}
        concepts = _parse_llm_concepts(data, [], [])
        assert len(concepts) == 0

    def test_parse_skips_nameless(self):
        data = {"concepts": [{"name": "", "definition": "test"}]}
        concepts = _parse_llm_concepts(data, [], [])
        assert len(concepts) == 0

    def test_parse_skips_short_names(self):
        data = {"concepts": [{"name": "AI", "definition": "test"}]}
        concepts = _parse_llm_concepts(data, [], [])
        assert len(concepts) == 0


# === Tests for chunk batching ===


class TestBatchChunks:
    def _make_chunk(self, content: str, index: int = 0) -> Chunk:
        return Chunk(
            content=content,
            source_file="test.md",
            turn_index=0,
            chunk_index=index,
            chunk_type="prose",
        )

    def test_single_batch(self):
        chunks = [self._make_chunk("short text", i) for i in range(3)]
        batches = _batch_chunks(chunks, max_chars=1000)
        assert len(batches) == 1

    def test_multiple_batches(self):
        chunks = [self._make_chunk("x" * 500, i) for i in range(5)]
        batches = _batch_chunks(chunks, max_chars=1000)
        assert len(batches) >= 2

    def test_empty_chunks(self):
        batches = _batch_chunks([], max_chars=1000)
        assert len(batches) == 0

    def test_large_single_chunk(self):
        chunks = [self._make_chunk("x" * 5000)]
        batches = _batch_chunks(chunks, max_chars=1000)
        assert len(batches) == 1  # Can't split a single chunk

    def test_never_mixes_source_files(self):
        # A batch must contain chunks from only one source file. Without this,
        # every concept extracted from a multi-file batch gets all of the
        # batch's source_files attached as provenance, breaking deletion-driven
        # soft-delete (a concept whose only source was the deleted file would
        # incorrectly still reference its batch-mates' files).
        chunks = [
            Chunk(
                content="x" * 100,
                source_file="a.md",
                turn_index=0,
                chunk_index=0,
                chunk_type="prose",
            ),
            Chunk(
                content="x" * 100,
                source_file="b.md",
                turn_index=0,
                chunk_index=0,
                chunk_type="prose",
            ),
            Chunk(
                content="x" * 100,
                source_file="b.md",
                turn_index=0,
                chunk_index=1,
                chunk_type="prose",
            ),
        ]
        batches = _batch_chunks(chunks, max_chars=10_000)
        assert len(batches) == 2
        assert {c.source_file for c in batches[0]} == {"a.md"}
        assert {c.source_file for c in batches[1]} == {"b.md"}


# === Tests for grounding filter ===


class TestNameInText:
    def test_exact_match_case_insensitive(self):
        assert _name_in_text("KV Cache", "We discussed KV Cache today.")
        assert _name_in_text("KV Cache", "we discussed kv cache today.")
        assert _name_in_text("KV Cache", "WE DISCUSSED KV CACHE TODAY.")

    def test_token_boundaries_reject_substring_matches(self):
        # "RAG" must not match inside "storage", "coverage", "drag", etc.
        assert not _name_in_text("RAG", "Long-term storage is fine.")
        assert not _name_in_text("RAG", "Test coverage was high.")
        assert not _name_in_text("RAG", "drag-and-drop works.")
        assert not _name_in_text("RAG", "this is a paragraph.")

    def test_standalone_acronym_matches(self):
        assert _name_in_text("RAG", "We tried RAG with embeddings.")
        assert _name_in_text("RAG", "RAG is a technique.")  # leading
        assert _name_in_text("RAG", "We tried RAG.")  # trailing punctuation

    def test_multi_word_phrase(self):
        assert _name_in_text("Vector Embeddings", "We use vector embeddings here.")
        assert not _name_in_text("Vector Embeddings", "We discussed embeddings (no qualifier).")

    def test_plural_strip_fallback(self):
        # Concept "Vector Embeddings" should match source "vector embedding".
        assert _name_in_text("Vector Embeddings", "Each vector embedding is dense.")
        # But the singular 's' should only strip if the result is reasonable.
        assert not _name_in_text(
            "Vector Embeddings", "Vectorize the input."
        )  # no "embedding" anywhere

    def test_unrelated_concept_rejected(self):
        # The hallucination case the filter is built to catch.
        text = "We're building a Rust PDF tool. It uses pdfium for parsing."
        assert not _name_in_text("KV Cache", text)
        assert not _name_in_text("Vector Embeddings", text)
        assert not _name_in_text("Retrieval-Augmented Generation", text)

    def test_short_name_no_plural_strip(self):
        # Don't strip 's' from very short names; "is" → "i" would always match.
        assert not _name_in_text("Is", "this contains the word i alone.")

    def test_empty_name_returns_false(self):
        assert not _name_in_text("", "any text at all")

    def test_punctuation_in_name(self):
        # Concept names can contain punctuation; the filter should still find them.
        assert _name_in_text("NSApp.delegate", "We override NSApp.delegate to ...")
        assert _name_in_text("CI/CD", "Standard CI/CD pipeline runs on push.")


# === Tests for per-concept provenance ===


class TestPerConceptProvenance:
    """Verify each extracted concept's source_chunks contains only chunks
    whose content includes the concept name (not the full batch)."""

    def _make_chunk(self, content: str, index: int = 0) -> Chunk:
        return Chunk(
            content=content,
            source_file="test.md",
            turn_index=index,  # 1 chunk == 1 turn for simplicity
            chunk_index=0,
            chunk_type="prose",
        )

    def test_concept_attributed_to_only_matching_chunks(self):
        # Chunks 0 and 2 mention 'KV Cache'; chunk 1 does not.
        chunks = [
            self._make_chunk("We discussed KV Cache at length.", 0),
            self._make_chunk("Unrelated stuff about pasta sauce.", 1),
            self._make_chunk("Back to KV Cache for the conclusion.", 2),
        ]
        client = MagicMock(spec=LLMClient)
        client.generate.return_value = LLMResponse(
            content=json.dumps({"concepts": [{"name": "KV Cache", "definition": "A cache."}]}),
            success=True,
        )
        concepts, stats = extract_concepts_llm(chunks, client, max_chars_per_call=10_000)
        assert len(concepts) == 1
        assert stats.rejected_by_grounding == 0
        kv = concepts[0]
        assert set(kv.source_chunks) == {chunks[0].id, chunks[2].id}
        assert chunks[1].id not in kv.source_chunks

    def test_single_chunk_match(self):
        chunks = [
            self._make_chunk("Only chunk 0 mentions Async Queue.", 0),
            self._make_chunk("Chunk 1 talks about something else.", 1),
        ]
        client = MagicMock(spec=LLMClient)
        client.generate.return_value = LLMResponse(
            content=json.dumps(
                {"concepts": [{"name": "Async Queue", "definition": "An async queue."}]}
            ),
            success=True,
        )
        concepts, _ = extract_concepts_llm(chunks, client, max_chars_per_call=10_000)
        assert len(concepts) == 1
        assert concepts[0].source_chunks == [chunks[0].id]

    def test_plural_strip_fallback_in_attribution(self):
        # Concept name is plural; source uses singular. The grounding filter's
        # plural-strip should also drive the per-chunk attribution.
        chunks = [
            self._make_chunk("We use vector embedding for retrieval.", 0),
            self._make_chunk("Unrelated paragraph.", 1),
        ]
        client = MagicMock(spec=LLMClient)
        client.generate.return_value = LLMResponse(
            content=json.dumps(
                {"concepts": [{"name": "Vector Embeddings", "definition": "Dense reps."}]}
            ),
            success=True,
        )
        concepts, _ = extract_concepts_llm(chunks, client, max_chars_per_call=10_000)
        assert len(concepts) == 1
        assert concepts[0].source_chunks == [chunks[0].id]

    def test_same_chunk_supports_multiple_concepts(self):
        chunks = [
            self._make_chunk("Vector Embeddings power Semantic Search here.", 0),
            self._make_chunk("Just filler text.", 1),
        ]
        client = MagicMock(spec=LLMClient)
        client.generate.return_value = LLMResponse(
            content=VALID_LLM_RESPONSE,  # emits both Vector Embeddings and Semantic Search
            success=True,
        )
        concepts, _ = extract_concepts_llm(chunks, client, max_chars_per_call=10_000)
        assert len(concepts) == 2
        # Both concepts cite chunk 0; neither cites chunk 1.
        for c in concepts:
            assert c.source_chunks == [chunks[0].id]

    def test_token_boundary_prevents_substring_attribution(self):
        # 'RAG' must not match inside 'storage' / 'paragraph'. So if the LLM
        # emits 'RAG' as a concept and the only chunks contain 'storage', the
        # concept should be REJECTED by grounding (zero supporting chunks),
        # not silently attributed to those chunks.
        chunks = [
            self._make_chunk("Long-term storage is a paragraph apart.", 0),
            self._make_chunk("More storage discussion here.", 1),
        ]
        client = MagicMock(spec=LLMClient)
        client.generate.return_value = LLMResponse(
            content=json.dumps(
                {"concepts": [{"name": "RAG", "definition": "Retrieval-Augmented Generation."}]}
            ),
            success=True,
        )
        concepts, stats = extract_concepts_llm(chunks, client, max_chars_per_call=10_000)
        assert len(concepts) == 0
        assert stats.rejected_by_grounding == 1

    def test_concept_in_all_chunks_lists_all_chunks(self):
        # Sanity: when every chunk supports a concept, source_chunks contains
        # all of them. This is the case where batch-level and per-chunk
        # attribution produce the same result.
        chunks = [
            self._make_chunk("KV Cache discussion here.", 0),
            self._make_chunk("More KV Cache analysis.", 1),
            self._make_chunk("Final KV Cache notes.", 2),
        ]
        client = MagicMock(spec=LLMClient)
        client.generate.return_value = LLMResponse(
            content=json.dumps({"concepts": [{"name": "KV Cache", "definition": "A cache."}]}),
            success=True,
        )
        concepts, _ = extract_concepts_llm(chunks, client, max_chars_per_call=10_000)
        assert len(concepts) == 1
        assert set(concepts[0].source_chunks) == {chunks[0].id, chunks[1].id, chunks[2].id}


# === Tests for LLM-aware distillation ===


class TestEmbeddedMetadata:
    def test_extract_relationships(self):
        text = "Definition here.\n\n[[rel:uses:Vector DB]]\n[[rel:enables:Search]]"
        rels = _extract_embedded_relationships(text, "my-concept")
        assert len(rels) == 2
        assert rels[0].source == "my-concept"
        assert rels[0].target == "vector-db"
        assert rels[0].rel_type.value == "uses"

    def test_extract_tags(self):
        text = "Some content.\n\n[[tags:ml,vectors,search]]"
        tags = _extract_embedded_tags(text)
        assert tags == ["ml", "vectors", "search"]

    def test_no_tags(self):
        tags = _extract_embedded_tags("No tags here.")
        assert tags == []

    def test_clean_markers(self):
        text = "Definition.\n\n[[rel:uses:X]]\n[[tags:a,b]]"
        cleaned = _clean_markers(text)
        assert "[[rel" not in cleaned
        assert "[[tags" not in cleaned
        assert "Definition." in cleaned


class TestDistillLLMConcept:
    def test_distill_with_metadata(self):
        raw = RawConcept(
            name="Vector Database",
            raw_content=(
                "A specialized database for storing and querying vectors.\n\n"
                "Enables fast nearest-neighbor search at scale.\n\n"
                "- Supports HNSW indexing\n"
                "- Handles metadata filtering\n\n"
                "Examples:\n- Qdrant\n- Pinecone\n\n"
                "[[rel:enables:Semantic Search]]\n"
                "[[tags:database,vectors,search]]"
            ),
            source_files=["test.md"],
            extraction_method="llm",
            confidence=0.9,
        )
        concept = distill_llm_concept(raw)
        assert concept.name == "Vector Database"
        assert "specialized database" in concept.definition
        assert "nearest-neighbor" in concept.explanation
        assert len(concept.insights) == 2
        assert "Qdrant" in concept.examples
        assert "database" in concept.tags
        assert len(concept.relationships) == 1
        assert concept.relationships[0].rel_type.value == "enables"

    def test_distill_minimal(self):
        raw = RawConcept(
            name="Test",
            raw_content="A simple test concept.",
            extraction_method="llm",
            confidence=0.9,
        )
        concept = distill_llm_concept(raw)
        assert concept.name == "Test"
        assert concept.definition == "A simple test concept."


class TestDistillConceptSmart:
    def test_routes_llm_concepts(self):
        raw = RawConcept(
            name="LLM Concept",
            raw_content="An LLM-extracted concept.\n\n[[tags:test]]",
            extraction_method="llm",
            confidence=0.9,
        )
        concept = distill_concept_smart(raw)
        assert concept.name == "LLM Concept"
        assert "test" in concept.tags

    def test_routes_heuristic_concepts(self):
        raw = RawConcept(
            name="Heuristic Concept",
            raw_content="Heuristic Concept is a pattern-matched concept for testing purposes.",
            extraction_method="definition_pattern",
            confidence=0.8,
        )
        concept = distill_concept_smart(raw)
        assert concept.name == "Heuristic Concept"


# === Tests for LLM client ===


class TestLLMConfig:
    def test_ollama_defaults(self):
        config = LLMConfig()
        assert config.provider == "ollama"
        assert config.base_url == "http://localhost:11434"

    def test_openai_defaults(self):
        config = LLMConfig(provider="openai")
        assert config.base_url == "https://api.openai.com"

    def test_custom_url(self):
        config = LLMConfig(base_url="http://my-server:8080")
        assert config.base_url == "http://my-server:8080"

    def test_ollama_think_default_is_none(self):
        config = LLMConfig()
        assert config.ollama_think is None


class TestOllamaPayload:
    """Verify the Ollama request payload reflects config flags."""

    def _capture_payload(self, config: LLMConfig) -> dict:
        client = LLMClient(config)
        captured: dict = {}

        def fake_post(url: str, payload: dict, parser):
            captured.update(payload)
            return LLMResponse(content='{"concepts": []}', success=True)

        client._post_json = fake_post  # type: ignore[method-assign]
        client.generate("test prompt", system="sys", response_format="json")
        return captured

    def test_think_omitted_when_none(self):
        payload = self._capture_payload(LLMConfig(ollama_think=None))
        assert "think" not in payload

    def test_think_false_in_payload(self):
        payload = self._capture_payload(LLMConfig(ollama_think=False))
        assert payload["think"] is False

    def test_think_true_in_payload(self):
        payload = self._capture_payload(LLMConfig(ollama_think=True))
        assert payload["think"] is True

    def test_keep_alive_in_payload(self):
        payload = self._capture_payload(LLMConfig(ollama_keep_alive="30m"))
        assert payload["keep_alive"] == "30m"


class TestLLMClient:
    def test_unavailable_server(self):
        config = LLMConfig(base_url="http://localhost:99999")
        client = LLMClient(config)
        assert not client.available

    def test_generate_on_unavailable(self):
        config = LLMConfig(base_url="http://localhost:99999")
        client = LLMClient(config)
        response = client.generate("test prompt")
        assert not response.success


# === Tests for extract_concepts_llm with mocked client ===


class TestExtractConceptsLLM:
    def _make_chunk(self, content: str, index: int = 0) -> Chunk:
        return Chunk(
            content=content,
            source_file="test.md",
            turn_index=0,
            chunk_index=index,
            chunk_type="prose",
        )

    def test_successful_extraction(self):
        # Chunk content must contain both concept names from VALID_LLM_RESPONSE
        # so the grounding filter accepts them.
        chunks = [
            self._make_chunk(
                "Vector embeddings are dense representations. "
                "Semantic search retrieves by meaning instead of keywords."
            )
        ]

        client = MagicMock(spec=LLMClient)
        client.generate.return_value = LLMResponse(
            content=VALID_LLM_RESPONSE,
            success=True,
        )

        concepts, stats = extract_concepts_llm(chunks, client)
        assert len(concepts) == 2
        assert stats.llm_calls == 1
        assert stats.parse_failures == 0
        assert stats.rejected_by_grounding == 0
        assert concepts[0].name == "Vector Embeddings"

    def test_failed_llm_call(self):
        chunks = [self._make_chunk("Some text.")]

        client = MagicMock(spec=LLMClient)
        client.generate.return_value = LLMResponse(
            content="",
            success=False,
            error="Connection refused",
        )

        concepts, stats = extract_concepts_llm(chunks, client)
        assert len(concepts) == 0
        assert stats.parse_failures == 1

    def test_invalid_json_response(self):
        chunks = [self._make_chunk("Some text.")]

        client = MagicMock(spec=LLMClient)
        client.generate.return_value = LLMResponse(
            content="I don't know how to respond in JSON.",
            success=True,
        )

        concepts, stats = extract_concepts_llm(chunks, client)
        assert len(concepts) == 0
        assert stats.parse_failures == 1

    def test_deduplicates_across_batches(self):
        # Each chunk must contain "Same Concept" so the grounding filter
        # accepts the LLM's emission across all batches; dedup then collapses.
        chunks = [self._make_chunk("Same Concept appears here. " + "x" * 3500, i) for i in range(3)]

        client = MagicMock(spec=LLMClient)
        # Both batches return the same concept
        client.generate.return_value = LLMResponse(
            content=json.dumps(
                {"concepts": [{"name": "Same Concept", "definition": "A concept."}]}
            ),
            success=True,
        )

        concepts, stats = extract_concepts_llm(chunks, client, max_chars_per_call=5000)
        # Should deduplicate by name
        names = [c.name for c in concepts]
        assert names.count("Same Concept") == 1
