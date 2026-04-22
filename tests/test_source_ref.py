"""Tests for the SourceRef dataclass."""

from __future__ import annotations

from mindforge.distillation.concept import Concept
from mindforge.distillation.source_ref import SNIPPET_MAX_CHARS, SourceRef


def test_to_dict_roundtrip():
    ref = SourceRef(
        transcript_path="2025-03-14_llm_internals.md",
        transcript_hash="abcd1234",
        turn_indices=[4, 7],
        extracted_at="2025-03-14T11:22:00Z",
        chunk_id="2025-03-14_llm_internals.md:t4:c0",
        snippet="KV Cache stores the Key and Value matrices.",
    )
    data = ref.to_dict()
    assert data["transcript_path"] == "2025-03-14_llm_internals.md"
    assert data["turn_indices"] == [4, 7]
    restored = SourceRef.from_dict(data)
    assert restored == ref


def test_defaults():
    ref = SourceRef(
        transcript_path="t.md",
        transcript_hash="h",
        turn_indices=[0],
        extracted_at="2025-01-01T00:00:00Z",
    )
    assert ref.chunk_id is None
    assert ref.snippet is None


def test_snippet_is_capped_on_init():
    long = "x" * 5000
    ref = SourceRef(
        transcript_path="t.md", transcript_hash="h",
        turn_indices=[0], extracted_at="2025-01-01T00:00:00Z",
        snippet=long,
    )
    assert len(ref.snippet) == SNIPPET_MAX_CHARS


def test_concept_roundtrip_preserves_sources():
    c = Concept(
        name="KV Cache",
        definition="d",
        explanation="e",
        sources=[SourceRef(
            transcript_path="t.md", transcript_hash="h",
            turn_indices=[0], extracted_at="2025-01-01T00:00:00Z",
        )],
    )
    restored = Concept.from_dict(c.to_dict())
    assert restored.sources == c.sources


def test_concept_merge_dedups_sources():
    ref = SourceRef(
        transcript_path="t.md", transcript_hash="h",
        turn_indices=[0], extracted_at="2025-01-01T00:00:00Z",
    )
    a = Concept(name="X", definition="d", explanation="e", sources=[ref])
    b = Concept(name="X", definition="d", explanation="e", sources=[ref])
    merged = a.merge_with(b)
    assert len(merged.sources) == 1
