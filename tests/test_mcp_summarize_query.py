"""Tests for the context pack composer (used by summarize_query) and the
summarize_query tool itself."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from mindforge.distillation.concept import (
    Concept,
    ConceptStore,
    Relationship,
    RelationshipType,
)
from mindforge.graph.builder import KnowledgeGraph
from mindforge.mcp.tools.summarize_query import handle_summarize_query
from mindforge.query.context_pack import compose_context_pack


def _c(
    name: str,
    defn: str,
    rels: list[tuple[str, RelationshipType, float]] | None = None,
) -> Concept:
    """Build a Concept with relationships sourced from itself.

    ``slug`` is derived from ``name`` via slugify, so test names are chosen so
    their slugs match the targets used in assertions (e.g. "RAG" -> "rag").
    """
    c = Concept(
        name=name,
        definition=defn,
        explanation="",
        insights=[],
        examples=[],
        tags=[],
        confidence=1.0,
        links=[],
        relationships=[],
        sources=[],
    )
    parent_slug = c.slug
    c.relationships = [
        Relationship(source=parent_slug, target=t, rel_type=r, confidence=conf)
        for t, r, conf in (rels or [])
    ]
    return c


@pytest.fixture
def kb() -> tuple[ConceptStore, KnowledgeGraph]:
    store = ConceptStore()
    store.add(
        _c(
            "RAG",
            "Retrieval-augmented generation pattern.",
            [("vector", RelationshipType.USES, 0.9)],
        )
    )
    store.add(_c("Vector", "Numeric representation of text."))
    graph = KnowledgeGraph.from_store(store)
    return store, graph


def test_compose_returns_top_k_concepts(kb) -> None:
    store, graph = kb
    pack = compose_context_pack(store, graph, "retrieval generation", top_k=2)
    assert len(pack.concepts) <= 2


def test_compose_includes_one_hop_neighbors(kb) -> None:
    store, graph = kb
    pack = compose_context_pack(store, graph, "retrieval generation", top_k=1, max_hops=1)
    slugs = [c.slug for c in pack.concepts] + pack.neighbor_slugs
    assert "vector" in slugs


def test_compose_records_relationships(kb) -> None:
    store, graph = kb
    pack = compose_context_pack(store, graph, "retrieval generation", top_k=2)
    assert any(r.source == "rag" and r.target == "vector" for r in pack.relationships)


def test_compose_drops_neighbor_already_in_top_k(kb) -> None:
    store, graph = kb
    pack = compose_context_pack(store, graph, "retrieval generation", top_k=2)
    primary = {c.slug for c in pack.concepts}
    assert primary.isdisjoint(set(pack.neighbor_slugs))


def test_summarize_query_returns_prose_and_concepts(kb) -> None:
    store, graph = kb
    fake_llm = MagicMock()
    fake_llm.generate.return_value = MagicMock(
        success=True,
        content="RAG combines retrieval with generation. It uses vector embeddings.",
    )
    fake_llm.available = True
    out = handle_summarize_query(
        store=store,
        graph=graph,
        llm_client=fake_llm,
        question="What is RAG?",
        top_k=2,
        max_hops=1,
    )
    assert "<mindforge_retrieved_content>" in out
    body = out.split("<mindforge_retrieved_content>", 1)[1].split(
        "</mindforge_retrieved_content>", 1
    )[0]
    parsed = json.loads(body.strip())
    assert "answer" in parsed
    assert parsed["answer"].startswith("RAG combines")
    assert "concepts_consulted" in parsed
    assert isinstance(parsed["concepts_consulted"], list)
    assert "suggested_followup" in parsed
    assert "confidence" in parsed


def test_summarize_query_returns_error_when_llm_unavailable(kb) -> None:
    store, graph = kb
    fake_llm = MagicMock()
    fake_llm.available = False
    out = handle_summarize_query(
        store=store,
        graph=graph,
        llm_client=fake_llm,
        question="q",
        top_k=2,
        max_hops=1,
    )
    assert "synthesis_backend_unavailable" in out
    fake_llm.generate.assert_not_called()


def test_summarize_query_handles_llm_failure(kb) -> None:
    store, graph = kb
    fake_llm = MagicMock()
    fake_llm.available = True
    fake_llm.generate.return_value = MagicMock(success=False, error="boom")
    out = handle_summarize_query(
        store=store,
        graph=graph,
        llm_client=fake_llm,
        question="q",
        top_k=2,
        max_hops=1,
    )
    assert "synthesis_failed" in out
    body = out.split("<mindforge_retrieved_content>", 1)[1].split(
        "</mindforge_retrieved_content>", 1
    )[0]
    parsed = json.loads(body.strip())
    assert parsed["error"] == "synthesis_failed"
    assert parsed["message"] == "boom"


def test_summarize_query_includes_provenance_when_requested(kb) -> None:
    store, graph = kb
    fake_llm = MagicMock()
    fake_llm.available = True
    fake_llm.generate.return_value = MagicMock(success=True, content="answer")
    out = handle_summarize_query(
        store=store,
        graph=graph,
        llm_client=fake_llm,
        question="q",
        top_k=2,
        max_hops=1,
        include_provenance=True,
    )
    body = out.split("<mindforge_retrieved_content>", 1)[1].split(
        "</mindforge_retrieved_content>", 1
    )[0]
    parsed = json.loads(body.strip())
    assert "provenance" in parsed


def test_summarize_query_strips_hidden_unicode_from_llm_output(kb) -> None:
    """Steganographic injection in the LLM response is stripped before wrapping."""
    store, graph = kb
    fake_llm = MagicMock()
    fake_llm.available = True
    poisoned = "RAG​ is a pattern.‮"
    fake_llm.generate.return_value = MagicMock(success=True, content=poisoned)
    out = handle_summarize_query(
        store=store,
        graph=graph,
        llm_client=fake_llm,
        question="q",
        top_k=2,
        max_hops=1,
    )
    assert "​" not in out
    assert "‮" not in out
