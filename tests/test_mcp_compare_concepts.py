"""Tests for MCP tool: compare_concepts."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from mindforge.distillation.concept import (
    Concept,
    ConceptStore,
    Relationship,
    RelationshipType,
)
from mindforge.graph.builder import KnowledgeGraph
from mindforge.mcp.tools.compare_concepts import handle_compare_concepts


def _c(
    name: str,
    defn: str,
    rels: list[tuple[str, RelationshipType, float]] | None = None,
) -> Concept:
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


def _body(out: str) -> dict:
    inner = out.split("<mindforge_retrieved_content>", 1)[1].split(
        "</mindforge_retrieved_content>", 1
    )[0]
    return json.loads(inner.strip())


def test_compare_returns_prose_and_relationship_types() -> None:
    store = ConceptStore()
    store.add(_c("RAG", "RAG def.", [("vector", RelationshipType.USES, 0.9)]))
    store.add(_c("Vector", "Vector def."))
    graph = KnowledgeGraph.from_store(store)
    llm = MagicMock()
    llm.available = True
    llm.generate.return_value = MagicMock(success=True, content="RAG uses Vector...")
    out = handle_compare_concepts(
        store=store, graph=graph, llm_client=llm, concepts=["rag", "vector"]
    )
    parsed = _body(out)
    assert "comparison" in parsed
    assert "uses" in parsed["relationship_types"]
    assert parsed["concepts_consulted"] == ["rag", "vector"]


def test_compare_unavailable_when_llm_down() -> None:
    store = ConceptStore()
    store.add(_c("RAG", "x"))
    store.add(_c("Vector", "y"))
    graph = KnowledgeGraph.from_store(store)
    llm = MagicMock()
    llm.available = False
    out = handle_compare_concepts(
        store=store, graph=graph, llm_client=llm, concepts=["rag", "vector"]
    )
    parsed = _body(out)
    assert parsed["error"] == "synthesis_backend_unavailable"


def test_compare_insufficient_concepts() -> None:
    store = ConceptStore()
    store.add(_c("RAG", "x"))
    graph = KnowledgeGraph.from_store(store)
    llm = MagicMock()
    llm.available = True
    out = handle_compare_concepts(
        store=store, graph=graph, llm_client=llm, concepts=["rag", "nonexistent"]
    )
    parsed = _body(out)
    assert parsed["error"] == "insufficient_concepts"


def test_compare_resolves_human_names() -> None:
    store = ConceptStore()
    store.add(_c("RAG", "x"))
    store.add(_c("Vector", "y"))
    graph = KnowledgeGraph.from_store(store)
    llm = MagicMock()
    llm.available = True
    llm.generate.return_value = MagicMock(success=True, content="comp")
    out = handle_compare_concepts(
        store=store, graph=graph, llm_client=llm, concepts=["RAG", "Vector"]
    )
    parsed = _body(out)
    assert parsed["concepts_consulted"] == ["rag", "vector"]


def test_compare_aspect_threads_into_prompt() -> None:
    store = ConceptStore()
    store.add(_c("RAG", "x"))
    store.add(_c("Vector", "y"))
    graph = KnowledgeGraph.from_store(store)
    llm = MagicMock()
    llm.available = True
    llm.generate.return_value = MagicMock(success=True, content="comp")
    handle_compare_concepts(
        store=store,
        graph=graph,
        llm_client=llm,
        concepts=["rag", "vector"],
        aspect="failure modes",
    )
    prompt = llm.generate.call_args[0][0]
    assert "focused on failure modes" in prompt
