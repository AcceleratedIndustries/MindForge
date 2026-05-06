"""Tests for MCP tool: explain_concept."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from mindforge.distillation.concept import Concept, ConceptStore
from mindforge.graph.builder import KnowledgeGraph
from mindforge.mcp.tools.explain_concept import handle_explain_concept


def _c(name: str, defn: str, insights: list[str] | None = None) -> Concept:
    return Concept(
        name=name,
        definition=defn,
        explanation="long explanation about the concept",
        insights=insights or ["insight one"],
        examples=[],
        tags=[],
        confidence=1.0,
        links=[],
        relationships=[],
        sources=[],
    )


def _body(out: str) -> dict:
    inner = out.split("<mindforge_retrieved_content>", 1)[1].split(
        "</mindforge_retrieved_content>", 1
    )[0]
    return json.loads(inner.strip())


def test_brief_skips_llm() -> None:
    store = ConceptStore()
    store.add(_c("RAG", "Retrieval-augmented generation pattern.", ["uses retrieval"]))
    graph = KnowledgeGraph.from_store(store)
    fake_llm = MagicMock()
    fake_llm.available = True

    out = handle_explain_concept(
        store=store, graph=graph, llm_client=fake_llm, concept="rag", depth="brief"
    )
    fake_llm.generate.assert_not_called()
    parsed = _body(out)
    assert parsed["slug"] == "rag"
    assert "Retrieval" in parsed["explanation"]


def test_standard_calls_llm() -> None:
    store = ConceptStore()
    store.add(_c("RAG", "Retrieval-augmented generation pattern."))
    graph = KnowledgeGraph.from_store(store)
    fake_llm = MagicMock()
    fake_llm.available = True
    fake_llm.generate.return_value = MagicMock(success=True, content="paraphrased explanation")

    out = handle_explain_concept(
        store=store, graph=graph, llm_client=fake_llm, concept="rag", depth="standard"
    )
    fake_llm.generate.assert_called_once()
    parsed = _body(out)
    assert parsed["explanation"] == "paraphrased explanation"


def test_detailed_calls_llm_with_longer_target() -> None:
    store = ConceptStore()
    store.add(_c("RAG", "RAG pattern."))
    graph = KnowledgeGraph.from_store(store)
    fake_llm = MagicMock()
    fake_llm.available = True
    fake_llm.generate.return_value = MagicMock(success=True, content="long answer")

    handle_explain_concept(
        store=store, graph=graph, llm_client=fake_llm, concept="rag", depth="detailed"
    )
    prompt_arg = fake_llm.generate.call_args[0][0]
    assert "200 words" in prompt_arg


def test_unknown_slug_returns_error() -> None:
    store = ConceptStore()
    graph = KnowledgeGraph.from_store(store)
    fake_llm = MagicMock()
    fake_llm.available = True
    out = handle_explain_concept(
        store=store, graph=graph, llm_client=fake_llm, concept="nonexistent", depth="brief"
    )
    parsed = _body(out)
    assert parsed["error"] == "concept_not_found"
    assert "nonexistent" in parsed["message"]


def test_standard_returns_synthesis_unavailable_when_llm_down() -> None:
    store = ConceptStore()
    store.add(_c("RAG", "RAG pattern."))
    graph = KnowledgeGraph.from_store(store)
    fake_llm = MagicMock()
    fake_llm.available = False

    out = handle_explain_concept(
        store=store, graph=graph, llm_client=fake_llm, concept="rag", depth="standard"
    )
    parsed = _body(out)
    assert parsed["error"] == "synthesis_backend_unavailable"
    fake_llm.generate.assert_not_called()


def test_brief_works_when_llm_unavailable() -> None:
    """Brief mode is the no-LLM fallback agents reach for when synthesis is down."""
    store = ConceptStore()
    store.add(_c("RAG", "RAG pattern."))
    graph = KnowledgeGraph.from_store(store)
    fake_llm = MagicMock()
    fake_llm.available = False

    out = handle_explain_concept(
        store=store, graph=graph, llm_client=fake_llm, concept="rag", depth="brief"
    )
    parsed = _body(out)
    assert "RAG pattern" in parsed["explanation"]


def test_resolves_by_human_name() -> None:
    store = ConceptStore()
    store.add(_c("Vector Embedding", "A numeric representation."))
    graph = KnowledgeGraph.from_store(store)
    fake_llm = MagicMock()
    fake_llm.available = True
    out = handle_explain_concept(
        store=store,
        graph=graph,
        llm_client=fake_llm,
        concept="Vector Embedding",
        depth="brief",
    )
    parsed = _body(out)
    assert parsed["slug"] == "vector-embedding"
