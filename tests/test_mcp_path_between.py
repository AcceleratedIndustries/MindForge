"""Tests for KnowledgeGraph.shortest_paths and the path_between MCP tool."""

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
from mindforge.mcp.tools.path_between import handle_path_between


def _c(
    name: str,
    rels: list[tuple[str, RelationshipType, float]] | None = None,
) -> Concept:
    c = Concept(
        name=name,
        definition="d",
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


def _abc_graph() -> tuple[ConceptStore, KnowledgeGraph]:
    store = ConceptStore()
    store.add(_c("a", [("b", RelationshipType.USES, 0.9)]))
    store.add(_c("b", [("c", RelationshipType.DEPENDS_ON, 0.8)]))
    store.add(_c("c"))
    return store, KnowledgeGraph.from_store(store)


def test_shortest_paths_finds_chain() -> None:
    _, graph = _abc_graph()
    paths = graph.shortest_paths("a", "c", max_length=4, max_paths=3)
    assert len(paths) == 1
    assert paths[0] == ["a", "b", "c"]


def test_shortest_paths_returns_empty_for_disconnected() -> None:
    store = ConceptStore()
    store.add(_c("a"))
    store.add(_c("z"))
    graph = KnowledgeGraph.from_store(store)
    assert graph.shortest_paths("a", "z") == []


def test_shortest_paths_returns_empty_for_unknown_node() -> None:
    _, graph = _abc_graph()
    assert graph.shortest_paths("a", "missing") == []
    assert graph.shortest_paths("missing", "c") == []


def test_shortest_paths_respects_max_length() -> None:
    _, graph = _abc_graph()
    # max_length=1 means only direct edges (single-hop)
    paths = graph.shortest_paths("a", "c", max_length=1)
    assert paths == []


def test_path_between_finds_short_chain_and_narrates() -> None:
    store, graph = _abc_graph()

    llm = MagicMock()
    llm.available = True
    llm.generate.return_value = MagicMock(success=True, content="A uses B which depends on C.")

    out = handle_path_between(
        store=store,
        graph=graph,
        llm_client=llm,
        from_concept="a",
        to_concept="c",
        max_hops=4,
    )
    parsed = _body(out)
    assert parsed["found"] is True
    assert parsed["path"] == ["a", "b", "c"]
    assert parsed["narrative"] == "A uses B which depends on C."
    assert "uses" in parsed["edge_types"]
    assert "depends_on" in parsed["edge_types"]


def test_path_between_returns_not_found_for_disconnected() -> None:
    store = ConceptStore()
    store.add(_c("a"))
    store.add(_c("z"))
    graph = KnowledgeGraph.from_store(store)
    llm = MagicMock()
    llm.available = True
    out = handle_path_between(
        store=store, graph=graph, llm_client=llm, from_concept="a", to_concept="z"
    )
    parsed = _body(out)
    assert parsed["found"] is False
    assert parsed["path"] == []
    assert parsed["narrative"] == ""
    llm.generate.assert_not_called()


def test_path_between_omits_narrative_when_llm_unavailable() -> None:
    store, graph = _abc_graph()
    llm = MagicMock()
    llm.available = False
    out = handle_path_between(
        store=store, graph=graph, llm_client=llm, from_concept="a", to_concept="c"
    )
    parsed = _body(out)
    assert parsed["found"] is True
    assert parsed["path"] == ["a", "b", "c"]
    assert parsed["narrative"] == ""
    llm.generate.assert_not_called()
