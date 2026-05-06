"""Tests for KnowledgeGraph.subgraph and the get_subgraph MCP tool."""

from __future__ import annotations

import json

from mindforge.distillation.concept import (
    Concept,
    ConceptStore,
    Relationship,
    RelationshipType,
)
from mindforge.graph.builder import KnowledgeGraph
from mindforge.mcp.tools.subgraph import handle_get_subgraph


def _c(
    name: str,
    rels: list[tuple[str, RelationshipType, float]] | None = None,
) -> Concept:
    c = Concept(
        name=name,
        definition=f"{name} definition",
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


def _abc_graph() -> tuple[ConceptStore, KnowledgeGraph]:
    """``a`` -uses-> ``b`` -depends_on-> ``c``;  ``a`` -related_to-> ``d`` (sibling)."""
    store = ConceptStore()
    store.add(
        _c(
            "a",
            [
                ("b", RelationshipType.USES, 0.9),
                ("d", RelationshipType.RELATED_TO, 0.7),
            ],
        )
    )
    store.add(_c("b", [("c", RelationshipType.DEPENDS_ON, 0.8)]))
    store.add(_c("c"))
    store.add(_c("d"))
    return store, KnowledgeGraph.from_store(store)


def _body(out: str) -> dict:
    inner = out.split("<mindforge_retrieved_content>", 1)[1].split(
        "</mindforge_retrieved_content>", 1
    )[0]
    return json.loads(inner.strip())


def test_subgraph_depth_1_returns_center_and_neighbors() -> None:
    _, graph = _abc_graph()
    sub = graph.subgraph("a", depth=1)
    node_ids = {n["id"] for n in sub["nodes"]}
    assert node_ids == {"a", "b", "d"}
    edge_pairs = {(e["source"], e["target"]) for e in sub["edges"]}
    assert ("a", "b") in edge_pairs
    assert ("a", "d") in edge_pairs


def test_subgraph_depth_2_reaches_two_hops() -> None:
    _, graph = _abc_graph()
    sub = graph.subgraph("a", depth=2)
    node_ids = {n["id"] for n in sub["nodes"]}
    assert node_ids == {"a", "b", "c", "d"}


def test_subgraph_unknown_center_returns_empty() -> None:
    _, graph = _abc_graph()
    assert graph.subgraph("missing", depth=1) == {"nodes": [], "edges": []}


def test_subgraph_filters_by_edge_type() -> None:
    _, graph = _abc_graph()
    sub = graph.subgraph("a", depth=1, edge_types=["uses"])
    edge_types = {e["type"] for e in sub["edges"]}
    assert edge_types == {"uses"}


def test_subgraph_node_labels_use_human_names() -> None:
    store = ConceptStore()
    store.add(_c("Vector Embedding", [("rag", RelationshipType.USES, 0.9)]))
    store.add(_c("RAG"))
    graph = KnowledgeGraph.from_store(store)
    sub = graph.subgraph("vector-embedding", depth=1)
    by_id = {n["id"]: n["label"] for n in sub["nodes"]}
    assert by_id["vector-embedding"] == "Vector Embedding"
    assert by_id["rag"] == "RAG"


def test_handle_get_subgraph_returns_json_and_markdown() -> None:
    store, graph = _abc_graph()
    out = handle_get_subgraph(store=store, graph=graph, center="a", depth=1)
    parsed = _body(out)
    assert "json" in parsed
    assert "markdown" in parsed
    assert {n["id"] for n in parsed["json"]["nodes"]} == {"a", "b", "d"}
    md = parsed["markdown"]
    assert "Subgraph centered on `a`" in md
    assert "**uses**" in md or "**related_to**" in md


def test_handle_get_subgraph_unknown_center_returns_empty_payload() -> None:
    store, graph = _abc_graph()
    out = handle_get_subgraph(store=store, graph=graph, center="missing", depth=1)
    parsed = _body(out)
    assert parsed["json"] == {"nodes": [], "edges": []}


def test_handle_get_subgraph_strips_hidden_unicode_from_labels() -> None:
    """Hidden chars in node labels (e.g. injected during ingestion) are stripped
    via the safety wrapper."""
    store = ConceptStore()
    poisoned_name = "Vec​tor"
    store.add(_c(poisoned_name))
    graph = KnowledgeGraph.from_store(store)
    out = handle_get_subgraph(store=store, graph=graph, center=store.all()[0].slug, depth=1)
    assert "​" not in out
