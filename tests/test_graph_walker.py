"""Tests for graph walker (hop reinforcement)."""

from mindforge.distillation.concept import Concept, Relationship, RelationshipType
from mindforge.graph.builder import KnowledgeGraph
from mindforge.query.graph_walker import GraphWalker


def _concept(name: str, rels: list[tuple[str, RelationshipType, float]] | None = None) -> Concept:
    source_slug = name.lower()
    relationships = [
        Relationship(source=source_slug, target=t, rel_type=r, confidence=c)
        for t, r, c in (rels or [])
    ]
    return Concept(
        name=name,
        definition="d",
        explanation="",
        insights=[],
        examples=[],
        tags=[],
        confidence=1.0,
        links=[],
        relationships=relationships,
        sources=[],
    )


def _build_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    a = _concept("A", [("b", RelationshipType.USES, 0.9), ("c", RelationshipType.RELATED_TO, 0.5)])
    b = _concept("B", [("d", RelationshipType.USES, 0.8)])
    c = _concept("C")
    d = _concept("D")
    for cn in (a, b, c, d):
        g.add_concept(cn)
    for cn in (a, b):
        g.add_relationships(cn)
    return g


def test_seed_concept_carries_full_score():
    g = _build_graph()
    walker = GraphWalker(g)
    scores = walker.walk(seed_scores={"a": 1.0})
    assert scores["a"] > 0.0
    # Seed gets the full hop-0 weight before normalization; after max-normalization
    # the seed should be the maximum (1.0) because hop-0 weight is largest.
    assert scores["a"] == max(scores.values())


def test_hop1_neighbor_receives_reinforcement():
    g = _build_graph()
    walker = GraphWalker(g)
    scores = walker.walk(seed_scores={"a": 1.0})
    # b is hop-1 from a via uses with confidence 0.9; should be present and < seed
    assert "b" in scores
    assert 0.0 < scores["b"] < scores["a"]


def test_hop2_neighbor_present_when_max_hops_2():
    g = _build_graph()
    walker = GraphWalker(g)
    scores = walker.walk(seed_scores={"a": 1.0}, max_hops=2)
    assert "d" in scores
    assert scores["d"] > 0.0


def test_disconnected_seed_returns_self_only():
    g = _build_graph()
    walker = GraphWalker(g)
    scores = walker.walk(seed_scores={"c": 1.0})
    # c has no outgoing relationships, so only c itself should be scored
    assert scores == {"c": 1.0}


def test_unknown_seed_skipped_silently():
    g = _build_graph()
    walker = GraphWalker(g)
    scores = walker.walk(seed_scores={"nonexistent": 1.0})
    assert scores == {}


def test_results_normalized_to_unit_max():
    g = _build_graph()
    walker = GraphWalker(g)
    scores = walker.walk(seed_scores={"a": 1.0})
    assert max(scores.values()) <= 1.0
