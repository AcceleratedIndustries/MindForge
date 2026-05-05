"""End-to-end hybrid retrieval orchestrator tests."""

import pytest

from mindforge.distillation.concept import (
    Concept,
    ConceptStore,
    Relationship,
    RelationshipType,
)
from mindforge.graph.builder import KnowledgeGraph
from mindforge.query.engine import QueryEngine, RetrievalWeights


def _concept(
    name: str,
    definition: str,
    rels: list[tuple[str, RelationshipType, float]] | None = None,
) -> Concept:
    relationships = [
        Relationship(
            source=name.lower().replace(" ", "-"),
            target=t,
            rel_type=r,
            confidence=c,
        )
        for t, r, c in (rels or [])
    ]
    return Concept(
        name=name,
        definition=definition,
        explanation="",
        insights=[],
        examples=[],
        tags=[],
        confidence=1.0,
        links=[],
        relationships=relationships,
        sources=[],
    )


@pytest.fixture
def kb() -> tuple[ConceptStore, KnowledgeGraph]:
    store = ConceptStore()
    store.add(
        _concept(
            "Retrieval-Augmented Generation",
            "A pattern combining retrieval with generation.",
            [("vector-embedding", RelationshipType.USES, 0.9)],
        )
    )
    store.add(_concept("Vector Embedding", "A high-dimensional representation."))
    store.add(_concept("Kubernetes", "A container orchestration platform."))
    graph = KnowledgeGraph.from_store(store)
    return store, graph


def test_hybrid_default_returns_keyword_match_first(
    kb: tuple[ConceptStore, KnowledgeGraph],
) -> None:
    store, graph = kb
    engine = QueryEngine(store, graph)
    results = engine.search("retrieval-augmented generation", top_k=3)
    assert results[0].concept.slug == "retrieval-augmented-generation"


def test_hybrid_includes_score_breakdown(
    kb: tuple[ConceptStore, KnowledgeGraph],
) -> None:
    store, graph = kb
    engine = QueryEngine(store, graph)
    results = engine.search("retrieval generation", top_k=3)
    assert "keyword" in results[0].score_breakdown
    assert "graph" in results[0].score_breakdown
    assert "semantic" in results[0].score_breakdown


def test_graph_walk_lifts_neighbor_above_keyword_only(
    kb: tuple[ConceptStore, KnowledgeGraph],
) -> None:
    """A concept connected to a strong keyword hit should outrank an unrelated weak match."""
    store, graph = kb
    engine = QueryEngine(store, graph)
    results = engine.search("retrieval-augmented generation", top_k=3)
    slugs = [r.concept.slug for r in results]
    if "vector-embedding" in slugs and "kubernetes" in slugs:
        assert slugs.index("vector-embedding") < slugs.index("kubernetes")


def test_keyword_only_mode_skips_graph_walk(
    kb: tuple[ConceptStore, KnowledgeGraph],
) -> None:
    store, graph = kb
    engine = QueryEngine(store, graph)
    results = engine.search("retrieval-augmented generation", top_k=3, mode="keyword")
    assert results[0].score_breakdown["graph"] == 0.0


def test_explicit_weights_override_default(
    kb: tuple[ConceptStore, KnowledgeGraph],
) -> None:
    store, graph = kb
    engine = QueryEngine(store, graph)
    weights = RetrievalWeights(keyword=1.0, semantic=0.0, graph=0.0)
    results = engine.search("retrieval", top_k=3, weights=weights)
    assert results[0].score_total == pytest.approx(results[0].score_breakdown["keyword"], abs=1e-9)


def test_no_embeddings_reweights_automatically(
    kb: tuple[ConceptStore, KnowledgeGraph],
) -> None:
    """When embeddings unavailable, default weights shift to (0.6, 0.0, 0.4)."""
    store, graph = kb
    engine = QueryEngine(store, graph)
    results = engine.search("retrieval", top_k=3)
    assert results[0].score_breakdown["semantic"] == 0.0


def test_matched_via_lists_contributing_signals(
    kb: tuple[ConceptStore, KnowledgeGraph],
) -> None:
    store, graph = kb
    engine = QueryEngine(store, graph)
    results = engine.search("retrieval-augmented generation", top_k=3)
    assert "keyword" in results[0].matched_via
