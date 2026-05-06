"""Tests for retrieval tuner: judgment synthesis + recall@k + weight sweep."""

from mindforge.distillation.concept import Concept, ConceptStore
from mindforge.eval.retrieval_tuner import (
    recall_at_k,
    sweep_weights,
    synthesize_judgments,
)
from mindforge.graph.builder import KnowledgeGraph
from mindforge.query.engine import RetrievalWeights


def _c(name: str, definition: str, tags: list[str]) -> Concept:
    return Concept(
        name=name,
        definition=definition,
        explanation="",
        insights=[],
        examples=[],
        tags=tags,
        confidence=1.0,
        links=[],
        relationships=[],
        sources=[],
    )


def test_recall_at_k_full_match_returns_one():
    relevant = {"a", "b"}
    retrieved = ["a", "b", "c"]
    assert recall_at_k(relevant, retrieved, k=2) == 1.0


def test_recall_at_k_partial():
    relevant = {"a", "b"}
    retrieved = ["a", "x", "y"]
    assert recall_at_k(relevant, retrieved, k=3) == 0.5


def test_recall_at_k_empty_relevant_returns_zero():
    assert recall_at_k(set(), ["a", "b"], k=2) == 0.0


def test_synthesize_judgments_uses_tag_overlap():
    store = ConceptStore()
    store.add(_c("RAG", "Retrieval-augmented gen.", ["llm", "retrieval"]))
    store.add(_c("Vector", "High-dim repr.", ["embeddings"]))
    store.add(_c("HyDE", "Hypothetical doc emb.", ["llm", "retrieval"]))
    judgments = synthesize_judgments(store)
    # query "rag" should consider hyde relevant via tag overlap, vector not
    assert "hyde" in judgments["rag"]
    assert "vector" not in judgments["rag"]


def test_synthesize_judgments_skips_self():
    store = ConceptStore()
    store.add(_c("RAG", "x", ["llm"]))
    store.add(_c("HyDE", "y", ["llm"]))
    judgments = synthesize_judgments(store)
    assert "rag" not in judgments["rag"]
    assert "hyde" in judgments["rag"]


def test_synthesize_judgments_no_tags_yields_empty():
    store = ConceptStore()
    store.add(_c("Lone", "no tags", []))
    store.add(_c("Other", "also no tags", []))
    judgments = synthesize_judgments(store)
    assert judgments["lone"] == set()
    assert judgments["other"] == set()


def test_sweep_weights_returns_sorted_descending():
    store = ConceptStore()
    store.add(_c("RAG", "Retrieval-augmented gen.", ["llm", "retrieval"]))
    store.add(_c("HyDE", "Hypothetical doc emb.", ["llm", "retrieval"]))
    store.add(_c("Vector", "Embeddings.", ["embeddings"]))
    graph = KnowledgeGraph.from_store(store)
    candidates = sweep_weights(store, graph, k=3, step=0.5)
    # Sorted by recall descending
    scores = [s for _, s in candidates]
    assert scores == sorted(scores, reverse=True)
    # Each weight tuple sums to ~1.0
    for w, _ in candidates:
        assert abs(w.keyword + w.semantic + w.graph - 1.0) < 1e-6


def test_sweep_weights_step_controls_grid_size():
    store = ConceptStore()
    store.add(_c("A", "x", ["t"]))
    store.add(_c("B", "y", ["t"]))
    graph = KnowledgeGraph.from_store(store)
    # step=0.5 → values [0.0, 0.5, 1.0]; combos summing to 1.0:
    # (1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5) = 6
    candidates = sweep_weights(store, graph, k=2, step=0.5)
    assert len(candidates) == 6


def test_sweep_weights_uses_provided_weights_type():
    store = ConceptStore()
    store.add(_c("A", "x", ["t"]))
    store.add(_c("B", "y", ["t"]))
    graph = KnowledgeGraph.from_store(store)
    candidates = sweep_weights(store, graph, k=2, step=0.5)
    assert all(isinstance(w, RetrievalWeights) for w, _ in candidates)
