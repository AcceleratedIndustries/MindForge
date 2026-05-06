"""Retrieval weight tuner.

Synthesizes relevance judgments from any concept corpus by treating each
concept's name+tags as the query and tag-overlapping concepts as the gold
set. Sweeps (keyword, semantic, graph) weight combinations and reports
recall@k so we can pick the winner against the docced 0.4/0.4/0.2 baseline.
"""

from __future__ import annotations

from itertools import product

from mindforge.distillation.concept import ConceptStore
from mindforge.graph.builder import KnowledgeGraph
from mindforge.query.engine import QueryEngine, RetrievalWeights


def synthesize_judgments(store: ConceptStore) -> dict[str, set[str]]:
    """For each concept, return the slugs sharing >=1 tag (excluding self)."""
    judgments: dict[str, set[str]] = {}
    for slug, concept in store.concepts.items():
        tag_set = set(concept.tags)
        if not tag_set:
            judgments[slug] = set()
            continue
        relevant: set[str] = set()
        for other_slug, other in store.concepts.items():
            if other_slug == slug:
                continue
            if tag_set & set(other.tags):
                relevant.add(other_slug)
        judgments[slug] = relevant
    return judgments


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def evaluate_weights(
    store: ConceptStore,
    graph: KnowledgeGraph,
    judgments: dict[str, set[str]],
    weights: RetrievalWeights,
    k: int = 5,
) -> float:
    """Mean recall@k across all concepts whose judgment set is non-empty."""
    engine = QueryEngine(store, graph)
    recalls: list[float] = []
    for slug, relevant in judgments.items():
        if not relevant:
            continue
        concept = store.get(slug)
        if concept is None:
            continue
        query = concept.name + " " + " ".join(concept.tags)
        # top_k+1 so that after dropping self we still have at least k candidates
        results = engine.search(query, top_k=k + 1, weights=weights)
        retrieved = [r.concept.slug for r in results if r.concept.slug != slug]
        recalls.append(recall_at_k(relevant, retrieved, k))
    return sum(recalls) / max(1, len(recalls))


def sweep_weights(
    store: ConceptStore,
    graph: KnowledgeGraph,
    judgments: dict[str, set[str]] | None = None,
    k: int = 5,
    step: float = 0.1,
) -> list[tuple[RetrievalWeights, float]]:
    """Return all (weights, recall@k) candidates sorted descending by score."""
    if judgments is None:
        judgments = synthesize_judgments(store)
    candidates: list[tuple[RetrievalWeights, float]] = []
    grid = [round(i * step, 2) for i in range(int(round(1.0 / step)) + 1)]
    for kw, sm, gr in product(grid, repeat=3):
        if abs(kw + sm + gr - 1.0) > 1e-6:
            continue
        weights = RetrievalWeights(keyword=kw, semantic=sm, graph=gr)
        score = evaluate_weights(store, graph, judgments, weights, k=k)
        candidates.append((weights, score))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates
