"""Compose retrieved concepts + 1-hop neighborhood into a structured payload.

Used by MCP synthesis tools (``summarize_query``, ``explain_concept``,
``compare_concepts``, ``path_between``) so each tool doesn't re-implement
retrieval + graph traversal + relationship gathering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mindforge.distillation.concept import Concept, ConceptStore
from mindforge.graph.builder import KnowledgeGraph
from mindforge.query.engine import QueryEngine


@dataclass
class RelationshipRef:
    source: str
    target: str
    rel_type: str
    confidence: float


@dataclass
class ContextPack:
    query: str
    concepts: list[Concept] = field(default_factory=list)
    neighbor_slugs: list[str] = field(default_factory=list)
    relationships: list[RelationshipRef] = field(default_factory=list)
    confidence: float = 0.0


def compose_context_pack(
    store: ConceptStore,
    graph: KnowledgeGraph,
    query: str,
    top_k: int = 5,
    max_hops: int = 2,
) -> ContextPack:
    """Run hybrid retrieval, gather direct neighbors, return a ContextPack.

    ``max_hops`` is reserved for future multi-hop expansion; today only the
    direct (1-hop) relationships of the top-k concepts are recorded.
    """
    engine = QueryEngine(store, graph)
    results = engine.search(query, top_k=top_k)
    concepts = [r.concept for r in results]

    neighbor_set: set[str] = set()
    relationships: list[RelationshipRef] = []
    for c in concepts:
        for rel in c.relationships:
            relationships.append(
                RelationshipRef(
                    source=c.slug,
                    target=rel.target,
                    rel_type=rel.rel_type.value,
                    confidence=rel.confidence,
                )
            )
            neighbor_set.add(rel.target)

    primary_slugs = {c.slug for c in concepts}
    neighbor_slugs = sorted(neighbor_set - primary_slugs)

    confidence = sum(r.score_total for r in results) / len(results) if results else 0.0

    # ``max_hops`` is a forward-compat knob; suppress unused-arg lint.
    _ = max_hops

    return ContextPack(
        query=query,
        concepts=concepts,
        neighbor_slugs=neighbor_slugs,
        relationships=relationships,
        confidence=confidence,
    )
