"""Graph-walk scorer for hybrid retrieval.

Given a set of seed concepts with seed scores, propagate reinforcement
to 1-2 hop neighbors weighted by edge confidence. Concepts adjacent to
multiple strong seeds rank above lone keyword/semantic hits.
"""

from __future__ import annotations

from mindforge.graph.builder import KnowledgeGraph

_HOP_WEIGHTS = {0: 1.0, 1: 0.5, 2: 0.2}


class GraphWalker:
    """Walks the KnowledgeGraph from a set of seeds, accumulating reinforcement."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self._graph = graph

    def walk(
        self,
        seed_scores: dict[str, float],
        max_hops: int = 2,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        nodes = set(self._graph.nodes())
        for seed, sigma in seed_scores.items():
            if seed not in nodes:
                continue
            scores[seed] = scores.get(seed, 0.0) + sigma * _HOP_WEIGHTS[0]
            self._propagate(seed, sigma, scores, current_hop=1, max_hops=max_hops, visited={seed})
        max_score = max(scores.values(), default=0.0)
        if max_score <= 0.0:
            return scores
        return {slug: s / max_score for slug, s in scores.items()}

    def _propagate(
        self,
        node: str,
        sigma: float,
        scores: dict[str, float],
        current_hop: int,
        max_hops: int,
        visited: set[str],
    ) -> None:
        if current_hop > max_hops:
            return
        weight = _HOP_WEIGHTS.get(current_hop, 0.0)
        for neighbor, edge_confidence in self._graph.neighbors_with_confidence(node):
            if neighbor in visited:
                continue
            scores[neighbor] = scores.get(neighbor, 0.0) + sigma * edge_confidence * weight
            self._propagate(
                neighbor, sigma, scores, current_hop + 1, max_hops, visited | {neighbor}
            )
