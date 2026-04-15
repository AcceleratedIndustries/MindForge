"""Query engine: provides a simple interface for searching the knowledge base.

Supports two search modes:
1. Keyword search (always available) - TF-IDF-like matching
2. Semantic search (when embeddings are available) - vector similarity

Results are combined and ranked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from mindforge.distillation.concept import Concept, ConceptStore
from mindforge.embeddings.index import EmbeddingIndex
from mindforge.graph.builder import KnowledgeGraph
from mindforge.utils.text import extract_keywords


@dataclass
class QueryResult:
    """A single result from a knowledge base query."""
    concept: Concept
    score: float
    match_type: str  # "keyword", "semantic", "combined"
    neighbors: list[str]  # related concept slugs


class QueryEngine:
    """Search interface for the MindForge knowledge base."""

    def __init__(
        self,
        store: ConceptStore,
        graph: KnowledgeGraph | None = None,
        embedding_index: EmbeddingIndex | None = None,
    ) -> None:
        self._store = store
        self._graph = graph
        self._index = embedding_index

    def search(self, query: str, top_k: int = 5) -> list[QueryResult]:
        """Search the knowledge base with a natural language query.

        Combines keyword and semantic search when available.
        """
        scores: dict[str, tuple[float, str]] = {}  # slug -> (score, match_type)

        # Keyword search (always available)
        kw_results = self._keyword_search(query, top_k=top_k * 2)
        for slug, score in kw_results:
            scores[slug] = (score, "keyword")

        # Semantic search (when available)
        if self._index and self._index.available:
            sem_results = self._index.query(query, top_k=top_k * 2)
            for slug, score in sem_results:
                if slug in scores:
                    # Combine scores
                    old_score = scores[slug][0]
                    scores[slug] = (old_score * 0.4 + score * 0.6, "combined")
                else:
                    scores[slug] = (score, "semantic")

        # Sort by score and build results
        ranked = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        results: list[QueryResult] = []

        for slug, (score, match_type) in ranked[:top_k]:
            concept = self._store.get(slug)
            if concept is None:
                continue

            neighbors = []
            if self._graph:
                neighbors = self._graph.neighbors(slug)

            results.append(QueryResult(
                concept=concept,
                score=score,
                match_type=match_type,
                neighbors=neighbors,
            ))

        return results

    def _keyword_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search concepts by keyword matching."""
        query_terms = set(re.findall(r"\w+", query.lower()))
        query_keywords = set(extract_keywords(query, top_n=10))
        all_query_terms = query_terms | query_keywords

        if not all_query_terms:
            return []

        scored: list[tuple[str, float]] = []

        for concept in self._store.all():
            score = 0.0

            # Name match (highest weight)
            name_lower = concept.name.lower()
            name_words = set(re.findall(r"\w+", name_lower))
            name_overlap = len(all_query_terms & name_words)
            if name_overlap:
                score += name_overlap * 0.4

            # Exact name substring match
            if query.lower() in name_lower or name_lower in query.lower():
                score += 0.5

            # Tag match
            tag_set = set(t.lower() for t in concept.tags)
            tag_overlap = len(all_query_terms & tag_set)
            score += tag_overlap * 0.2

            # Definition keyword match
            def_words = set(re.findall(r"\w+", concept.definition.lower()))
            def_overlap = len(all_query_terms & def_words)
            score += def_overlap * 0.05

            if score > 0:
                scored.append((concept.slug, min(score, 1.0)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def format_results(self, results: list[QueryResult]) -> str:
        """Format query results as human-readable text."""
        if not results:
            return "No matching concepts found."

        lines: list[str] = []
        for i, result in enumerate(results, 1):
            c = result.concept
            lines.append(f"{'─' * 60}")
            lines.append(f"  [{i}] {c.name}  (score: {result.score:.2f}, {result.match_type})")
            lines.append(f"      {c.definition[:150]}...")
            if result.neighbors:
                neighbor_names = []
                for ns in result.neighbors[:5]:
                    nc = self._store.get(ns)
                    neighbor_names.append(nc.name if nc else ns)
                lines.append(f"      Related: {', '.join(neighbor_names)}")
            lines.append("")

        lines.append(f"{'─' * 60}")
        return "\n".join(lines)
