"""Query engine: hybrid retrieval orchestrator.

Combines three signals:

1. Keyword (BM25-lite via :class:`KeywordScorer`) — always available.
2. Semantic (vector similarity via :class:`EmbeddingIndex`) — optional.
3. Graph walk (reinforcement from neighbors via :class:`GraphWalker`).

Results are ranked by a weighted blend; weights are tunable per call via
:class:`RetrievalWeights`. The legacy single-mode behaviors are reachable
via ``mode="keyword"`` or ``mode="semantic"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from mindforge.distillation.concept import Concept, ConceptStore
from mindforge.graph.builder import KnowledgeGraph
from mindforge.query.graph_walker import GraphWalker
from mindforge.query.keyword_scorer import KeywordScorer

if TYPE_CHECKING:
    from mindforge.embeddings.index import EmbeddingIndex


def filter_concepts(
    concepts: list[Concept],
    tag: str | None = None,
    min_confidence: float | None = None,
    since: str | None = None,
) -> list[Concept]:
    """Filter concepts by tag, minimum confidence, and/or last-reinforced-since date."""
    out = list(concepts)
    if tag:
        out = [c for c in out if tag in c.tags]
    if min_confidence is not None:
        out = [c for c in out if c.confidence >= min_confidence]
    if since:
        from datetime import timezone

        cutoff = datetime.fromisoformat(since.replace("Z", "+00:00"))
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        kept: list[Concept] = []
        for c in out:
            if not c.last_reinforced_at:
                continue
            ts = datetime.fromisoformat(c.last_reinforced_at.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                kept.append(c)
        out = kept
    return out


@dataclass
class RetrievalWeights:
    """Weighting for the three retrieval signals.

    Defaults assume embeddings are available. If they aren't, callers should
    use :meth:`no_embeddings` or let :meth:`QueryEngine.search` substitute it
    automatically when ``mode="hybrid"``.
    """

    keyword: float = 0.4
    semantic: float = 0.4
    graph: float = 0.2

    @classmethod
    def no_embeddings(cls) -> RetrievalWeights:
        return cls(keyword=0.6, semantic=0.0, graph=0.4)

    @classmethod
    def keyword_only(cls) -> RetrievalWeights:
        return cls(keyword=1.0, semantic=0.0, graph=0.0)

    @classmethod
    def semantic_only(cls) -> RetrievalWeights:
        return cls(keyword=0.0, semantic=1.0, graph=0.0)


@dataclass
class QueryResult:
    """A single result from a knowledge base query.

    ``score`` and ``neighbors`` are preserved for backward compatibility with
    the existing MCP server payload shape.
    """

    concept: Concept
    score_total: float
    score_breakdown: dict[str, float]
    matched_via: list[str]
    neighbors: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.score_total

    @property
    def match_type(self) -> str:
        if not self.matched_via:
            return "none"
        if len(self.matched_via) == 1:
            return self.matched_via[0]
        return "combined"


class QueryEngine:
    """Search interface for the MindForge knowledge base."""

    SEED_POOL_SIZE = 10

    def __init__(
        self,
        store: ConceptStore,
        graph: KnowledgeGraph | None = None,
        embedding_index: EmbeddingIndex | None = None,
    ) -> None:
        self._store = store
        self._graph = graph
        self._index = embedding_index
        self._keyword_scorer = KeywordScorer(store.all())
        self._graph_walker = GraphWalker(graph) if graph is not None else None

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        weights: RetrievalWeights | None = None,
    ) -> list[QueryResult]:
        """Search the knowledge base with a natural language query.

        ``mode`` is validated even when ``weights`` is supplied so a typo in
        ``mode`` cannot pass silently.
        """
        if mode not in ("hybrid", "keyword", "semantic"):
            raise ValueError(f"Unknown mode: {mode!r}")
        if weights is None:
            weights = self._default_weights(mode)

        kw_scores = self._keyword_scorer.score(query) if weights.keyword > 0 else {}

        sem_scores: dict[str, float] = {}
        if weights.semantic > 0 and self._index is not None and self._index.available:
            sem_scores = self._semantic_scores(query)

        seed_pool = self._build_seed_pool(kw_scores, sem_scores)
        graph_scores: dict[str, float] = {}
        if weights.graph > 0 and self._graph_walker is not None:
            graph_scores = self._graph_walker.walk(seed_pool)

        all_slugs = set(kw_scores) | set(sem_scores) | set(graph_scores)
        results: list[QueryResult] = []
        for slug in all_slugs:
            k = kw_scores.get(slug, 0.0)
            s = sem_scores.get(slug, 0.0)
            g = graph_scores.get(slug, 0.0)
            kw_part = weights.keyword * k
            sem_part = weights.semantic * s
            graph_part = weights.graph * g
            total = kw_part + sem_part + graph_part
            if total <= 0.0:
                continue
            matched_via = [
                label for label, raw in (("keyword", k), ("semantic", s), ("graph", g)) if raw > 0.0
            ]
            concept = self._store.get(slug)
            if concept is None:
                continue
            neighbors = self._graph.neighbors(slug) if self._graph is not None else []
            results.append(
                QueryResult(
                    concept=concept,
                    score_total=total,
                    score_breakdown={
                        "keyword": kw_part,
                        "semantic": sem_part,
                        "graph": graph_part,
                    },
                    matched_via=matched_via,
                    neighbors=neighbors,
                )
            )
        results.sort(key=lambda r: r.score_total, reverse=True)
        return results[:top_k]

    def _default_weights(self, mode: str) -> RetrievalWeights:
        if mode == "hybrid":
            if self._index is not None and self._index.available:
                return RetrievalWeights()
            return RetrievalWeights.no_embeddings()
        if mode == "keyword":
            return RetrievalWeights.keyword_only()
        if mode == "semantic":
            return RetrievalWeights.semantic_only()
        raise ValueError(f"Unknown mode: {mode}")

    def _build_seed_pool(
        self,
        kw_scores: dict[str, float],
        sem_scores: dict[str, float],
    ) -> dict[str, float]:
        top_kw = sorted(kw_scores.items(), key=lambda x: x[1], reverse=True)[: self.SEED_POOL_SIZE]
        top_sem = sorted(sem_scores.items(), key=lambda x: x[1], reverse=True)[
            : self.SEED_POOL_SIZE
        ]
        seed_pool: dict[str, float] = {}
        for slug, s in top_kw + top_sem:
            if s <= 0.0:
                continue
            seed_pool[slug] = max(seed_pool.get(slug, 0.0), s)
        return seed_pool

    def _semantic_scores(self, query: str) -> dict[str, float]:
        if self._index is None:
            return {}
        raw = self._index.query(query, top_k=self.SEED_POOL_SIZE * 5)
        if not raw:
            return {}
        max_score = max((score for _, score in raw), default=0.0)
        if max_score <= 0.0:
            return {slug: 0.0 for slug, _ in raw}
        return {slug: score / max_score for slug, score in raw}

    def format_results(self, results: list[QueryResult]) -> str:
        """Format query results as human-readable text."""
        if not results:
            return "No matching concepts found."

        lines: list[str] = []
        for i, result in enumerate(results, 1):
            c = result.concept
            lines.append(f"{'─' * 60}")
            lines.append(
                f"  [{i}] {c.name}  (score: {result.score_total:.2f}, {result.match_type})"
            )
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
