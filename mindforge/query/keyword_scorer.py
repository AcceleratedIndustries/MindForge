"""BM25-lite keyword scorer.

Scores a query against a corpus of Concepts using TF-IDF over their
name + definition + insights text. Intentionally simple: full BM25 is
overkill for KB-sized corpora (< 10k concepts typical); this gives
fast, deterministic, embedding-free baseline relevance.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from mindforge.distillation.concept import Concept

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _concept_text(concept: Concept) -> str:
    parts = [concept.name, concept.definition]
    parts.extend(concept.insights)
    return " ".join(p for p in parts if p)


@dataclass
class _DocStats:
    counts: Counter[str]
    length: int


class KeywordScorer:
    """TF-IDF over the concept name + definition + insights string."""

    def __init__(self, concepts: list[Concept]) -> None:
        self._docs: dict[str, _DocStats] = {}
        df: Counter[str] = Counter()
        for c in concepts:
            tokens = _tokenize(_concept_text(c))
            self._docs[c.slug] = _DocStats(counts=Counter(tokens), length=len(tokens))
            for term in set(tokens):
                df[term] += 1
        n = max(1, len(concepts))
        self._idf: dict[str, float] = {
            term: math.log((n - n_t + 0.5) / (n_t + 0.5) + 1.0) for term, n_t in df.items()
        }

    def score(self, query: str) -> dict[str, float]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return {slug: 0.0 for slug in self._docs}
        raw_scores: dict[str, float] = {}
        for slug, stats in self._docs.items():
            if stats.length == 0:
                raw_scores[slug] = 0.0
                continue
            score = 0.0
            for q in query_tokens:
                tf = stats.counts.get(q, 0) / stats.length
                idf = self._idf.get(q, 0.0)
                score += tf * idf
            raw_scores[slug] = score
        max_score = max(raw_scores.values(), default=0.0)
        if max_score <= 0.0:
            return {slug: 0.0 for slug in raw_scores}
        return {slug: s / max_score for slug, s in raw_scores.items()}
