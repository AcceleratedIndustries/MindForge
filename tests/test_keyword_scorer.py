"""Tests for BM25-lite keyword scorer."""

from mindforge.distillation.concept import Concept
from mindforge.query.keyword_scorer import KeywordScorer


def _make_concept(name: str, definition: str, insights: list[str]) -> Concept:
    return Concept(
        name=name,
        definition=definition,
        explanation="",
        insights=insights,
    )


def test_exact_name_match_scores_highest():
    a = _make_concept("Retrieval-Augmented Generation", "A pattern.", [])
    b = _make_concept("Vector Embedding", "A representation.", [])
    scorer = KeywordScorer([a, b])
    scores = scorer.score("retrieval-augmented generation")
    assert scores[a.slug] > scores[b.slug]


def test_definition_match_scores_above_zero():
    a = _make_concept("RAG", "Retrieval-augmented generation pattern.", [])
    scorer = KeywordScorer([a])
    scores = scorer.score("retrieval augmented")
    assert scores[a.slug] > 0.0


def test_unknown_query_scores_zero():
    a = _make_concept("RAG", "A pattern.", [])
    scorer = KeywordScorer([a])
    scores = scorer.score("kubernetes")
    assert scores[a.slug] == 0.0


def test_scores_are_normalized_between_0_and_1():
    a = _make_concept(
        "Retrieval-Augmented Generation",
        "Generation augmented by retrieval.",
        ["uses retrieval"],
    )
    scorer = KeywordScorer([a])
    scores = scorer.score("retrieval generation")
    assert 0.0 <= scores[a.slug] <= 1.0
