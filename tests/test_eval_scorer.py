"""Tests for the eval scoring metrics."""

from __future__ import annotations

from mindforge.eval.scorer import score_concepts, score_relationships


def test_concept_recall_exact_slug_match():
    expected = [{"slug": "kv-cache", "name": "KV Cache"}]
    actual = [{"slug": "kv-cache", "name": "KV Cache", "definition": "..."}]
    r = score_concepts(expected, actual)
    assert r["recall"] == 1.0
    assert r["precision"] == 1.0


def test_concept_recall_fuzzy_name_match():
    expected = [{"slug": "kv-cache", "name": "KV Cache"}]
    actual = [{"slug": "kv-caches", "name": "KV Caches", "definition": "..."}]
    r = score_concepts(expected, actual)
    # fuzzy threshold 0.85 — slight name variance should still match.
    assert r["recall"] == 1.0


def test_concept_missing_drops_recall():
    expected = [
        {"slug": "kv-cache", "name": "KV Cache"},
        {"slug": "attention", "name": "Attention"},
    ]
    actual = [{"slug": "kv-cache", "name": "KV Cache", "definition": "..."}]
    r = score_concepts(expected, actual)
    assert r["recall"] == 0.5


def test_phrase_grounding():
    expected = [{"slug": "x", "name": "X", "key_phrases": ["foo bar", "baz"]}]
    actual = [{
        "slug": "x", "name": "X",
        "definition": "This concerns foo bar.",
        "insights": ["baz happens"],
    }]
    r = score_concepts(expected, actual)
    assert r["phrase_grounding"] == 1.0


def test_phrase_grounding_partial():
    expected = [{"slug": "x", "name": "X", "key_phrases": ["foo", "nope"]}]
    actual = [{"slug": "x", "name": "X", "definition": "foo exists", "insights": []}]
    r = score_concepts(expected, actual)
    assert r["phrase_grounding"] == 0.5


def test_relationship_recall_and_type_accuracy():
    expected = [{"source": "a", "target": "b", "type": "related_to"}]
    actual = [{"source": "a", "target": "b", "type": "related_to"}]
    r = score_relationships(expected, actual)
    assert r["recall"] == 1.0
    assert r["type_accuracy"] == 1.0


def test_relationship_type_mismatch():
    expected = [{"source": "a", "target": "b", "type": "related_to"}]
    actual = [{"source": "a", "target": "b", "type": "uses"}]
    r = score_relationships(expected, actual)
    assert r["recall"] == 1.0
    assert r["type_accuracy"] == 0.0


def test_empty_expected_is_clean():
    r = score_concepts([], [])
    assert r["recall"] == 1.0
    r = score_relationships([], [])
    assert r["recall"] == 1.0
