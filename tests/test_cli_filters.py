"""Tests for the query/list filter helpers."""

from __future__ import annotations

from mindforge.distillation.concept import Concept
from mindforge.query.engine import filter_concepts


def test_filter_by_tag() -> None:
    concepts = [
        Concept(name="A", definition="d", explanation="e", tags=["rag"]),
        Concept(name="B", definition="d", explanation="e", tags=["ml"]),
    ]
    out = filter_concepts(concepts, tag="rag")
    assert [c.name for c in out] == ["A"]


def test_filter_by_min_confidence() -> None:
    concepts = [
        Concept(name="A", definition="d", explanation="e", confidence=0.9),
        Concept(name="B", definition="d", explanation="e", confidence=0.3),
    ]
    out = filter_concepts(concepts, min_confidence=0.5)
    assert [c.name for c in out] == ["A"]


def test_filter_by_since() -> None:
    concepts = [
        Concept(
            name="A",
            definition="d",
            explanation="e",
            last_reinforced_at="2026-04-22T00:00:00+00:00",
        ),
        Concept(
            name="B",
            definition="d",
            explanation="e",
            last_reinforced_at="2026-01-01T00:00:00+00:00",
        ),
    ]
    out = filter_concepts(concepts, since="2026-03-01")
    assert [c.name for c in out] == ["A"]


def test_filter_combined() -> None:
    concepts = [
        Concept(name="A", definition="d", explanation="e", tags=["rag"], confidence=0.9),
        Concept(name="B", definition="d", explanation="e", tags=["rag"], confidence=0.1),
        Concept(name="C", definition="d", explanation="e", tags=["ml"], confidence=0.9),
    ]
    out = filter_concepts(concepts, tag="rag", min_confidence=0.5)
    assert [c.name for c in out] == ["A"]
