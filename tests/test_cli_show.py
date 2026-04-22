"""Tests for the mindforge show render path."""

from __future__ import annotations

from mindforge.cli import _render_show
from mindforge.distillation.concept import (
    Concept,
    ConceptStore,
    Relationship,
    RelationshipType,
)


def test_show_default_output() -> None:
    c = Concept(name="X", definition="d", explanation="e")
    text = _render_show(c, sources=False, neighbors=False, raw=False, store=None)
    assert "X" in text
    assert "d" in text


def test_show_sources_output() -> None:
    from mindforge.distillation.source_ref import SourceRef

    c = Concept(
        name="X",
        definition="d",
        explanation="e",
        sources=[
            SourceRef(
                transcript_path="t.md",
                transcript_hash="h",
                turn_indices=[4, 7],
                extracted_at="2025-01-01T00:00:00Z",
            )
        ],
    )
    text = _render_show(c, sources=True, neighbors=False, raw=False, store=None)
    assert "t.md" in text
    assert "4, 7" in text


def test_show_neighbors_output() -> None:
    store = ConceptStore()
    store.add(
        Concept(
            name="X",
            definition="d",
            explanation="e",
            relationships=[Relationship("x", "y", RelationshipType.USES)],
        )
    )
    store.add(Concept(name="Y", definition="d", explanation="e"))
    text = _render_show(store.get("x"), sources=False, neighbors=True, raw=False, store=store)
    assert "y" in text.lower()
    assert "uses" in text.lower()


def test_show_raw_prints_markdown() -> None:
    from mindforge.distillation.renderer import render_concept

    c = Concept(name="X", definition="d", explanation="e")
    raw_md = render_concept(c)
    text = _render_show(c, sources=False, neighbors=False, raw=True, store=None)
    assert text.strip() == raw_md.strip()
