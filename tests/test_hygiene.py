"""Tests for knowledge hygiene: model, conflict detection, decay, queue, TUI."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import StringIO

from mindforge.distillation.concept import Concept, ConceptStore
from mindforge.distillation.source_ref import SourceRef
from mindforge.hygiene.conflict_detector import (
    detect_definition_conflict,
    detect_insight_conflicts,
)
from mindforge.hygiene.decay import adjusted_confidence, age_days, is_stale
from mindforge.hygiene.markers import ConflictMarker, ConflictVariant
from mindforge.hygiene.review_queue import build_review_queue
from mindforge.hygiene.tui import review_loop


def _iso(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


# ── model roundtrip ─────────────────────────────────────────────────────────


def test_concept_hygiene_roundtrip():
    c = Concept(
        name="X", definition="d", explanation="e",
        status="conflicted",
        conflicts=[ConflictMarker(field="definition", variants=[
            ConflictVariant(
                source=SourceRef("t.md", "h", [0], "2025-01-01T00:00:00Z"),
                text="A",
            ),
            ConflictVariant(
                source=SourceRef("t2.md", "h2", [1], "2025-01-02T00:00:00Z"),
                text="B",
            ),
        ])],
        last_reinforced_at="2025-01-03T00:00:00Z",
    )
    restored = Concept.from_dict(c.to_dict())
    assert restored.status == "conflicted"
    assert len(restored.conflicts[0].variants) == 2
    assert restored.last_reinforced_at == "2025-01-03T00:00:00Z"


# ── conflict detector ───────────────────────────────────────────────────────


def test_definition_similar_no_conflict():
    a = "KV Cache stores Key and Value matrices."
    b = "KV cache stores the Key and Value matrices."
    assert detect_definition_conflict(a, b) is False


def test_definition_divergent_flags_conflict():
    a = "Context window is measured in tokens."
    b = "A context window holds approximately 128,000 characters in older APIs."
    assert detect_definition_conflict(a, b) is True


def test_insight_units_conflict():
    insights = [
        "Context window is always measured in tokens.",
        "Context window is sometimes measured in characters.",
    ]
    pairs = detect_insight_conflicts(insights)
    assert pairs  # at least one found


def test_insight_no_conflict():
    insights = ["KV cache trades memory for speed.", "MQA reduces KV cache size."]
    assert detect_insight_conflicts(insights) == []


# ── decay ────────────────────────────────────────────────────────────────────


def test_fresh_concept_keeps_confidence():
    c = adjusted_confidence(base=0.9, last_reinforced_at=_iso(1), source_count=3)
    assert c >= 0.8


def test_old_unreinforced_decays():
    c = adjusted_confidence(base=0.9, last_reinforced_at=_iso(365), source_count=1)
    assert c < 0.7


def test_reinforcement_boost_counteracts_age():
    old = adjusted_confidence(base=0.9, last_reinforced_at=_iso(180), source_count=1)
    reinforced = adjusted_confidence(base=0.9, last_reinforced_at=_iso(180), source_count=16)
    assert reinforced > old


def test_is_stale_thresholds():
    assert is_stale(adjusted=0.25, age_days_value=120) is True
    assert is_stale(adjusted=0.80, age_days_value=365) is False


def test_age_days_handles_none():
    assert age_days(None) > 1000


# ── review queue ────────────────────────────────────────────────────────────


def test_conflicted_enters_queue():
    store = ConceptStore()
    store.add(Concept(name="X", definition="d", explanation="e", status="conflicted"))
    queue = build_review_queue(store)
    assert any(item["reason"] == "conflicted" for item in queue)


def test_orphan_enters_queue():
    store = ConceptStore()
    store.add(Concept(name="Y", definition="d", explanation="e"))  # no sources
    queue = build_review_queue(store)
    assert any(item["reason"] == "orphaned" for item in queue)


def test_stale_enters_queue():
    store = ConceptStore()
    store.add(Concept(
        name="Z", definition="d", explanation="e",
        confidence=0.2,
        last_reinforced_at=_iso(200),
        source_files=["t.md"],
    ))
    queue = build_review_queue(store)
    assert any(item["reason"] == "stale" for item in queue)


# ── TUI ──────────────────────────────────────────────────────────────────────


def test_tui_empty_queue_prints_message():
    store = ConceptStore()
    out = StringIO()
    actions = review_loop(store, stdin=StringIO(""), stdout=out)
    assert actions == []
    assert "empty" in out.getvalue().lower()


def test_tui_skip_and_quit():
    store = ConceptStore()
    store.add(Concept(name="X", definition="d", explanation="e", status="conflicted"))
    store.add(Concept(name="Y", definition="d", explanation="e", status="conflicted"))
    out = StringIO()
    actions = review_loop(store, stdin=StringIO("s\nq\n"), stdout=out)
    assert actions == [("X", "skip"), ("Y", "quit")]


def test_tui_delete_removes_from_store():
    store = ConceptStore()
    store.add(Concept(name="X", definition="d", explanation="e", status="conflicted"))
    out = StringIO()
    actions = review_loop(store, stdin=StringIO("d\n"), stdout=out)
    assert actions == [("X", "delete")]
    assert store.get("x") is None
