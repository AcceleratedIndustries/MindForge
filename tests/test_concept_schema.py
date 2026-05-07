"""Tests for Concept schema fields, round-trip serialization, and merge invariants."""

from mindforge.distillation.concept import Concept


def test_concept_deleted_at_round_trips_through_dict() -> None:
    c = Concept(
        name="X",
        definition="d",
        explanation="e",
        status="deleted",
        deleted_at="2026-05-07T12:00:00+00:00",
    )
    data = c.to_dict()
    assert data["deleted_at"] == "2026-05-07T12:00:00+00:00"
    assert data["status"] == "deleted"

    restored = Concept.from_dict(data)
    assert restored.deleted_at == "2026-05-07T12:00:00+00:00"
    assert restored.status == "deleted"


def test_concept_deleted_at_defaults_to_none() -> None:
    c = Concept(name="X", definition="d", explanation="e")
    assert c.deleted_at is None
    assert "deleted_at" in c.to_dict()
    assert c.to_dict()["deleted_at"] is None


def test_concept_from_dict_handles_old_manifest_without_deleted_at() -> None:
    data = {
        "name": "X",
        "definition": "d",
        "explanation": "e",
    }
    c = Concept.from_dict(data)
    assert c.deleted_at is None


def test_merge_with_active_undeletes() -> None:
    """Merging a soft-deleted concept with an active one un-deletes it."""
    deleted = Concept(
        name="X",
        definition="d",
        explanation="e",
        source_files=["a.md"],
        status="deleted",
        deleted_at="2026-05-07T12:00:00+00:00",
    )
    active = Concept(
        name="X",
        definition="d",
        explanation="e",
        source_files=["b.md"],
        status="active",
    )
    merged = deleted.merge_with(active)
    assert merged.status == "active"
    assert merged.deleted_at is None


def test_merge_with_both_deleted_keeps_earlier_timestamp() -> None:
    earlier = "2026-05-01T12:00:00+00:00"
    later = "2026-05-07T12:00:00+00:00"
    a = Concept(
        name="X",
        definition="d",
        explanation="e",
        status="deleted",
        deleted_at=later,
    )
    b = Concept(
        name="X",
        definition="d",
        explanation="e",
        status="deleted",
        deleted_at=earlier,
    )
    merged = a.merge_with(b)
    assert merged.status == "deleted"
    assert merged.deleted_at == earlier
