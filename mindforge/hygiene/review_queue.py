"""Aggregate conflicted, stale, and orphaned concepts into a single queue."""

from __future__ import annotations

from typing import Any

from mindforge.distillation.concept import ConceptStore
from mindforge.hygiene.decay import (
    DEFAULT_HALF_LIFE_DAYS,
    adjusted_confidence,
    age_days,
    is_stale,
)


def build_review_queue(
    store: ConceptStore,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> list[dict[str, Any]]:
    """Return a list of queue items with slug/name/reason/adjusted fields."""
    queue: list[dict[str, Any]] = []
    for c in store.all():
        if c.status == "conflicted" or c.conflicts:
            queue.append({"slug": c.slug, "name": c.name, "reason": "conflicted"})
            continue
        has_sources = bool(c.sources) or bool(c.source_files)
        if not has_sources:
            queue.append({"slug": c.slug, "name": c.name, "reason": "orphaned"})
            continue
        source_count = len(c.sources) or len(c.source_files)
        adj = adjusted_confidence(
            base=c.confidence,
            last_reinforced_at=c.last_reinforced_at,
            source_count=source_count,
            half_life_days=half_life_days,
        )
        age = age_days(c.last_reinforced_at)
        if is_stale(adj, age):
            queue.append(
                {
                    "slug": c.slug,
                    "name": c.name,
                    "reason": "stale",
                    "adjusted": adj,
                }
            )
    return queue
