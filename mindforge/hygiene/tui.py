"""Minimal stdlib review TUI for the hygiene queue.

Actions per concept:
    s  skip       (leave as-is)
    d  delete     (remove from store)
    e  edit-noop  (records an edit intent; wiring $EDITOR is future work)
    q  quit

Returns a list of (concept_name, action) tuples so callers can verify
behavior in tests.
"""

from __future__ import annotations

from typing import TextIO

from mindforge.distillation.concept import ConceptStore
from mindforge.hygiene.review_queue import build_review_queue


def review_loop(
    store: ConceptStore,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    half_life_days: float | None = None,
) -> list[tuple[str, str]]:
    import sys

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    if half_life_days is None:
        from mindforge.hygiene.decay import DEFAULT_HALF_LIFE_DAYS
        half_life_days = DEFAULT_HALF_LIFE_DAYS

    queue = build_review_queue(store, half_life_days=half_life_days)
    actions: list[tuple[str, str]] = []

    if not queue:
        print("Review queue is empty.", file=stdout)
        return actions

    print(f"MindForge Review Queue ({len(queue)} items)", file=stdout)
    print("=" * 40, file=stdout)

    for i, item in enumerate(queue, start=1):
        concept = store.get(item["slug"])
        if concept is None:
            continue
        print(f"[{i}/{len(queue)}] {item['slug']} ({item['reason']})", file=stdout)
        print(f"  {concept.definition}", file=stdout)
        print("  [s] skip  [d] delete  [e] edit-noop  [q] quit", file=stdout)

        choice = (stdin.readline() or "").strip().lower()
        if choice == "q":
            actions.append((concept.name, "quit"))
            return actions
        if choice == "d":
            store.concepts.pop(concept.slug, None)
            actions.append((concept.name, "delete"))
        elif choice == "e":
            actions.append((concept.name, "edit-noop"))
        else:
            actions.append((concept.name, "skip"))

    return actions
