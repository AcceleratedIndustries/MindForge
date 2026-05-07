"""Tests for mindforge prune."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from mindforge.config import MindForgeConfig
from mindforge.distillation.concept import Concept, ConceptStore
from mindforge.prune import prune_orphans


def _make(name: str, status: str = "active", deleted_at: str | None = None) -> Concept:
    return Concept(
        name=name,
        definition=f"def {name}",
        explanation=f"exp {name}",
        status=status,
        deleted_at=deleted_at,
    )


def _seed_kb(output_dir: Path, concepts: list[Concept]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "concepts").mkdir(exist_ok=True)
    (output_dir / "provenance").mkdir(exist_ok=True)

    store = ConceptStore()
    for c in concepts:
        store.add(c)
    store.save(output_dir / "concepts.json")

    for c in concepts:
        (output_dir / "concepts" / f"{c.slug}.md").write_text(f"# {c.name}\n")
        (output_dir / "provenance" / f"{c.slug}.json").write_text("{}")


def test_prune_removes_soft_deleted_concept(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    now = datetime.now(timezone.utc).isoformat()
    _seed_kb(
        output_dir,
        [_make("A"), _make("B", status="deleted", deleted_at=now)],
    )

    config = MindForgeConfig(output_dir=output_dir)
    summary = prune_orphans(config, dry_run=False)
    assert summary.removed == 1

    store = ConceptStore.load(output_dir / "concepts.json")
    assert "b" not in store.concepts
    assert "a" in store.concepts
    assert not (output_dir / "concepts" / "b.md").exists()
    assert not (output_dir / "provenance" / "b.json").exists()
    assert (output_dir / "concepts" / "a.md").exists()


def test_prune_dry_run_changes_nothing(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    now = datetime.now(timezone.utc).isoformat()
    _seed_kb(output_dir, [_make("B", status="deleted", deleted_at=now)])

    config = MindForgeConfig(output_dir=output_dir)
    summary = prune_orphans(config, dry_run=True)
    assert summary.would_remove == 1
    assert summary.removed == 0

    store = ConceptStore.load(output_dir / "concepts.json")
    assert "b" in store.concepts
    assert (output_dir / "concepts" / "b.md").exists()


def test_prune_older_than_days_filters_recent_orphans(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    recent = datetime.now(timezone.utc).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    _seed_kb(
        output_dir,
        [
            _make("Recent", status="deleted", deleted_at=recent),
            _make("Old", status="deleted", deleted_at=old),
        ],
    )

    config = MindForgeConfig(output_dir=output_dir)
    summary = prune_orphans(config, dry_run=False, older_than_days=30)
    assert summary.removed == 1

    store = ConceptStore.load(output_dir / "concepts.json")
    assert "old" not in store.concepts
    assert "recent" in store.concepts


def test_prune_idempotent(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    now = datetime.now(timezone.utc).isoformat()
    _seed_kb(output_dir, [_make("B", status="deleted", deleted_at=now)])

    config = MindForgeConfig(output_dir=output_dir)
    prune_orphans(config, dry_run=False)
    summary = prune_orphans(config, dry_run=False)
    assert summary.removed == 0


def test_prune_rebuilds_embeddings_index_when_present(tmp_path: Path) -> None:
    """After pruning, the embeddings index should not contain pruned slugs."""
    output_dir = tmp_path / "output"
    now = datetime.now(timezone.utc).isoformat()
    _seed_kb(
        output_dir,
        [_make("A"), _make("B", status="deleted", deleted_at=now)],
    )
    # Simulate an existing embeddings dir (any file in it triggers the rebuild path).
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    (embeddings_dir / "metadata.json").write_text("{}")

    config = MindForgeConfig(output_dir=output_dir)
    summary = prune_orphans(config, dry_run=False)
    assert summary.removed == 1

    # If embeddings extras are installed, the actual rebuilt index should not list 'b'.
    # If they're not, the function should still complete cleanly without error.
    from mindforge.embeddings.index import EmbeddingIndex

    index = EmbeddingIndex(config.embedding_model)
    if index.available:
        # Reload from disk:
        loaded = EmbeddingIndex.load(embeddings_dir)
        if loaded.available:
            assert "b" not in loaded._slugs  # noqa: SLF001
