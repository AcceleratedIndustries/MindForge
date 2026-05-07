"""Hard-delete soft-marked concepts and their on-disk artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from mindforge.config import MindForgeConfig
from mindforge.distillation.concept import ConceptStore
from mindforge.graph.builder import KnowledgeGraph


@dataclass
class PruneSummary:
    removed: int = 0
    would_remove: int = 0
    slugs: list[str] = field(default_factory=list)


def prune_orphans(
    config: MindForgeConfig,
    dry_run: bool = False,
    older_than_days: int | None = None,
) -> PruneSummary:
    """Remove soft-deleted concepts (and their artifacts) from the KB."""
    summary = PruneSummary()

    manifest = config.output_dir / "concepts.json"
    if not manifest.exists():
        return summary

    store = ConceptStore.load(manifest)

    cutoff: datetime | None = None
    if older_than_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)

    to_remove: list[str] = []
    for slug, concept in list(store.concepts.items()):
        if concept.status != "deleted":
            continue
        if cutoff is not None:
            if not concept.deleted_at:
                continue
            ts = datetime.fromisoformat(concept.deleted_at.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                continue
        to_remove.append(slug)

    summary.slugs = to_remove

    if dry_run:
        summary.would_remove = len(to_remove)
        return summary

    for slug in to_remove:
        store.concepts.pop(slug, None)
        (config.output_dir / "concepts" / f"{slug}.md").unlink(missing_ok=True)
        (config.output_dir / "provenance" / f"{slug}.json").unlink(missing_ok=True)

    if to_remove:
        store.save(manifest)

        graph_path = config.graph_dir / "knowledge_graph.json"
        if graph_path.exists():
            graph = KnowledgeGraph.from_store(store)
            graph.save(graph_path)

        embeddings_dir = config.embeddings_dir
        if embeddings_dir.exists() and any(embeddings_dir.iterdir()):
            from mindforge.embeddings.index import EmbeddingIndex

            index = EmbeddingIndex(config.embedding_model)
            if index.available:
                index.build(store.all())
                index.save(embeddings_dir)

    summary.removed = len(to_remove)
    return summary
