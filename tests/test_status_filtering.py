"""Read sites must filter out soft-deleted concepts by default."""

from __future__ import annotations

from pathlib import Path

from mindforge.distillation.concept import Concept, ConceptStore
from mindforge.distillation.renderer import write_all_concepts
from mindforge.embeddings.index import EmbeddingIndex
from mindforge.graph.builder import KnowledgeGraph
from mindforge.query.engine import filter_concepts


def _make(name: str, status: str = "active") -> Concept:
    return Concept(name=name, definition=f"def {name}", explanation=f"exp {name}", status=status)


def test_filter_concepts_excludes_deleted_by_default() -> None:
    concepts = [_make("A"), _make("B", status="deleted")]
    out = filter_concepts(concepts)
    assert [c.name for c in out] == ["A"]


def test_filter_concepts_include_deleted_returns_all() -> None:
    concepts = [_make("A"), _make("B", status="deleted")]
    out = filter_concepts(concepts, include_deleted=True)
    assert {c.name for c in out} == {"A", "B"}


def test_graph_from_store_skips_deleted() -> None:
    store = ConceptStore()
    store.add(_make("A"))
    store.add(_make("B", status="deleted"))
    graph = KnowledgeGraph.from_store(store)
    assert "a" in graph.nodes()
    assert "b" not in graph.nodes()


def test_embedding_index_build_skips_deleted() -> None:
    store = ConceptStore()
    store.add(_make("A"))
    store.add(_make("B", status="deleted"))
    index = EmbeddingIndex("all-MiniLM-L6-v2")
    # Pass raw store contents (including deleted) so build's internal filter runs.
    index.build(store.all())
    if index.available:
        assert "b" not in index._slugs  # noqa: SLF001
        assert "a" in index._slugs  # noqa: SLF001


def test_write_all_concepts_skips_deleted_and_removes_stale_file(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "concepts"
    out_dir.mkdir()
    stale = out_dir / "b.md"
    stale.write_text("old content")

    concepts = [_make("A"), _make("B", status="deleted")]
    written = write_all_concepts(concepts, out_dir)

    written_names = {p.name for p in written}
    assert "a.md" in written_names
    assert "b.md" not in written_names
    assert not stale.exists(), "stale markdown for deleted slug should be removed"
