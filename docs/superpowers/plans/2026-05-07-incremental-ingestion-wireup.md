# Incremental ingestion wire-up — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-07-incremental-ingestion-wireup-design.md`](../specs/2026-05-07-incremental-ingestion-wireup-design.md)

**Goal:** Wire file-hash-based incrementality into `mindforge ingest` so a re-run against unchanged transcripts returns in seconds, modified-file changes are correctly re-extracted with stale concepts soft-deleted, and a new `mindforge prune` command performs orphan cleanup.

**Architecture:** A new slim `FileHashStore` persists per-file SHA-256 hashes at `output/.ingest/content_hashes.json`. The pipeline auto-detects incremental mode when the cache exists. Modified-or-deleted files are dropped from the loaded `ConceptStore` first (concepts whose `source_files` go empty are soft-marked `status="deleted"`), then new+modified files are extracted and merged in via the existing `store.add()` path. Read sites (query, list, show, graph, embeddings, MCP, renderer) filter to `status=="active"` by default. A separate `mindforge prune` command performs hard delete of soft-marked concepts.

**Tech Stack:** Python 3.11+, stdlib (hashlib, json, pathlib), pytest. No new third-party deps.

---

## File Structure

**New files:**
- `mindforge/ingestion/file_hash_store.py` — `ContentHasher` and `FileHashStore` (the only surviving primitives from `incremental.py`).
- `tests/test_file_hash_store.py` — unit tests for the new module.
- `tests/test_incremental_pipeline.py` — integration tests against real `MindForgePipeline`.
- `tests/test_prune.py` — tests for `mindforge prune`.

**Modified files:**
- `mindforge/distillation/concept.py` — add `deleted_at: str | None = None` field, update `to_dict`/`from_dict`.
- `mindforge/distillation/renderer.py` — `write_all_concepts` skips deleted concepts and removes stale on-disk markdown files for slugs that just transitioned to deleted.
- `mindforge/query/engine.py` — `filter_concepts` gains `include_deleted: bool = False`; default behavior excludes `status=="deleted"`.
- `mindforge/graph/builder.py` — `KnowledgeGraph.from_store` skips deleted concepts.
- `mindforge/embeddings/index.py` — `EmbeddingIndex.build` skips deleted concepts.
- `mindforge/mcp/server.py` — `list_concepts` and `get_concept` filter out deleted (silently; not exposed to MCP clients).
- `mindforge/pipeline.py` — incremental flow: auto-detect, drop logic, fast path, new `PipelineResult` fields (`skipped`, `files_*`, `concepts_soft_deleted`).
- `mindforge/cli.py` — `--full` flag on `ingest`, new `prune` subcommand, `--include-deleted` flag on `query`/`list`/`show`.

**Deleted files:**
- `mindforge/ingestion/incremental.py` — gutted module; `ContentHasher` moved out, everything else removed.
- `tests/test_incremental_ingestion.py` — corresponding test file; coverage of `ContentHasher` moves to `test_file_hash_store.py`.

---

## Task 1: `FileHashStore` module

Pure new module, no dependencies on the rest of the change set. Build it TDD-first.

**Files:**
- Create: `mindforge/ingestion/file_hash_store.py`
- Test: `tests/test_file_hash_store.py`

- [ ] **Step 1.1: Write failing tests for `ContentHasher`**

Create `tests/test_file_hash_store.py`:

```python
"""Tests for FileHashStore and ContentHasher."""

from __future__ import annotations

from pathlib import Path

import pytest

from mindforge.ingestion.file_hash_store import ContentHasher, FileHashStore


class TestContentHasher:
    def test_hash_string_is_stable(self) -> None:
        h = ContentHasher()
        assert h.hash_string("hello") == h.hash_string("hello")

    def test_hash_string_differs_for_different_input(self) -> None:
        h = ContentHasher()
        assert h.hash_string("hello") != h.hash_string("world")

    def test_hash_bytes_matches_hash_string_for_utf8(self) -> None:
        h = ContentHasher()
        assert h.hash_bytes(b"hello") == h.hash_string("hello")

    def test_hash_file_matches_hash_bytes(self, tmp_path: Path) -> None:
        h = ContentHasher()
        f = tmp_path / "a.txt"
        f.write_bytes(b"hello world")
        assert h.hash_file(f) == h.hash_bytes(b"hello world")
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
pytest tests/test_file_hash_store.py -v
```
Expected: collection error or `ImportError` because the module does not exist yet.

- [ ] **Step 1.3: Implement `ContentHasher`**

Create `mindforge/ingestion/file_hash_store.py`:

```python
"""File-hash manifest for incremental ingestion.

Persists per-file SHA-256 hashes to ``output/.ingest/content_hashes.json``
so the pipeline can skip re-parsing/re-extracting unchanged transcripts on
subsequent runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ContentHasher:
    """SHA-256 content hashing for file change detection."""

    def hash_file(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def hash_bytes(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def hash_string(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
```

- [ ] **Step 1.4: Run tests to verify `ContentHasher` passes**

```bash
pytest tests/test_file_hash_store.py::TestContentHasher -v
```
Expected: 4 passed.

- [ ] **Step 1.5: Write failing tests for `FileHashStore`**

Append to `tests/test_file_hash_store.py`:

```python
class TestFileHashStore:
    def test_new_file_status(self, tmp_path: Path) -> None:
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        f = transcripts / "a.md"
        f.write_text("hello")

        store = FileHashStore.load(tmp_path / ".ingest", transcripts)
        status = store.status_of(f)
        assert status.is_new is True
        assert status.is_modified is False
        assert status.is_unchanged is False
        assert status.current_hash is not None
        assert status.previous_hash is None

    def test_unchanged_file_status_after_update(self, tmp_path: Path) -> None:
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        f = transcripts / "a.md"
        f.write_text("hello")

        store = FileHashStore.load(tmp_path / ".ingest", transcripts)
        store.update(f, store.hasher.hash_file(f))
        status = store.status_of(f)
        assert status.is_unchanged is True

    def test_modified_file_status(self, tmp_path: Path) -> None:
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        f = transcripts / "a.md"
        f.write_text("hello")

        store = FileHashStore.load(tmp_path / ".ingest", transcripts)
        store.update(f, store.hasher.hash_file(f))
        f.write_text("hello world")
        status = store.status_of(f)
        assert status.is_modified is True

    def test_known_paths_returns_stored_relative_paths(self, tmp_path: Path) -> None:
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        f = transcripts / "a.md"
        f.write_text("x")

        store = FileHashStore.load(tmp_path / ".ingest", transcripts)
        store.update(f, "abc")
        assert (transcripts / "a.md").resolve() in store.known_paths()

    def test_forget_removes_path(self, tmp_path: Path) -> None:
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        f = transcripts / "a.md"
        f.write_text("x")

        store = FileHashStore.load(tmp_path / ".ingest", transcripts)
        store.update(f, "abc")
        store.forget(f)
        assert f.resolve() not in store.known_paths()

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        f = transcripts / "a.md"
        f.write_text("hello")

        ingest_dir = tmp_path / ".ingest"
        store = FileHashStore.load(ingest_dir, transcripts)
        store.update(f, store.hasher.hash_file(f))
        store.save()

        reloaded = FileHashStore.load(ingest_dir, transcripts)
        assert reloaded.status_of(f).is_unchanged is True

    def test_load_missing_file_returns_empty_store(self, tmp_path: Path) -> None:
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        store = FileHashStore.load(tmp_path / ".ingest", transcripts)
        assert store.known_paths() == set()

    def test_path_normalization_relative_when_under_transcripts_dir(
        self, tmp_path: Path
    ) -> None:
        """Stored keys are relative to transcripts_dir so the cache is portable."""
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        f = transcripts / "subdir" / "a.md"
        f.parent.mkdir()
        f.write_text("x")

        ingest_dir = tmp_path / ".ingest"
        store = FileHashStore.load(ingest_dir, transcripts)
        store.update(f, "abc")
        store.save()

        raw = json.loads((ingest_dir / "content_hashes.json").read_text())
        assert "subdir/a.md" in raw or str(Path("subdir/a.md")) in raw
```

- [ ] **Step 1.6: Run tests to verify they fail**

```bash
pytest tests/test_file_hash_store.py::TestFileHashStore -v
```
Expected: 8 fails — `FileHashStore` not defined.

- [ ] **Step 1.7: Implement `FileHashStore`**

Append to `mindforge/ingestion/file_hash_store.py`:

```python
class _Status(str, Enum):
    NEW = "new"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class FileStatus:
    path: Path
    current_hash: str
    previous_hash: str | None
    is_new: bool = False
    is_modified: bool = False
    is_unchanged: bool = False


class FileHashStore:
    """Persistent per-file SHA-256 manifest.

    Keys in the on-disk JSON are normalized to be relative to ``transcripts_dir``
    when possible, so the cache survives moving the project directory.
    """

    HASH_FILE_NAME = "content_hashes.json"

    def __init__(
        self,
        ingest_dir: Path,
        transcripts_dir: Path,
        hashes: dict[str, str] | None = None,
    ) -> None:
        self.ingest_dir = Path(ingest_dir)
        self.transcripts_dir = Path(transcripts_dir).resolve()
        self.hasher = ContentHasher()
        self._hashes: dict[str, str] = dict(hashes or {})

    @classmethod
    def load(cls, ingest_dir: Path, transcripts_dir: Path) -> FileHashStore:
        path = Path(ingest_dir) / cls.HASH_FILE_NAME
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {}
        return cls(ingest_dir=ingest_dir, transcripts_dir=transcripts_dir, hashes=data)

    def save(self) -> None:
        self.ingest_dir.mkdir(parents=True, exist_ok=True)
        path = self.ingest_dir / self.HASH_FILE_NAME
        path.write_text(json.dumps(self._hashes, indent=2, sort_keys=True), encoding="utf-8")

    def _key(self, file_path: Path) -> str:
        resolved = Path(file_path).resolve()
        try:
            return str(resolved.relative_to(self.transcripts_dir))
        except ValueError:
            return str(resolved)

    def status_of(self, file_path: Path) -> FileStatus:
        current = self.hasher.hash_file(file_path)
        previous = self._hashes.get(self._key(file_path))
        status = FileStatus(
            path=Path(file_path),
            current_hash=current,
            previous_hash=previous,
        )
        if previous is None:
            status.is_new = True
        elif previous == current:
            status.is_unchanged = True
        else:
            status.is_modified = True
        return status

    def update(self, file_path: Path, hash_value: str) -> None:
        self._hashes[self._key(file_path)] = hash_value

    def forget(self, file_path: Path) -> None:
        self._hashes.pop(self._key(file_path), None)

    def known_paths(self) -> set[Path]:
        """Return absolute paths corresponding to keys in the store."""
        out: set[Path] = set()
        for key in self._hashes:
            p = Path(key)
            if p.is_absolute():
                out.add(p)
            else:
                out.add((self.transcripts_dir / p).resolve())
        return out
```

- [ ] **Step 1.8: Run all tests in this module**

```bash
pytest tests/test_file_hash_store.py -v
```
Expected: 12 passed.

- [ ] **Step 1.9: Run lint + type check**

```bash
ruff check mindforge/ingestion/file_hash_store.py tests/test_file_hash_store.py
ruff format --check mindforge/ingestion/file_hash_store.py tests/test_file_hash_store.py
mypy mindforge/ingestion/file_hash_store.py
```
Expected: clean.

- [ ] **Step 1.10: Commit**

```bash
git add mindforge/ingestion/file_hash_store.py tests/test_file_hash_store.py
git commit -m "feat(ingestion): add FileHashStore for per-file SHA-256 manifest"
```

---

## Task 2: Add `deleted_at` to Concept schema

Pure schema change, no behavior change. Adds the field to the dataclass and JSON serialization so later tasks can use it.

**Files:**
- Modify: `mindforge/distillation/concept.py:62-130`
- Test: extend an existing test file or add a small case in `tests/test_distillation.py` if present; otherwise inline a tiny test in `tests/test_file_hash_store.py` is wrong — add to `tests/test_concept_schema.py`.

- [ ] **Step 2.1: Find the existing concept tests**

```bash
ls tests/ | grep -i concept
ls tests/ | grep -i distill
```

If `tests/test_concept.py` or similar exists, use it. Otherwise create `tests/test_concept_schema.py`.

- [ ] **Step 2.2: Write failing test for `deleted_at` round-trip**

In the chosen test file, add:

```python
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
```

- [ ] **Step 2.3: Run tests to verify failure**

```bash
pytest tests/test_concept_schema.py -v
```
Expected: 3 fails — `Concept.__init__` got unexpected keyword `deleted_at`, or `data["deleted_at"]` KeyError.

- [ ] **Step 2.4: Add the `deleted_at` field**

In `mindforge/distillation/concept.py`, edit the `Concept` dataclass (after line 79 `last_reviewed_at`):

```python
    last_reinforced_at: str | None = None
    last_reviewed_at: str | None = None
    deleted_at: str | None = None
```

In `to_dict()` (around line 89), add inside the returned dict:

```python
            "last_reviewed_at": self.last_reviewed_at,
            "deleted_at": self.deleted_at,
        }
```

In `from_dict()` (around line 109), add to the `cls(...)` kwargs:

```python
            last_reinforced_at=data.get("last_reinforced_at"),
            last_reviewed_at=data.get("last_reviewed_at"),
            deleted_at=data.get("deleted_at"),
        )
```

- [ ] **Step 2.5: Update `merge_with` to preserve `deleted_at`**

In `merge_with` (around line 132–171), add a rule: a concept being merged should *un-delete* if either side is active. If both are deleted, take the older `deleted_at`.

Add to the returned `Concept(...)`:

```python
        deleted_at = None
        if self.status == "deleted" and other.status == "deleted":
            # Both deleted; keep the earlier timestamp
            deleted_at = min(
                ts for ts in (self.deleted_at, other.deleted_at) if ts is not None
            ) if (self.deleted_at or other.deleted_at) else None
        # else: leave as None (un-delete on merge with active concept)

        return Concept(
            ...,
            deleted_at=deleted_at,
        )
```

Also derive `status` for the merged concept: `"active"` unless both inputs are deleted.

```python
        merged_status = (
            "deleted"
            if self.status == "deleted" and other.status == "deleted"
            else "active"
        )
```

Pass `status=merged_status` in the returned `Concept(...)`. Note: existing `merge_with` does not currently pass `status` — it defaults to `"active"`, which happens to match the desired "un-delete on merge" behavior. Adding the explicit status keeps it correct in the both-deleted case.

- [ ] **Step 2.6: Run tests**

```bash
pytest tests/test_concept_schema.py -v
pytest tests/ -v -k concept
```
Expected: new tests pass; existing tests remain green.

- [ ] **Step 2.7: Lint + type check**

```bash
ruff check mindforge/distillation/concept.py tests/test_concept_schema.py
ruff format --check mindforge/distillation/concept.py tests/test_concept_schema.py
mypy mindforge/distillation/concept.py
```

- [ ] **Step 2.8: Commit**

```bash
git add mindforge/distillation/concept.py tests/test_concept_schema.py
git commit -m "feat(concept): add deleted_at field for soft-delete tracking"
```

---

## Task 3: Read-site filtering by `status="active"`

Every read path filters out soft-deleted concepts by default. Renderer additionally removes any stale on-disk markdown file for slugs that just transitioned to deleted.

**Files:**
- Modify: `mindforge/query/engine.py` (filter_concepts + QueryEngine.search)
- Modify: `mindforge/graph/builder.py` (KnowledgeGraph.from_store)
- Modify: `mindforge/embeddings/index.py` (EmbeddingIndex.build)
- Modify: `mindforge/distillation/renderer.py` (write_all_concepts)
- Modify: `mindforge/mcp/server.py` (list_concepts, get_concept)
- Modify: `mindforge/cli.py` (`--include-deleted` on query/list/show)
- Test: `tests/test_status_filtering.py` (new)

- [ ] **Step 3.1: Write failing tests**

Create `tests/test_status_filtering.py`:

```python
"""Read sites must filter out soft-deleted concepts by default."""

from __future__ import annotations

from pathlib import Path

from mindforge.distillation.concept import Concept, ConceptStore
from mindforge.distillation.renderer import write_all_concepts
from mindforge.embeddings.index import EmbeddingIndex
from mindforge.graph.builder import KnowledgeGraph
from mindforge.query.engine import filter_concepts


def _make(name: str, status: str = "active") -> Concept:
    return Concept(
        name=name, definition=f"def {name}", explanation=f"exp {name}", status=status
    )


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
    # Smoke test: build does not include deleted slugs.
    # If embeddings deps are missing, build is a no-op; just ensure no crash.
    store = ConceptStore()
    store.add(_make("A"))
    store.add(_make("B", status="deleted"))
    index = EmbeddingIndex("all-MiniLM-L6-v2")
    index.build([c for c in store.all() if c.status == "active"])
    if index.available:
        assert "b" not in index._slugs  # noqa: SLF001
        assert "a" in index._slugs  # noqa: SLF001


def test_write_all_concepts_skips_deleted_and_removes_stale_file(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "concepts"
    out_dir.mkdir()
    # Pre-existing markdown file for a slug that is now deleted
    stale = out_dir / "b.md"
    stale.write_text("old content")

    concepts = [_make("A"), _make("B", status="deleted")]
    written = write_all_concepts(concepts, out_dir)

    written_names = {p.name for p in written}
    assert "a.md" in written_names
    assert "b.md" not in written_names
    assert not stale.exists(), "stale markdown for deleted slug should be removed"
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
pytest tests/test_status_filtering.py -v
```
Expected: 5 fails.

- [ ] **Step 3.3: Update `filter_concepts` to take `include_deleted`**

Edit `mindforge/query/engine.py:29-57`. Change the signature and add the filter:

```python
def filter_concepts(
    concepts: list[Concept],
    tag: str | None = None,
    min_confidence: float | None = None,
    since: str | None = None,
    include_deleted: bool = False,
) -> list[Concept]:
    """Filter concepts by tag, minimum confidence, last-reinforced-since, and status."""
    out = list(concepts)
    if not include_deleted:
        out = [c for c in out if c.status != "deleted"]
    if tag:
        out = [c for c in out if tag in c.tags]
    if min_confidence is not None:
        out = [c for c in out if c.confidence >= min_confidence]
    if since:
        # ... existing block unchanged ...
```

- [ ] **Step 3.4: Update `QueryEngine.search` to filter deleted concepts**

In `mindforge/query/engine.py`, find `QueryEngine.search`. Before the keyword/semantic/graph scoring, restrict the candidate set to active concepts (or pass through the underlying scorers' inputs). Easiest: filter at `_collect_candidates` or wherever the store is iterated.

Find the lines that iterate `self.store.all()` inside `QueryEngine`. Wrap them with:

```python
active_concepts = [c for c in self.store.all() if c.status != "deleted"]
```

…and use `active_concepts` everywhere the unfiltered list was used.

If `KeywordScorer` or `GraphWalker` access `store` directly (rather than receiving a list), they will continue to see all concepts — but the graph already excludes deleted nodes (Step 3.5 below), and `KeywordScorer` should be reviewed: if it iterates `store.all()`, restrict it the same way.

- [ ] **Step 3.5: Update `KnowledgeGraph.from_store` to skip deleted**

Edit `mindforge/graph/builder.py:71-79`:

```python
    @classmethod
    def from_store(cls, store: ConceptStore) -> KnowledgeGraph:
        """Build a graph from a ConceptStore. Deleted concepts are excluded."""
        graph = cls()
        active = [c for c in store.all() if c.status != "deleted"]
        for concept in active:
            graph.add_concept(concept)
        for concept in active:
            graph.add_relationships(concept)
        return graph
```

Note: `add_relationships` may add edges referencing deleted slugs (if an active concept has a relationship pointing to a now-deleted slug). Filter those edges:

```python
    def add_relationships(self, concept: Concept) -> None:
        for rel in concept.relationships:
            if rel.target not in self._nodes:
                continue  # target was filtered out (e.g., deleted)
            ...
```

(Read the current implementation before editing — apply the `if rel.target not in self._nodes: continue` guard at the top of the loop.)

- [ ] **Step 3.6: Update `EmbeddingIndex.build` to skip deleted**

Edit `mindforge/embeddings/index.py` `build()` (around line 111):

```python
    def build(self, concepts: list[Concept]) -> None:
        """Build the index from a list of concepts. Deleted concepts are excluded."""
        if not self._available:
            return

        active = [c for c in concepts if c.status != "deleted"]
        # ... rest of method uses `active` instead of `concepts` ...
```

- [ ] **Step 3.7: Update `write_all_concepts` to skip deleted and remove stale files**

Edit `mindforge/distillation/renderer.py:110-112`:

```python
def write_all_concepts(concepts: list[Concept], output_dir: Path) -> list[Path]:
    """Write all active concepts; remove stale markdown for deleted slugs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    active = [c for c in concepts if c.status != "deleted"]
    deleted_slugs = {c.slug for c in concepts if c.status == "deleted"}
    for slug in deleted_slugs:
        stale = output_dir / f"{slug}.md"
        if stale.exists():
            stale.unlink()
    return [write_concept(c, output_dir) for c in active]
```

- [ ] **Step 3.8: Update MCP read paths**

In `mindforge/mcp/server.py`, find `list_concepts` and `get_concept` handler bodies (around lines 854 and 884). Filter out `status=="deleted"` from the results returned to MCP clients. No new tool parameter — clients should never see soft-deleted concepts.

For `list_concepts` (around line 884), wherever it iterates concepts, add:

```python
concepts = [c for c in concepts if c.status != "deleted"]
```

For `get_concept` (around line 854), if the requested slug resolves to a deleted concept, return the same "not found" path as a missing slug.

Read the current handler bodies before editing to find the exact insertion points.

- [ ] **Step 3.9: Add `--include-deleted` to query/list/show CLI**

Edit `mindforge/cli.py`:

For the `query` subparser (around line 166), add:

```python
query.add_argument(
    "--include-deleted",
    action="store_true",
    help="Include soft-deleted concepts in results",
)
```

Same for `list` (around line 250) and `show` (around line 397).

In `cmd_query` (around line 523), pass `include_deleted=args.include_deleted` into the post-search `filter_concepts(...)` call (around line 555).

In `cmd_list` (around line 568), pass the same to `filter_concepts(...)` (around line 580).

In `cmd_show` (around line 803), if loading by slug, allow showing a deleted concept when `--include-deleted` is passed; otherwise treat it as not found.

- [ ] **Step 3.10: Run tests**

```bash
pytest tests/test_status_filtering.py -v
pytest tests/ -v
```
Expected: new tests pass; full suite stays green.

- [ ] **Step 3.11: Lint + type check**

```bash
ruff check mindforge/ tests/
ruff format --check mindforge/ tests/
mypy mindforge
```

- [ ] **Step 3.12: Commit**

```bash
git add mindforge/query/engine.py mindforge/graph/builder.py \
        mindforge/embeddings/index.py mindforge/distillation/renderer.py \
        mindforge/mcp/server.py mindforge/cli.py tests/test_status_filtering.py
git commit -m "feat(read): filter soft-deleted concepts from query/graph/embeddings/renderer/MCP"
```

---

## Task 4: Pipeline integration — auto-detect + trivial fast path

Wire `FileHashStore` into `MindForgePipeline.run()`. First sub-step: classify files and short-circuit when nothing changed. Drop logic comes in Task 5.

**Files:**
- Modify: `mindforge/pipeline.py` (PipelineResult fields, run() classification + fast path, save cache at end)
- Test: `tests/test_incremental_pipeline.py` (new)

- [ ] **Step 4.1: Write a failing test for first-run + cache creation**

Create `tests/test_incremental_pipeline.py`:

```python
"""Integration tests for incremental ingestion in MindForgePipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mindforge.config import MindForgeConfig
from mindforge.pipeline import MindForgePipeline


def _seed_transcripts(transcripts_dir: Path, contents: dict[str, str]) -> None:
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    for name, body in contents.items():
        (transcripts_dir / name).write_text(body, encoding="utf-8")


@pytest.fixture
def fixture_paths(tmp_path: Path) -> tuple[Path, Path]:
    transcripts_dir = tmp_path / "transcripts"
    output_dir = tmp_path / "output"
    return transcripts_dir, output_dir


def test_first_run_creates_hash_cache(fixture_paths: tuple[Path, Path]) -> None:
    transcripts_dir, output_dir = fixture_paths
    _seed_transcripts(
        transcripts_dir,
        {
            "a.md": "# Alpha\n\nAlpha is the first letter of the Greek alphabet.\n",
        },
    )

    config = MindForgeConfig(
        transcripts_dir=transcripts_dir, output_dir=output_dir, use_llm=False
    )
    pipeline = MindForgePipeline(config)
    result = pipeline.run()

    assert (output_dir / ".ingest" / "content_hashes.json").exists()
    assert result.skipped is False
    assert result.files_new == 1
```

- [ ] **Step 4.2: Run test to verify failure**

```bash
pytest tests/test_incremental_pipeline.py::test_first_run_creates_hash_cache -v
```
Expected: fail — `PipelineResult` has no `skipped` / `files_new` field, and no `.ingest/` is written.

- [ ] **Step 4.3: Extend `PipelineResult`**

Edit `mindforge/pipeline.py:80-108`:

```python
@dataclass
class PipelineResult:
    concepts_extracted: int
    concepts_after_dedup: int
    concept_files_written: int
    edges_in_graph: int
    embeddings_built: bool
    extraction_method: str = "heuristic"
    dry_run: bool = False
    new: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    # Incremental tracking
    skipped: bool = False
    files_new: int = 0
    files_modified: int = 0
    files_unchanged: int = 0
    files_deleted: int = 0
    concepts_soft_deleted: int = 0

    def summary(self) -> str:
        lines = [
            "MindForge Pipeline Complete",
            "=" * 40,
            f"  Extraction method:       {self.extraction_method}",
            f"  Files (new/mod/unch/del):"
            f"  {self.files_new}/{self.files_modified}/"
            f"{self.files_unchanged}/{self.files_deleted}",
            f"  Concepts extracted:      {self.concepts_extracted}",
            f"  After deduplication:     {self.concepts_after_dedup}",
            f"  Soft-deleted concepts:   {self.concepts_soft_deleted}",
            f"  Markdown files written:  {self.concept_files_written}",
            f"  Graph edges:             {self.edges_in_graph}",
            f"  Embeddings built:        "
            f"{'yes' if self.embeddings_built else 'no (optional deps not installed)'}",
        ]
        return "\n".join(lines)
```

- [ ] **Step 4.4: Add classification + fast path to `run()`**

Edit `mindforge/pipeline.py:121-273`. After `parse_all_transcripts(...)` (line 133) and before chunking, insert classification:

```python
        from mindforge.ingestion.file_hash_store import FileHashStore

        ingest_dir = self.config.output_dir / ".ingest"
        hash_store = FileHashStore.load(ingest_dir, self.config.transcripts_dir)

        on_disk_paths = {Path(t.source_path).resolve() for t in transcripts}
        known = hash_store.known_paths()
        deleted_paths = known - on_disk_paths

        new_files: list[Path] = []
        modified_files: list[Path] = []
        unchanged_files: list[Path] = []
        for transcript in transcripts:
            p = Path(transcript.source_file)
            status = hash_store.status_of(p)
            if status.is_new:
                new_files.append(p)
            elif status.is_modified:
                modified_files.append(p)
            else:
                unchanged_files.append(p)

        cache_existed = (ingest_dir / FileHashStore.HASH_FILE_NAME).exists()
        no_changes = not new_files and not modified_files and not deleted_paths

        if cache_existed and no_changes and not dry_run:
            print(
                f"  Files (new/mod/unch/del): "
                f"0/0/{len(unchanged_files)}/0"
            )
            print("  Nothing to do.")
            self._load_state()
            return PipelineResult(
                concepts_extracted=0,
                concepts_after_dedup=0,
                concept_files_written=0,
                edges_in_graph=0,
                embeddings_built=self.embedding_index is not None,
                extraction_method="incremental-skip",
                skipped=True,
                files_new=0,
                files_modified=0,
                files_unchanged=len(unchanged_files),
                files_deleted=0,
            )

        print(
            f"  Files (new/mod/unch/del): "
            f"{len(new_files)}/{len(modified_files)}/"
            f"{len(unchanged_files)}/{len(deleted_paths)}"
        )
```

(`transcript.source_file: str` is defined at `mindforge/ingestion/parser.py:24`.)

At the very end of the non-dry-run path (after `self.store.save(manifest_path)` near line 264), add:

```python
        # Update FileHashStore: write hashes for the files we processed.
        # In the trivial-or-full case this is "all files we just parsed."
        # The drop logic in Task 5 will refine which files are "processed."
        for transcript in transcripts:
            p = Path(transcript.source_file)
            hash_store.update(p, hash_store.hasher.hash_file(p))
        for p in deleted_paths:
            hash_store.forget(p)
        hash_store.save()
```

For now (Task 4), the pipeline still re-extracts all files — Task 5 will trim the work. This task only adds classification, fast-path, and cache writing.

- [ ] **Step 4.5: Run the first-run test**

```bash
pytest tests/test_incremental_pipeline.py::test_first_run_creates_hash_cache -v
```
Expected: pass.

- [ ] **Step 4.6: Add a test for the trivial fast path**

Append to `tests/test_incremental_pipeline.py`:

```python
def test_rerun_with_no_changes_triggers_fast_path(
    fixture_paths: tuple[Path, Path],
) -> None:
    transcripts_dir, output_dir = fixture_paths
    _seed_transcripts(
        transcripts_dir,
        {
            "a.md": "# Alpha\n\nAlpha is the first letter of the Greek alphabet.\n",
        },
    )

    config = MindForgeConfig(
        transcripts_dir=transcripts_dir, output_dir=output_dir, use_llm=False
    )
    MindForgePipeline(config).run()  # first full run

    result = MindForgePipeline(config).run()  # second run, no changes
    assert result.skipped is True
    assert result.files_unchanged == 1
    assert result.files_new == 0
    assert result.files_modified == 0
```

- [ ] **Step 4.7: Run the fast-path test**

```bash
pytest tests/test_incremental_pipeline.py -v
```
Expected: both tests pass.

- [ ] **Step 4.8: Lint + type check**

```bash
ruff check mindforge/pipeline.py tests/test_incremental_pipeline.py
ruff format --check mindforge/pipeline.py tests/test_incremental_pipeline.py
mypy mindforge/pipeline.py
```

- [ ] **Step 4.9: Commit**

```bash
git add mindforge/pipeline.py tests/test_incremental_pipeline.py
git commit -m "feat(pipeline): auto-detect incremental mode with trivial fast path"
```

---

## Task 5: Pipeline integration — drop + re-extract logic

Now make the pipeline only extract from new+modified files, drop modified+deleted files' contributions from the loaded store, soft-mark concepts whose `source_files` go empty, and reuse old extracted concepts via `ConceptStore.load`.

**Files:**
- Modify: `mindforge/pipeline.py:121-273` (run() body)
- Modify: `tests/test_incremental_pipeline.py` (add scenarios 3–5 from the spec)

- [ ] **Step 5.1: Write failing test — add a new transcript reuses old concepts**

Append to `tests/test_incremental_pipeline.py`:

```python
def test_adding_new_transcript_preserves_old_concepts(
    fixture_paths: tuple[Path, Path],
) -> None:
    transcripts_dir, output_dir = fixture_paths
    _seed_transcripts(
        transcripts_dir,
        {
            "a.md": "# Alpha\n\nAlpha is the first letter of the Greek alphabet.\n",
        },
    )

    config = MindForgeConfig(
        transcripts_dir=transcripts_dir, output_dir=output_dir, use_llm=False
    )
    MindForgePipeline(config).run()

    # Capture concept slugs from the first run.
    from mindforge.distillation.concept import ConceptStore

    first_store = ConceptStore.load(output_dir / "concepts.json")
    first_slugs = set(first_store.slugs())
    assert first_slugs, "first run should have produced some concepts"

    # Add a second transcript and re-run.
    (transcripts_dir / "b.md").write_text(
        "# Beta\n\nBeta is the second letter of the Greek alphabet.\n",
        encoding="utf-8",
    )
    result = MindForgePipeline(config).run()
    assert result.skipped is False
    assert result.files_new == 1
    assert result.files_unchanged == 1

    # Old concepts must still be present.
    second_store = ConceptStore.load(output_dir / "concepts.json")
    assert first_slugs.issubset(set(second_store.slugs())), (
        "concepts from unchanged files should be preserved"
    )
```

- [ ] **Step 5.2: Run to verify failure**

```bash
pytest tests/test_incremental_pipeline.py::test_adding_new_transcript_preserves_old_concepts -v
```
Expected: fail — currently the pipeline re-extracts everything from scratch with empty starting store, so old concepts may be re-derived but the test mostly checks behavior; this test may actually pass coincidentally if extraction is deterministic. If it passes, that's fine — we still need the next two tests.

- [ ] **Step 5.3: Write failing test — modifying a transcript drops its prior contributions**

Append:

```python
def test_modifying_transcript_soft_marks_removed_concepts(
    fixture_paths: tuple[Path, Path],
) -> None:
    transcripts_dir, output_dir = fixture_paths
    _seed_transcripts(
        transcripts_dir,
        {
            "a.md": (
                "# Alpha\n\nAlpha is the first letter of the Greek alphabet.\n\n"
                "# Beta\n\nBeta is the second letter of the Greek alphabet.\n"
            ),
        },
    )

    config = MindForgeConfig(
        transcripts_dir=transcripts_dir, output_dir=output_dir, use_llm=False
    )
    MindForgePipeline(config).run()

    from mindforge.distillation.concept import ConceptStore

    first_store = ConceptStore.load(output_dir / "concepts.json")
    assert "alpha" in first_store.concepts or "beta" in first_store.concepts

    # Rewrite the file to remove the Beta section entirely.
    (transcripts_dir / "a.md").write_text(
        "# Alpha\n\nAlpha is the first letter of the Greek alphabet.\n",
        encoding="utf-8",
    )
    result = MindForgePipeline(config).run()
    assert result.files_modified == 1

    second_store = ConceptStore.load(output_dir / "concepts.json")
    # Beta concept (if it existed) should now be soft-marked, not gone.
    if "beta" in second_store.concepts:
        assert second_store.concepts["beta"].status == "deleted"
        assert second_store.concepts["beta"].deleted_at is not None
```

- [ ] **Step 5.4: Write failing test — deleting a transcript soft-marks its orphaned concepts**

Append:

```python
def test_deleting_transcript_soft_marks_orphans(
    fixture_paths: tuple[Path, Path],
) -> None:
    transcripts_dir, output_dir = fixture_paths
    _seed_transcripts(
        transcripts_dir,
        {
            "a.md": "# Alpha\n\nAlpha is the first letter of the Greek alphabet.\n",
            "b.md": "# Beta\n\nBeta is the second letter of the Greek alphabet.\n",
        },
    )

    config = MindForgeConfig(
        transcripts_dir=transcripts_dir, output_dir=output_dir, use_llm=False
    )
    MindForgePipeline(config).run()

    (transcripts_dir / "b.md").unlink()
    result = MindForgePipeline(config).run()
    assert result.files_deleted == 1

    from mindforge.distillation.concept import ConceptStore

    store = ConceptStore.load(output_dir / "concepts.json")
    if "beta" in store.concepts:
        assert store.concepts["beta"].status == "deleted"
```

- [ ] **Step 5.5: Run the failing tests**

```bash
pytest tests/test_incremental_pipeline.py -v
```
Expected: at least the modify-and-delete tests fail — concepts re-appear or stay active because the pipeline still re-extracts everything from empty store.

- [ ] **Step 5.6: Implement drop + re-extract logic in `run()`**

Edit `mindforge/pipeline.py` `run()`. After classification (Task 4) and before stage 2 chunking, insert:

```python
        # Decide which files actually need re-extraction.
        is_incremental = cache_existed and not getattr(self, "_force_full", False)

        files_to_process: list[Path]
        if is_incremental:
            files_to_process = new_files + modified_files
            # Load existing store as the starting point.
            self.store = ConceptStore.load(self.config.output_dir / "concepts.json")

            # Drop contributions from modified and deleted files.
            drop_paths = {p.resolve() for p in modified_files} | {
                p.resolve() for p in deleted_paths
            }
            now_iso = datetime.now(timezone.utc).isoformat()
            soft_deleted_count = 0
            for slug, concept in list(self.store.concepts.items()):
                # Drop file paths from source_files
                concept.source_files = [
                    sf for sf in concept.source_files
                    if Path(sf).resolve() not in drop_paths
                ]
                # Drop matching SourceRefs
                concept.sources = [
                    s for s in concept.sources
                    if Path(s.transcript_path).resolve() not in drop_paths
                ]
                if not concept.source_files and concept.status != "deleted":
                    concept.status = "deleted"
                    concept.deleted_at = now_iso
                    soft_deleted_count += 1
        else:
            files_to_process = [Path(t.source_path) for t in transcripts]
            soft_deleted_count = 0

        # Restrict the transcripts list to the ones we need to chunk.
        process_set = {p.resolve() for p in files_to_process}
        transcripts_to_extract = [
            t for t in transcripts
            if Path(t.source_path).resolve() in process_set
        ]
```

Then change stage 2 (around line 142) to chunk only `transcripts_to_extract`:

```python
        all_chunks = []
        for transcript in transcripts_to_extract:
            chunks = chunk_turns(transcript.assistant_turns)
            all_chunks.extend(chunks)
        print(f"  Generated {len(all_chunks)} semantic chunks")
```

Stage 3 (dedup) and stage 4 (distill) operate on the freshly extracted concepts as before. The crucial existing call at line 176–177 already merges-by-slug into `self.store`:

```python
        for concept in concepts:
            self.store.add(concept)
```

So fresh extractions correctly merge with the loaded (and partially drop-pruned) store. No additional change needed there.

Add the soft-deleted count to the final `PipelineResult`:

```python
        return PipelineResult(
            concepts_extracted=len(raw_concepts),
            concepts_after_dedup=len(deduped),
            concept_files_written=len(written),
            edges_in_graph=stats["edges"],
            embeddings_built=embeddings_built,
            extraction_method=extraction_method,
            files_new=len(new_files),
            files_modified=len(modified_files),
            files_unchanged=len(unchanged_files),
            files_deleted=len(deleted_paths),
            concepts_soft_deleted=soft_deleted_count,
        )
```

Same for the dry-run branch — populate the file_* fields there too.

- [ ] **Step 5.7: Run all incremental tests**

```bash
pytest tests/test_incremental_pipeline.py -v
```
Expected: all 5 tests pass.

- [ ] **Step 5.8: Verify the full test suite still passes**

```bash
pytest
```
Expected: all green.

- [ ] **Step 5.9: Lint + type check**

```bash
ruff check mindforge/pipeline.py
ruff format --check mindforge/pipeline.py
mypy mindforge/pipeline.py
```

- [ ] **Step 5.10: Commit**

```bash
git add mindforge/pipeline.py tests/test_incremental_pipeline.py
git commit -m "feat(pipeline): incremental drop+re-extract with soft-delete on orphan"
```

---

## Task 6: `--full` flag on `mindforge ingest`

Add the explicit "redo everything from scratch" escape hatch.

**Files:**
- Modify: `mindforge/cli.py` (ingest subparser + cmd_ingest)
- Modify: `mindforge/pipeline.py` (consume `_force_full` flag set by CLI)
- Test: `tests/test_incremental_pipeline.py` (add `--full` scenario)

- [ ] **Step 6.1: Write failing test**

Append to `tests/test_incremental_pipeline.py`:

```python
def test_full_flag_forces_full_rebuild(fixture_paths: tuple[Path, Path]) -> None:
    transcripts_dir, output_dir = fixture_paths
    _seed_transcripts(
        transcripts_dir,
        {
            "a.md": "# Alpha\n\nAlpha is the first letter of the Greek alphabet.\n",
        },
    )

    config = MindForgeConfig(
        transcripts_dir=transcripts_dir, output_dir=output_dir, use_llm=False
    )
    MindForgePipeline(config).run()  # first run — populates cache

    # Force full rebuild via the API the CLI uses.
    pipeline = MindForgePipeline(config)
    pipeline._force_full = True
    result = pipeline.run()

    assert result.skipped is False
    # All files treated as "new" again because cache was cleared.
    assert result.files_new == 1
    assert result.files_unchanged == 0
    # Cache file is repopulated.
    assert (output_dir / ".ingest" / "content_hashes.json").exists()
```

- [ ] **Step 6.2: Run to verify failure**

```bash
pytest tests/test_incremental_pipeline.py::test_full_flag_forces_full_rebuild -v
```
Expected: fail — `_force_full` is not consumed; the cache is not cleared, so files appear unchanged.

- [ ] **Step 6.3: Honor `_force_full` in the pipeline**

Edit `mindforge/pipeline.py` `run()`. At the top of `run()`, before classification:

```python
        if getattr(self, "_force_full", False):
            cache_path = self.config.output_dir / ".ingest" / FileHashStore.HASH_FILE_NAME
            cache_path.unlink(missing_ok=True)
```

(Move the `from mindforge.ingestion.file_hash_store import FileHashStore` to module-top so this works.)

This deletes the cache file, after which the existing logic naturally treats `cache_existed = False` and runs as a full rebuild.

- [ ] **Step 6.4: Run the test**

```bash
pytest tests/test_incremental_pipeline.py::test_full_flag_forces_full_rebuild -v
```
Expected: pass.

- [ ] **Step 6.5: Add `--full` to the CLI subparser**

Edit `mindforge/cli.py` ingest subparser (around line 77–165). After the existing flags, add:

```python
    ingest.add_argument(
        "--full",
        action="store_true",
        help="Force a full rebuild, ignoring the incremental hash cache",
    )
```

- [ ] **Step 6.6: Wire the flag in `cmd_ingest`**

Edit `cmd_ingest` (around line 463–500). After `pipeline = MindForgePipeline(config)`:

```python
    pipeline = MindForgePipeline(config)
    if args.full:
        pipeline._force_full = True
    result = pipeline.run(dry_run=args.dry_run)
```

- [ ] **Step 6.7: Run lint + type check**

```bash
ruff check mindforge/pipeline.py mindforge/cli.py tests/test_incremental_pipeline.py
ruff format --check mindforge/pipeline.py mindforge/cli.py tests/test_incremental_pipeline.py
mypy mindforge/pipeline.py mindforge/cli.py
```

- [ ] **Step 6.8: Commit**

```bash
git add mindforge/pipeline.py mindforge/cli.py tests/test_incremental_pipeline.py
git commit -m "feat(cli): --full flag on ingest forces full rebuild"
```

---

## Task 7: `mindforge prune` subcommand

New top-level subcommand. Removes soft-deleted concepts and all their on-disk artifacts. Apply by default; `--dry-run` previews.

**Files:**
- Create: `mindforge/prune.py` (the prune logic, isolated from CLI plumbing)
- Modify: `mindforge/cli.py` (subparser + cmd_prune)
- Test: `tests/test_prune.py` (new)

- [ ] **Step 7.1: Write failing tests**

Create `tests/test_prune.py`:

```python
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

    # Pre-write a markdown + provenance file for each concept.
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
```

- [ ] **Step 7.2: Run to verify failure**

```bash
pytest tests/test_prune.py -v
```
Expected: collection error — `mindforge.prune` does not exist.

- [ ] **Step 7.3: Implement `mindforge/prune.py`**

Create `mindforge/prune.py`:

```python
"""Hard-delete soft-marked concepts and their on-disk artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mindforge.config import MindForgeConfig
from mindforge.distillation.concept import ConceptStore
from mindforge.graph.builder import KnowledgeGraph


@dataclass
class PruneSummary:
    removed: int = 0
    would_remove: int = 0
    slugs: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.slugs is None:
            self.slugs = []


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
        for sub in ("concepts", "provenance"):
            f = config.output_dir / sub / (
                f"{slug}.md" if sub == "concepts" else f"{slug}.json"
            )
            f.unlink(missing_ok=True)

    if to_remove:
        store.save(manifest)

        # Rebuild the graph so deleted slugs and their edges disappear.
        graph_path = config.graph_dir / "knowledge_graph.json"
        if graph_path.exists():
            graph = KnowledgeGraph.from_store(store)
            graph.save(graph_path)

        # Embedding index rebuild is left to the next ingest run; pruning
        # alone does not invalidate the index in a way reads care about
        # (filtered by slug membership).

    summary.removed = len(to_remove)
    return summary
```

- [ ] **Step 7.4: Wire the CLI subcommand**

Edit `mindforge/cli.py`. Add a new subparser near the other subcommand registrations (after the `config` subparser around line 444–458):

```python
    # --- prune ---
    prune_p = subparsers.add_parser(
        "prune",
        help="Hard-delete soft-marked concepts and their on-disk artifacts",
    )
    prune_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be removed without changing anything",
    )
    prune_p.add_argument(
        "--older-than-days",
        type=int,
        default=None,
        metavar="N",
        help="Only prune concepts soft-deleted at least N days ago",
    )
    prune_p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )
```

Add `cmd_prune`:

```python
def cmd_prune(args: argparse.Namespace) -> int:
    """Hard-delete soft-marked concepts."""
    from mindforge.prune import prune_orphans

    config = MindForgeConfig(output_dir=args.output)
    summary = prune_orphans(
        config, dry_run=args.dry_run, older_than_days=args.older_than_days
    )

    if args.dry_run:
        print(f"Would remove {summary.would_remove} soft-deleted concept(s):")
        for slug in summary.slugs:
            print(f"  - {slug}")
        print("Re-run without --dry-run to apply.")
    else:
        print(f"Removed {summary.removed} soft-deleted concept(s).")
        for slug in summary.slugs:
            print(f"  - {slug}")
    return 0
```

Wire it into the `main()` dispatch (find the `if args.command == "ingest":` chain near the bottom of `cli.py` — usually a dict or if/elif tree). Add:

```python
    elif args.command == "prune":
        return cmd_prune(args)
```

- [ ] **Step 7.5: Run prune tests**

```bash
pytest tests/test_prune.py -v
```
Expected: 4 passed.

- [ ] **Step 7.6: Smoke-test the CLI**

```bash
mindforge prune --help
```
Expected: help text printed; non-zero exit only for genuine errors.

- [ ] **Step 7.7: Lint + type check**

```bash
ruff check mindforge/prune.py mindforge/cli.py tests/test_prune.py
ruff format --check mindforge/prune.py mindforge/cli.py tests/test_prune.py
mypy mindforge/prune.py mindforge/cli.py
```

- [ ] **Step 7.8: Commit**

```bash
git add mindforge/prune.py mindforge/cli.py tests/test_prune.py
git commit -m "feat(cli): add mindforge prune subcommand for orphan cleanup"
```

---

## Task 8: Delete the old `incremental.py` module and its test

Now that no caller depends on `incremental.py` (the pipeline imports from `file_hash_store.py`), remove the dead module and its test file.

**Files:**
- Delete: `mindforge/ingestion/incremental.py`
- Delete: `tests/test_incremental_ingestion.py`

- [ ] **Step 8.1: Confirm no remaining imports**

```bash
grep -rn "from mindforge.ingestion.incremental\|import mindforge.ingestion.incremental" \
    mindforge/ tests/ --include="*.py"
```
Expected: no results. If any results appear, update those imports first to point at `file_hash_store` (or remove the import entirely if it referenced now-deleted symbols).

- [ ] **Step 8.2: Delete the files**

```bash
git rm mindforge/ingestion/incremental.py tests/test_incremental_ingestion.py
```

- [ ] **Step 8.3: Run the full suite**

```bash
pytest
```
Expected: all green. No collection errors from the deleted test file.

- [ ] **Step 8.4: Lint + type check**

```bash
ruff check .
ruff format --check .
mypy mindforge
```

- [ ] **Step 8.5: Commit**

```bash
git commit -m "chore(ingestion): remove obsolete incremental.py and its test

Functionality replaced by FileHashStore + pipeline drop+re-extract logic.
ContentHasher moved to mindforge/ingestion/file_hash_store.py."
```

---

## Task 9: Final validation — eval suite + manual dogfood smoke

Confirm incremental on/off produce equivalent KBs and the heuristic eval is stable.

- [ ] **Step 9.1: Run full test suite**

```bash
pytest
```
Expected: all green.

- [ ] **Step 9.2: Run heuristic eval**

```bash
mindforge eval --mode heuristic
```
Expected: completes without error; new report dropped under `eval/reports/`. Compare scores to the most recent prior report — they should be within noise (±1 on heuristic metrics).

- [ ] **Step 9.3: Manual smoke test against examples/transcripts**

```bash
rm -rf /tmp/mf-incremental-smoke
mindforge ingest --input examples/transcripts --output /tmp/mf-incremental-smoke
# expect: full first run; .ingest/content_hashes.json created.
mindforge ingest --input examples/transcripts --output /tmp/mf-incremental-smoke
# expect: "Nothing to do." in seconds.
mindforge ingest --input examples/transcripts --output /tmp/mf-incremental-smoke --full
# expect: full rebuild again.
```

- [ ] **Step 9.4: Commit any incidental fixes from validation**

If the eval or smoke test surfaces a bug, fix it in a small commit:

```bash
git commit -am "fix(<area>): <what was wrong, in one line>"
```

Otherwise this task has no commit.

---

## Out of scope (deliberately deferred)

Per the spec, the following are *not* implemented in this plan:

- Per-concept embedding cache keyed by content hash.
- Extractor-version stamp in `content_hashes.json` for automatic invalidation.
- Concurrent-ingest manifest locking.
- File-system watcher mode.
- Non-transcript inputs.

If dogfooding surfaces a need for any of these, file a follow-up spec.
