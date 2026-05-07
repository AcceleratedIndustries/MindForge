# Incremental ingestion wire-up — design

**Date:** 2026-05-07
**Status:** approved, awaiting implementation plan
**Scope:** integrate the existing `mindforge/ingestion/incremental.py` primitives into the real ingestion pipeline so unchanged transcripts are skipped on re-runs. Pre-v0.4.0 dogfood enabler.

## Context

`mindforge ingest` currently re-parses every transcript, re-chunks every transcript, and re-runs LLM extraction over every chunk on every run. The cost is dominated by LLM extraction, which scales with chunk count, which scales with total transcript volume — not with what changed since the last run. This makes the tool uncomfortable to use in-situ inside Claude Code / Cowork sessions, which is the friction blocker preventing the daily-driver dogfood loop that should inform v0.4.0 (HTTP API + Web UI + concept versioning) design choices.

Library half-shipped: `mindforge/ingestion/incremental.py` (410 lines) implements `ContentHasher`, `IncrementalIngest`, `FileStatus`, `IncrementalResult` plus a parallel `concepts_meta.json` store and a toy regex extractor. None of it is imported by the rest of the package. Some of it is the right shape; most of it duplicates capabilities the real pipeline already has.

## Goals

- A second invocation of `mindforge ingest` against an unchanged transcripts dir returns in seconds, not minutes, with no flags.
- A re-run that touches one transcript only re-extracts that transcript's content, while preserving (and correctly merging) concepts from the rest.
- A re-run after a transcript is deleted soft-marks the orphaned concepts; a separate `mindforge prune` command performs the actual deletion when the user invokes it.
- Existing behavior (`--full`, eval suite, dry-run reporting) keeps working without surprise regressions.

## Non-goals

- Per-concept embedding cache. The bottleneck is LLM extraction; embedding rebuild is a separate optimization to revisit if dogfood signal demands it.
- Non-transcript inputs (markdown notes, source code). The current pipeline only ingests transcripts; this design inherits that scope.
- Concurrent-ingest safety. Single-user dogfood; no manifest locking.
- Extractor-version cache invalidation. Document `--full` as the recipe for "I changed how extraction works."

## Architecture

### New module

`mindforge/ingestion/file_hash_store.py` — slim file-hash manifest:

- `ContentHasher` (moved from `incremental.py`) — SHA-256 over file bytes.
- `FileHashStore` — load/save a `{relative_path → sha256}` map persisted at `output/.ingest/content_hashes.json`. Methods:
  - `load(ingest_dir: Path) -> FileHashStore`
  - `save() -> None`
  - `status_of(file_path: Path) -> FileStatus` returning one of `new | modified | unchanged`
  - `update(file_path: Path, hash: str) -> None`
  - `forget(file_path: Path) -> None`
  - `known_paths() -> set[Path]` (for deletion detection: paths in store but not on disk)

Path keys are normalized to relative-to-`transcripts_dir` via `Path(p).resolve().relative_to(transcripts_dir.resolve())`, falling back to absolute when the relative form fails (out-of-tree files).

### Refactor

`mindforge/ingestion/incremental.py` is **deleted** entirely. `ContentHasher` moves to the new `file_hash_store.py`; everything else has no surviving home:

- `IncrementalIngest` class (the parallel `concepts_meta.json`, `embeddings_cache.json`, `deleted_concepts.json` state — all duplicate what `ConceptStore` and the existing pipeline already do).
- `IncrementalIngest.run()` and `_extract_from_file()` — the toy regex extractor that competes with `ingestion.extractor` and `llm.extractor`.
- `FileStatus` / `IncrementalResult` dataclasses (replaced by simpler equivalents in `file_hash_store.py` and existing `PipelineResult` fields).
- `mark_deleted` / `gc_deleted` / `get_active_concepts` / `upsert_concept` — replaced by direct manipulation of `Concept.status` and a new `mindforge prune` command.

The corresponding test file `tests/test_incremental_ingestion.py` is **deleted**. New coverage lives in `tests/test_file_hash_store.py` and `tests/test_incremental_pipeline.py` (see Testing strategy below). No re-export shims; callers should import `ContentHasher` from the new module.

### Pipeline change

`mindforge/pipeline.py` `MindForgePipeline.run()` becomes:

```
1. parse_all_transcripts(transcripts_dir)  → list of transcripts on disk
2. Load FileHashStore from output/.ingest/content_hashes.json
   If the file does not exist, this is a full rebuild (auto-detect rule);
   skip steps 3–6 and run the legacy code path, then write the cache at the end.
3. Classify each on-disk transcript via store.status_of(...) →
        unchanged_files, modified_files, new_files
   Detect deleted_files = store.known_paths() − {paths on disk}
4. If no files changed AND no deletions:
        print "Nothing to do." ; return PipelineResult(skipped=True, ...)
5. Load existing ConceptStore from output/concepts.json
   (else start empty; should not happen if cache exists, but defensive)
6. For each modified_files ∪ deleted_files:
        For every concept in the store:
            remove this file from concept.source_files
            remove SourceRefs whose transcript_path == this file
        For every concept whose source_files ended up empty:
            concept.status = "deleted"
            concept.deleted_at = now_iso
7. Run chunking + extraction over (new_files ∪ modified_files) only.
   This reuses the existing chunk_turns / extract_concepts / extract_concepts_llm
   code paths unchanged — they just operate on a smaller transcript list.
8. Distill + store.add(...) — existing merge-by-slug behavior unions
   source_files / sources / insights / examples / tags into the loaded store.
9. Run dedup + linking + graph build + embedding rebuild on the FULL store
   (status=="active" subset for graph/embeddings; see "Read filtering" below).
10. Update FileHashStore: write hashes for processed files, forget deleted files,
    save to output/.ingest/content_hashes.json.
11. Save store, manifest snapshot, provenance — unchanged.
```

The key invariant: **modified = drop + re-extract**. Without step 6, a concept that was removed from a modified file would silently linger in the store with that file still listed as a source. Step 6 ensures the modified file's prior contributions are cleared before its new content is added back in step 7–8, and that orphaned concepts are correctly soft-marked.

### State matrix per file

| File state | Drop prior contributions | Re-extract |
|---|---|---|
| new | no (nothing to drop) | yes |
| modified | yes | yes |
| deleted | yes | no |
| unchanged | no | no |

### `PipelineResult` additions

- `skipped: bool = False` — true when the trivial fast path triggered (no work done).
- `files_unchanged: int = 0`
- `files_new: int = 0`
- `files_modified: int = 0`
- `files_deleted: int = 0`
- `concepts_soft_deleted: int = 0`

The existing `new / updated / unchanged / removed` concept-level counts are unchanged in semantics (they reflect the manifest diff against the previous `concepts.json`).

## Soft delete & orphan pruning

A "soft-marked" concept stays in `concepts.json` with `status="deleted"` and a `deleted_at` ISO timestamp. The `Concept` dataclass already has `status: str = "active"`; this is a field flip and a new `deleted_at: str | None = None` field, not a schema rewrite.

### Read filtering

By default, every read site filters to `status == "active"`:

- `mindforge query` → `query/engine.py` filtering before scoring.
- `mindforge list` / `mindforge show` → filter after `ConceptStore.load`.
- `KnowledgeGraph.from_store(...)` → exclude deleted nodes (and any edges touching them).
- `EmbeddingIndex.build(...)` → exclude deleted concepts.
- MCP tool surface → filter on each list/get path.
- `distillation/renderer.write_all_concepts` → skip writing markdown for deleted concepts AND remove any stale on-disk file for slugs that just transitioned to deleted (so `output/concepts/<slug>.md` doesn't outlive the soft-delete).

A new `--include-deleted` flag on `query`, `list`, `show` exposes them when wanted (debugging / forensics).

### `mindforge prune` subcommand

New top-level subcommand (not folded into `ingest`, since pruning is a deliberate destructive operation that should stand on its own):

```
mindforge prune                       # delete all soft-marked concepts
mindforge prune --dry-run             # preview only, no changes
mindforge prune --older-than-days 30  # only prune things soft-deleted >= N days ago
```

`--older-than-days` takes an integer N; concepts with `deleted_at` older than `now - N days` are pruned. Combine with no other flag = apply; combine with `--dry-run` = preview.

Default action is to apply (consistent with `mindforge ingest`'s pattern: active by default, `--dry-run` for preview). When applied, removes per slug:

- The entry from `concepts.json`
- The markdown file under `output/concepts/<slug>.md`
- The provenance JSON under `output/provenance/<slug>.json`
- The embedding row from the embeddings index
- Any graph edges incident to the slug (rebuild graph after pruning)

The file-hash manifest is untouched — the transcript was already gone before pruning.

## CLI surface

### `mindforge ingest`

No new required flags. Behavior:

- Auto-detect: if `output/.ingest/content_hashes.json` exists → incremental; else full rebuild.
- New flag `--full`: forces full rebuild. Deletes `.ingest/content_hashes.json` first, runs the legacy path end-to-end, repopulates the cache. Recipe for "I changed extractor logic / similarity threshold / LLM model and want everything regenerated."
- Existing `--dry-run` continues to work and now also reports the file-level breakdown before the concept-level diff.

### Output during incremental run

Replaces stage 1's "Found N file(s), M turns":

```
[1/6] Parsing transcripts (incremental)...
  Found 47 file(s) total
    unchanged: 44   new: 2   modified: 1   deleted: 0
  Processing 3 file(s), 19 turns
[2/6] Chunking and extracting concepts...
  ...
```

When the trivial fast path triggers:

```
[1/6] Parsing transcripts (incremental)...
  Found 47 file(s) total
    unchanged: 47   new: 0   modified: 0   deleted: 0
  Nothing to do.
```

Pipeline returns immediately with `PipelineResult(skipped=True, ...)` and all-zero counts.

### Config file

Nothing new. `incremental: true|false` is **not** added — auto-detect is the policy. Revisit if dogfooding shows people want a "always full rebuild" project setting.

## Testing strategy

### Unit tests

`tests/test_file_hash_store.py`:

- `ContentHasher` returns stable SHA-256 for byte / str / file inputs.
- `FileHashStore.status_of` returns `new` for unknown paths, `unchanged` for matching hashes, `modified` for differing hashes.
- `known_paths()` returns the set of paths in the store.
- Path normalization: paths are stored relative to `transcripts_dir`; absolute fallback when out of tree.
- Persistence round-trip: save, load fresh instance, identical state.

`tests/test_incremental_ingestion.py` (delete):

- File is removed entirely; coverage of `ContentHasher` moves to `tests/test_file_hash_store.py`. The other tests covered removed functionality (parallel `concepts_meta`, soft-delete primitives, `IncrementalIngest.run()`) and have no surviving target.

### Integration tests

`tests/test_incremental_pipeline.py` — real `MindForgePipeline` against tmp transcripts and tmp output:

1. **First run, no cache** → full ingest runs; `.ingest/content_hashes.json` is created with one entry per transcript.
2. **Re-run, no changes** → trivial fast path triggers, `PipelineResult.skipped == True`, no markdown files re-written, manifest history not appended.
3. **Add a new transcript** → only the new file is processed; concepts from the original transcripts are preserved unchanged in `concepts.json`.
4. **Modify a transcript to remove a previously-present concept** → that concept gets `status="deleted"`; new content from the modified file is extracted and merged.
5. **Delete a transcript** → concepts whose only source was that file get soft-marked; concepts shared with other files keep them in their `source_files` and remain `active`.
6. **`--full` flag** → cache ignored, full rebuild runs, cache rewritten.

`tests/test_prune.py`:

- Orphan detection: concepts with `status=="deleted"` are pruned; `status=="active"` concepts are not touched.
- Dry-run vs apply: `--dry-run` reports what would be removed and changes nothing on disk.
- `--older-than-days N`: only soft-deleted concepts whose `deleted_at` is older than the cutoff are pruned.
- All artifacts removed per pruned slug: manifest entry, markdown file, provenance JSON, embedding row, graph edges.
- Idempotency: running `prune` twice in a row is a no-op the second time.

### Eval suite

`mindforge eval --mode heuristic` continues to pass. Incremental on/off must produce equivalent KBs given the same input set (modulo `last_reinforced_at` timestamps).

## Migration & backwards compatibility

- Existing KBs (`output/concepts.json` exists, no `.ingest/` dir) → first run after upgrade is automatically a full rebuild via the auto-detect rule. Cache populates on that run; subsequent runs are fast. No user action required.
- `Concept` schema gains `deleted_at: str | None = None`. `ConceptStore.from_dict` already uses `data.get(...)` for optional fields, so old manifests without the new field load cleanly.
- No CLI breakage: all existing flags work; `--full` and `mindforge prune` are additive.

## Risks & mitigations

- **Stale cache after extractor logic changes.** If extraction logic or the LLM model changes but file hashes did not, the cache will skip files that should have been re-extracted. Mitigation: document `--full` as the recipe and surface this in the `--help` text. Follow-up if it bites in practice: stamp an extractor-version into `content_hashes.json` and auto-invalidate on mismatch.
- **Soft-deleted concepts cluttering reads.** Addressed by `status=="active"` filtering at every read site. Tests must cover each (query, list, show, graph, embeddings, MCP, renderer).
- **Modified-file drop wiping a shared concept.** The drop only removes the *modified file* from `source_files` and `sources`; soft-mark only triggers when the resulting list is empty. So a concept that also appears in an unchanged file survives.
- **Path normalization edge cases.** Symlinked transcript dirs and out-of-tree files. Use `Path.resolve().relative_to(transcripts_dir.resolve())` with absolute-path fallback. Document the normalization rule in the `FileHashStore` docstring.
- **Cache poisoning by manual edits.** If a user hand-edits `concepts.json` (e.g., deletes a concept) without touching the transcripts, the next incremental run will not bring it back (the source file is unchanged). Acceptable: `--full` is the documented escape hatch.

## Follow-ups (out of scope here, on the radar)

- Per-concept embedding cache keyed by concept content hash (skip re-embedding unchanged concepts).
- Extractor-version stamp in `content_hashes.json` for automatic cache invalidation on logic changes.
- Concurrent-ingest manifest locking (only matters if MindForge starts being used by multi-process tooling).
- File-system watcher mode (`mindforge watch`) so saved transcripts auto-ingest in the background.
