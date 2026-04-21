# Feature: Concept Provenance

**Phase:** 1.1
**Depends on:** incremental ingestion (shipped)
**Unblocks:** knowledge hygiene (1.3), conflict detection, evaluation harness verification, web UI citations

---

## Motivation

Today, when a concept like "KV Cache" appears in the output, there's no way to see *which transcript* and *which turn(s)* produced it. That breaks three things:

1. **Trust.** A user who disagrees with a distillation can't check the source.
2. **Conflict detection.** When two transcripts disagree, there's no way to surface both.
3. **Debuggability.** A bad extraction is untraceable.

Every concept must cite its sources.

---

## User-facing behavior

### In the Markdown file

```markdown
---
title: "KV Cache"
slug: "kv-cache"
tags: [transformers, inference]
confidence: 0.90
sources:
  - transcript: "2025-03-14_llm_internals.md"
    turns: [4, 7]
    extracted_at: "2025-03-14T11:22:00Z"
  - transcript: "2025-04-02_cache_optimization.md"
    turns: [2]
    extracted_at: "2025-04-02T09:10:00Z"
---
```

### In CLI

```bash
mindforge show kv-cache --sources
# → kv-cache
# → Sources:
# →   2025-03-14_llm_internals.md (turns 4, 7)
# →   2025-04-02_cache_optimization.md (turn 2)
```

### In MCP

`get_concept` response gains a `sources` field.

---

## Design

### Data model

Add to `mindforge/distillation/concept.py`:

```python
@dataclass
class SourceRef:
    transcript_path: str       # Relative to transcripts_dir
    transcript_hash: str        # Matches hash from incremental ingestion
    turn_indices: list[int]     # 0-indexed turn numbers
    extracted_at: str           # ISO 8601 UTC
    chunk_id: str | None = None # If available, for exact location

@dataclass
class Concept:
    # ...existing fields...
    sources: list[SourceRef] = field(default_factory=list)
```

### Where provenance is captured

Provenance is threaded through the pipeline at the point of extraction:

1. **Parser** (`ingestion/parser.py`) already knows transcript path + turn indices. No change needed; it already produces `Turn` objects.
2. **Chunker** (`ingestion/chunker.py`) must **preserve** which turns a chunk came from. Add `source_turns: list[int]` to `Chunk`.
3. **Extractor** (heuristic and LLM both) must **attach** the source chunk(s) to each `RawConcept`. Add `source_chunks: list[Chunk]` to `RawConcept`.
4. **Deduplicator** (`distillation/deduplicator.py`) **merges** source lists when it merges concepts.
5. **Distiller** produces final `Concept` with `sources` populated from the chunks.
6. **Store** persists sources to the manifest and to YAML frontmatter.

### Storage

Two locations:

- **Concept frontmatter** (human-readable): a `sources` YAML list.
- **`output/provenance/<slug>.json`** (detailed, machine-readable): same data plus any context snippets.

Rationale: keep the frontmatter small; put verbose stuff (e.g., extracted snippet text) in the JSON.

### Merging rules

When an incremental run re-extracts a concept from a transcript that has changed:

- If transcript hash is unchanged: skip re-extraction entirely (incremental ingestion already handles this).
- If changed: remove old `SourceRef` for that transcript, add new one.
- If a transcript is deleted: prune its `SourceRef` from all concepts. If a concept ends up with zero sources, mark it for review queue (handled in 1.3) rather than deleting.

---

## Files touched

### New
- `mindforge/distillation/source_ref.py` — `SourceRef` dataclass + serialization

### Modified
- `mindforge/distillation/concept.py` — add `sources` field
- `mindforge/ingestion/chunker.py` — thread turn indices through `Chunk`
- `mindforge/ingestion/extractor.py` — attach chunks to `RawConcept`
- `mindforge/llm/extractor.py` — same
- `mindforge/distillation/deduplicator.py` — merge source lists on dedup
- `mindforge/distillation/distiller.py` — populate `concept.sources` from chunks
- `mindforge/distillation/renderer.py` — emit `sources` in frontmatter
- `mindforge/pipeline.py` — write `output/provenance/<slug>.json` per concept
- `mindforge/mcp/server.py` — include `sources` in `get_concept` response
- `mindforge/cli.py` — new `mindforge show <slug> [--sources]` subcommand

---

## Testing

- `tests/test_provenance.py` (new):
  - A transcript with N turns extracts N concepts; each has correct turn indices.
  - Deduplicated concepts from two transcripts have both source refs.
  - Incremental run on a deleted transcript removes its source refs only.
  - YAML frontmatter roundtrips correctly.
- Update `tests/test_ingestion.py` to assert `Chunk.source_turns` is populated.
- Update `tests/test_mcp.py` to assert `get_concept` returns `sources`.

---

## Open questions

- **Snippet capture:** should `output/provenance/<slug>.json` include the literal extracted text? Pros: grounding for UI. Cons: storage bloat on large corpora. **Proposed default:** yes, but cap each snippet at 500 chars.
- **Migration:** older KBs have no `sources`. `ConceptStore.load()` should default to `[]` and emit a one-time warning suggesting re-ingestion. No destructive migration.
- **Storage abstraction:** this is a good moment to introduce a `Storage` protocol (see `ARCHITECTURE.md`) rather than hardcoding filesystem paths. Recommend doing it here.
