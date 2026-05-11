# Per-concept chunk-level provenance — design

**Date:** 2026-05-11
**Status:** approved, awaiting implementation plan
**Scope:** narrow LLM-extractor provenance from batch-level to per-concept using deterministic substring matching against chunk text. No schema migration; on-disk format unchanged.

## Context

The LLM extractor (`mindforge/llm/extractor.py`) batches up to ~6KB of chunk content per call and assigns *every* chunk in the batch as the `source_chunks` for *every* concept the LLM extracts from that batch. Downstream, the distiller constructs one `SourceRef` per (concept, transcript) pair with `turn_indices` containing the full batch's turns. Result: `mindforge show <slug> --sources` returns a wide range of turns, most of which don't actually support the concept being inspected.

Concrete example from `~/.mindforge/unified/output/provenance/accessibility-misc.json` (the unified KB built 2026-05-09):

```
"turn_indices": [194, 196, 198, 200, 201, 202, 203, 204, 205, 206, 207, 208,
                  210, 211, 212, 213, 214, 216, 217, 219, 220, 221, 223, 225,
                  226, 228, 229, 231, 232, 233, 235, 237]
```

32 turns are listed as sources for "Accessibility & misc." The actual concept is grounded in 1-2 of them. Drilling down via `--sources` is currently useless: the wide range doesn't lead the reader to the supporting span.

This was flagged in the `mindforge-extraction-quality-issues` memory and listed as a follow-up in the heuristic-removal spec (`docs/superpowers/specs/2026-05-08-heuristic-removal-and-mock-llm-design.md`). The grounding filter that shipped with the heuristic removal (commit `39ba00b`) introduced the `_name_in_text` helper, which provides exactly the primitive this work needs: "does this concept name appear in this text?" The helper is reused here so substring-match semantics stay consistent across the two filters.

## Goals

- Each extracted concept's `source_chunks` contains only chunks whose content includes the concept name (matched by `_name_in_text`: case-insensitive, token-bounded, plural-strip fallback).
- Each `SourceRef.turn_indices` derives from the supporting chunks' turns, not the whole batch's turns.
- `mindforge show <slug> --sources` returns the narrow supporting range automatically — no CLI changes needed.
- No on-disk schema changes. `SourceRef.chunk_id` (singular) stays singular and now points at the first supporting chunk; `snippet` is the first supporting chunk's content (capped at the existing 500-char limit).

## Non-goals

- **Schema changes.** `SourceRef` shape, the on-disk JSON layout, and CLI output formats stay identical. The semantic change is only what data lives in each field.
- **Migration of existing KBs.** Concepts already on disk retain whatever provenance shape they had when extracted. The next `mindforge ingest --full` rebuilds them under the new shape; incremental ingests don't touch unchanged concepts.
- **Synonym / acronym resolution.** A concept the LLM canonicalizes to a form the source doesn't use ("TLS" in source → "Transport Layer Security" in LLM output) is still rejected by the grounding filter. Captured in Future followups.
- **LLM-emitted chunk hints.** The Approach A alternative (LLM tags each concept with the chunk IDs that support it) is deferred. If substring matching proves insufficient, that's the natural next move.
- **Sub-chunk char-span attribution.** Chunk-level granularity is sufficient for `mindforge show --sources`. Finer (offset-within-chunk) attribution is out of scope.

## Architecture

### Data flow

```
extract_concepts_llm(chunks, client):
  for batch in _batch_chunks(chunks):
    response = client.generate(prompt_built_from_batch)
    llm_concepts = _parse_llm_concepts(response, ...)
    for concept in llm_concepts:
      # Existing grounding filter: reject if name appears in zero chunks
      if not _name_in_text(concept.name, batch_text):
        stats.rejected_by_grounding += 1
        continue
      # NEW: narrow source_chunks to chunks whose content contains the name
      supporting = [c for c in batch if _name_in_text(concept.name, c.content)]
      concept.source_chunks = [c.id for c in supporting]
      # source_files derives from supporting chunks (unchanged behavior post-T9
      # batcher: all chunks in a batch share the same source_file anyway)
```

The change is localized to the per-batch loop inside `extract_concepts_llm`. The grounding filter (kept) guarantees `len(supporting) >= 1`, so there's no zero-support edge case.

### Distillation

Where the distiller currently builds a `SourceRef` from a `RawConcept`'s `source_chunks`, it looks up the actual `Chunk` objects via the existing `chunk_map: dict[str, Chunk]` and derives:

- `turn_indices` = sorted unique turn indices from those chunks
- `chunk_id` = `source_chunks[0]` (the first supporting chunk)
- `snippet` = the first supporting chunk's `content[:500]`
- `transcript_path`, `transcript_hash`, `extracted_at` — unchanged

Since `source_chunks` is now per-concept rather than per-batch, no logic change is needed at this layer — the same code reads the narrower list and produces a narrower SourceRef. The distiller doesn't need to know about the filtering that happened upstream.

### Reuse of `_name_in_text`

The grounding filter and the per-concept attribution call the same helper:

```python
# mindforge/llm/extractor.py
def _name_in_text(name: str, text: str) -> bool:
    # case-insensitive, alphanumeric word boundaries, plural-strip fallback
```

A concept passes the grounding filter iff `_name_in_text(name, batch_text)`. The same call (against each individual chunk's content) determines which chunks support the concept. Consistent semantics means a concept that grounds also has ≥1 supporting chunk, and the same plural/acronym/case rules apply at both gates.

## Data shape

No changes to `SourceRef`:

```python
@dataclass
class SourceRef:
    transcript_path: str
    transcript_hash: str
    turn_indices: list[int]
    extracted_at: str
    chunk_id: str | None = None
    snippet: str | None = None
```

The on-disk JSON in `output/provenance/<slug>.json` and `output/concepts.json` keeps the same field names and types. The values differ:

| Field | Before | After |
|---|---|---|
| `turn_indices` | every turn in the batch | only turns of supporting chunks |
| `chunk_id` | first chunk in the batch | first supporting chunk |
| `snippet` | first batch chunk's content[:500] | first supporting chunk's content[:500] |

Readers that count or iterate `turn_indices` see fewer entries on new ingests. No reader needs to change.

## Edge cases & error handling

1. **Many supporting chunks.** A concept discussed in 10 chunks gets 10 entries in `source_chunks` and up to 10 distinct turns in `turn_indices`. No cap; previous batch-level model was also unbounded.

2. **Same chunk supports multiple concepts.** Each concept records that chunk in its own `source_chunks`. Standard case; already handled by the existing data model.

3. **Concepts with overlapping names** (e.g., "App" inside "App Group"). `_name_in_text` is token-bounded — "App" matches "App" but not "App Group." (And vice versa: "App Group" doesn't match if only "App" is in source.) Both concepts get their own attribution; chunks containing only one form support only that one.

4. **LLM canonicalization drift on single-form sources.** Source contains only "TLS"; LLM emits concept named "Transport Layer Security". `_name_in_text` fails; the grounding filter rejects the concept. The concept is lost. *This is the same trade-off the grounding filter already makes — we'd rather drop a concept than misattribute it.* If a source contains *both* forms, the LLM typically emits a concept for each, both ground successfully, and the deduplicator may or may not merge them post-extraction. That outcome (possible redundancy, no loss) is acceptable; canonicalization drift on single-form sources is the genuinely-lost case and is captured in Future followups.

5. **Snippet truncation.** `SNIPPET_MAX_CHARS = 500` is unchanged. Snippet from a chunk shorter than 500 chars is the full chunk; longer chunks are truncated.

6. **Existing KBs with batch-level provenance.** A concept on disk before this change keeps its batch-level shape. `mindforge show --sources` against an old concept still works; it displays the (over-broad) turn list it was written with. No `if old_shape vs new_shape` branches in the code — the reader doesn't care about provenance breadth.

7. **`raw_content` size.** Unchanged. This work operates on `Chunk.content` (the original text), not `RawConcept.raw_content` (the LLM's definition/explanation).

No new error modes are introduced. The implementation is a pure refinement of existing assignment logic.

## Testing

### Unit tests (`tests/test_llm.py`, new class `TestPerConceptProvenance`)

- Multi-chunk batch, concept name in chunks 1 and 3 only → `source_chunks` contains those two chunks (not chunks 0, 2, 4...).
- Single-chunk match → `source_chunks` length 1.
- Plural-strip fallback: concept "Vector Embeddings" in chunks containing "vector embedding" → attribution targets the matching chunks.
- Same chunk supports multiple concepts → each concept independently lists that chunk.
- Token-boundary correctness: concept "RAG" with a chunk containing "storage" → that chunk is NOT a supporting chunk (regression test against substring false-positives).

### Snippet tests

- First supporting chunk < 500 chars → snippet equals the chunk content.
- First supporting chunk > 500 chars → snippet equals `content[:500]`.

### Integration test (extend `tests/test_pipeline.py` or co-locate)

- Two-transcript fixture: transcript A contains the concept text; transcript B does not. After ingest, the concept's `SourceRef` list contains transcript A only, and `turn_indices` is restricted to the supporting chunks within A. Verify against the on-disk `concepts.json` / `provenance/*.json`.

### Existing tests

- `tests/test_llm.py::TestExtractConceptsLLM::test_successful_extraction` and `test_deduplicates_across_batches` already include concept-name strings in their chunk fixtures (updated in the grounding-filter PR). They should continue to pass without changes; the attribution refinement is downstream of those fixtures.

## Migration & documentation

- **No migration.** Existing on-disk KBs keep their data. Next `mindforge ingest --full` rebuilds with the new shape.
- **No docs changes.** Schema, CLI surface, and on-disk format are identical. The user-visible difference ("fewer turns listed in `show --sources`") is self-evident and doesn't need documentation.
- **Memory update on landing.** Note in `mindforge-extraction-quality-issues` that batch-level provenance is fixed; followups are now the canonicalization-drift edge case and the broader synonym resolution problem.

## Future followups

1. **LLM canonicalization drift on single-form sources** (Edge #4). Cases where the LLM emits a concept name that doesn't textually appear in source (e.g., "Transport Layer Security" when source only says "TLS"). Three possible fixes, in increasing complexity:
   - Acronym / alias tables in `_name_in_text` — small, targeted, addresses the most common drift.
   - Embedding-similarity match — broader, slower, opens a tuning surface.
   - Approach A (LLM emits per-concept chunk IDs in the JSON response) — sidesteps substring matching entirely; richer prompt; more parsing fragility. Best long-term if substring proves insufficient.

2. **Post-extraction dedup that catches synonyms.** Two concepts in the KB that are the same idea under different names (e.g., "TLS" and "Transport Layer Security" both extracted from a multi-form source). This is a deduplication concern, not a provenance concern; separate spec.

3. **Sub-chunk char-span attribution.** Currently a SourceRef points at a chunk; a future enhancement could narrow to a character offset within the chunk. Useful for very long chunks, less so for the current 200-600 char chunks. Out of scope unless evidence warrants it.

4. **Cross-batch `source_chunks` merging.** When the same concept name is emitted by the LLM in multiple batches (e.g., a transcript large enough to span two batches that both mention "Vector Embeddings"), the existing `seen_names` dedup at `mindforge/llm/extractor.py:294-298` keeps only the first batch's `RawConcept` and drops the later ones. Pre-existing behavior, not introduced by this spec. But now that provenance is per-concept, this drop becomes more visible: a concept discussed across batches loses provenance from all batches except the first. Fix: merge `source_chunks` (preserving order, deduplicating) before dropping the duplicate, so per-concept provenance reflects every batch that mentioned the concept. Small follow-up; flagged during the post-implementation review on 2026-05-11.

## Open questions

None at design time — every decision in this spec was explicitly chosen during the brainstorming session on 2026-05-11.
