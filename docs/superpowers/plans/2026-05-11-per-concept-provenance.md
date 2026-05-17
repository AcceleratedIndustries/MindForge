# Per-concept chunk-level provenance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Narrow LLM-extractor provenance from batch-level to per-concept so `mindforge show <slug> --sources` returns the supporting turns, not the full batch's turns.

**Architecture:** In `extract_concepts_llm`, after the grounding filter accepts a concept, re-scan the batch's chunks individually and assign `RawConcept.source_chunks` to only those whose content contains the concept name (matched by the existing `_name_in_text` helper). The distiller's `_build_source_refs` already derives `turn_indices`, `chunk_id`, and `snippet` from `source_chunks`, so no distillation changes are needed.

**Tech Stack:** Python 3.10+, pytest, mypy strict, ruff. No new dependencies.

**Spec:** [docs/superpowers/specs/2026-05-11-per-concept-provenance-design.md](../specs/2026-05-11-per-concept-provenance-design.md)

---

## File map

**Modified:**
- `mindforge/llm/extractor.py` — `extract_concepts_llm` assigns per-concept `source_chunks` after grounding (Task 1)
- `tests/test_llm.py` — new `TestPerConceptProvenance` class (Task 1)
- `tests/test_pipeline.py` or new integration test file — end-to-end fixture verifying narrow `turn_indices` (Task 2)

**Created:** none.

**Deleted:** none.

---

## Task 1: Per-concept attribution in `extract_concepts_llm`

**Files:**
- Modify: `mindforge/llm/extractor.py` (the per-batch loop inside `extract_concepts_llm`, currently around lines 245-300 — verify line numbers with `grep -n "for i, batch in enumerate" mindforge/llm/extractor.py`)
- Modify: `tests/test_llm.py` (add new test class `TestPerConceptProvenance`)

TDD-style: write failing tests first, then narrow the assignment.

### Step 1: Write the failing tests

Append to `tests/test_llm.py` (insert directly above `# === Tests for LLM-aware distillation ===` line so the class lands with the other extractor tests):

```python
# === Tests for per-concept provenance ===


class TestPerConceptProvenance:
    """Verify each extracted concept's source_chunks contains only chunks
    whose content includes the concept name (not the full batch)."""

    def _make_chunk(self, content: str, index: int = 0) -> Chunk:
        return Chunk(
            content=content,
            source_file="test.md",
            turn_index=index,  # 1 chunk == 1 turn for simplicity
            chunk_index=0,
            chunk_type="prose",
        )

    def test_concept_attributed_to_only_matching_chunks(self):
        # Chunks 0 and 2 mention 'KV Cache'; chunk 1 does not.
        chunks = [
            self._make_chunk("We discussed KV Cache at length.", 0),
            self._make_chunk("Unrelated stuff about pasta sauce.", 1),
            self._make_chunk("Back to KV Cache for the conclusion.", 2),
        ]
        client = MagicMock(spec=LLMClient)
        client.generate.return_value = LLMResponse(
            content=json.dumps(
                {"concepts": [{"name": "KV Cache", "definition": "A cache."}]}
            ),
            success=True,
        )
        concepts, stats = extract_concepts_llm(chunks, client, max_chars_per_call=10_000)
        assert len(concepts) == 1
        assert stats.rejected_by_grounding == 0
        kv = concepts[0]
        assert set(kv.source_chunks) == {chunks[0].id, chunks[2].id}
        assert chunks[1].id not in kv.source_chunks

    def test_single_chunk_match(self):
        chunks = [
            self._make_chunk("Only chunk 0 mentions Async Queue.", 0),
            self._make_chunk("Chunk 1 talks about something else.", 1),
        ]
        client = MagicMock(spec=LLMClient)
        client.generate.return_value = LLMResponse(
            content=json.dumps(
                {"concepts": [{"name": "Async Queue", "definition": "An async queue."}]}
            ),
            success=True,
        )
        concepts, _ = extract_concepts_llm(chunks, client, max_chars_per_call=10_000)
        assert len(concepts) == 1
        assert concepts[0].source_chunks == [chunks[0].id]

    def test_plural_strip_fallback_in_attribution(self):
        # Concept name is plural; source uses singular. The grounding filter's
        # plural-strip should also drive the per-chunk attribution.
        chunks = [
            self._make_chunk("We use vector embedding for retrieval.", 0),
            self._make_chunk("Unrelated paragraph.", 1),
        ]
        client = MagicMock(spec=LLMClient)
        client.generate.return_value = LLMResponse(
            content=json.dumps(
                {"concepts": [{"name": "Vector Embeddings", "definition": "Dense reps."}]}
            ),
            success=True,
        )
        concepts, _ = extract_concepts_llm(chunks, client, max_chars_per_call=10_000)
        assert len(concepts) == 1
        assert concepts[0].source_chunks == [chunks[0].id]

    def test_same_chunk_supports_multiple_concepts(self):
        chunks = [
            self._make_chunk("Vector Embeddings power Semantic Search here.", 0),
            self._make_chunk("Just filler text.", 1),
        ]
        client = MagicMock(spec=LLMClient)
        client.generate.return_value = LLMResponse(
            content=VALID_LLM_RESPONSE,  # emits both Vector Embeddings and Semantic Search
            success=True,
        )
        concepts, _ = extract_concepts_llm(chunks, client, max_chars_per_call=10_000)
        assert len(concepts) == 2
        # Both concepts cite chunk 0; neither cites chunk 1.
        for c in concepts:
            assert c.source_chunks == [chunks[0].id]

    def test_token_boundary_prevents_substring_attribution(self):
        # 'RAG' must not match inside 'storage' / 'paragraph'. So if the LLM
        # emits 'RAG' as a concept and the only chunks contain 'storage', the
        # concept should be REJECTED by grounding (zero supporting chunks),
        # not silently attributed to those chunks.
        chunks = [
            self._make_chunk("Long-term storage is a paragraph apart.", 0),
            self._make_chunk("More storage discussion here.", 1),
        ]
        client = MagicMock(spec=LLMClient)
        client.generate.return_value = LLMResponse(
            content=json.dumps(
                {"concepts": [{"name": "RAG", "definition": "Retrieval-Augmented Generation."}]}
            ),
            success=True,
        )
        concepts, stats = extract_concepts_llm(chunks, client, max_chars_per_call=10_000)
        assert len(concepts) == 0
        assert stats.rejected_by_grounding == 1

    def test_concept_in_all_chunks_lists_all_chunks(self):
        # Sanity: when every chunk supports a concept, source_chunks contains
        # all of them. This is the case where batch-level and per-chunk
        # attribution produce the same result.
        chunks = [
            self._make_chunk("KV Cache discussion here.", 0),
            self._make_chunk("More KV Cache analysis.", 1),
            self._make_chunk("Final KV Cache notes.", 2),
        ]
        client = MagicMock(spec=LLMClient)
        client.generate.return_value = LLMResponse(
            content=json.dumps(
                {"concepts": [{"name": "KV Cache", "definition": "A cache."}]}
            ),
            success=True,
        )
        concepts, _ = extract_concepts_llm(chunks, client, max_chars_per_call=10_000)
        assert len(concepts) == 1
        assert set(concepts[0].source_chunks) == {chunks[0].id, chunks[1].id, chunks[2].id}
```

### Step 2: Run the tests to verify they fail

Run: `.venv/bin/pytest tests/test_llm.py::TestPerConceptProvenance -v`
Expected: most tests fail because `source_chunks` currently contains *all* batch chunks, not only matching ones.

Specifically:
- `test_concept_attributed_to_only_matching_chunks` fails — current behavior gives all 3 chunks; test expects 2.
- `test_single_chunk_match` fails — current gives 2 chunks; test expects 1.
- `test_plural_strip_fallback_in_attribution` fails — current gives 2 chunks; test expects 1.
- `test_same_chunk_supports_multiple_concepts` fails — current gives 2 chunks per concept; test expects 1 each.
- `test_token_boundary_prevents_substring_attribution` may already pass if the grounding filter's existing behavior matches (zero matches → rejection). Verify and adjust phrasing if it does.
- `test_concept_in_all_chunks_lists_all_chunks` may pass even pre-change (since batch == supporting set in this case).

### Step 3: Update `extract_concepts_llm` to narrow `source_chunks` per concept

Open `mindforge/llm/extractor.py` and locate the per-batch loop (around lines 245-300). The current grounding-filter block looks like this:

```python
        concepts = _parse_llm_concepts(data, source_chunks, source_files)

        # Grounding filter: reject concepts whose name doesn't appear in
        # the source text the LLM saw. Catches stock-AI hallucinations
        # (KV Cache, Vector Embeddings, RAG) the model emits unprompted
        # in projects unrelated to those topics.
        grounded = []
        for concept in concepts:
            if _name_in_text(concept.name, batch_text):
                grounded.append(concept)
            else:
                stats.rejected_by_grounding += 1
                logger.info(
                    "grounding filter rejected '%s' (not in source text)", concept.name
                )
```

Replace with:

```python
        concepts = _parse_llm_concepts(data, source_chunks, source_files)

        # Grounding filter: reject concepts whose name doesn't appear in
        # the source text the LLM saw. Catches stock-AI hallucinations
        # (KV Cache, Vector Embeddings, RAG) the model emits unprompted
        # in projects unrelated to those topics. For surviving concepts,
        # narrow source_chunks to only the chunks whose content actually
        # contains the name — produces accurate per-concept provenance
        # so `mindforge show <slug> --sources` returns the supporting
        # span instead of the whole batch.
        grounded = []
        for concept in concepts:
            if not _name_in_text(concept.name, batch_text):
                stats.rejected_by_grounding += 1
                logger.info(
                    "grounding filter rejected '%s' (not in source text)", concept.name
                )
                continue
            supporting_chunks = [c for c in batch if _name_in_text(concept.name, c.content)]
            concept.source_chunks = [c.id for c in supporting_chunks]
            grounded.append(concept)
```

Key changes vs. current code:
- Replaced `if … : grounded.append() else: …` with `if not …: continue; …` so the per-chunk filter runs only on accepted concepts.
- The `supporting_chunks` list collects only chunks whose `content` matches `_name_in_text(name, ...)` — same predicate the grounding filter uses, applied per chunk instead of against the full `batch_text`.
- `concept.source_chunks` is reassigned in place. `_parse_llm_concepts` initially set it to the batch-wide list (`source_chunks`); this overwrites that with the narrow list.
- `source_files` on the concept is unchanged — post-T9 batcher every chunk in a batch has the same `source_file`, so the value remains correct (all supporting chunks come from the same file).

### Step 4: Run the tests to verify they pass

Run: `.venv/bin/pytest tests/test_llm.py::TestPerConceptProvenance -v`
Expected: all 6 tests pass.

### Step 5: Run the full suite

Run: `.venv/bin/pytest -q`
Expected: all tests pass. The existing `TestExtractConceptsLLM` tests should keep working — their fixtures already contain the concept names in chunk content (updated in the grounding-filter PR).

If a previously-passing test fails, it likely depended on `source_chunks` containing the full batch. Check the assertion and either:
- Update the test to reflect per-chunk attribution (if the test's intent was to check provenance, the narrower value is more correct).
- Or update the fixture so the concept name appears in all batch chunks (if the test was checking something else and incidentally relied on the wider list).

### Step 6: Run mypy and ruff

Run: `.venv/bin/mypy mindforge && .venv/bin/ruff check . && .venv/bin/ruff format --check .`
Expected: clean. If ruff format suggests changes, run `.venv/bin/ruff format .` and re-stage.

### Step 7: Commit

```bash
git add mindforge/llm/extractor.py tests/test_llm.py
git commit -m "feat(llm): narrow source_chunks to supporting chunks per concept

After the grounding filter accepts a concept, re-scan the batch's
chunks individually and assign source_chunks to only those whose
content contains the concept name. The distiller's _build_source_refs
already derives turn_indices/chunk_id/snippet from source_chunks, so
this single change narrows on-disk provenance from batch-level to
per-concept without any schema migration.

\`mindforge show <slug> --sources\` will now display the supporting
turn range (typically 1-3 turns) instead of the full batch (often
30+ turns). Closes the batch-level-provenance overreach flagged in
the extraction-quality-issues memory.

Reuses _name_in_text (shipped with the grounding filter, commit
39ba00b) for consistent token-bounded match semantics across both
the grounding gate and per-chunk attribution."
```

---

## Task 2: Integration test for end-to-end provenance accuracy

**Files:**
- Modify: `tests/test_pipeline.py` (add `TestProvenanceAccuracy` class) OR create a focused file `tests/test_provenance_accuracy.py` if `test_pipeline.py` is already large

### Step 1: Inspect `tests/test_pipeline.py` size and decide placement

Run: `wc -l tests/test_pipeline.py`
- If under ~400 lines, append to it.
- If 400+ lines, create `tests/test_provenance_accuracy.py` instead.

### Step 2: Write the integration test

Add the following test class (in whichever file you chose). The test runs the real pipeline end-to-end against a two-transcript fixture using the mock LLM provider, then inspects `concepts.json` on disk to verify narrow `turn_indices`.

```python
"""End-to-end test that per-concept provenance narrows turn_indices to
supporting chunks, not the full batch."""

from __future__ import annotations

import json
from pathlib import Path

from mindforge.config import MindForgeConfig
from mindforge.pipeline import MindForgePipeline


class TestProvenanceAccuracy:
    def test_turn_indices_narrowed_to_supporting_turns(self, tmp_path: Path) -> None:
        # Build a transcript where 'KV Cache' is mentioned in exactly one
        # assistant turn, with many other turns of unrelated content. After
        # ingest, the concept's SourceRef should list only the supporting
        # turn(s), not every turn the LLM saw in the same batch.

        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()

        # Mock LLM produces deterministic title-case extraction. The transcript
        # is shaped so 'KV Cache' only appears in turn 4 (the third assistant
        # turn). All other turns mention only other things (e.g., 'Cache' alone
        # without 'KV', or unrelated content). _name_in_text("KV Cache", "Cache
        # alone") is False (token-bounded), so attribution should target turn 4
        # only.
        transcript = transcripts / "session.md"
        transcript.write_text(
            "Human: hi\n\n"
            "Assistant: We talked about Cache eviction.\n\n"
            "Human: ok\n\n"
            "Assistant: Then we discussed Cache hit rates.\n\n"
            "Human: continue\n\n"
            "Assistant: And finally, KV Cache specifically came up.\n\n"
            "Human: cool\n\n"
            "Assistant: Wrapping up with Cache invalidation.\n",
            encoding="utf-8",
        )

        out = tmp_path / "out"
        cfg = MindForgeConfig(
            transcripts_dir=transcripts,
            output_dir=out,
            llm_provider="mock",
        )
        pipe = MindForgePipeline(cfg)
        result = pipe.run()
        assert result.concept_files_written >= 1

        # Load concepts.json and find the KV Cache concept.
        store = json.loads((out / "concepts.json").read_text(encoding="utf-8"))
        kv = store.get("kv-cache")
        assert kv is not None, "expected 'kv-cache' concept (mock client always emits title-case)"

        sources = kv.get("sources", [])
        assert len(sources) >= 1
        # Crucial assertion: turn_indices contains the supporting turn(s) only.
        # The transcript has 4 assistant turns (indices depend on the parser
        # numbering; check that turn_indices is NOT the full set of assistant
        # turns).
        for src in sources:
            assert isinstance(src["turn_indices"], list)
            assert len(src["turn_indices"]) >= 1
            # The full batch would have included all 4 assistant turns
            # (indices roughly [1, 3, 5, 7] depending on parser conventions).
            # The narrow result should have FEWER than 4 entries.
            assert len(src["turn_indices"]) < 4, (
                f"turn_indices too broad: {src['turn_indices']} — should "
                "contain only the supporting turn(s), not every batch turn"
            )

    def test_unsupported_chunks_not_listed(self, tmp_path: Path) -> None:
        # Two-transcript fixture: transcript A mentions 'KV Cache', transcript
        # B does not. The KV Cache concept's source_files / sources should
        # contain transcript A only.

        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()

        (transcripts / "a.md").write_text(
            "Human: q\n\nAssistant: We use KV Cache for inference.\n",
            encoding="utf-8",
        )
        (transcripts / "b.md").write_text(
            "Human: q\n\nAssistant: Today's lunch was pasta.\n",
            encoding="utf-8",
        )

        out = tmp_path / "out"
        cfg = MindForgeConfig(
            transcripts_dir=transcripts,
            output_dir=out,
            llm_provider="mock",
        )
        pipe = MindForgePipeline(cfg)
        pipe.run()

        store = json.loads((out / "concepts.json").read_text(encoding="utf-8"))
        kv = store.get("kv-cache")
        assert kv is not None
        # The concept's sources should reference only a.md (the supporting
        # transcript), not b.md.
        source_paths = {src["transcript_path"] for src in kv["sources"]}
        assert any(str(transcripts / "a.md") in p for p in source_paths)
        assert not any(str(transcripts / "b.md") in p for p in source_paths)
```

### Step 3: Run the integration tests

Run: `.venv/bin/pytest tests/test_pipeline.py::TestProvenanceAccuracy -v` (or `tests/test_provenance_accuracy.py` if you created a new file).
Expected: both tests pass.

If `kv-cache` isn't extracted by the mock client on this fixture, inspect what the mock produces and adjust either the fixture or the assertions. (The mock's `_mock_concepts_from_text` extracts title-case phrases; "KV Cache" in the transcript should yield a concept named "KV Cache".)

### Step 4: Run the full suite

Run: `.venv/bin/pytest -q`
Expected: all tests pass.

### Step 5: Commit

```bash
git add tests/test_pipeline.py  # or tests/test_provenance_accuracy.py
git commit -m "test(provenance): end-to-end check that turn_indices narrows

Two integration tests verifying per-concept provenance through the
full pipeline (parse -> chunk -> extract -> distill -> store):

- A multi-turn transcript where one concept appears in only one
  assistant turn produces a SourceRef whose turn_indices contains
  only the supporting turn(s), not every batch turn.
- A two-transcript fixture where one transcript mentions a concept
  and the other doesn't produces sources referencing only the
  supporting transcript.

Uses the mock LLM provider so the test runs in CI without an
LLM endpoint."
```

---

## Task 3: Final verification and memory update

**Files:**
- Modify: `/Users/will/.claude/projects/-Users-will-src-MindForgeForHermes/memory/mindforge-extraction-quality-issues.md` (mark batch-level provenance fixed; preserve other findings)

### Step 1: Run all checks

Run:
```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy mindforge
```
Expected: all clean.

### Step 2: Smoke-test against a real KB

Re-ingest one small dogfood KB to verify the change works on real data:

```bash
rm -rf ~/.mindforge/AccelSTT/output
.venv/bin/mindforge ingest --input ~/.mindforge/AccelSTT/transcripts --output ~/.mindforge/AccelSTT/output --full
```

Then inspect one concept's provenance file:

```bash
ls ~/.mindforge/AccelSTT/output/provenance/*.json | head -1 | xargs cat | python3 -m json.tool
```

Expected: the `turn_indices` list is short (1-3 entries typically), not the full batch's turns. If AccelSTT only has a few concepts, repeat with `souschef` or `doc-ingestor` for a sample with more chunks per batch.

(If the smoke test reveals problems — e.g., a concept whose provenance file is empty or whose turn_indices is unexpectedly broad — investigate before proceeding.)

### Step 3: Update the extraction-quality-issues memory

Open `/Users/will/.claude/projects/-Users-will-src-MindForgeForHermes/memory/mindforge-extraction-quality-issues.md`.

Find the section that describes Issue 2 (batch-level provenance over-claims chunks). Update it to note the fix:

```markdown
**Issue 2: batch-level provenance over-claims chunks.** `mindforge/llm/extractor.py:213-218` packs multiple chunks per LLM call (up to 6KB) and assigns ALL of the batch's `source_chunks` as provenance for every concept that batch produced. Result: a single hallucinated concept can claim 20+ turns as sources. Drill-down is impossible — `mindforge show <slug> --sources` returns the entire batch range, not the specific chunk that supports the concept.

**FIXED 2026-05-11** (commit TBD on the per-concept-provenance branch). `extract_concepts_llm` now narrows `source_chunks` per concept using the same `_name_in_text` predicate as the grounding filter. SourceRef.turn_indices reflects only the supporting chunks' turns. Schema unchanged. Existing KBs keep batch-level data until rebuilt with `--full`.
```

Leave the rest of the memory intact (Issue 1 / hallucinations and the "how to apply" guidance are still relevant — hallucinations are addressed by the grounding filter, but the broader extraction-quality framing remains valuable).

Replace `TBD` with the actual commit SHA of Task 1's commit (e.g., look up via `git log --oneline -5`).

### Step 4: Final commit (if anything was tweaked)

If the memory update or smoke-test exposed a small fix, stage and commit:

```bash
git add -A
git commit -m "chore: per-concept provenance verification cleanups"
```

If nothing needed fixing beyond the memory update, commit that alone:

```bash
git add /Users/will/.claude/projects/-Users-will-src-MindForgeForHermes/memory/mindforge-extraction-quality-issues.md
git commit -m "docs(memory): mark batch-level provenance issue fixed"
```

(Memory files live outside the repo, but they're tracked separately. If your environment doesn't track the memory dir as a git repo, just save the file and skip the commit step for memory.)

### Step 5: Summarize for the user

Print a brief verification report including:
- Final test count (expect 355 + 6 new = ~361)
- Mypy/ruff status
- Sample provenance file showing narrow turn_indices
- Commit list since main (expect 2-3 commits)

The user will decide whether to push & PR via `superpowers:finishing-a-development-branch`.
