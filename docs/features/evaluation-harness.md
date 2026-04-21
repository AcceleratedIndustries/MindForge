# Feature: Evaluation Harness

**Phase:** 1.2
**Depends on:** —
**Unblocks:** trustworthy changes to extraction and distillation prompts/heuristics

---

## Motivation

Every change to the heuristic patterns, LLM prompts, or distillation logic silently changes output quality. Today there's no way to answer: "does this change make extraction better or worse?"

Without that answer, every tweak is vibes-based, and regressions ship unnoticed.

---

## User-facing behavior

```bash
# Run the eval suite against the fixture corpus
mindforge eval

# → MindForge Evaluation Report
# → ========================================
# →   Corpus:    eval/fixtures/ (12 transcripts, 2,341 turns)
# →   Mode:      heuristic
# →
# →   Concepts
# →     Expected:          87
# →     Extracted:         82
# →     Recall:            0.89
# →     Precision:         0.94
# →
# →   Relationships
# →     Expected edges:    54
# →     Found:             41
# →     Recall:            0.76
# →
# →   Compared to last run (2025-04-18):
# →     Recall:            0.89 (+0.03) ✓
# →     Precision:         0.94 (-0.01)

# Run with a specific mode
mindforge eval --mode llm --llm-provider ollama --llm-model llama3.2

# Compare two modes
mindforge eval --compare heuristic llm
```

A CI workflow runs `mindforge eval --mode heuristic` on every PR and fails if recall drops by more than a configurable threshold.

---

## Design

### Corpus

Create `eval/fixtures/` — a curated set of ~12 transcripts covering:

- Role-prefixed, heading-style, separator-based formats (parser coverage)
- Short (< 5 turns) and long (> 100 turns) transcripts
- Highly technical (LLM internals, distributed systems) and softer topics (product decisions, conversations)
- Transcripts that share concepts across files (dedup stress)
- Transcripts with explicit contradictions (conflict detection stress)

For each transcript, a sibling YAML file specifies ground truth:

```yaml
# eval/fixtures/2024-05-10_kv_cache.gt.yaml
expected_concepts:
  - name: "KV Cache"
    slug: "kv-cache"
    must_have_tags: ["transformers", "inference"]
    key_phrases: ["stores the Key and Value matrices", "autoregressive"]
  - name: "Multi-Query Attention"
    ...
expected_relationships:
  - source: "kv-cache"
    target: "attention-mechanism"
    type: "related_to"
```

### Scoring

- **Concept recall:** fraction of expected concepts matched by slug or fuzzy-name (threshold 0.85).
- **Concept precision:** fraction of extracted concepts that match an expected concept.
- **Phrase grounding:** for each matched concept, fraction of `key_phrases` appearing in the distilled definition or insights.
- **Relationship recall:** fraction of expected edges present in the graph.
- **Relationship type accuracy:** of matched edges, fraction with the correct type.

### Report format

- Markdown summary to stdout (above)
- JSON to `eval/reports/<timestamp>.json` for diffing
- Comparison to the most recent prior run if available

### CI integration

`.github/workflows/eval.yml` runs on PRs that touch `mindforge/ingestion/`, `mindforge/distillation/`, `mindforge/llm/`, or `mindforge/linking/`. Heuristic-only (LLM eval requires secrets; skip in CI, run locally or on cron).

---

## Files touched

### New
- `eval/fixtures/` — curated transcripts + ground-truth YAMLs
- `eval/README.md` — how to add to the corpus
- `mindforge/eval/__init__.py`
- `mindforge/eval/corpus.py` — load fixtures + ground truth
- `mindforge/eval/scorer.py` — scoring metrics
- `mindforge/eval/runner.py` — run pipeline, compute scores, render report
- `.github/workflows/eval.yml`

### Modified
- `mindforge/cli.py` — add `eval` subcommand
- `pyproject.toml` — add optional `[eval]` extra (already listed: `jsonschema`, `pytest`)

---

## Testing

- `tests/test_eval_scorer.py` — unit tests for each metric with tiny hand-built inputs
- `tests/test_eval_runner.py` — runs the whole pipeline on a 1-transcript fixture and asserts report shape
- Dog-food: CI eval gates itself

---

## Open questions

- **LLM non-determinism:** LLM extraction outputs vary run-to-run. Options: (a) seed where possible, (b) run N=3 and report mean±stdev, (c) only gate on heuristic mode in CI. **Proposed:** (c), with (b) on local eval runs.
- **Ground truth maintenance:** who updates GT YAMLs when the team consciously improves extraction? Adopt a convention: GT changes must land in the same PR as the extractor change, with reviewer sign-off.
- **Corpus licensing:** synthetic transcripts are safe. If we include real chat logs, sanitize and get explicit permission. **Proposed:** 100% synthetic to start.
