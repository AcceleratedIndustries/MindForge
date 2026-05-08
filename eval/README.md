# MindForge Evaluation Fixtures

Every transcript under `eval/fixtures/` has a sibling `<name>.gt.yaml` ground-truth file.

## Adding a fixture

1. Drop `<name>.md` into `eval/fixtures/`.
2. Create `<name>.gt.yaml` with expected concepts and relationships.
3. Run `mindforge eval` locally to see the score; include the GT file in your PR.

## Ground-truth schema

```yaml
expected_concepts:
  - name: "KV Cache"
    slug: "kv-cache"
    must_have_tags: ["transformers", "inference"]
    key_phrases: ["stores the Key and Value", "autoregressive"]

expected_relationships:
  - source: "kv-cache"
    target: "attention-mechanism"
    type: "related_to"
```

## Running

```
mindforge eval --fixtures eval/fixtures --mode mock
```

Report lands on stdout (markdown) and `eval/reports/<timestamp>.json` (machine-readable).

### Eval modes

- `mindforge eval --mode mock` (default) — deterministic smoke test. Verifies the pipeline runs end-to-end and the scorer produces a structurally valid report. Recall/precision/phrase-grounding numbers are not asserted in this mode; the mock client's content is not the gate.
- `mindforge eval --mode llm` — real-LLM quality gate. Runs against a configured LLM endpoint; recall/precision are meaningful. Run separately from per-PR CI.

## CI

`.github/workflows/eval.yml` runs mock-mode eval on any PR that touches `mindforge/ingestion/`, `mindforge/distillation/`, `mindforge/llm/`, `mindforge/linking/`, or `eval/fixtures/`. The real-LLM quality gate runs separately.
