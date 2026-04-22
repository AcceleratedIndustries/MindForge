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
mindforge eval --fixtures eval/fixtures --mode heuristic
```

Report lands on stdout (markdown) and `eval/reports/<timestamp>.json` (machine-readable).

## CI

`.github/workflows/eval.yml` runs heuristic-mode eval on any PR that touches `mindforge/ingestion/`, `mindforge/distillation/`, `mindforge/llm/`, `mindforge/linking/`, or `eval/fixtures/`.
