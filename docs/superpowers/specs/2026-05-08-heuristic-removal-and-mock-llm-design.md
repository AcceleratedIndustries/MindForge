# Heuristic removal and mock-LLM client — design

**Date:** 2026-05-08
**Status:** approved, awaiting implementation plan
**Scope:** remove the heuristic concept extractor from `mindforge ingest`, drop the `--llm` flag, hard-fail when the LLM endpoint is unreachable, and introduce a `MockLLMClient` to replace the de-facto "use heuristic for fast pipeline tests" pattern. Pre-grading-test cleanup so the unified KB can be re-ingested on a single-extractor pipeline.

## Context

The current pipeline runs the heuristic extractor (`mindforge/ingestion/extractor.py`) in parallel with the LLM extractor and merges all unique heuristic concepts into the final concept set (`mindforge/pipeline.py:430-447`, comment: "LLM may miss things that pattern matching catches"). On the unified dogfood KB built 2026-05-07 (`~/.mindforge/unified/output`, 1132 concepts), a post-hoc audit found that **178 concepts (15.7%)** are pure heuristic-extractor noise — sentence fragments and bare-word names like "Now let me check the app", "Fixing both", "PR B open", "Lock", "Delete", "Found", "Lives". Every one of those names is a regex catch from raw transcript text, and they pollute every cross-project query against the KB.

Two further observations made this redesign decisive:

1. The heuristic-as-fallback was an early-project assumption (some users couldn't run local LLMs). Project shape has shifted: the cross-project synthesis goal (per `mindforge-cross-project-direction` memory) requires extraction quality the heuristic can't deliver, and local LLM access (`qwen3:30b-a3b` on `inpherence01` via Tailscale) is now the verified-working baseline.
2. The actual value the heuristic was providing was **not** "fallback for users without LLM." It was **fast pipeline testing** — running the full pipeline (parse → chunk → extract → distill → graph → store) without paying LLM round-trip costs. That use case is real and deserves a proper replacement, not a low-quality fallback masquerading as one.

This spec covers heuristic removal and the mock-LLM replacement together because they're tightly coupled: the eval suite uses `mode="heuristic"` as its default in CI (`mindforge/eval/runner.py:19`, `mindforge/cli.py:387`), so dropping heuristic without a replacement breaks the eval gate.

## Goals

- `mindforge ingest` always uses the LLM extractor; heuristic concept generation is gone from the pipeline.
- A `MockLLMClient` provides deterministic, content-derivative concept extraction for fast pipeline tests, selectable via the existing `llm.provider` config knob (no new CLI flag).
- The eval suite continues to run end-to-end in CI without an LLM endpoint, but reframed as a smoke test (pipeline doesn't crash, scoring code runs); content-quality validation moves to `--mode llm` runs against a real endpoint.
- Mock-mode and real-mode runs cannot share an output directory — the pipeline refuses to mix them at startup, with a clear error message.
- LLM-unreachable conditions hard-fail with a clear message rather than silently degrading to heuristic.

## Non-goals

- **Post-LLM rule layer** (grounding filter, name-shape rules, hallucination guards). Captured in the `mindforge-heuristic-mode-redesign` memory as the next pillar; gets its own design spec.
- **Per-concept chunk-level provenance.** Separate extraction-quality work. Tracked in the `mindforge-extraction-quality-issues` memory.
- **Re-ingesting the unified KB** on the cleaned pipeline. Operational follow-up; one-shot LLM run; not design work.
- **Concurrent-ingest safety, manifest locking, multi-user.** Single-user dogfood; out of scope.
- **Backward compatibility for the `--llm` flag.** Hard-failure on use was an explicit decision; no transition release.

## Architecture

### Component changes

```
mindforge/
├── cli.py                         (modified)
│   ├── ingest: drop --llm flag
│   └── eval:   --mode choices: heuristic|llm|tune-retrieval
│                              → mock|llm|tune-retrieval
├── pipeline.py                    (modified)
│   ├── Drop the LLM-unavailable heuristic fallback (lines 419-422)
│   │   → raise RuntimeError with config-pointing message
│   ├── Drop the LLM∪heuristic merge (lines 430-447)
│   ├── Drop the plain-mode heuristic-only path (line 249)
│   ├── Replace `LLMClient(llm_config)` with
│   │   `make_llm_client(llm_config)` factory call
│   └── Run the KB pollution guard before extraction starts
├── llm/
│   ├── client.py                  (modified)
│   │   └── + make_llm_client(config) factory
│   ├── mock.py                    (new)
│   │   └── MockLLMClient subclass
│   └── extractor.py               (unchanged)
├── ingestion/
│   ├── extractor.py               (DELETED)
│   ├── chunker.py                 (unchanged)
│   ├── parser.py                  (unchanged)
│   └── sources.py                 (unchanged)
├── eval/runner.py                 (modified)
│   └── default mode "heuristic" → "mock"
└── storage/manifest.py            (modified)
    └── + provider field on history entries; + provider-compat check
```

### MockLLMClient

```python
# mindforge/llm/mock.py
class MockLLMClient(LLMClient):
    """Deterministic content-derivative LLM client for pipeline tests.

    Produces wire-format JSON responses derived from chunk text via a fixed
    rule. No network. Reproducible byte-for-byte across runs.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        super().__init__(config)
        self._available = True

    @property
    def available(self) -> bool:
        return True

    def generate(
        self, prompt: str, system: str = "", response_format: str = ""
    ) -> LLMResponse:
        chunk_text = _extract_chunk_text_from_prompt(prompt)
        concepts = _mock_concepts_from_text(chunk_text)
        body = json.dumps({"concepts": concepts})
        return LLMResponse(text=body, success=True)
```

The factory:

```python
# mindforge/llm/client.py
def make_llm_client(config: LLMConfig) -> LLMClient:
    if config.provider == "mock":
        from mindforge.llm.mock import MockLLMClient
        return MockLLMClient(config)
    if config.provider in ("ollama", "openai"):
        return LLMClient(config)
    raise ValueError(
        f"unknown LLM provider: {config.provider!r}; "
        f"expected one of ollama|openai|mock"
    )
```

### Mock content-derivative rule

Per chunk, the mock applies this deterministic rule:

```python
TITLECASE_PHRASE = re.compile(
    r"\b([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){0,3})\b"
)

def _mock_concepts_from_text(text: str) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for m in TITLECASE_PHRASE.finditer(text):
        phrase = m.group(1).strip()
        if len(phrase) < 3 or phrase in seen:
            continue
        seen.add(phrase)
        definition = _surrounding_sentence(text, m.start())[:300]
        out.append({
            "name": phrase,
            "definition": definition,
            "tags": ["mock"],
            "insights": [],
            "examples": [],
            "relationships": [],
        })
        if len(out) >= 3:
            break
    return out
```

Properties:

- **Deterministic**: same input → same output, byte-for-byte. No randomness, no time, no environment dependencies.
- **Content-derivative**: real chunks produce realistic-looking concepts (e.g., "KV Cache", "Multi-Query Attention" if those phrases appear); empty/code-only chunks produce zero concepts.
- **Capped at 3 concepts per chunk**: bounds total concept count for predictable pipeline behavior; still exercises dedup logic when the same phrase appears in multiple chunks.
- **Tagged `["mock"]`**: lets test assertions and debug tooling distinguish mock output from real LLM output.
- **Confidence 0.5**: signals "this is test data" if anything ever leaks into a real KB.

The chunk text the mock sees is whatever appears in the prompt the LLM extractor builds. The mock does not inspect prompt structure beyond extracting the chunk-text payload — the prompt template is the LLM extractor's concern, not the mock's.

### KB pollution guard

Mock-mode and real-mode runs cannot share an output directory. Enforced by adding a `provider` field to manifest history entries and checking compatibility before extraction starts.

**Manifest history entry shape:**

```json
{
  "timestamp": "2026-05-08T...",
  "provider": "ollama" | "openai" | "mock",
  "slug_hashes": { ... }
}
```

(Existing manifests have no `provider` field — treated as "real" for back-compat; see table below.)

**Compatibility check** (runs before extraction, fails fast):

| State of output dir       | Run mode | Result                                          |
|---------------------------|----------|-------------------------------------------------|
| Empty / no manifest       | mock     | proceeds; sets `provider: mock` marker          |
| Empty / no manifest       | real     | proceeds; sets `provider: ollama/openai` marker |
| Last run = mock           | mock     | proceeds (incremental update)                   |
| Last run = real           | real     | proceeds (incremental update)                   |
| Last run = mock           | real     | refuses with clear error                        |
| Last run = real           | mock     | refuses with clear error                        |
| Legacy KB (no field)      | mock     | refuses (treats unmarked as real)               |
| Legacy KB (no field)      | real     | proceeds; first ingest sets the field           |

**Error message** (both directions):

```
Output dir <path> was last built with provider '<last>'; current provider
is '<current>'. Mock and real runs cannot share a KB. Either point
output_dir at a fresh location, or wipe the dir to rebuild under the new
provider.
```

### CLI surface changes

| Command                       | Before                                          | After                                       |
|-------------------------------|-------------------------------------------------|---------------------------------------------|
| `mindforge ingest`            | accepts `--llm` boolean flag                    | flag rejected (argparse unknown-arg error)  |
| `mindforge ingest`            | runs heuristic when `--llm` absent              | always uses configured `llm.provider`       |
| `mindforge ingest --llm-provider mock` | not supported                          | activates `MockLLMClient`                   |
| `mindforge eval --mode heuristic` | runs heuristic extractor                    | flag value rejected (choices error)         |
| `mindforge eval --mode mock`  | not supported                                   | runs pipeline with mock client (smoke test) |
| `mindforge eval` (no --mode)  | defaults to heuristic                           | defaults to mock                            |

The `--llm-provider`, `--llm-model`, `--llm-base-url`, `--llm-api-key` overrides on `ingest` and `mcp` are unchanged; `--llm-provider mock` works without code change.

### Eval suite reframing

`mindforge eval --mode mock` is a **smoke test** in CI: it asserts the pipeline runs end-to-end and the report JSON has the expected shape. Recall, precision, and phrase-grounding scores are not asserted in mock mode — the mock's content-derivative rule produces concept names that may or may not match the fixtures' `expected_concepts`, and the scores carry no semantic meaning.

`mindforge eval --mode llm` is the **quality gate**: run separately (manually or scheduled CI), against a real LLM endpoint, with recall/precision thresholds. Not part of every-PR CI.

## Data flow

```
mindforge ingest
  ↓
load config (config.yaml + CLI overrides)
  ↓
LLMConfig (provider = "ollama" | "openai" | "mock")
  ↓
client = make_llm_client(llm_config)
  ↓
if not client.available:
    raise RuntimeError(<config-pointing message>)
  ↓
load manifest if exists; check_kb_provider_compat(config, manifest)
  ↓
extract_concepts_llm(chunks, client)         ← shared path; mock returns
  ├─ batches chunks                            real-shaped JSON
  ├─ client.generate(prompt, response_format="json")
  ├─ parse JSON via existing _parse_llm_concepts
  └─ list[RawConcept]
  ↓
distillation → graph build → storage         (unchanged)
  ↓
manifest.history.append({timestamp, provider: <current>, slug_hashes})
```

## Error handling

- **Unknown provider value** (e.g., `llm.provider: "qwen"`): factory raises `ValueError("unknown LLM provider: ...")`. CLI surfaces it; user sees the expected provider list.
- **LLM unreachable** (real provider): `client.available` returns False; pipeline raises `RuntimeError` with a message including the configured `base_url` and instructions to either configure the endpoint or set `llm.provider: mock`. No fallback.
- **Provider mismatch** (KB pollution guard): `RuntimeError` with the documented message before any extraction work runs. Manifest is not modified on mismatch.
- **Mock can't fail at the network layer.** Bad input chunks produce zero concepts (same shape as real LLM returning empty `concepts` array). Pipeline continues; resulting KB is small but valid.
- **JSON parse failures** in `extract_concepts_llm`: handled by existing code (`stats.parse_failures`). Mock should never produce parse failures; CI can assert `parse_failures == 0` in mock-mode runs as a smoke-test invariant.
- **Legacy KB upgrade**: a real-mode ingest against a manifest with no `provider` field proceeds normally and writes the field on the next history entry. No user action required.

## Testing

### `MockLLMClient` unit tests (new, `tests/test_mock_llm_client.py`)

- Deterministic byte-equality: same chunk text → same JSON, twice in a row.
- Empty / whitespace-only / code-only chunks → empty `concepts` array.
- Chunk with > 3 capitalized phrases → exactly 3 concepts emitted.
- Duplicate phrase within a chunk → deduped to one concept.
- All emitted concepts have `tags == ["mock"]`, `confidence == 0.5` (after distillation), and a non-empty `name` of length ≥ 3.
- Round-trip: `MockLLMClient.generate(...).text` parsed by `_parse_llm_concepts` produces a non-empty `list[RawConcept]` with `extraction_method == "llm"` (the parser doesn't distinguish mock from real — by design).

### Pipeline integration tests (modified)

- All existing tests using `use_llm=False` migrate to `llm.provider="mock"`. Same coverage; same assertion shapes (concept count, slug presence, graph edges).
- New test: provider mismatch refusal. Build a tiny KB in mock mode → run real-mode ingest against same dir → expect `RuntimeError` with the documented message; manifest unchanged.
- New test: legacy KB upgrade. Build a manifest by hand without `provider` field → run real-mode ingest → expect success; expect `provider: ollama` (or whatever) on the new history entry.
- New test: mock JSON parse failures stay at 0 across a representative fixture run.

### Eval suite tests (modified)

- `mindforge eval --mode mock` runs end-to-end against the existing fixtures; assertion is structural only (report has `mode`, `corpus_size`, `concepts`, `relationships` keys; no recall/precision threshold).
- `mindforge eval --mode llm` test stays as-is, gated on LLM endpoint availability (skipped in CI without one).

### Deletions

- `tests/test_extractor.py` (or whatever heuristic-extractor unit tests exist) — remove alongside the module.
- Any test asserting `mode="heuristic"` literal — migrate to `"mock"`.

### Out of scope for this spec's CI plan

- The 20-concept manual grading test (`~/.mindforge/scripts/grade-sample.py`) is an empirical research instrument, not a regression gate. Captured in `mindforge-scale-experiment-findings` memory as a follow-up against the post-cleanup unified KB.

## Migration & documentation

- **`CLAUDE.md`** — update the "Running the eval suite" example: `mindforge eval --mode heuristic` → `mindforge eval --mode mock`. The "Style" section's legacy-module list does not need editing (`mindforge.ingestion.extractor` was not in the list — it had strict mypy already; only `ingestion.incremental` is in the legacy list, and that module is unaffected by this work).
- **`README.md`** — Modes section gets the mock callout (KB-pollution warning); remove `--llm` flag mentions.
- **`eval/README.md`** — explicit framing: mock = smoke test, llm = quality gate; default for CI is mock.
- **`docs/ARCHITECTURE.md`** — extraction-layer description switches from "heuristic + LLM" to "LLM extractor; mock provider for tests."
- **`pyproject.toml`** — no mypy override entry exists for `mindforge.ingestion.extractor` (verified during spec review against `pyproject.toml:117-124`). Module deletion is a clean removal; no override list edits required.
- **`mindforge/llm/client.py:24`** — comment on `LLMConfig.provider` currently reads `# "ollama" or "openai"`; update to `# "ollama" | "openai" | "mock"`.
- **PR description / CHANGELOG entry** — explicit breaking-change callout:
  - `--llm` flag removed; LLM is always on.
  - Heuristic extraction removed; use `llm.provider: mock` for pipeline-test mode.
  - `mindforge eval --mode heuristic` removed; `--mode mock` replaces it.
  - Existing KBs continue to work; first ingest after upgrade adds the `provider` marker.

## Open questions

None at design time — every decision in this spec was explicitly chosen during the brainstorming session on 2026-05-08.

## Follow-ups (separate specs)

1. **Post-LLM rule layer** — grounding filter, name-shape rejection, fragment-shape rejection, list-position rejection, stock-concept gating. Captured in `mindforge-heuristic-mode-redesign` memory.
2. **Per-concept chunk-level provenance** — `mindforge show <slug> --sources` should return the supporting span, not the whole batch. Captured in `mindforge-extraction-quality-issues` memory.
3. **Unified KB re-ingest** — operational follow-up to run after this spec lands. Single LLM run; produces clean LLM-only corpus; enables the 20-concept manual grading test.
