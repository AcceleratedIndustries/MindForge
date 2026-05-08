# Heuristic removal and mock-LLM client — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the heuristic concept extractor from MindForge, drop the `--llm` flag, hard-fail when the LLM endpoint is unreachable, and add a `MockLLMClient` for fast pipeline-test mode. Output: `mindforge ingest` always runs LLM extraction (or its mock counterpart, selected via `llm.provider: mock`); the eval suite stays runnable in CI as a smoke test.

**Architecture:** A factory function (`make_llm_client`) selects between `LLMClient` (real Ollama/OpenAI) and a new `MockLLMClient` based on `config.llm.provider`. A small per-manifest `provider` marker prevents mock and real runs from sharing an output directory. The heuristic extractor module is deleted; the `RawConcept` dataclass that lived there moves to `mindforge/distillation/raw.py` so the LLM extraction → distillation flow keeps working unchanged.

**Tech Stack:** Python 3.10+, pytest, mypy strict, ruff. No new third-party deps.

**Spec:** [docs/superpowers/specs/2026-05-08-heuristic-removal-and-mock-llm-design.md](../specs/2026-05-08-heuristic-removal-and-mock-llm-design.md)

---

## File map

**Created:**
- `mindforge/distillation/raw.py` — moves `RawConcept` dataclass (Task 1)
- `mindforge/llm/mock.py` — `MockLLMClient` (Task 2-3)
- `tests/test_mock_llm_client.py` — unit tests for MockLLMClient (Task 2-3)
- `tests/test_kb_provider_guard.py` — guard tests (Task 5)

**Modified:**
- `mindforge/llm/client.py` — add `make_llm_client` factory; update provider comment (Task 2)
- `mindforge/pipeline.py` — manifest provider field, guard call, factory call, remove heuristic plumbing (Tasks 4, 5, 6, 9)
- `mindforge/cli.py` — drop `--llm` flag, add "mock" to `--llm-provider` choices, eval `--mode` choices (Tasks 7, 10, 11)
- `mindforge/config.py` — remove `use_llm` field (Task 10)
- `mindforge/eval/runner.py` — default mode `"heuristic"` → `"mock"` (Task 11)
- `mindforge/llm/extractor.py`, `mindforge/llm/distiller.py`, `mindforge/distillation/distiller.py`, `mindforge/distillation/deduplicator.py`, `tests/test_llm.py`, `tests/test_distillation.py` — `RawConcept` import path update (Task 1)
- `tests/test_incremental_pipeline.py` — `use_llm=False` → `llm.provider="mock"` (Task 8)
- `CLAUDE.md`, `README.md`, `eval/README.md`, `docs/ARCHITECTURE.md` — docs (Task 13)

**Deleted:**
- `mindforge/ingestion/extractor.py` — heuristic extractor module (Task 12)
- `tests/test_ingestion.py` — heuristic-extractor unit tests (Task 12)

---

## Task 1: Move `RawConcept` out of the heuristic module

**Why this is the first task:** every consumer (LLM extractor, distillers, dedup, several tests) imports `RawConcept` from `mindforge.ingestion.extractor`. We can't delete that file later until the dataclass has a new home. This task is a pure mechanical move; no behavior change.

**Files:**
- Create: `mindforge/distillation/raw.py`
- Modify: `mindforge/ingestion/extractor.py:19-29` (remove `RawConcept`; keep file otherwise intact for now — cleared in Task 12)
- Modify: `mindforge/llm/extractor.py:19` (import path)
- Modify: `mindforge/llm/distiller.py:22` (import path)
- Modify: `mindforge/distillation/distiller.py:16` (import path)
- Modify: `mindforge/distillation/deduplicator.py:9` (import path)
- Modify: `mindforge/pipeline.py:22` (import path; keep `extract_concepts` import for now — removed in Task 9)
- Modify: `tests/test_llm.py:16` (import path)
- Modify: `tests/test_distillation.py:7` (import path)

- [ ] **Step 1: Create the new module**

Create `mindforge/distillation/raw.py` with:

```python
"""RawConcept: candidate concept produced by an extractor before distillation.

Lives here (not in extractors) because it's the input shape the distillation
pipeline consumes. Both the LLM extractor and the mock LLM client produce
RawConcept instances; the distiller is the consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RawConcept:
    """A candidate concept before distillation."""

    name: str
    raw_content: str
    source_chunks: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    extraction_method: str = "unknown"
    confidence: float = 0.5
    source_hash: str = ""
```

- [ ] **Step 2: Update every importer**

Replace `from mindforge.ingestion.extractor import RawConcept` (and any combined-import variants) with `from mindforge.distillation.raw import RawConcept` in every file listed in the Files block above. For `mindforge/pipeline.py:22` specifically, the current line is:

```python
from mindforge.ingestion.extractor import RawConcept, extract_concepts
```

Split into two imports for now (Task 9 deletes the second one):

```python
from mindforge.distillation.raw import RawConcept
from mindforge.ingestion.extractor import extract_concepts
```

- [ ] **Step 3: Remove `RawConcept` from `mindforge/ingestion/extractor.py`**

Delete lines 19-29 (the `@dataclass class RawConcept:` block). Leave the rest of the file (the regex patterns, `extract_concepts`, helpers) intact — Task 12 deletes the whole file.

- [ ] **Step 4: Run the full test suite to verify the move is transparent**

Run: `pytest -q`
Expected: all tests pass (this is a pure refactor; if any fail, an importer was missed).

- [ ] **Step 5: Run mypy to verify type integrity**

Run: `mypy mindforge`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add mindforge/distillation/raw.py mindforge/ingestion/extractor.py mindforge/llm/extractor.py mindforge/llm/distiller.py mindforge/distillation/distiller.py mindforge/distillation/deduplicator.py mindforge/pipeline.py tests/test_llm.py tests/test_distillation.py
git commit -m "refactor(distillation): move RawConcept out of ingestion.extractor

Prepares for heuristic-extractor module removal: RawConcept is the
input shape distillation consumes, not an artifact of any one extractor.
Pure import-path move; no behavior change."
```

---

## Task 2: Add `make_llm_client` factory and `MockLLMClient` skeleton

**Files:**
- Modify: `mindforge/llm/client.py` (add factory + update provider comment)
- Create: `mindforge/llm/mock.py`
- Create: `tests/test_mock_llm_client.py`

- [ ] **Step 1: Write the failing factory test**

Create `tests/test_mock_llm_client.py`:

```python
"""Tests for MockLLMClient and make_llm_client factory."""

from __future__ import annotations

import json

import pytest

from mindforge.llm.client import LLMClient, LLMConfig, make_llm_client
from mindforge.llm.mock import MockLLMClient


class TestFactory:
    def test_factory_returns_mock_for_mock_provider(self) -> None:
        cfg = LLMConfig(provider="mock")
        client = make_llm_client(cfg)
        assert isinstance(client, MockLLMClient)

    def test_factory_returns_real_client_for_ollama(self) -> None:
        cfg = LLMConfig(provider="ollama")
        client = make_llm_client(cfg)
        assert isinstance(client, LLMClient)
        assert not isinstance(client, MockLLMClient)

    def test_factory_returns_real_client_for_openai(self) -> None:
        cfg = LLMConfig(provider="openai")
        client = make_llm_client(cfg)
        assert isinstance(client, LLMClient)
        assert not isinstance(client, MockLLMClient)

    def test_factory_rejects_unknown_provider(self) -> None:
        cfg = LLMConfig(provider="qwen")
        with pytest.raises(ValueError, match="unknown LLM provider: 'qwen'"):
            make_llm_client(cfg)


class TestMockClientAvailability:
    def test_mock_is_always_available(self) -> None:
        client = MockLLMClient(LLMConfig(provider="mock"))
        assert client.available is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mock_llm_client.py -v`
Expected: ImportError on `from mindforge.llm.mock import MockLLMClient` (module doesn't exist) AND `make_llm_client` not exported from `mindforge.llm.client`.

- [ ] **Step 3: Create `mindforge/llm/mock.py` skeleton**

```python
"""Deterministic content-derivative LLM client for pipeline tests.

Produces wire-format JSON responses derived from chunk text via a fixed
rule. No network. Reproducible byte-for-byte across runs. Selected via
`config.llm.provider == "mock"` and the make_llm_client factory.
"""

from __future__ import annotations

from mindforge.llm.client import LLMClient, LLMConfig, LLMResponse


class MockLLMClient(LLMClient):
    """Deterministic mock that returns canned JSON without network calls."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        super().__init__(config)
        self._available = True

    def generate(
        self, prompt: str, system: str = "", response_format: str = ""
    ) -> LLMResponse:
        # Implementation in Task 3.
        return LLMResponse(text='{"concepts": []}', success=True)
```

- [ ] **Step 4: Add `make_llm_client` factory to `mindforge/llm/client.py`**

At the end of `mindforge/llm/client.py`, append:

```python
def make_llm_client(config: LLMConfig) -> LLMClient:
    """Construct the right LLM client for ``config.provider``.

    Returns ``MockLLMClient`` when provider is "mock"; otherwise returns the
    standard ``LLMClient`` (which dispatches Ollama vs. OpenAI internally).
    Raises ``ValueError`` for unknown providers.
    """
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

Also update the `provider` field comment in `LLMConfig` (currently around line 24):

Find:
```python
    provider: str = "ollama"  # "ollama" or "openai"
```

Replace with:
```python
    provider: str = "ollama"  # "ollama" | "openai" | "mock"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_mock_llm_client.py -v`
Expected: all tests in `TestFactory` and `TestMockClientAvailability` pass.

- [ ] **Step 6: Run mypy**

Run: `mypy mindforge`
Expected: no new errors.

- [ ] **Step 7: Commit**

```bash
git add mindforge/llm/client.py mindforge/llm/mock.py tests/test_mock_llm_client.py
git commit -m "feat(llm): add make_llm_client factory and MockLLMClient skeleton

Factory dispatches on config.provider: 'mock' -> MockLLMClient, 'ollama'
or 'openai' -> LLMClient, anything else -> ValueError. Mock client
generate() returns empty concepts; content-derivative rule lands in
the next task."
```

---

## Task 3: Implement `MockLLMClient.generate` content-derivative rule

**Files:**
- Modify: `mindforge/llm/mock.py`
- Modify: `tests/test_mock_llm_client.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mock_llm_client.py`:

```python
class TestMockGenerate:
    def setup_method(self) -> None:
        self.client = MockLLMClient(LLMConfig(provider="mock"))

    def test_extracts_titlecase_phrase_as_concept_name(self) -> None:
        prompt = "Some text mentioning KV Cache in passing."
        resp = self.client.generate(prompt, response_format="json")
        data = json.loads(resp.text)
        names = [c["name"] for c in data["concepts"]]
        assert "KV Cache" in names

    def test_deterministic_byte_for_byte(self) -> None:
        prompt = "KV Cache and Multi-Query Attention come up here."
        a = self.client.generate(prompt, response_format="json").text
        b = self.client.generate(prompt, response_format="json").text
        assert a == b

    def test_caps_at_three_concepts_per_call(self) -> None:
        # Five distinct titlecase phrases.
        prompt = "Alpha One and Beta Two and Gamma Three and Delta Four and Epsilon Five."
        resp = self.client.generate(prompt, response_format="json")
        data = json.loads(resp.text)
        assert len(data["concepts"]) == 3

    def test_dedupes_within_a_single_call(self) -> None:
        prompt = "KV Cache. Then more text. KV Cache again. KV Cache once more."
        resp = self.client.generate(prompt, response_format="json")
        data = json.loads(resp.text)
        names = [c["name"] for c in data["concepts"]]
        assert names.count("KV Cache") == 1

    def test_empty_text_returns_no_concepts(self) -> None:
        resp = self.client.generate("", response_format="json")
        data = json.loads(resp.text)
        assert data["concepts"] == []

    def test_lowercase_only_text_returns_no_concepts(self) -> None:
        resp = self.client.generate("just lowercase words here, nothing capitalized.", response_format="json")
        data = json.loads(resp.text)
        assert data["concepts"] == []

    def test_concepts_tagged_mock(self) -> None:
        prompt = "KV Cache exists."
        resp = self.client.generate(prompt, response_format="json")
        data = json.loads(resp.text)
        for c in data["concepts"]:
            assert c["tags"] == ["mock"]

    def test_definition_is_surrounding_sentence(self) -> None:
        prompt = "First sentence. KV Cache is a memory structure. Third sentence."
        resp = self.client.generate(prompt, response_format="json")
        data = json.loads(resp.text)
        kv = next(c for c in data["concepts"] if c["name"] == "KV Cache")
        assert "memory structure" in kv["definition"]
        assert "First sentence" not in kv["definition"]
        assert "Third sentence" not in kv["definition"]

    def test_short_phrases_under_3_chars_excluded(self) -> None:
        # "I" and "A" are titlecase but len < 3.
        prompt = "I think A is fine but Real Concept matters."
        resp = self.client.generate(prompt, response_format="json")
        data = json.loads(resp.text)
        names = [c["name"] for c in data["concepts"]]
        assert "Real Concept" in names
        assert "I" not in names
        assert "A" not in names

    def test_round_trips_through_real_parser(self) -> None:
        from mindforge.llm.extractor import _parse_llm_concepts

        prompt = "KV Cache and Multi-Query Attention are concepts."
        resp = self.client.generate(prompt, response_format="json")
        data = json.loads(resp.text)
        concepts = _parse_llm_concepts(
            data,
            source_chunks=["chunk-1"],
            source_files=["/tmp/x.md"],
        )
        names = {c.name for c in concepts}
        assert "KV Cache" in names
        assert "Multi-Query Attention" in names
        for c in concepts:
            assert c.extraction_method == "llm"  # parser stamps this; mock indistinguishable to consumer
            assert c.confidence == 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mock_llm_client.py::TestMockGenerate -v`
Expected: most tests fail (current `generate` returns empty concepts).

- [ ] **Step 3: Implement the content-derivative rule**

Replace the body of `mindforge/llm/mock.py` with:

```python
"""Deterministic content-derivative LLM client for pipeline tests.

Produces wire-format JSON responses derived from chunk text via a fixed
rule. No network. Reproducible byte-for-byte across runs. Selected via
`config.llm.provider == "mock"` and the make_llm_client factory.
"""

from __future__ import annotations

import json
import re

from mindforge.llm.client import LLMClient, LLMConfig, LLMResponse

_TITLECASE_PHRASE = re.compile(
    r"\b([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){0,3})\b"
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MAX_CONCEPTS_PER_CALL = 3
_MIN_NAME_LEN = 3
_MAX_DEFINITION_CHARS = 300


def _surrounding_sentence(text: str, position: int) -> str:
    """Return the sentence containing ``position`` in ``text``.

    Sentence boundaries are simple [.!?] followed by whitespace. Falls back to
    the whole text when no boundary is found.
    """
    start = 0
    end = len(text)
    for m in re.finditer(r"[.!?]\s+", text):
        if m.end() <= position:
            start = m.end()
        elif m.start() >= position:
            end = m.start()
            break
    sentence = text[start:end].strip()
    return sentence or text.strip()


def _mock_concepts_from_text(text: str) -> list[dict]:
    """Apply the deterministic content-derivative rule to ``text``."""
    seen: set[str] = set()
    out: list[dict] = []
    for m in _TITLECASE_PHRASE.finditer(text):
        phrase = m.group(1).strip()
        if len(phrase) < _MIN_NAME_LEN or phrase in seen:
            continue
        seen.add(phrase)
        definition = _surrounding_sentence(text, m.start())[:_MAX_DEFINITION_CHARS]
        out.append({
            "name": phrase,
            "definition": definition,
            "tags": ["mock"],
            "insights": [],
            "examples": [],
            "relationships": [],
        })
        if len(out) >= _MAX_CONCEPTS_PER_CALL:
            break
    return out


class MockLLMClient(LLMClient):
    """Deterministic mock that returns canned JSON without network calls."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        super().__init__(config)
        self._available = True

    def generate(
        self, prompt: str, system: str = "", response_format: str = ""
    ) -> LLMResponse:
        concepts = _mock_concepts_from_text(prompt)
        body = json.dumps({"concepts": concepts})
        return LLMResponse(text=body, success=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mock_llm_client.py -v`
Expected: all tests pass (factory + availability + generate).

- [ ] **Step 5: Run mypy**

Run: `mypy mindforge`
Expected: no new errors.

- [ ] **Step 6: Run ruff to verify formatting**

Run: `ruff check mindforge tests && ruff format --check mindforge tests`
Expected: no issues. If format suggests changes, run `ruff format mindforge tests` and re-stage.

- [ ] **Step 7: Commit**

```bash
git add mindforge/llm/mock.py tests/test_mock_llm_client.py
git commit -m "feat(llm): MockLLMClient content-derivative concept extraction

Deterministic per-prompt rule: extract up to 3 distinct title-case
phrases (>=3 chars) from chunk text, emit each as a mock concept tagged
['mock'] with the surrounding sentence as definition. Round-trips
through the real LLM-extractor JSON parser unchanged."
```

---

## Task 4: Add `provider` field to manifest snapshots

**Files:**
- Modify: `mindforge/pipeline.py:32-46` (`write_manifest_snapshot`)
- Modify: `mindforge/pipeline.py:49-55` (`read_manifest_history` — no change needed, just verifying the read continues to work)

- [ ] **Step 1: Write the failing test**

Append to a new test file (or your preferred existing pipeline-test file). Recommended: append to `tests/test_kb_provider_guard.py` (created in Task 5). For now, place these in a new file `tests/test_manifest_snapshot.py`:

```python
"""Manifest snapshot/history tests focused on the provider field."""

from __future__ import annotations

import json
from pathlib import Path

from mindforge.pipeline import read_manifest_history, write_manifest_snapshot
from mindforge.storage.fs import ConceptStore


def _empty_store() -> ConceptStore:
    return ConceptStore()


def test_snapshot_records_provider(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    write_manifest_snapshot(_empty_store(), manifest, provider="mock")
    history = read_manifest_history(manifest)
    assert len(history) == 1
    assert history[0]["provider"] == "mock"


def test_snapshot_records_ollama_provider(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    write_manifest_snapshot(_empty_store(), manifest, provider="ollama")
    history = read_manifest_history(manifest)
    assert history[0]["provider"] == "ollama"


def test_snapshot_appends_preserving_history(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    write_manifest_snapshot(_empty_store(), manifest, provider="ollama")
    write_manifest_snapshot(_empty_store(), manifest, provider="ollama")
    history = read_manifest_history(manifest)
    assert len(history) == 2


def test_legacy_manifest_without_provider_field_loads_cleanly(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    legacy = {
        "version": 1,
        "history": [
            {"timestamp": "2026-04-01T00:00:00+00:00", "slug_hashes": {}},
        ],
    }
    manifest.write_text(json.dumps(legacy), encoding="utf-8")
    history = read_manifest_history(manifest)
    assert len(history) == 1
    assert "provider" not in history[0]  # legacy entries stay legacy
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_manifest_snapshot.py -v`
Expected: TypeError or signature mismatch — `write_manifest_snapshot` doesn't accept `provider` kwarg yet.

- [ ] **Step 3: Add `provider` parameter to `write_manifest_snapshot`**

In `mindforge/pipeline.py`, current code at lines 32-46:

```python
def write_manifest_snapshot(store: ConceptStore, manifest_path: Path) -> None:
    """Append a timestamped snapshot of slug hashes to manifest_path."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    slug_hashes = {c.slug: c.hash for c in store.all()}
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "slug_hashes": slug_hashes,
    }
    data: dict[str, Any]
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        data = {"version": 1, "history": []}
    data.setdefault("history", []).append(snapshot)
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
```

Replace with:

```python
def write_manifest_snapshot(
    store: ConceptStore, manifest_path: Path, *, provider: str
) -> None:
    """Append a timestamped snapshot of slug hashes to manifest_path.

    ``provider`` records which LLM provider produced this snapshot
    ("ollama" | "openai" | "mock") so future runs can refuse to mix
    real and mock data in the same KB.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    slug_hashes = {c.slug: c.hash for c in store.all()}
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "slug_hashes": slug_hashes,
    }
    data: dict[str, Any]
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        data = {"version": 1, "history": []}
    data.setdefault("history", []).append(snapshot)
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Update the existing call site that writes the manifest**

In `mindforge/pipeline.py`, the only `write_manifest_snapshot` call is at line 316:

```python
        write_manifest_snapshot(self.store, self.config.output_dir / "manifest.json")
```

Replace with:

```python
        write_manifest_snapshot(
            self.store,
            self.config.output_dir / "manifest.json",
            provider=self.config.llm_provider,
        )
```

(`MindForgeConfig.llm_provider` already exists — defined at `mindforge/config.py:35` with default `"ollama"`.)

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `pytest tests/test_manifest_snapshot.py -v`
Expected: all four tests pass.

- [ ] **Step 6: Run the full suite to check no other call sites broke**

Run: `pytest -q`
Expected: all pass. If a test uses `write_manifest_snapshot` directly without `provider`, fix it to pass `provider="ollama"` (since that's the historical default behavior).

- [ ] **Step 7: Run mypy**

Run: `mypy mindforge`
Expected: no new errors.

- [ ] **Step 8: Commit**

```bash
git add mindforge/pipeline.py tests/test_manifest_snapshot.py
git commit -m "feat(pipeline): record provider in manifest history snapshots

Adds a required 'provider' kwarg to write_manifest_snapshot so each
ingest run records which LLM provider produced the slug hashes.
Legacy manifests without the field still load via read_manifest_history.
Used by the KB pollution guard in the next task."
```

---

## Task 5: Add `check_kb_provider_compat` guard

**Files:**
- Modify: `mindforge/pipeline.py` (add `check_kb_provider_compat` near `read_manifest_history`)
- Create: `tests/test_kb_provider_guard.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_kb_provider_guard.py`:

```python
"""KB pollution guard: mock and real runs cannot share an output dir."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindforge.pipeline import check_kb_provider_compat


def _write_manifest(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "history": history}), encoding="utf-8")


class TestProviderGuard:
    def test_no_manifest_proceeds(self, tmp_path: Path) -> None:
        # No concepts.json on disk; any provider proceeds.
        manifest = tmp_path / "manifest.json"
        check_kb_provider_compat(manifest, current_provider="mock")
        check_kb_provider_compat(manifest, current_provider="ollama")

    def test_empty_history_proceeds(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.json"
        _write_manifest(manifest, [])
        check_kb_provider_compat(manifest, current_provider="mock")

    def test_mock_on_mock_proceeds(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.json"
        _write_manifest(manifest, [{"timestamp": "...", "provider": "mock", "slug_hashes": {}}])
        check_kb_provider_compat(manifest, current_provider="mock")

    def test_real_on_real_proceeds(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.json"
        _write_manifest(manifest, [{"timestamp": "...", "provider": "ollama", "slug_hashes": {}}])
        check_kb_provider_compat(manifest, current_provider="ollama")
        check_kb_provider_compat(manifest, current_provider="openai")  # any real provider

    def test_real_on_mock_marked_dir_refuses(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.json"
        _write_manifest(manifest, [{"timestamp": "...", "provider": "mock", "slug_hashes": {}}])
        with pytest.raises(RuntimeError, match="last built with provider 'mock'"):
            check_kb_provider_compat(manifest, current_provider="ollama")

    def test_mock_on_real_marked_dir_refuses(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.json"
        _write_manifest(manifest, [{"timestamp": "...", "provider": "ollama", "slug_hashes": {}}])
        with pytest.raises(RuntimeError, match="last built with provider 'ollama'"):
            check_kb_provider_compat(manifest, current_provider="mock")

    def test_legacy_kb_with_real_proceeds(self, tmp_path: Path) -> None:
        # Legacy entries (no 'provider' field) treated as real for back-compat.
        manifest = tmp_path / "manifest.json"
        _write_manifest(manifest, [{"timestamp": "...", "slug_hashes": {}}])
        check_kb_provider_compat(manifest, current_provider="ollama")

    def test_legacy_kb_with_mock_refuses(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.json"
        _write_manifest(manifest, [{"timestamp": "...", "slug_hashes": {}}])
        with pytest.raises(RuntimeError, match="last built with provider 'ollama'"):
            check_kb_provider_compat(manifest, current_provider="mock")

    def test_only_last_history_entry_matters(self, tmp_path: Path) -> None:
        # If somehow a KB has mixed history (shouldn't happen post-guard, but
        # be defensive), the most recent entry decides compatibility.
        manifest = tmp_path / "manifest.json"
        _write_manifest(manifest, [
            {"timestamp": "older", "provider": "mock", "slug_hashes": {}},
            {"timestamp": "newer", "provider": "ollama", "slug_hashes": {}},
        ])
        check_kb_provider_compat(manifest, current_provider="ollama")
        with pytest.raises(RuntimeError):
            check_kb_provider_compat(manifest, current_provider="mock")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_kb_provider_guard.py -v`
Expected: ImportError on `check_kb_provider_compat`.

- [ ] **Step 3: Implement the guard**

Add to `mindforge/pipeline.py`, immediately after the `read_manifest_history` function:

```python
def check_kb_provider_compat(manifest_path: Path, *, current_provider: str) -> None:
    """Refuse to mix mock and real LLM runs in the same output directory.

    Reads the most recent manifest history entry's ``provider`` field and
    compares it to ``current_provider``. Mock and real runs cannot share a
    KB; mixing would corrupt either the test data or the production data.

    Legacy manifests without a ``provider`` field are treated as real
    (mock didn't exist before this guard, so any unmarked KB is real by
    construction).

    Raises ``RuntimeError`` with a user-actionable message on mismatch.
    """
    history = read_manifest_history(manifest_path)
    if not history:
        return
    last = history[-1]
    last_provider = last.get("provider", "ollama")  # legacy = real
    last_is_mock = last_provider == "mock"
    current_is_mock = current_provider == "mock"
    if last_is_mock != current_is_mock:
        raise RuntimeError(
            f"Output dir {manifest_path.parent} was last built with provider "
            f"'{last_provider}'; current provider is '{current_provider}'. "
            f"Mock and real runs cannot share a KB. Either point output_dir "
            f"at a fresh location, or wipe the dir to rebuild under the new "
            f"provider."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kb_provider_guard.py -v`
Expected: all 9 tests pass.

- [ ] **Step 5: Run mypy**

Run: `mypy mindforge`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add mindforge/pipeline.py tests/test_kb_provider_guard.py
git commit -m "feat(pipeline): KB provider guard refuses mock/real mixing

check_kb_provider_compat reads the last manifest snapshot's provider
field and refuses to run if mock and real would mix. Legacy manifests
without the field are treated as real, so existing KBs upgrade
transparently on the next ingest."
```

---

## Task 6: Wire factory and guard into `MindForgePipeline.run()`

**Files:**
- Modify: `mindforge/pipeline.py` (`_extract_with_llm`, `run` method)

- [ ] **Step 1: Locate the relevant pipeline code**

In `mindforge/pipeline.py`, find:
- `_extract_with_llm` method (around line 405-447): currently does `client = LLMClient(llm_config)`.
- `run()` method (around line 200): runs extraction; manifest write happens later.

- [ ] **Step 2: Replace `LLMClient(llm_config)` with `make_llm_client`**

In `_extract_with_llm`, find:

```python
        client = LLMClient(llm_config)
```

Replace with:

```python
        client = make_llm_client(llm_config)
```

Also update the import at the top of the file. The current line at `mindforge/pipeline.py:26` is:

```python
from mindforge.llm.client import LLMClient, LLMConfig
```

Replace with:

```python
from mindforge.llm.client import LLMClient, LLMConfig, make_llm_client
```

- [ ] **Step 3: Wire the guard before extraction**

In `MindForgePipeline.run()`, the method begins at line 132. Currently line 140 reads `self.config.ensure_dirs()`, followed by `# === Stage 1: Ingestion ===` at line 142. Insert the guard call between them — right after `ensure_dirs()`, before any disk reads:

```python
        self.config.ensure_dirs()

        # Refuse to mix mock and real runs in the same KB.
        check_kb_provider_compat(
            self.config.output_dir / "manifest.json",
            current_provider=self.config.llm_provider,
        )

        # === Stage 1: Ingestion ===
```

This MUST run before any extraction or ingestion work so the failure is fast and the existing KB stays untouched.

- [ ] **Step 4: Add an integration test for the wiring**

Create or append to `tests/test_kb_provider_guard.py`:

```python
class TestPipelineWiring:
    def test_pipeline_run_refuses_mock_on_real_kb(self, tmp_path: Path) -> None:
        """Pipeline.run() must hit the guard before doing extraction work."""
        from mindforge.config import MindForgeConfig
        from mindforge.pipeline import MindForgePipeline

        # Create a tiny "real" KB by hand: manifest with provider="ollama".
        out = tmp_path / "out"
        out.mkdir()
        _write_manifest(out / "manifest.json", [
            {"timestamp": "...", "provider": "ollama", "slug_hashes": {}},
        ])

        # Now try to run with provider=mock pointed at the same dir.
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        (transcripts / "x.md").write_text("Human: test\n\nAssistant: ok", encoding="utf-8")

        cfg = MindForgeConfig(
            transcripts_dir=transcripts,
            output_dir=out,
            llm_provider="mock",
        )
        pipe = MindForgePipeline(cfg)
        with pytest.raises(RuntimeError, match="last built with provider 'ollama'"):
            pipe.run()
```

- [ ] **Step 5: Run all guard tests**

Run: `pytest tests/test_kb_provider_guard.py -v`
Expected: all tests pass, including the new pipeline-wiring one.

- [ ] **Step 6: Run the full suite to confirm no regressions**

Run: `pytest -q`
Expected: all pass. If failures appear in pipeline tests, they likely depend on the heuristic path that's about to be removed in Task 9 — leave them alone for now and fix in Task 8/9.

- [ ] **Step 7: Commit**

```bash
git add mindforge/pipeline.py tests/test_kb_provider_guard.py
git commit -m "feat(pipeline): wire factory and provider guard into run()

_extract_with_llm now uses make_llm_client to dispatch ollama/openai/mock.
MindForgePipeline.run() calls check_kb_provider_compat before any
extraction work, so a mismatch fails fast without modifying the KB."
```

---

## Task 7: Add "mock" to CLI `--llm-provider` choices

**Files:**
- Modify: `mindforge/cli.py:124-129` (ingest `--llm-provider`)
- Modify: `mindforge/cli.py:330-334` (mcp `--llm-provider`, if it has the same choices restriction)

- [ ] **Step 1: Update the ingest argparse choices**

In `mindforge/cli.py`, find:

```python
    llm_group.add_argument(
        "--llm-provider",
        choices=["ollama", "openai"],
        default=None,
        help="LLM provider (overrides config; default: from config or ollama)",
    )
```

Replace with:

```python
    llm_group.add_argument(
        "--llm-provider",
        choices=["ollama", "openai", "mock"],
        default=None,
        help="LLM provider (overrides config; default: from config or ollama). "
             "'mock' uses the deterministic content-derivative test client.",
    )
```

- [ ] **Step 2: Check the mcp subcommand for the same choices**

Run: `grep -n -A3 'add_argument.*--llm-provider' mindforge/cli.py`

If the mcp subcommand also has `choices=["ollama", "openai"]`, update it the same way. (No semantic test needed — `mindforge mcp` doesn't run extraction; mock is informational here.)

- [ ] **Step 3: Quick smoke test**

Run: `python -m mindforge ingest --llm-provider mock --help`
Expected: shows `--llm-provider {ollama,openai,mock}` in the help text. (No actual ingest run.)

- [ ] **Step 4: Commit**

```bash
git add mindforge/cli.py
git commit -m "feat(cli): add 'mock' to --llm-provider choices

Lets users select MockLLMClient via the existing provider knob:
mindforge ingest --llm-provider mock"
```

---

## Task 8: Migrate tests using `use_llm=False` to `llm.provider="mock"`

**Files:**
- Modify: `tests/test_incremental_pipeline.py` (7 occurrences of `use_llm=False`)

- [ ] **Step 1: Inspect the existing tests**

Run: `grep -n "use_llm" tests/test_incremental_pipeline.py`
Expected: 7 lines, each constructing `MindForgeConfig(transcripts_dir=..., output_dir=..., use_llm=False)`.

- [ ] **Step 2: Replace `use_llm=False` with `use_llm=True, llm_provider="mock"`**

For each occurrence in `tests/test_incremental_pipeline.py`, replace:

```python
    config = MindForgeConfig(transcripts_dir=transcripts_dir, output_dir=output_dir, use_llm=False)
```

with:

```python
    config = MindForgeConfig(
        transcripts_dir=transcripts_dir,
        output_dir=output_dir,
        use_llm=True,
        llm_provider="mock",
    )
```

Both fields are required at this point because:
- The pipeline's `if self.config.use_llm:` branch (still alive until Task 9) gates whether LLM extraction runs at all. With `use_llm=False`, the pipeline takes the heuristic path and ignores the provider setting.
- `llm_provider="mock"` selects the `MockLLMClient` inside the LLM extraction path.

`use_llm=True` is redundant once Task 9 drops the branch; Task 10 Step 7 removes it from these tests in the same pass that drops the field from `MindForgeConfig`.

(`MindForgeConfig.llm_provider` is defined at `mindforge/config.py:35`.)

- [ ] **Step 3: Run the migrated tests**

Run: `pytest tests/test_incremental_pipeline.py -v`
Expected: all 7 (or however many) tests pass against MockLLMClient. Some assertions about specific concept counts/names may need adjustment because mock produces different concepts than heuristic — update assertions to match what mock actually produces (it's deterministic, so the new numbers stay stable).

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_incremental_pipeline.py
git commit -m "test(pipeline): migrate use_llm=False tests to mock provider

Tests previously relied on the heuristic extractor for fast, no-LLM
runs. They now run against MockLLMClient. Concept-count assertions
adjusted to mock's deterministic output."
```

---

## Task 9: Remove heuristic call sites from `pipeline.py`

**Files:**
- Modify: `mindforge/pipeline.py:22` (drop `extract_concepts` import)
- Modify: `mindforge/pipeline.py:243-249` (drop `if self.config.use_llm:` branch)
- Modify: `mindforge/pipeline.py:419-422` (replace fallback with hard error)
- Modify: `mindforge/pipeline.py:430-447` (drop merge)

- [ ] **Step 1: Drop the heuristic import**

In `mindforge/pipeline.py`, find:

```python
from mindforge.distillation.raw import RawConcept
from mindforge.ingestion.extractor import extract_concepts
```

Remove the second line:

```python
from mindforge.distillation.raw import RawConcept
```

- [ ] **Step 2: Always run LLM extraction in `run()`**

Find the block at lines ~243-251:

```python
        extraction_method = "heuristic"
        raw_concepts: list[RawConcept] = []

        if self.config.use_llm:
            raw_concepts, extraction_method = self._extract_with_llm(all_chunks)
        else:
            raw_concepts = extract_concepts(all_chunks)

        print(f"  Extracted {len(raw_concepts)} candidate concepts")
```

Replace with:

```python
        raw_concepts, extraction_method = self._extract_with_llm(all_chunks)

        print(f"  Extracted {len(raw_concepts)} candidate concepts")
```

- [ ] **Step 3: Replace the LLM-unavailable fallback with a hard error**

In `_extract_with_llm`, find lines ~419-422:

```python
        if not client.available:
            print(f"  LLM server not reachable ({llm_config.base_url})")
            print("  Falling back to heuristic extraction")
            return extract_concepts(chunks), "heuristic (LLM unavailable)"
```

Replace with:

```python
        if not client.available:
            raise RuntimeError(
                f"LLM endpoint not reachable at {llm_config.base_url}. "
                f"Configure ~/.config/mindforge/config.yaml or pass "
                f"--llm-base-url. For pipeline-test mode without a real LLM, "
                f"set llm.provider: mock."
            )
```

- [ ] **Step 4: Drop the LLM∪heuristic merge**

Still in `_extract_with_llm`, find lines ~425-447:

```python
        print(f"  Using LLM: {llm_config.provider}/{llm_config.model}")
        llm_concepts, stats = extract_concepts_llm(chunks, client)

        if stats.parse_failures > 0:
            print(f"  Warning: {stats.parse_failures} LLM parse failure(s)")

        # Also run heuristic extraction and merge (LLM may miss things
        # that pattern matching catches, and vice versa)
        heuristic_concepts = extract_concepts(chunks)
        print(
            f"  LLM extracted {len(llm_concepts)} concepts, "
            f"heuristic found {len(heuristic_concepts)}"
        )

        # Merge: LLM concepts take priority, then add unique heuristic ones
        seen_names = {c.name.lower() for c in llm_concepts}
        merged = list(llm_concepts)
        for hc in heuristic_concepts:
            if hc.name.lower() not in seen_names:
                seen_names.add(hc.name.lower())
                merged.append(hc)

        method = f"llm ({llm_config.provider}/{llm_config.model}) + heuristic"
        return merged, method
```

Replace with:

```python
        print(f"  Using LLM: {llm_config.provider}/{llm_config.model}")
        llm_concepts, stats = extract_concepts_llm(chunks, client)

        if stats.parse_failures > 0:
            print(f"  Warning: {stats.parse_failures} LLM parse failure(s)")

        print(f"  LLM extracted {len(llm_concepts)} concepts")
        method = f"llm ({llm_config.provider}/{llm_config.model})"
        return llm_concepts, method
```

- [ ] **Step 5: Run the test suite**

Run: `pytest -q`
Expected: all tests pass. The migrated `test_incremental_pipeline.py` tests now run through mock; no test exercises the deleted heuristic-on-LLM-unavailable path because mock is always available.

- [ ] **Step 6: Run mypy**

Run: `mypy mindforge`
Expected: no new errors. (`extract_concepts` is no longer imported in pipeline.py; if mypy complains about an unused import elsewhere, address it.)

- [ ] **Step 7: Commit**

```bash
git add mindforge/pipeline.py
git commit -m "refactor(pipeline): drop heuristic extractor call sites

- Always runs LLM extraction (no use_llm branch).
- Hard-fails when LLM endpoint is unreachable instead of falling back.
- Drops the LLM∪heuristic merge that was dumping ~16% noise into every
  ingest (per the unified-KB audit on 2026-05-08).

The heuristic extractor module itself is removed in a later task."
```

---

## Task 10: Remove `--llm` flag and `use_llm` config field

**Files:**
- Modify: `mindforge/cli.py:119-123` (drop `--llm` argument)
- Modify: `mindforge/cli.py:516` (drop `use_llm=args.llm`)
- Modify: `mindforge/cli.py:529` (drop `if config.use_llm:` block)
- Modify: `mindforge/config.py:34` (drop `use_llm` field)

- [ ] **Step 1: Remove `--llm` argument from ingest parser**

In `mindforge/cli.py`, find:

```python
    llm_group.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM-assisted concept extraction (requires Ollama or API)",
    )
```

Delete the entire block.

- [ ] **Step 2: Remove `use_llm=args.llm` from config construction**

In `mindforge/cli.py` around line 516, find the `use_llm=args.llm` line in the config-construction call. Remove it.

- [ ] **Step 3: Remove `if config.use_llm:` print block**

In `mindforge/cli.py` around line 529, find:

```python
    if config.use_llm:
        print(f"LLM:    {config.llm_provider}/{config.llm_model}")
```

Replace with the unconditional version (LLM is always on now):

```python
    print(f"LLM:    {config.llm_provider}/{config.llm_model}")
```

- [ ] **Step 4: Remove `use_llm` from `MindForgeConfig`**

In `mindforge/config.py`, find:

```python
    use_llm: bool = False
```

Delete the line.

- [ ] **Step 5: Update `mindforge/eval/runner.py` to drop `use_llm`**

The eval runner constructs `MindForgeConfig` with `cfg_kwargs["use_llm"] = mode == "llm"` at `mindforge/eval/runner.py:34`. Since `use_llm` is gone, replace that mapping with `llm_provider` selection. The current block:

```python
        cfg_kwargs: dict[str, Any] = {
            "transcripts_dir": fixtures_dir,
            "output_dir": out,
            "use_llm": mode == "llm",
        }
```

Becomes:

```python
        cfg_kwargs: dict[str, Any] = {
            "transcripts_dir": fixtures_dir,
            "output_dir": out,
            # mode == "heuristic" maps to mock for the duration of the
            # transition window between this task and Task 11 (which removes
            # "heuristic" as a valid CLI choice). After Task 11, only
            # "mock" and "llm" reach this code path.
            "llm_provider": "mock" if mode in ("heuristic", "mock") else "ollama",
        }
```

(Existing kwargs handling for `llm_*` overrides is unchanged.)

- [ ] **Step 6: Remove `use_llm=True` from migrated tests**

Task 8 added `use_llm=True` to the `tests/test_incremental_pipeline.py` config fixtures so the pipeline would take the LLM branch. Now that the field is gone from `MindForgeConfig`, those `use_llm=True` lines would TypeError on construction. Remove them:

For each occurrence in `tests/test_incremental_pipeline.py`, replace:

```python
    config = MindForgeConfig(
        transcripts_dir=transcripts_dir,
        output_dir=output_dir,
        use_llm=True,
        llm_provider="mock",
    )
```

with:

```python
    config = MindForgeConfig(
        transcripts_dir=transcripts_dir,
        output_dir=output_dir,
        llm_provider="mock",
    )
```

- [ ] **Step 7: Verify no `use_llm` references remain anywhere**

Run: `grep -rn "use_llm" mindforge/ tests/ 2>/dev/null | grep -v __pycache__`
Expected: zero results.

- [ ] **Step 8: Run the test suite**

Run: `pytest -q`
Expected: all pass.

- [ ] **Step 9: Verify the `--llm` flag is rejected**

Run: `python -m mindforge ingest --llm 2>&1 | head -5`
Expected: argparse error like `unrecognized arguments: --llm`.

- [ ] **Step 10: Run mypy and ruff**

Run: `mypy mindforge && ruff check . && ruff format --check .`
Expected: no errors. (Format may suggest changes; apply with `ruff format .` if needed.)

- [ ] **Step 11: Commit**

```bash
git add mindforge/cli.py mindforge/config.py mindforge/eval/runner.py tests/test_incremental_pipeline.py
git commit -m "refactor(cli,config): drop --llm flag and use_llm field

LLM is always on; mock provider replaces 'no LLM' as the fast-test mode.
Calling 'mindforge ingest --llm' now hard-fails with argparse error."
```

---

## Task 11: Eval mode rename — `heuristic` → `mock`

**Files:**
- Modify: `mindforge/cli.py:385-388` (eval `--mode` choices and default)
- Modify: `mindforge/eval/runner.py:19` (default mode signature)

- [ ] **Step 1: Update eval `--mode` choices and default in CLI**

In `mindforge/cli.py`, find:

```python
    eval_p.add_argument(
        "--mode",
        choices=["heuristic", "llm", "tune-retrieval"],
        default="heuristic",
```

Replace with:

```python
    eval_p.add_argument(
        "--mode",
        choices=["mock", "llm", "tune-retrieval"],
        default="mock",
```

- [ ] **Step 2: Update default in `runner.py`**

In `mindforge/eval/runner.py`, find:

```python
def run_eval(fixtures_dir: Path, mode: str = "heuristic", **llm_kwargs: Any) -> dict[str, Any]:
    """Run the pipeline on a fixture directory and score against ground truth.

    ``mode`` is "heuristic" (default) or "llm". For LLM mode, pass
    ``llm_provider``, ``llm_model``, ``llm_base_url``, ``llm_api_key`` via kwargs.
    """
```

Replace with:

```python
def run_eval(fixtures_dir: Path, mode: str = "mock", **llm_kwargs: Any) -> dict[str, Any]:
    """Run the pipeline on a fixture directory and score against ground truth.

    ``mode`` is "mock" (default; deterministic smoke test) or "llm" (real
    LLM, used as the quality gate). For LLM mode, pass ``llm_provider``,
    ``llm_model``, ``llm_base_url``, ``llm_api_key`` via kwargs.
    """
```

- [ ] **Step 3: Simplify the provider mapping in `runner.py`**

Task 10 Step 5 left a transitional mapping that handled both `"heuristic"` and `"mock"` mode names:

```python
            "llm_provider": "mock" if mode in ("heuristic", "mock") else "ollama",
```

Now that `--mode heuristic` is rejected at argparse, the `"heuristic"` branch is dead. Simplify to:

```python
            "llm_provider": "mock" if mode == "mock" else "ollama",
```

- [ ] **Step 4: Run the eval suite end-to-end in mock mode**

Run: `mindforge eval --mode mock`
Expected: completes without error; produces a report file at `eval/reports/<timestamp>.json`. Recall/precision values are not asserted here; the test is structural.

- [ ] **Step 5: Run unit tests**

Run: `pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add mindforge/cli.py mindforge/eval/runner.py
git commit -m "refactor(eval): rename --mode heuristic to --mode mock

Eval default flips from heuristic (deleted) to mock (deterministic
smoke test). --mode llm stays as the real-LLM quality gate, run
separately from per-PR CI."
```

---

## Task 12: Delete the heuristic extractor module and its tests

**Files:**
- Delete: `mindforge/ingestion/extractor.py`
- Delete: `tests/test_ingestion.py`

- [ ] **Step 1: Verify no consumers remain**

Run: `grep -rn "from mindforge.ingestion.extractor\|import.*ingestion.extractor\|extract_concepts\b" mindforge/ tests/ 2>/dev/null | grep -v __pycache__`

Expected: zero results (RawConcept moved in Task 1; pipeline.py import dropped in Task 9; tests using `extract_concepts` are in `test_ingestion.py` which we delete now).

If anything still imports from `mindforge.ingestion.extractor`, fix it before proceeding.

- [ ] **Step 2: Delete the module**

Run:

```bash
rm mindforge/ingestion/extractor.py
rm tests/test_ingestion.py
```

- [ ] **Step 3: Run the test suite**

Run: `pytest -q`
Expected: all pass.

- [ ] **Step 4: Run mypy**

Run: `mypy mindforge`
Expected: no errors. (If a stale reference appears in a docstring or unused import, fix it.)

- [ ] **Step 5: Run ruff**

Run: `ruff check . && ruff format --check .`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add -u mindforge/ingestion/extractor.py tests/test_ingestion.py
git commit -m "chore(ingestion): delete heuristic extractor module

Module's contributions to ingest output were 100% noise per the
unified-KB audit. RawConcept moved to mindforge/distillation/raw.py
in an earlier commit; nothing else here is consumed elsewhere."
```

---

## Task 13: Documentation updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `eval/README.md`
- Modify: `docs/ARCHITECTURE.md`

- [ ] **Step 1: Update `CLAUDE.md`**

Find the line:

```
    mindforge eval --mode heuristic
```

Replace with:

```
    mindforge eval --mode mock
```

No other changes to CLAUDE.md required (the mypy-relaxation list does not mention `mindforge.ingestion.extractor`; `ingestion.incremental` reference is unaffected).

- [ ] **Step 2: Update `README.md` Modes section**

Open `README.md`, locate any section describing extraction modes / `--llm` flag. Replace mentions of:
- `--llm` flag → remove (LLM is always on)
- "heuristic mode" / "fallback mode" → describe as removed
- Add a Modes subsection:

```markdown
### Extraction modes

`mindforge ingest` always runs LLM extraction. The provider is selected via
`llm.provider` in `~/.config/mindforge/config.yaml` (or `--llm-provider`):

- `ollama` (default): local LLM via Ollama; requires a reachable endpoint.
- `openai`: OpenAI-compatible HTTP API; requires `llm.api_key`.
- `mock`: deterministic content-derivative test client; no network. Use
  this for fast pipeline tests in CI and local development.

> **Mock and real runs cannot share an output directory.** The pipeline
> refuses to mix them at startup. Use a separate `output_dir` for mock-mode
> test runs (convention: `<your-kb>/mock-test/` or a tempdir).
```

If the README has no "Modes" section yet, add it under Usage / Configuration.

- [ ] **Step 3: Update `eval/README.md`**

Replace any `--mode heuristic` with `--mode mock`. Add explicit framing:

```markdown
### Eval modes

- `mindforge eval --mode mock` (default) — deterministic smoke test. Verifies
  the pipeline runs end-to-end and the scorer produces a structurally valid
  report. Recall/precision/phrase-grounding numbers are not asserted in this
  mode; the mock client's content is not the gate.
- `mindforge eval --mode llm` — real-LLM quality gate. Runs against a
  configured LLM endpoint; recall/precision are meaningful. Run separately
  from per-PR CI.
```

- [ ] **Step 4: Update `docs/ARCHITECTURE.md`**

Locate the extraction-layer description. Replace any "heuristic + LLM" framing with:

> **Extraction:** A single LLM-driven extractor (`mindforge.llm.extractor`)
> consumes chunks and emits `RawConcept` candidates. The provider is
> pluggable via `make_llm_client(config)` in `mindforge.llm.client` —
> `ollama`, `openai`, or `mock` (deterministic test client). The mock
> provider is used for CI and pipeline tests; the real providers handle
> production ingest.

If the architecture doc has a diagram or flow, ensure it shows a single
extraction lane, not a heuristic-plus-LLM merge.

- [ ] **Step 5: Verify docs render reasonably**

Run: `grep -rn "heuristic" CLAUDE.md README.md eval/README.md docs/ARCHITECTURE.md`

Expected: no remaining references that imply heuristic extraction is supported. Any remaining occurrences should be in historical context only (e.g., changelog entries) — those are fine.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md eval/README.md docs/ARCHITECTURE.md
git commit -m "docs: update for heuristic removal and mock provider

- CLAUDE.md: eval example uses --mode mock.
- README.md: Modes section describes ollama/openai/mock + KB pollution
  warning; --llm flag mentions removed.
- eval/README.md: mock = smoke test, llm = quality gate.
- ARCHITECTURE.md: extraction layer is single-lane, mock-pluggable."
```

---

## Task 14: Final verification

- [ ] **Step 1: Run pytest with verbose output**

Run: `pytest`
Expected: all tests pass; no skips or warnings related to this work.

- [ ] **Step 2: Run linters and type-check**

Run:
```bash
ruff check . && ruff format --check . && mypy mindforge
```
Expected: clean. If formatting issues, fix with `ruff format .`.

- [ ] **Step 3: Run the eval suite as a smoke test**

Run: `mindforge eval --mode mock`
Expected: completes; report written to `eval/reports/<timestamp>.json`.

- [ ] **Step 4: Run the binaries-style sanity check**

Run: `python -m mindforge --help`
Expected: shows the subcommand list; no `--llm` reference for ingest.

Run: `python -m mindforge ingest --help | grep -A1 '\-\-llm-provider'`
Expected: `--llm-provider {ollama,openai,mock}`.

- [ ] **Step 5: Confirm the heuristic module is gone**

Run: `find mindforge -name "extractor.py" | xargs ls -la`
Expected: only `mindforge/llm/extractor.py` (the LLM extractor stays).

- [ ] **Step 6: Confirm `RawConcept` is at its new home**

Run: `grep -rn "from mindforge.distillation.raw import RawConcept" mindforge/ tests/ | wc -l`
Expected: ≥ 5 (every consumer imports from the new location).

- [ ] **Step 7: Final commit if anything was tweaked**

If any of the verification steps caused you to fix a small issue (formatting, etc.), stage and commit:

```bash
git add -A
git commit -m "chore: final verification cleanups for heuristic removal"
```

If nothing needed fixing, skip this step.

- [ ] **Step 8: Create the PR**

Push the branch and open a PR with the body summarizing the breaking changes:

```
## Summary
- Removes the heuristic concept extractor entirely.
- LLM is always on; --llm flag rejected.
- Adds MockLLMClient (selectable via llm.provider: mock) for fast pipeline
  tests; replaces the de-facto "use heuristic for tests" pattern.
- Adds a KB pollution guard so mock and real runs cannot share an output dir.

## Breaking changes
- `mindforge ingest --llm` is rejected (argparse).
- `mindforge eval --mode heuristic` is rejected; use `--mode mock`.
- `MindForgeConfig.use_llm` removed; use `llm_provider`.
- Existing KBs: continue to work; first ingest after upgrade tags the
  manifest with the current provider.

## Test plan
- [ ] pytest passes
- [ ] ruff + mypy clean
- [ ] mindforge eval --mode mock completes
- [ ] mindforge eval --mode llm completes against a real endpoint (run manually)
```
