"""Tests for the manifest.json history-snapshot writer."""

from __future__ import annotations

import json
from pathlib import Path

from mindforge.distillation.concept import Concept, ConceptStore
from mindforge.pipeline import read_manifest_history, write_manifest_snapshot


def test_write_snapshot_creates_history_file(tmp_path: Path) -> None:
    store = ConceptStore()
    store.add(Concept(name="A", definition="d", explanation="e"))
    manifest_path = tmp_path / "manifest.json"
    write_manifest_snapshot(store, manifest_path)

    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert "history" in data
    assert len(data["history"]) == 1
    snap = data["history"][0]
    assert "timestamp" in snap
    assert snap["slug_hashes"]["a"]


def test_multiple_snapshots_accumulate(tmp_path: Path) -> None:
    store = ConceptStore()
    store.add(Concept(name="A", definition="d", explanation="e"))
    manifest_path = tmp_path / "manifest.json"
    write_manifest_snapshot(store, manifest_path)

    store.add(Concept(name="B", definition="d", explanation="e"))
    write_manifest_snapshot(store, manifest_path)

    history = read_manifest_history(manifest_path)
    assert len(history) == 2


def test_read_history_missing_file(tmp_path: Path) -> None:
    assert read_manifest_history(tmp_path / "nope.json") == []
