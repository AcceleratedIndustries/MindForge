"""Manifest snapshot/history tests focused on the provider field."""

from __future__ import annotations

import json
from pathlib import Path

from mindforge.distillation.concept import ConceptStore
from mindforge.pipeline import read_manifest_history, write_manifest_snapshot


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
