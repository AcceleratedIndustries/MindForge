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
        # No manifest.json on disk; any provider proceeds.
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
        _write_manifest(
            manifest,
            [
                {"timestamp": "older", "provider": "mock", "slug_hashes": {}},
                {"timestamp": "newer", "provider": "ollama", "slug_hashes": {}},
            ],
        )
        check_kb_provider_compat(manifest, current_provider="ollama")
        with pytest.raises(RuntimeError):
            check_kb_provider_compat(manifest, current_provider="mock")


class TestPipelineWiring:
    def test_pipeline_run_refuses_mock_on_real_kb(self, tmp_path: Path) -> None:
        """Pipeline.run() must hit the guard before doing extraction work."""
        from mindforge.config import MindForgeConfig
        from mindforge.pipeline import MindForgePipeline

        # Create a tiny "real" KB by hand: manifest with provider="ollama".
        out = tmp_path / "out"
        out.mkdir()
        _write_manifest(
            out / "manifest.json",
            [
                {"timestamp": "...", "provider": "ollama", "slug_hashes": {}},
            ],
        )

        # Now try to run with provider=mock pointed at the same dir.
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        (transcripts / "x.md").write_text("Human: test\n\nAssistant: ok", encoding="utf-8")

        cfg = MindForgeConfig(
            transcripts_dir=transcripts,
            output_dir=out,
            llm_provider="mock",
            use_llm=True,
        )
        pipe = MindForgePipeline(cfg)
        with pytest.raises(RuntimeError, match="last built with provider 'ollama'"):
            pipe.run()
