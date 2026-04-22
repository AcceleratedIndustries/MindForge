"""Tests for centralized path resolution."""

from __future__ import annotations

from pathlib import Path

from mindforge.paths import MindForgePaths, resolve_root


def test_default_root_is_user_mindforge(monkeypatch):
    monkeypatch.delenv("MINDFORGE_ROOT", raising=False)
    monkeypatch.delenv("MINDFORGE_CONFIG", raising=False)
    paths = MindForgePaths.resolve()
    assert paths.root == (Path.home() / ".mindforge").resolve()


def test_env_var_overrides_default(monkeypatch, tmp_path):
    monkeypatch.setenv("MINDFORGE_ROOT", str(tmp_path))
    paths = MindForgePaths.resolve()
    assert paths.root == tmp_path.resolve()


def test_explicit_root_wins_over_env(monkeypatch, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("MINDFORGE_ROOT", str(tmp_path))
    paths = MindForgePaths.resolve(explicit_root=other)
    assert paths.root == other.resolve()


def test_derived_paths(tmp_path):
    paths = MindForgePaths.resolve(explicit_root=tmp_path)
    assert paths.kbs_dir == (tmp_path / "kbs").resolve()
    assert paths.trash_dir == (tmp_path / "trash").resolve()
    assert paths.registry_file == (tmp_path / "registry.json").resolve()


def test_config_file_path_respects_env(monkeypatch, tmp_path):
    custom_config = tmp_path / "config.yaml"
    monkeypatch.setenv("MINDFORGE_CONFIG", str(custom_config))
    paths = MindForgePaths.resolve()
    assert paths.config_file == custom_config.resolve()


def test_ensure_dirs_creates_structure(tmp_path):
    paths = MindForgePaths.resolve(explicit_root=tmp_path)
    paths.ensure_dirs()
    assert paths.kbs_dir.is_dir()
    assert paths.trash_dir.is_dir()


def test_resolve_root_env_expansion(monkeypatch):
    monkeypatch.setenv("MINDFORGE_ROOT", "~/custom-mindforge")
    resolved = resolve_root()
    assert resolved == (Path.home() / "custom-mindforge").resolve()
