"""Tests for shared config file."""

from pathlib import Path

import pytest

from mindforge.config_file import (
    DEFAULT_CONFIG,
    ConfigFile,
    default_config_path,
    load_config,
    merge_with_overrides,
)


def test_default_config_loads_when_no_file(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg.llm.provider == "ollama"
    assert cfg.retrieval.weights["keyword"] == 0.4


def test_default_config_constant_matches_factory() -> None:
    assert DEFAULT_CONFIG.llm.provider == "ollama"
    assert DEFAULT_CONFIG.retrieval.weights == {
        "keyword": 0.4,
        "semantic": 0.4,
        "graph": 0.2,
    }


def test_config_file_overrides_defaults(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text(
        "llm:\n  provider: openai\n  model: gpt-4\n"
        "retrieval:\n  weights: {keyword: 0.5, semantic: 0.3, graph: 0.2}\n"
    )
    cfg = load_config(p)
    assert cfg.llm.provider == "openai"
    assert cfg.llm.model == "gpt-4"
    assert cfg.retrieval.weights["keyword"] == 0.5


def test_cli_overrides_win_over_config_file(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text("llm:\n  provider: ollama\n  model: qwen3:30b-a3b\n")
    cfg = load_config(p)
    merged = merge_with_overrides(cfg, llm_model="llama3.2")
    assert merged.llm.model == "llama3.2"
    # untouched fields stay put
    assert merged.llm.provider == "ollama"


def test_cli_override_skips_none_and_empty(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text("llm:\n  model: qwen3:30b-a3b\n")
    cfg = load_config(p)
    merged = merge_with_overrides(cfg, llm_model=None, llm_provider="")
    assert merged.llm.model == "qwen3:30b-a3b"
    assert merged.llm.provider == "ollama"


def test_invalid_yaml_raises_clear_error(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text("not valid: : yaml: : :")
    with pytest.raises(ValueError, match="Invalid config file"):
        load_config(p)


def test_load_returns_configfile_instance(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "nope.yaml")
    assert isinstance(cfg, ConfigFile)


def test_default_config_path_uses_env_override(monkeypatch, tmp_path: Path) -> None:
    custom = tmp_path / "custom.yaml"
    monkeypatch.setenv("MINDFORGE_CONFIG", str(custom))
    assert default_config_path() == custom


def test_default_config_path_falls_back_to_home(monkeypatch) -> None:
    monkeypatch.delenv("MINDFORGE_CONFIG", raising=False)
    p = default_config_path()
    assert p.name == "config.yaml"
    assert p.parent.name == "mindforge"


def test_merge_overrides_retrieval_weights_dict(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "x.yaml")
    merged = merge_with_overrides(
        cfg, retrieval_weights={"keyword": 0.5, "semantic": 0.3, "graph": 0.2}
    )
    assert merged.retrieval.weights["keyword"] == 0.5


def test_unknown_yaml_keys_are_ignored(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text(
        "llm:\n  provider: ollama\n  bogus_field: 42\nmade_up_section:\n  whatever: true\n"
    )
    cfg = load_config(p)
    assert cfg.llm.provider == "ollama"
    assert not hasattr(cfg.llm, "bogus_field")
