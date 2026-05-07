"""Shared MindForge config file (~/.config/mindforge/config.yaml).

Resolution order (highest priority wins):
1. CLI flags
2. Config file
3. Hard-coded dataclass defaults

Override location: set MINDFORGE_CONFIG to point at a different YAML.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def default_config_path() -> Path:
    override = os.environ.get("MINDFORGE_CONFIG")
    if override:
        return Path(override)
    return Path.home() / ".config" / "mindforge" / "config.yaml"


@dataclass
class LLMConfigSection:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:30b-a3b"
    summarize_model: str = ""  # empty = use the same as `model`
    keep_alive: int | str = -1
    timeout: int = 120
    api_key: str = ""
    # When False, reasoning models (qwen3, deepseek-r1, gpt-oss, ...) skip
    # the chain-of-thought phase. Roughly 5x faster on structured-extraction
    # workloads. None = leave at server default. Ollama-only.
    think: bool | None = None


@dataclass
class EmbeddingsConfigSection:
    provider: str = "sentence-transformers"
    base_url: str = ""
    model: str = ""
    api_key: str = ""


@dataclass
class RetrievalConfigSection:
    weights: dict[str, float] = field(
        default_factory=lambda: {"keyword": 0.4, "semantic": 0.4, "graph": 0.2}
    )
    seed_pool_size: int = 10
    walk_depth: int = 2


@dataclass
class ConfigFile:
    llm: LLMConfigSection = field(default_factory=LLMConfigSection)
    embeddings: EmbeddingsConfigSection = field(default_factory=EmbeddingsConfigSection)
    retrieval: RetrievalConfigSection = field(default_factory=RetrievalConfigSection)


DEFAULT_CONFIG = ConfigFile()


def load_config(path: Path | None = None) -> ConfigFile:
    """Load YAML from ``path`` (or the default location). Missing file → defaults."""
    p = path if path is not None else default_config_path()
    if not p.exists():
        return ConfigFile()
    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid config file at {p}: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config file at {p}: top-level must be a mapping")
    cfg = ConfigFile()
    if isinstance(raw.get("llm"), dict):
        for k, v in raw["llm"].items():
            if hasattr(cfg.llm, k):
                setattr(cfg.llm, k, v)
    if isinstance(raw.get("embeddings"), dict):
        for k, v in raw["embeddings"].items():
            if hasattr(cfg.embeddings, k):
                setattr(cfg.embeddings, k, v)
    if isinstance(raw.get("retrieval"), dict):
        retrieval = raw["retrieval"]
        if isinstance(retrieval.get("weights"), dict):
            cfg.retrieval.weights = dict(retrieval["weights"])
        for k in ("seed_pool_size", "walk_depth"):
            if k in retrieval:
                setattr(cfg.retrieval, k, retrieval[k])
    return cfg


def merge_with_overrides(cfg: ConfigFile, **overrides: Any) -> ConfigFile:
    """Apply CLI-level overrides. Keys named ``llm_model`` map to ``cfg.llm.model``.

    A value of ``None`` or empty string is treated as 'not provided' and skipped,
    so only explicit CLI flags override file/dataclass defaults.
    """
    for key, value in overrides.items():
        if value is None or value == "":
            continue
        if "_" not in key:
            continue
        section, field_name = key.split("_", 1)
        section_obj = getattr(cfg, section, None)
        if section_obj is None:
            continue
        if hasattr(section_obj, field_name):
            setattr(section_obj, field_name, value)
    return cfg
