"""Centralized path resolution for MindForge.

Precedence for the root directory:
    1. Explicit (passed by caller, e.g. CLI --root)
    2. Env var MINDFORGE_ROOT
    3. Default ~/.mindforge

The optional config file path is $MINDFORGE_CONFIG or <root>/config.yaml.

Stdlib only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _expand(p: str | Path) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(p)))).resolve()


def resolve_root(explicit: str | Path | None = None) -> Path:
    """Return the MindForge root directory per the documented precedence."""
    if explicit is not None:
        return _expand(explicit)
    env = os.environ.get("MINDFORGE_ROOT")
    if env:
        return _expand(env)
    return _expand("~/.mindforge")


def resolve_config_file(root: Path | None = None) -> Path:
    """Return the path to the optional config file."""
    env = os.environ.get("MINDFORGE_CONFIG")
    if env:
        return _expand(env)
    r = root if root is not None else resolve_root()
    return r / "config.yaml"


@dataclass(frozen=True)
class MindForgePaths:
    """Resolved paths for a MindForge installation."""

    root: Path
    kbs_dir: Path
    trash_dir: Path
    registry_file: Path
    config_file: Path

    @classmethod
    def resolve(cls, explicit_root: str | Path | None = None) -> "MindForgePaths":
        root = resolve_root(explicit_root)
        return cls(
            root=root,
            kbs_dir=(root / "kbs").resolve(),
            trash_dir=(root / "trash").resolve(),
            registry_file=(root / "registry.json").resolve(),
            config_file=resolve_config_file(root),
        )

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.kbs_dir.mkdir(parents=True, exist_ok=True)
        self.trash_dir.mkdir(parents=True, exist_ok=True)
