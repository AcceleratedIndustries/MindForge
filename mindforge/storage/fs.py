"""Minimal Storage protocol + filesystem implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Storage(Protocol):
    """A minimal blob-by-path read/write contract."""

    def write_text(self, path: Path, text: str) -> None: ...
    def read_text(self, path: Path) -> str: ...
    def exists(self, path: Path) -> bool: ...


class FilesystemStorage:
    """Write and read text files on the local filesystem."""

    def write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def exists(self, path: Path) -> bool:
        return path.exists()
