"""Tests for the Storage protocol + FilesystemStorage implementation."""

from __future__ import annotations

from pathlib import Path

from mindforge.storage import FilesystemStorage, Storage


def test_filesystem_storage_satisfies_protocol():
    storage = FilesystemStorage()
    assert isinstance(storage, Storage)


def test_write_and_read_roundtrip(tmp_path: Path):
    storage = FilesystemStorage()
    target = tmp_path / "subdir" / "note.txt"
    storage.write_text(target, "hello")
    assert storage.exists(target)
    assert storage.read_text(target) == "hello"


def test_write_creates_parent_dirs(tmp_path: Path):
    storage = FilesystemStorage()
    target = tmp_path / "a" / "b" / "c.txt"
    storage.write_text(target, "x")
    assert target.is_file()


def test_exists_false_for_missing_file(tmp_path: Path):
    storage = FilesystemStorage()
    assert storage.exists(tmp_path / "nope.txt") is False
