"""Tests for FileHashStore and ContentHasher."""

from __future__ import annotations

import json
from pathlib import Path

from mindforge.ingestion.file_hash_store import ContentHasher, FileHashStore


class TestContentHasher:
    def test_hash_string_is_stable(self) -> None:
        h = ContentHasher()
        assert h.hash_string("hello") == h.hash_string("hello")

    def test_hash_string_differs_for_different_input(self) -> None:
        h = ContentHasher()
        assert h.hash_string("hello") != h.hash_string("world")

    def test_hash_bytes_matches_hash_string_for_utf8(self) -> None:
        h = ContentHasher()
        assert h.hash_bytes(b"hello") == h.hash_string("hello")

    def test_hash_file_matches_hash_bytes(self, tmp_path: Path) -> None:
        h = ContentHasher()
        f = tmp_path / "a.txt"
        f.write_bytes(b"hello world")
        assert h.hash_file(f) == h.hash_bytes(b"hello world")


class TestFileHashStore:
    def test_new_file_status(self, tmp_path: Path) -> None:
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        f = transcripts / "a.md"
        f.write_text("hello")

        store = FileHashStore.load(tmp_path / ".ingest", transcripts)
        status = store.status_of(f)
        assert status.is_new is True
        assert status.is_modified is False
        assert status.is_unchanged is False
        assert status.current_hash is not None
        assert status.previous_hash is None

    def test_unchanged_file_status_after_update(self, tmp_path: Path) -> None:
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        f = transcripts / "a.md"
        f.write_text("hello")

        store = FileHashStore.load(tmp_path / ".ingest", transcripts)
        store.update(f, store.hasher.hash_file(f))
        status = store.status_of(f)
        assert status.is_unchanged is True

    def test_modified_file_status(self, tmp_path: Path) -> None:
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        f = transcripts / "a.md"
        f.write_text("hello")

        store = FileHashStore.load(tmp_path / ".ingest", transcripts)
        store.update(f, store.hasher.hash_file(f))
        f.write_text("hello world")
        status = store.status_of(f)
        assert status.is_modified is True

    def test_known_paths_returns_stored_relative_paths(self, tmp_path: Path) -> None:
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        f = transcripts / "a.md"
        f.write_text("x")

        store = FileHashStore.load(tmp_path / ".ingest", transcripts)
        store.update(f, "abc")
        assert (transcripts / "a.md").resolve() in store.known_paths()

    def test_forget_removes_path(self, tmp_path: Path) -> None:
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        f = transcripts / "a.md"
        f.write_text("x")

        store = FileHashStore.load(tmp_path / ".ingest", transcripts)
        store.update(f, "abc")
        store.forget(f)
        assert f.resolve() not in store.known_paths()

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        f = transcripts / "a.md"
        f.write_text("hello")

        ingest_dir = tmp_path / ".ingest"
        store = FileHashStore.load(ingest_dir, transcripts)
        store.update(f, store.hasher.hash_file(f))
        store.save()

        reloaded = FileHashStore.load(ingest_dir, transcripts)
        assert reloaded.status_of(f).is_unchanged is True

    def test_load_missing_file_returns_empty_store(self, tmp_path: Path) -> None:
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        store = FileHashStore.load(tmp_path / ".ingest", transcripts)
        assert store.known_paths() == set()

    def test_path_normalization_relative_when_under_transcripts_dir(self, tmp_path: Path) -> None:
        """Stored keys are relative to transcripts_dir so the cache is portable."""
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        f = transcripts / "subdir" / "a.md"
        f.parent.mkdir()
        f.write_text("x")

        ingest_dir = tmp_path / ".ingest"
        store = FileHashStore.load(ingest_dir, transcripts)
        store.update(f, "abc")
        store.save()

        raw = json.loads((ingest_dir / "content_hashes.json").read_text())
        assert "subdir/a.md" in raw or str(Path("subdir/a.md")) in raw
