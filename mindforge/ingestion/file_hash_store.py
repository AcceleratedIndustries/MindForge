"""File-hash manifest for incremental ingestion.

Persists per-file SHA-256 hashes to ``output/.ingest/content_hashes.json``
so the pipeline can skip re-parsing/re-extracting unchanged transcripts on
subsequent runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


class ContentHasher:
    """SHA-256 content hashing for file change detection."""

    def hash_file(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def hash_bytes(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def hash_string(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass
class FileStatus:
    path: Path
    current_hash: str
    previous_hash: str | None
    is_new: bool = False
    is_modified: bool = False
    is_unchanged: bool = False


class FileHashStore:
    """Persistent per-file SHA-256 manifest.

    Keys in the on-disk JSON are normalized to be relative to ``transcripts_dir``
    when possible, so the cache survives moving the project directory.
    """

    HASH_FILE_NAME = "content_hashes.json"

    def __init__(
        self,
        ingest_dir: Path,
        transcripts_dir: Path,
        hashes: dict[str, str] | None = None,
    ) -> None:
        self.ingest_dir = Path(ingest_dir)
        self.transcripts_dir = Path(transcripts_dir).resolve()
        self.hasher = ContentHasher()
        self._hashes: dict[str, str] = dict(hashes or {})

    @classmethod
    def load(cls, ingest_dir: Path, transcripts_dir: Path) -> FileHashStore:
        path = Path(ingest_dir) / cls.HASH_FILE_NAME
        data: dict[str, str] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(
                    f"warning: {path} is corrupted ({exc}); treating as empty cache",
                    file=sys.stderr,
                )
        return cls(ingest_dir=ingest_dir, transcripts_dir=transcripts_dir, hashes=data)

    def save(self) -> None:
        self.ingest_dir.mkdir(parents=True, exist_ok=True)
        path = self.ingest_dir / self.HASH_FILE_NAME
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._hashes, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def _key(self, file_path: Path) -> str:
        resolved = Path(file_path).resolve()
        try:
            return str(resolved.relative_to(self.transcripts_dir))
        except ValueError:
            return str(resolved)

    def status_of(self, file_path: Path) -> FileStatus:
        current = self.hasher.hash_file(file_path)
        previous = self._hashes.get(self._key(file_path))
        status = FileStatus(
            path=Path(file_path),
            current_hash=current,
            previous_hash=previous,
        )
        if previous is None:
            status.is_new = True
        elif previous == current:
            status.is_unchanged = True
        else:
            status.is_modified = True
        return status

    def update(self, file_path: Path, hash_value: str) -> None:
        self._hashes[self._key(file_path)] = hash_value

    def forget(self, file_path: Path) -> None:
        self._hashes.pop(self._key(file_path), None)

    def known_paths(self) -> set[Path]:
        """Return absolute paths corresponding to keys in the store."""
        out: set[Path] = set()
        for key in self._hashes:
            p = Path(key)
            if p.is_absolute():
                out.add(p)
            else:
                out.add((self.transcripts_dir / p).resolve())
        return out
