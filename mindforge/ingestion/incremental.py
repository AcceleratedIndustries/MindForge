"""Incremental ingestion with content hashing for MindForge.

Enables efficient updates without full rebuilds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from mindforge.ingestion.extractor import RawConcept


@dataclass
class FileStatus:
    """Status of a file for incremental processing."""

    path: Path
    is_new: bool = False
    is_modified: bool = False
    is_unchanged: bool = False
    previous_hash: str | None = None
    current_hash: str | None = None


@dataclass
class IncrementalResult:
    """Result of an incremental ingestion run."""

    files_processed: int = 0
    files_skipped: int = 0
    concepts_extracted: int = 0
    concepts_updated: int = 0
    embeddings_reused: int = 0
    embeddings_computed: int = 0


class ContentHasher:
    """SHA-256 content hashing for file change detection."""

    def hash_file(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file content."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def hash_bytes(self, content: bytes) -> str:
        """Hash raw bytes."""
        return hashlib.sha256(content).hexdigest()

    def hash_string(self, content: str) -> str:
        """Hash a string."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


class IncrementalIngest:
    """Manages incremental ingestion state and operations."""

    def __init__(self, kb_dir: Path):
        self.kb_dir = Path(kb_dir)
        self.ingest_dir = self.kb_dir / ".ingest"
        self.hashes_file = self.ingest_dir / "content_hashes.json"
        self.embeddings_cache_file = self.ingest_dir / "embeddings_cache.json"
        self.deleted_file = self.ingest_dir / "deleted_concepts.json"
        self.concepts_meta_file = self.ingest_dir / "concepts_meta.json"

        self._ensure_structure()
        self.hasher = ContentHasher()

        # In-memory caches
        self._hashes_cache: dict[str, str] = {}
        self._embeddings_cache: dict[str, Any] = {}
        self._concepts_meta: dict[str, dict] = {}
        self._deleted: dict[str, dict] = {}

        self._load_state()

    def _ensure_structure(self) -> None:
        """Create .ingest/ directory structure."""
        self.ingest_dir.mkdir(parents=True, exist_ok=True)
        (self.ingest_dir / "embeddings").mkdir(exist_ok=True)

        # Initialize empty state files
        if not self.hashes_file.exists():
            self.hashes_file.write_text("{}")
        if not self.embeddings_cache_file.exists():
            self.embeddings_cache_file.write_text("{}")
        if not self.deleted_file.exists():
            self.deleted_file.write_text("{}")
        if not self.concepts_meta_file.exists():
            self.concepts_meta_file.write_text("{}")

    def _load_state(self) -> None:
        """Load persisted state."""
        if self.hashes_file.exists():
            with open(self.hashes_file) as f:
                self._hashes_cache = json.load(f)

        if self.embeddings_cache_file.exists():
            with open(self.embeddings_cache_file) as f:
                self._embeddings_cache = json.load(f)

        if self.concepts_meta_file.exists():
            with open(self.concepts_meta_file) as f:
                self._concepts_meta = json.load(f)

        if self.deleted_file.exists():
            with open(self.deleted_file) as f:
                self._deleted = json.load(f)

    def _save_state(self) -> None:
        """Persist state to disk."""
        with open(self.hashes_file, "w") as f:
            json.dump(self._hashes_cache, f, indent=2)

        with open(self.embeddings_cache_file, "w") as f:
            json.dump(self._embeddings_cache, f, indent=2)

        with open(self.concepts_meta_file, "w") as f:
            json.dump(self._concepts_meta, f, indent=2)

        with open(self.deleted_file, "w") as f:
            json.dump(self._deleted, f, indent=2)

    # === Hash Management ===

    def hash_file(self, file_path: Path) -> str:
        """Compute current hash of file."""
        return self.hasher.hash_file(file_path)

    def store_hash(self, file_path: str, hash_value: str) -> None:
        """Store hash for a file."""
        self._hashes_cache[file_path] = hash_value
        self._save_state()

    def get_hash(self, file_path: str) -> str | None:
        """Retrieve stored hash for a file."""
        return self._hashes_cache.get(file_path)

    def get_file_status(self, file_path: Path) -> FileStatus:
        """Determine if file is new, modified, or unchanged."""
        file_path = Path(file_path)
        path_str = str(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        current_hash = self.hash_file(file_path)
        previous_hash = self.get_hash(path_str)

        status = FileStatus(
            path=file_path,
            current_hash=current_hash,
            previous_hash=previous_hash,
        )

        if previous_hash is None:
            status.is_new = True
        elif previous_hash == current_hash:
            status.is_unchanged = True
        else:
            status.is_modified = True

        return status

    def get_files_to_process(self, file_paths: list[Path]) -> list[Path]:
        """Filter to only files that need processing."""
        to_process = []
        for path in file_paths:
            status = self.get_file_status(path)
            if status.is_new or status.is_modified:
                to_process.append(path)
        return to_process

    # === Embedding Cache ===

    def store_embedding(self, concept_id: str, content_hash: str, embedding: list[float]) -> None:
        """Store embedding for a concept (keyed by content hash)."""
        key = f"{concept_id}:{content_hash}"
        self._embeddings_cache[key] = embedding
        self._save_state()

    def get_embedding(self, concept_id: str, content_hash: str) -> list[float] | None:
        """Retrieve cached embedding if content hash matches."""
        key = f"{concept_id}:{content_hash}"
        return self._embeddings_cache.get(key)

    def invalidate_embeddings(self, concept_id: str) -> None:
        """Remove all cached embeddings for a concept."""
        keys_to_remove = [k for k in self._embeddings_cache if k.startswith(f"{concept_id}:")]
        for k in keys_to_remove:
            del self._embeddings_cache[k]
        self._save_state()

    # === Concept Upsert ===

    def upsert_concept(self, concept: RawConcept, source_hash: str) -> None:
        """Insert or update a concept with source tracking."""
        now = datetime.now().isoformat()
        concept_name = concept.name

        # Set hash on concept object
        concept.source_hash = source_hash

        existing = self._concepts_meta.get(concept_name)

        if existing:
            # Update
            self._concepts_meta[concept_name].update(
                {
                    "raw_content": concept.raw_content,
                    "source_hash": source_hash,
                    "updated_at": now,
                    "extraction_method": concept.extraction_method,
                    "confidence": concept.confidence,
                    "source_chunks": concept.source_chunks,
                    "source_files": concept.source_files,
                }
            )
        else:
            # Insert
            self._concepts_meta[concept_name] = {
                "name": concept_name,
                "raw_content": concept.raw_content,
                "source_hash": source_hash,
                "created_at": now,
                "updated_at": now,
                "extraction_method": concept.extraction_method,
                "confidence": concept.confidence,
                "source_chunks": concept.source_chunks,
                "source_files": concept.source_files,
            }

        self._save_state()

    def get_concept_by_name(self, name: str) -> RawConcept | None:
        """Retrieve concept metadata."""
        meta = self._concepts_meta.get(name)
        if not meta:
            return None

        return RawConcept(
            name=meta["name"],
            raw_content=meta["raw_content"],
            source_chunks=meta.get("source_chunks", []),
            source_files=meta.get("source_files", []),
            extraction_method=meta.get("extraction_method", "unknown"),
            confidence=meta.get("confidence", 0.5),
            source_hash=meta.get("source_hash", ""),
        )

    # === Soft Delete ===

    def mark_deleted(self, concept_name: str) -> None:
        """Soft-delete a concept."""
        now = datetime.now().isoformat()
        self._deleted[concept_name] = {
            "name": concept_name,
            "deleted_at": now,
        }

        # Also mark in concepts meta
        if concept_name in self._concepts_meta:
            self._concepts_meta[concept_name]["deleted"] = True
            self._concepts_meta[concept_name]["deleted_at"] = now

        self._save_state()

    def get_active_concepts(self) -> list[RawConcept]:
        """Get non-deleted concepts."""
        active = []
        for name, meta in self._concepts_meta.items():
            if not meta.get("deleted", False) and name not in self._deleted:
                active.append(self._concept_from_meta(meta))
        return active

    def get_all_concepts(self, include_deleted: bool = False) -> list[RawConcept]:
        """Get all concepts, optionally including deleted."""
        concepts = []
        for name, meta in self._concepts_meta.items():
            if include_deleted or not meta.get("deleted", False):
                concepts.append(self._concept_from_meta(meta))
        return concepts

    def _concept_from_meta(self, meta: dict) -> RawConcept:
        """Convert metadata to RawConcept."""
        return RawConcept(
            name=meta["name"],
            raw_content=meta["raw_content"],
            source_chunks=meta.get("source_chunks", []),
            source_files=meta.get("source_files", []),
            extraction_method=meta.get("extraction_method", "unknown"),
            confidence=meta.get("confidence", 0.5),
            source_hash=meta.get("source_hash", ""),
        )

    def gc_deleted(self, older_than_days: int = 30) -> int:
        """Garbage collect deleted concepts older than threshold.

        Returns: number of concepts permanently removed.
        """
        cutoff = datetime.now() - timedelta(days=older_than_days)
        to_remove = []

        for name, meta in self._deleted.items():
            deleted_at = datetime.fromisoformat(meta["deleted_at"])
            if deleted_at < cutoff:
                to_remove.append(name)

        for name in to_remove:
            del self._deleted[name]
            if name in self._concepts_meta:
                del self._concepts_meta[name]
            # Also clean up embeddings
            self.invalidate_embeddings(name)

        if to_remove:
            self._save_state()

        return len(to_remove)

    # === Pipeline Integration ===

    def run(self, source_files: list[Path]) -> IncrementalResult:
        """Run incremental ingestion on source files.

        This is a simplified implementation. Full pipeline would integrate
        with the existing MindForgePipeline.
        """
        result = IncrementalResult()

        # Check each file
        for file_path in source_files:
            status = self.get_file_status(file_path)

            if status.is_unchanged:
                result.files_skipped += 1
            else:
                result.files_processed += 1
                # Extract concepts from file
                file_concepts = self._extract_from_file(file_path)
                for concept in file_concepts:
                    self.upsert_concept(concept, status.current_hash or "")
                    result.concepts_extracted += 1

                # Update the hash
                self.store_hash(str(file_path), status.current_hash or self.hash_file(file_path))

        return result

    def _extract_from_file(self, file_path: Path) -> list[RawConcept]:
        """Extract concepts from a single file.

        This is a minimal implementation. Full version would use
        the full extraction pipeline.
        """
        concepts = []
        content = file_path.read_text()

        # Simple extraction: look for heading patterns
        import re

        heading_pattern = re.compile(r"^#{1,4}\s+(.{3,60})$", re.MULTILINE)
        definition_pattern = re.compile(
            r"(?:^|\.\s+)([A-Z][\w\s]{1,40}?)\s+(?:is|are)\s+(?:a|an|the)\s+(.{20,300}?)[.!]",
            re.MULTILINE,
        )

        # Extract headings
        for match in heading_pattern.finditer(content):
            name = match.group(1).strip()
            # Get context after heading
            context_start = match.end()
            context_end = min(context_start + 500, len(content))
            context = content[context_start:context_end].strip()

            concept = RawConcept(
                name=name,
                raw_content=context,
                source_files=[str(file_path)],
                extraction_method="heading",
                confidence=0.7,
            )
            concepts.append(concept)

        # Extract definitions
        for match in definition_pattern.finditer(content):
            name = match.group(1).strip()
            definition = match.group(2).strip()

            # Skip if too generic
            if len(name) < 3 or name.lower() in {"the", "this", "that", "for"}:
                continue

            concept = RawConcept(
                name=name,
                raw_content=definition,
                source_files=[str(file_path)],
                extraction_method="definition_pattern",
                confidence=0.8,
            )
            concepts.append(concept)

        return concepts
