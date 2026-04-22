"""Tests for incremental ingestion with content hashing.

Key behaviors:
1. Content hashing tracks which files are unchanged
2. Incremental runs only process new or modified files
3. Embeddings are cached and reused for unchanged content
4. Soft-deleted concepts are preserved but excluded from search
"""

from __future__ import annotations

import hashlib
from datetime import datetime

import pytest

from mindforge.ingestion.extractor import RawConcept
from mindforge.ingestion.incremental import ContentHasher, IncrementalIngest


class TestContentHasher:
    """Test content hashing for change detection."""

    def test_hash_file_content(self, tmp_path):
        """SHA-256 hash of file content should be deterministic."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        hasher = ContentHasher()
        hash1 = hasher.hash_file(test_file)
        hash2 = hasher.hash_file(test_file)

        # Same content = same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length

    def test_hash_changes_with_content(self, tmp_path):
        """Different content should produce different hashes."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Version 1")

        hasher = ContentHasher()
        hash1 = hasher.hash_file(test_file)

        # Modify content
        test_file.write_text("Version 2")
        hash2 = hasher.hash_file(test_file)

        assert hash1 != hash2

    def test_hash_ignores_mtime(self, tmp_path):
        """Hash should only depend on content, not metadata."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Content")

        hasher = ContentHasher()
        hash1 = hasher.hash_file(test_file)

        # Touch file (update mtime without changing content)
        test_file.touch()
        hash2 = hasher.hash_file(test_file)

        # Same content = same hash
        assert hash1 == hash2


class TestIncrementalIngestState:
    """Test the incremental ingestion state management."""

    @pytest.fixture
    def ingest_dir(self, tmp_path):
        """Create a temporary KB directory with .ingest/ structure."""
        kb_dir = tmp_path / "test-kb"
        kb_dir.mkdir()
        ingest = IncrementalIngest(kb_dir)
        yield kb_dir

    def test_state_file_created_on_init(self, ingest_dir):
        """.ingest/ directory and content_hashes.json should exist."""
        ingest_state = ingest_dir / ".ingest" / "content_hashes.json"
        assert ingest_state.exists()

    def test_store_and_retrieve_hash(self, ingest_dir):
        """Should be able to store and retrieve content hashes."""
        ingest = IncrementalIngest(ingest_dir)

        ingest.store_hash("/path/to/file.md", "abc123")
        retrieved = ingest.get_hash("/path/to/file.md")

        assert retrieved == "abc123"

    def test_detect_unchanged_file(self, ingest_dir):
        """File with matching hash should be unchanged."""
        ingest = IncrementalIngest(ingest_dir)
        test_file = ingest_dir / "source.md"
        test_file.write_text("Content")

        # Simulate previous ingestion
        hash_value = hashlib.sha256(b"Content").hexdigest()
        ingest.store_hash(str(test_file), hash_value)

        # Check status
        status = ingest.get_file_status(test_file)

        assert status.is_unchanged is True
        assert status.is_new is False
        assert status.is_modified is False

    def test_detect_new_file(self, ingest_dir):
        """File not in hash database is new."""
        ingest = IncrementalIngest(ingest_dir)
        test_file = ingest_dir / "new_file.md"
        test_file.write_text("New content")

        status = ingest.get_file_status(test_file)

        assert status.is_new is True
        assert status.is_unchanged is False

    def test_detect_modified_file(self, ingest_dir):
        """File with different hash is modified."""
        ingest = IncrementalIngest(ingest_dir)
        test_file = ingest_dir / "modified.md"
        test_file.write_text("Original content")

        # Store old hash
        ingest.store_hash(str(test_file), "old_hash")

        # Modify content
        test_file.write_text("Modified content")

        status = ingest.get_file_status(test_file)

        assert status.is_modified is True
        assert status.is_unchanged is False


class TestIncrementalExtraction:
    """Test that extraction only runs on changed files."""

    def test_skip_unchanged_files(self, tmp_path):
        """Extraction should skip files with unchanged hashes."""
        ingest = IncrementalIngest(tmp_path)

        # Create a file and mark it as processed
        source_file = tmp_path / "transcript.md"
        source_file.write_text("Existing content")
        hash_value = hashlib.sha256(b"Existing content").hexdigest()
        ingest.store_hash(str(source_file), hash_value)

        # Run incremental extraction
        files_to_process = ingest.get_files_to_process([source_file])

        assert len(files_to_process) == 0

    def test_process_modified_files(self, tmp_path):
        """Extraction should process modified files."""
        ingest = IncrementalIngest(tmp_path)

        source_file = tmp_path / "transcript.md"
        source_file.write_text("Original content")
        ingest.store_hash(str(source_file), "old_hash")

        # Modify file
        source_file.write_text("Updated content")

        files_to_process = ingest.get_files_to_process([source_file])

        assert len(files_to_process) == 1
        assert files_to_process[0] == source_file

    def test_process_new_files(self, tmp_path):
        """Extraction should process new files."""
        ingest = IncrementalIngest(tmp_path)

        new_file = tmp_path / "new_transcript.md"
        new_file.write_text("New content")

        files_to_process = ingest.get_files_to_process([new_file])

        assert len(files_to_process) == 1


class TestEmbeddingCache:
    """Test embeddings caching for unchanged content."""

    def test_embeddings_stored_with_hash(self, tmp_path):
        """Embeddings should be associated with content hash."""
        ingest = IncrementalIngest(tmp_path)

        concept_id = "concept-123"
        content_hash = "abc123"
        embedding = [0.1, 0.2, 0.3]

        ingest.store_embedding(concept_id, content_hash, embedding)
        retrieved = ingest.get_embedding(concept_id, content_hash)

        assert retrieved == embedding

    def test_embeddings_invalidated_on_change(self, tmp_path):
        """Old embeddings should not be returned if content changed."""
        ingest = IncrementalIngest(tmp_path)

        concept_id = "concept-456"
        old_hash = "old_hash"
        new_hash = "new_hash"

        ingest.store_embedding(concept_id, old_hash, [0.1, 0.2])

        # Should return None for new hash
        retrieved = ingest.get_embedding(concept_id, new_hash)
        assert retrieved is None

    def test_cache_reuse_unchanged_concepts(self, tmp_path):
        """Embeddings for unchanged concepts should be reused."""
        ingest = IncrementalIngest(tmp_path)

        concept_hash = "stable_hash"
        embedding = [0.5, 0.6, 0.7]

        # Store once
        ingest.store_embedding("concept_id", concept_hash, embedding)

        # Retrieve multiple times
        retrieved1 = ingest.get_embedding("concept_id", concept_hash)
        retrieved2 = ingest.get_embedding("concept_id", concept_hash)

        assert retrieved1 == embedding
        assert retrieved2 == embedding


class TestConceptUpsert:
    """Test update/insert logic for concepts."""

    def test_insert_new_concept(self, tmp_path):
        """New concepts should be inserted with metadata."""
        ingest = IncrementalIngest(tmp_path)

        concept = RawConcept(
            name="Test Concept",
            raw_content="This is a test concept",
        )

        ingest.upsert_concept(concept, source_hash="abc123")

        # Should be retrievable
        retrieved = ingest.get_concept_by_name("Test Concept")
        assert retrieved is not None
        assert retrieved.name == "Test Concept"
        assert retrieved.source_hash == "abc123"

    def test_update_existing_concept(self, tmp_path):
        """Modified concepts should be updated."""
        ingest = IncrementalIngest(tmp_path)

        # Insert initial version
        concept_v1 = RawConcept(name="Updateable Concept", raw_content="Version 1")
        ingest.upsert_concept(concept_v1, source_hash="hash1")

        # Update with new version
        concept_v2 = RawConcept(name="Updateable Concept", raw_content="Version 2")
        concept_v2.updated_at = datetime.now().isoformat()
        ingest.upsert_concept(concept_v2, source_hash="hash2")

        retrieved = ingest.get_concept_by_name("Updateable Concept")
        assert retrieved.raw_content == "Version 2"
        assert retrieved.source_hash == "hash2"


class TestSoftDelete:
    """Test soft deletion for removed content."""

    def test_mark_concept_deleted(self, tmp_path):
        """Deleted concepts should be marked but retained."""
        ingest = IncrementalIngest(tmp_path)

        concept = RawConcept(name="To Delete", raw_content="Content")
        ingest.upsert_concept(concept, source_hash="abc123")

        # Mark as deleted
        ingest.mark_deleted("To Delete")

        # Should not appear in search
        active = ingest.get_active_concepts()
        assert "To Delete" not in [c.name for c in active]

        # But should still exist
        all_concepts = ingest.get_all_concepts(include_deleted=True)
        assert any(c.name == "To Delete" for c in all_concepts)

    def test_gc_deleted_concepts(self, tmp_path):
        """Permanently remove soft-deleted concepts."""
        ingest = IncrementalIngest(tmp_path)

        concept = RawConcept(name="Gonner", raw_content="Content")
        ingest.upsert_concept(concept, source_hash="abc123")
        ingest.mark_deleted("Gonner")

        # Garbage collect
        ingest.gc_deleted(older_than_days=0)

        # Should be completely gone
        all_concepts = ingest.get_all_concepts(include_deleted=True)
        assert not any(c.name == "Gonner" for c in all_concepts)


class TestIncrementalPipeline:
    """Integration tests for the full incremental pipeline."""

    def test_first_run_processes_all_files(self, tmp_path):
        """Initial run should process all files."""
        ingest = IncrementalIngest(tmp_path)

        # Create source files
        transcript1 = tmp_path / "session1.md"
        transcript1.write_text("# Topic 1\n\nContent about AI.")
        transcript2 = tmp_path / "session2.md"
        transcript2.write_text("# Topic 2\n\nMore content.")

        # First run
        result = ingest.run([transcript1, transcript2])

        assert result.files_processed == 2
        assert result.files_skipped == 0
        assert result.concepts_extracted > 0

    def test_second_run_skips_unchanged(self, tmp_path):
        """Second run should skip unchanged files."""
        ingest = IncrementalIngest(tmp_path)

        transcript = tmp_path / "session.md"
        transcript.write_text("# Topic\n\nContent about ML.")

        # First run
        ingest.run([transcript])

        # Second run (no changes)
        result = ingest.run([transcript])

        assert result.files_processed == 0
        assert result.files_skipped == 1

    def test_add_new_file_after_initial(self, tmp_path):
        """Adding a file should only process the new one."""
        ingest = IncrementalIngest(tmp_path)

        existing = tmp_path / "old_session.md"
        existing.write_text("Old content")
        ingest.run([existing])

        # Add new file
        new_file = tmp_path / "new_session.md"
        new_file.write_text("New content")

        result = ingest.run([existing, new_file])

        assert result.files_processed == 1  # Only new file
        assert result.files_skipped == 1  # Existing unchanged
