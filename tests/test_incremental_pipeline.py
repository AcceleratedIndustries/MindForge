"""Integration tests for incremental ingestion in MindForgePipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from mindforge.config import MindForgeConfig
from mindforge.pipeline import MindForgePipeline


def _seed_transcripts(transcripts_dir: Path, contents: dict[str, str]) -> None:
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    for name, body in contents.items():
        (transcripts_dir / name).write_text(body, encoding="utf-8")


@pytest.fixture
def fixture_paths(tmp_path: Path) -> tuple[Path, Path]:
    transcripts_dir = tmp_path / "transcripts"
    output_dir = tmp_path / "output"
    return transcripts_dir, output_dir


def test_first_run_creates_hash_cache(fixture_paths: tuple[Path, Path]) -> None:
    transcripts_dir, output_dir = fixture_paths
    _seed_transcripts(
        transcripts_dir,
        {
            "a.md": "# Alpha\n\nAlpha is the first letter of the Greek alphabet.\n",
        },
    )

    config = MindForgeConfig(
        transcripts_dir=transcripts_dir,
        output_dir=output_dir,
        llm_provider="mock",
    )
    pipeline = MindForgePipeline(config)
    result = pipeline.run()

    assert (output_dir / ".ingest" / "content_hashes.json").exists()
    assert result.skipped is False
    assert result.files_new == 1


def test_rerun_with_no_changes_triggers_fast_path(
    fixture_paths: tuple[Path, Path],
) -> None:
    transcripts_dir, output_dir = fixture_paths
    _seed_transcripts(
        transcripts_dir,
        {
            "a.md": "# Alpha\n\nAlpha is the first letter of the Greek alphabet.\n",
        },
    )

    config = MindForgeConfig(
        transcripts_dir=transcripts_dir,
        output_dir=output_dir,
        llm_provider="mock",
    )
    MindForgePipeline(config).run()  # first full run

    result = MindForgePipeline(config).run()  # second run, no changes
    assert result.skipped is True
    assert result.files_unchanged == 1
    assert result.files_new == 0
    assert result.files_modified == 0


def test_adding_new_transcript_preserves_old_concepts(
    fixture_paths: tuple[Path, Path],
) -> None:
    transcripts_dir, output_dir = fixture_paths
    _seed_transcripts(
        transcripts_dir,
        {
            "a.md": "# Alpha\n\nAlpha is the first letter of the Greek alphabet.\n",
        },
    )

    config = MindForgeConfig(
        transcripts_dir=transcripts_dir,
        output_dir=output_dir,
        llm_provider="mock",
    )
    MindForgePipeline(config).run()

    from mindforge.distillation.concept import ConceptStore

    first_store = ConceptStore.load(output_dir / "concepts.json")
    first_slugs = set(first_store.slugs())
    assert first_slugs, "first run should have produced some concepts"

    (transcripts_dir / "b.md").write_text(
        "# Beta\n\nBeta is the second letter of the Greek alphabet.\n",
        encoding="utf-8",
    )
    result = MindForgePipeline(config).run()
    assert result.skipped is False
    assert result.files_new == 1
    assert result.files_unchanged == 1

    second_store = ConceptStore.load(output_dir / "concepts.json")
    assert first_slugs.issubset(set(second_store.slugs())), (
        "concepts from unchanged files should be preserved"
    )


def test_modifying_transcript_soft_marks_removed_concepts(
    fixture_paths: tuple[Path, Path],
) -> None:
    transcripts_dir, output_dir = fixture_paths
    _seed_transcripts(
        transcripts_dir,
        {
            "a.md": (
                "# Alpha\n\n"
                "Alpha is the first letter of the Greek alphabet. "
                "Alpha is widely used as a placeholder name. "
                "Alpha denotes the start of a series.\n\n"
                "# Beta\n\n"
                "Beta is the second letter of the Greek alphabet. "
                "Beta is commonly used in software for early test releases. "
                "Beta indicates a candidate not yet ready for production.\n"
            ),
        },
    )

    config = MindForgeConfig(
        transcripts_dir=transcripts_dir,
        output_dir=output_dir,
        llm_provider="mock",
    )
    MindForgePipeline(config).run()

    from mindforge.distillation.concept import ConceptStore

    first_store = ConceptStore.load(output_dir / "concepts.json")
    assert "beta" in first_store.concepts, (
        "test is misconfigured: Beta block must produce a 'beta' concept on first run"
    )

    (transcripts_dir / "a.md").write_text(
        "# Alpha\n\n"
        "Alpha is the first letter of the Greek alphabet. "
        "Alpha is widely used as a placeholder name. "
        "Alpha denotes the start of a series.\n",
        encoding="utf-8",
    )
    result = MindForgePipeline(config).run()
    assert result.files_modified == 1
    assert result.concepts_soft_deleted >= 1, (
        "removing the Beta block should soft-delete at least one concept"
    )

    second_store = ConceptStore.load(output_dir / "concepts.json")
    assert "beta" in second_store.concepts, "soft-deleted concept should still be in the store"
    assert second_store.concepts["beta"].status == "deleted"
    assert second_store.concepts["beta"].deleted_at is not None


def test_deleting_transcript_soft_marks_orphans(
    fixture_paths: tuple[Path, Path],
) -> None:
    transcripts_dir, output_dir = fixture_paths
    _seed_transcripts(
        transcripts_dir,
        {
            "a.md": "# Alpha\n\nAlpha is the first letter of the Greek alphabet.\n",
            "b.md": (
                "# Beta\n\n"
                "Beta is the second letter of the Greek alphabet. "
                "Beta is commonly used in software for early test releases. "
                "Beta indicates a candidate not yet ready for production.\n"
            ),
        },
    )

    config = MindForgeConfig(
        transcripts_dir=transcripts_dir,
        output_dir=output_dir,
        llm_provider="mock",
    )
    MindForgePipeline(config).run()

    (transcripts_dir / "b.md").unlink()
    result = MindForgePipeline(config).run()
    assert result.files_deleted == 1
    assert result.concepts_soft_deleted >= 1

    from mindforge.distillation.concept import ConceptStore

    store = ConceptStore.load(output_dir / "concepts.json")
    assert "beta" in store.concepts
    assert store.concepts["beta"].status == "deleted"


def test_full_flag_forces_full_rebuild(fixture_paths: tuple[Path, Path]) -> None:
    transcripts_dir, output_dir = fixture_paths
    _seed_transcripts(
        transcripts_dir,
        {
            "a.md": "# Alpha\n\nAlpha is the first letter of the Greek alphabet.\n",
        },
    )

    config = MindForgeConfig(
        transcripts_dir=transcripts_dir,
        output_dir=output_dir,
        llm_provider="mock",
    )
    MindForgePipeline(config).run()  # first run — populates cache

    # Force full rebuild via the API the CLI uses.
    pipeline = MindForgePipeline(config)
    pipeline._force_full = True
    result = pipeline.run()

    assert result.skipped is False
    assert result.files_new == 1
    assert result.files_unchanged == 0
    assert (output_dir / ".ingest" / "content_hashes.json").exists()


def test_full_with_dry_run_does_not_delete_cache(
    fixture_paths: tuple[Path, Path],
) -> None:
    transcripts_dir, output_dir = fixture_paths
    _seed_transcripts(
        transcripts_dir,
        {"a.md": "# Alpha\n\nAlpha is the first letter of the Greek alphabet.\n"},
    )
    config = MindForgeConfig(
        transcripts_dir=transcripts_dir,
        output_dir=output_dir,
        llm_provider="mock",
    )
    MindForgePipeline(config).run()  # populate cache
    cache_path = output_dir / ".ingest" / "content_hashes.json"
    assert cache_path.exists()
    cache_content_before = cache_path.read_text()

    pipeline = MindForgePipeline(config)
    pipeline._force_full = True
    pipeline.run(dry_run=True)

    assert cache_path.exists(), "dry-run must not delete the cache"
    assert cache_path.read_text() == cache_content_before, "dry-run must not modify the cache"
