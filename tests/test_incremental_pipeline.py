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

    config = MindForgeConfig(transcripts_dir=transcripts_dir, output_dir=output_dir, use_llm=False)
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

    config = MindForgeConfig(transcripts_dir=transcripts_dir, output_dir=output_dir, use_llm=False)
    MindForgePipeline(config).run()  # first full run

    result = MindForgePipeline(config).run()  # second run, no changes
    assert result.skipped is True
    assert result.files_unchanged == 1
    assert result.files_new == 0
    assert result.files_modified == 0
