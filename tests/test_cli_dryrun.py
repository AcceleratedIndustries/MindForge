"""Tests for mindforge ingest --dry-run."""

from __future__ import annotations

from pathlib import Path

from mindforge.config import MindForgeConfig
from mindforge.pipeline import MindForgePipeline


def test_dry_run_writes_no_files(tmp_path: Path) -> None:
    transcripts = tmp_path / "t"
    transcripts.mkdir()
    (transcripts / "x.md").write_text(
        "Assistant: KV Cache is a mechanism that stores Key and Value matrices.\n"
    )
    out = tmp_path / "out"
    cfg = MindForgeConfig(transcripts_dir=transcripts, output_dir=out)
    result = MindForgePipeline(cfg).run(dry_run=True)
    assert result.dry_run is True
    # Output dir should exist (ensure_dirs ran) but carry no concept files.
    if out.exists():
        assert not any((out / "concepts").glob("*.md"))


def test_dry_run_reports_diff(tmp_path: Path) -> None:
    transcripts = tmp_path / "t"
    transcripts.mkdir()
    (transcripts / "x.md").write_text(
        "Assistant: KV Cache is a mechanism that stores Key and Value matrices.\n"
    )
    out = tmp_path / "out"
    cfg = MindForgeConfig(transcripts_dir=transcripts, output_dir=out)

    # First real run.
    MindForgePipeline(cfg).run()

    # Dry run on same input — everything should be unchanged.
    r = MindForgePipeline(cfg).run(dry_run=True)
    assert r.dry_run is True
    assert r.new == 0
