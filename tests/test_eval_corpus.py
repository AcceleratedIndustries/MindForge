"""Tests for the eval corpus loader."""

from __future__ import annotations

from pathlib import Path

from mindforge.eval.corpus import Fixture, load_corpus


def test_load_corpus_pairs_transcripts_and_gt(tmp_path: Path):
    (tmp_path / "a.md").write_text("Assistant: stuff")
    (tmp_path / "a.gt.yaml").write_text(
        "expected_concepts:\n"
        "  - name: Foo\n"
        "    slug: foo\n"
        "    key_phrases: [stuff]\n"
    )
    fixtures = load_corpus(tmp_path)
    assert len(fixtures) == 1
    assert isinstance(fixtures[0], Fixture)
    assert fixtures[0].transcript_path.name == "a.md"
    assert fixtures[0].expected_concepts[0]["slug"] == "foo"


def test_load_corpus_warns_on_missing_gt(tmp_path: Path, capsys):
    (tmp_path / "orphan.md").write_text("x")
    fixtures = load_corpus(tmp_path)
    assert fixtures == []
    captured = capsys.readouterr()
    assert "missing ground truth" in captured.err.lower()


def test_load_corpus_empty_dir(tmp_path: Path):
    assert load_corpus(tmp_path) == []
