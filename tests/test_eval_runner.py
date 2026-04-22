"""End-to-end tests for the eval runner."""

from __future__ import annotations

from pathlib import Path

from mindforge.eval.runner import render_markdown, run_eval


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_runner_empty_corpus_returns_zero_report(tmp_path: Path):
    report = run_eval(tmp_path, mode="heuristic")
    assert report["corpus_size"] == 0
    md = render_markdown(report)
    assert "no fixtures" in md.lower()


def test_runner_produces_report(tmp_path: Path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    _write(
        fixtures / "x.md",
        "Assistant: KV Cache is a mechanism that stores the Key and Value "
        "matrices from the attention computation of previously processed "
        "tokens, avoiding redundant recomputation during autoregressive "
        "generation.\n",
    )
    _write(
        fixtures / "x.gt.yaml",
        "expected_concepts:\n"
        "  - name: KV Cache\n"
        "    slug: kv-cache\n"
        "    key_phrases: [\"Key and Value\"]\n"
        "expected_relationships: []\n",
    )

    report = run_eval(fixtures, mode="heuristic")
    assert report["corpus_size"] == 1
    assert "concepts" in report
    c = report["concepts"]
    assert 0.0 <= c["recall"] <= 1.0
    assert c["expected"] == 1


def test_runner_report_markdown_format(tmp_path: Path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    _write(fixtures / "x.md", "Assistant: Foo is a thing.\n")
    _write(
        fixtures / "x.gt.yaml",
        "expected_concepts:\n  - name: Foo\n    slug: foo\n    key_phrases: []\n"
        "expected_relationships: []\n",
    )
    report = run_eval(fixtures, mode="heuristic")
    md = render_markdown(report)
    assert "MindForge Evaluation Report" in md
    assert "Recall" in md
    assert "Precision" in md
