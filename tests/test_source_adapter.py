"""Tests for SourceAdapter protocol and MarkdownSourceAdapter."""

from __future__ import annotations

from pathlib import Path

import pytest

# Import parser first so MarkdownSourceAdapter registers itself.
from mindforge.ingestion.parser import MarkdownSourceAdapter, parse_transcript
from mindforge.ingestion.sources import (
    SourceAdapter,
    get_adapter_for,
    registered_extensions,
)


def test_markdown_adapter_implements_protocol():
    adapter = MarkdownSourceAdapter()
    assert isinstance(adapter, SourceAdapter)


def test_markdown_adapter_matches_existing_parser(tmp_path: Path):
    transcript = tmp_path / "t.md"
    transcript.write_text(
        "Human: What is attention?\n\nAssistant: A mechanism that weighs tokens.\n"
    )
    adapter = MarkdownSourceAdapter()
    adapter_result = adapter.parse(transcript)
    direct_result = parse_transcript(transcript)
    assert [t.role for t in adapter_result.turns] == [t.role for t in direct_result.turns]
    assert [t.content for t in adapter_result.turns] == [t.content for t in direct_result.turns]


def test_md_and_txt_extensions_registered():
    exts = registered_extensions()
    assert ".md" in exts
    assert ".txt" in exts


def test_get_adapter_for_markdown_returns_markdown_adapter(tmp_path: Path):
    p = tmp_path / "x.md"
    p.write_text("Assistant: hi")
    adapter = get_adapter_for(p)
    assert isinstance(adapter, MarkdownSourceAdapter)


def test_get_adapter_for_unknown_extension_raises(tmp_path: Path):
    p = tmp_path / "x.pdf"
    p.write_text("")
    with pytest.raises(ValueError, match="No SourceAdapter"):
        get_adapter_for(p)
