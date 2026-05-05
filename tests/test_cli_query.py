"""Tests for CLI hybrid retrieval flags (--mode, --weights)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from mindforge.cli import _build_parser, _parse_weights, cmd_query
from mindforge.query.engine import QueryResult, RetrievalWeights


def test_parser_mode_default_is_hybrid() -> None:
    parser = _build_parser()
    args = parser.parse_args(["query", "X"])
    assert args.mode == "hybrid"
    assert args.weights is None


def test_parser_accepts_keyword_and_semantic_modes() -> None:
    parser = _build_parser()
    for mode in ("keyword", "semantic", "hybrid"):
        args = parser.parse_args(["query", "X", "--mode", mode])
        assert args.mode == mode


def test_parser_rejects_bogus_mode() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["query", "X", "--mode", "bogus"])
    assert exc.value.code == 2


def test_parse_weights_three_floats() -> None:
    weights = _parse_weights("0.5,0.3,0.2")
    assert isinstance(weights, RetrievalWeights)
    assert weights.keyword == 0.5
    assert weights.semantic == 0.3
    assert weights.graph == 0.2


def test_parse_weights_none_returns_none() -> None:
    assert _parse_weights(None) is None


def test_parse_weights_wrong_length_raises() -> None:
    with pytest.raises(SystemExit) as exc:
        _parse_weights("0.5,0.5")
    msg = str(exc.value.code)
    assert "three comma-separated floats" in msg


def test_parse_weights_non_numeric_raises() -> None:
    with pytest.raises(SystemExit) as exc:
        _parse_weights("0.5,foo,0.5")
    msg = str(exc.value.code)
    assert "three comma-separated floats" in msg


def test_cmd_query_passes_mode_and_weights_through(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """cmd_query should hand --mode and --weights to engine.search()."""

    captured: dict[str, Any] = {}

    class _FakeEngine:
        def search(
            self,
            query: str,
            top_k: int = 5,
            mode: str = "hybrid",
            weights: RetrievalWeights | None = None,
        ) -> list[QueryResult]:
            captured["query"] = query
            captured["top_k"] = top_k
            captured["mode"] = mode
            captured["weights"] = weights
            return []

        def format_results(self, results: list[QueryResult]) -> str:
            return "no results"

    class _FakePipeline:
        def __init__(self, _config: Any) -> None:
            self.query_engine = _FakeEngine()

        def _load_state(self) -> None:
            pass

    monkeypatch.setattr("mindforge.cli.MindForgePipeline", _FakePipeline)

    args = argparse.Namespace(
        question="what is rag",
        top_k=5,
        output=Path("output"),
        embeddings=False,
        tag=None,
        min_confidence=None,
        since=None,
        mode="keyword",
        weights="0.7,0.0,0.3",
    )
    rc = cmd_query(args)
    assert rc == 0
    assert captured["mode"] == "keyword"
    w = captured["weights"]
    assert isinstance(w, RetrievalWeights)
    assert w.keyword == 0.7
    assert w.semantic == 0.0
    assert w.graph == 0.3


def test_cmd_query_default_mode_and_no_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeEngine:
        def search(
            self,
            query: str,
            top_k: int = 5,
            mode: str = "hybrid",
            weights: RetrievalWeights | None = None,
        ) -> list[QueryResult]:
            captured["mode"] = mode
            captured["weights"] = weights
            return []

        def format_results(self, results: list[QueryResult]) -> str:
            return ""

    class _FakePipeline:
        def __init__(self, _config: Any) -> None:
            self.query_engine = _FakeEngine()

        def _load_state(self) -> None:
            pass

    monkeypatch.setattr("mindforge.cli.MindForgePipeline", _FakePipeline)

    args = argparse.Namespace(
        question="x",
        top_k=5,
        output=Path("output"),
        embeddings=False,
        tag=None,
        min_confidence=None,
        since=None,
        mode="hybrid",
        weights=None,
    )
    assert cmd_query(args) == 0
    assert captured["mode"] == "hybrid"
    assert captured["weights"] is None
