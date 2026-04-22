"""Tests for mindforge diff."""

from __future__ import annotations

import json
from pathlib import Path

from mindforge.cli import compute_diff


def _seed(tmp_path: Path, snapshots: list[tuple[str, dict[str, str]]]) -> Path:
    """Write a manifest.json with the given (timestamp, slug_hashes) snapshots."""
    mpath = tmp_path / "out" / "manifest.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {"version": 1, "history": []}
    for ts, hashes in snapshots:
        data["history"].append({"timestamp": ts, "slug_hashes": hashes})
    mpath.write_text(json.dumps(data))
    return mpath


def test_diff_no_changes(tmp_path: Path) -> None:
    mpath = _seed(
        tmp_path,
        [
            ("2026-04-22T00:00:00Z", {"a": "h1", "b": "h2"}),
            ("2026-04-22T01:00:00Z", {"a": "h1", "b": "h2"}),
        ],
    )
    d = compute_diff(mpath)
    assert d == {"added": [], "modified": [], "deleted": []}


def test_diff_detects_added(tmp_path: Path) -> None:
    mpath = _seed(
        tmp_path,
        [
            ("2026-04-22T00:00:00Z", {"a": "h1"}),
            ("2026-04-22T01:00:00Z", {"a": "h1", "b": "h2"}),
        ],
    )
    assert compute_diff(mpath)["added"] == ["b"]


def test_diff_detects_modified_and_deleted(tmp_path: Path) -> None:
    mpath = _seed(
        tmp_path,
        [
            ("2026-04-22T00:00:00Z", {"a": "h1", "b": "h2"}),
            ("2026-04-22T01:00:00Z", {"a": "h1-new"}),
        ],
    )
    d = compute_diff(mpath)
    assert d["modified"] == ["a"]
    assert d["deleted"] == ["b"]


def test_diff_empty_when_missing_manifest(tmp_path: Path) -> None:
    d = compute_diff(tmp_path / "none.json")
    assert d == {"added": [], "modified": [], "deleted": []}


def test_diff_single_snapshot_is_empty(tmp_path: Path) -> None:
    mpath = _seed(tmp_path, [("2026-04-22T00:00:00Z", {"a": "h1"})])
    d = compute_diff(mpath)
    assert d == {"added": [], "modified": [], "deleted": []}
