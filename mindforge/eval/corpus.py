"""Load eval/fixtures/ transcripts + .gt.yaml pairs."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Fixture:
    transcript_path: Path
    expected_concepts: list[dict[str, Any]]
    expected_relationships: list[dict[str, Any]]


def load_corpus(fixtures_dir: Path) -> list[Fixture]:
    """Pair every <name>.md in fixtures_dir with <name>.gt.yaml.

    Transcripts without ground truth are skipped with a stderr warning.
    """
    fixtures: list[Fixture] = []
    for t in sorted(fixtures_dir.glob("*.md")):
        gt = t.with_suffix(".gt.yaml")
        if not gt.exists():
            print(
                f"[eval] {t.name}: missing ground truth ({gt.name})",
                file=sys.stderr,
            )
            continue
        data = yaml.safe_load(gt.read_text(encoding="utf-8")) or {}
        fixtures.append(Fixture(
            transcript_path=t,
            expected_concepts=data.get("expected_concepts", []),
            expected_relationships=data.get("expected_relationships", []),
        ))
    return fixtures
