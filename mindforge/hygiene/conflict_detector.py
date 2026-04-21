"""Rule-based conflict detection for Concept fields.

Two detectors ship here:
- Definition-level: when pairwise SequenceMatcher ratio falls below
  DEFINITION_SIMILARITY_THRESHOLD, the definitions are considered to
  diverge materially.
- Insight-level: a small word-pair table catches common contradictions
  (quantifier mismatches: always/sometimes; unit mismatches: tokens/characters).

LLM-assisted conflict detection is a future session's work — this module
stays rule-based. The seam is in distiller.py, which calls these detectors.
"""

from __future__ import annotations

from difflib import SequenceMatcher


DEFINITION_SIMILARITY_THRESHOLD = 0.7

_QUANTIFIERS = [("always", "sometimes"), ("never", "sometimes"), ("all", "some")]
_UNITS = [("tokens", "characters"), ("bytes", "bits"), ("seconds", "minutes")]


def detect_definition_conflict(a: str, b: str) -> bool:
    """Return True when two definitions diverge materially."""
    if not a or not b:
        return False
    ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return ratio < DEFINITION_SIMILARITY_THRESHOLD


def _contradicts(a: str, b: str) -> bool:
    la, lb = a.lower(), b.lower()
    for q1, q2 in _QUANTIFIERS:
        if (q1 in la and q2 in lb) or (q2 in la and q1 in lb):
            return True
    for u1, u2 in _UNITS:
        if (u1 in la and u2 in lb) or (u2 in la and u1 in lb):
            return True
    return False


def detect_insight_conflicts(insights: list[str]) -> list[tuple[int, int]]:
    """Return pairs of insight indices that appear contradictory."""
    pairs: list[tuple[int, int]] = []
    for i in range(len(insights)):
        for j in range(i + 1, len(insights)):
            if _contradicts(insights[i], insights[j]):
                pairs.append((i, j))
    return pairs
