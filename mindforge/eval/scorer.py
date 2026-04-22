"""Scoring metrics for MindForge evaluation."""

from __future__ import annotations

from difflib import SequenceMatcher


FUZZY_NAME_THRESHOLD = 0.85


def _name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _match_concept(expected: dict, actuals: list[dict]) -> dict | None:
    """Match by slug first, then fuzzy-name at FUZZY_NAME_THRESHOLD."""
    for a in actuals:
        if a.get("slug") == expected.get("slug"):
            return a
    for a in actuals:
        if _name_similarity(a.get("name", ""), expected.get("name", "")) >= FUZZY_NAME_THRESHOLD:
            return a
    return None


def _phrase_found(phrase: str, concept: dict) -> bool:
    blobs = [concept.get("definition", ""), concept.get("explanation", "")]
    blobs.extend(concept.get("insights", []))
    blob = " ".join(blobs).lower()
    return phrase.lower() in blob


def score_concepts(expected: list[dict], actual: list[dict]) -> dict:
    """Compute recall, precision, and phrase grounding."""
    if not expected:
        return {
            "recall": 1.0,
            "precision": 1.0 if not actual else 0.0,
            "phrase_grounding": 1.0,
            "matched": 0,
            "expected": 0,
            "extracted": len(actual),
        }
    matched_pairs: list[tuple[dict, dict]] = []
    for e in expected:
        m = _match_concept(e, actual)
        if m is not None:
            matched_pairs.append((e, m))
    recall = len(matched_pairs) / len(expected)
    precision = len(matched_pairs) / max(len(actual), 1)
    phrases = [p for e, _ in matched_pairs for p in e.get("key_phrases", [])]
    if phrases:
        grounded = [
            p
            for e, m in matched_pairs
            for p in e.get("key_phrases", [])
            if _phrase_found(p, m)
        ]
        phrase_grounding = len(grounded) / len(phrases)
    else:
        phrase_grounding = 1.0
    return {
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "phrase_grounding": round(phrase_grounding, 3),
        "matched": len(matched_pairs),
        "expected": len(expected),
        "extracted": len(actual),
    }


def score_relationships(expected: list[dict], actual: list[dict]) -> dict:
    """Compute relationship recall and type accuracy."""
    if not expected:
        return {
            "recall": 1.0,
            "type_accuracy": 1.0,
            "matched": 0,
            "expected": 0,
            "found": len(actual),
        }

    def _same_edge(e: dict, a: dict) -> bool:
        return e.get("source") == a.get("source") and e.get("target") == a.get("target")

    matched = 0
    type_matches = 0
    for e in expected:
        for a in actual:
            if _same_edge(e, a):
                matched += 1
                if e.get("type") == a.get("type"):
                    type_matches += 1
                break
    recall = matched / len(expected)
    type_accuracy = type_matches / matched if matched else 1.0
    return {
        "recall": round(recall, 3),
        "type_accuracy": round(type_accuracy, 3),
        "matched": matched,
        "expected": len(expected),
        "found": len(actual),
    }
