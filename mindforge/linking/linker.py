"""Linking engine: detects relationships between concepts and inserts wiki-style links.

Relationship detection uses multiple signals:
1. Name co-occurrence: concept A's text mentions concept B by name
2. Keyword overlap: shared tags/keywords suggest relatedness
3. Structural patterns: "X uses Y", "X depends on Y", etc.
"""

from __future__ import annotations

import re

from mindforge.distillation.concept import Concept, ConceptStore, Relationship, RelationshipType
from mindforge.utils.text import compute_text_similarity


# Patterns for detecting typed relationships in text
_RELATIONSHIP_PATTERNS: list[tuple[re.Pattern[str], RelationshipType]] = [
    (re.compile(r"\buses\b|\butilizes\b|\bleverages\b|\bemploys\b", re.IGNORECASE), RelationshipType.USES),
    (re.compile(r"\bimproves\b|\benhances\b|\boptimizes\b|\bextends\b", re.IGNORECASE), RelationshipType.IMPROVES),
    (re.compile(r"\bdepends on\b|\brequires\b|\bneeds\b|\brelies on\b", re.IGNORECASE), RelationshipType.DEPENDS_ON),
    (re.compile(r"\bpart of\b|\bcomponent of\b|\bsubset of\b|\bbelongs to\b", re.IGNORECASE), RelationshipType.PART_OF),
    (re.compile(r"\bexample of\b|\binstance of\b|\btype of\b|\bkind of\b", re.IGNORECASE), RelationshipType.EXAMPLE_OF),
    (re.compile(r"\bcontrasts with\b|\bunlike\b|\bversus\b|\bvs\.?\b|\bcompared to\b", re.IGNORECASE), RelationshipType.CONTRASTS_WITH),
    (re.compile(r"\benables\b|\ballows\b|\bmakes possible\b|\bfacilitates\b", re.IGNORECASE), RelationshipType.ENABLES),
]


def _find_concept_mentions(text: str, target_name: str) -> list[int]:
    """Find all positions where a concept is mentioned in text."""
    pattern = re.compile(re.escape(target_name), re.IGNORECASE)
    return [m.start() for m in pattern.finditer(text)]


def _detect_relationship_type(
    text: str,
    source_name: str,
    target_name: str,
) -> RelationshipType:
    """Detect the type of relationship between two concepts based on context.

    Searches for relationship-indicating verbs near mentions of the target concept.
    """
    # Look for patterns in sentences that mention both concepts
    sentences = re.split(r"[.!?]\s+", text)
    source_lower = source_name.lower()
    target_lower = target_name.lower()

    for sentence in sentences:
        s_lower = sentence.lower()
        if target_lower in s_lower:
            for pattern, rel_type in _RELATIONSHIP_PATTERNS:
                if pattern.search(s_lower):
                    return rel_type

    return RelationshipType.RELATED_TO


def detect_links(
    store: ConceptStore,
    confidence_threshold: float = 0.3,
) -> None:
    """Detect and insert links between all concepts in the store.

    Modifies concepts in-place, adding:
    - wiki-link targets (concept.links)
    - typed relationships (concept.relationships)
    """
    all_concepts = store.all()
    all_slugs = store.slugs()

    for concept in all_concepts:
        combined_text = f"{concept.definition} {concept.explanation}"
        linked_slugs: set[str] = set()

        for other in all_concepts:
            if other.slug == concept.slug:
                continue

            # Signal 1: Name mention
            mentions = _find_concept_mentions(combined_text, other.name)
            name_score = min(len(mentions) * 0.3, 1.0) if mentions else 0.0

            # Signal 2: Tag overlap
            shared_tags = set(concept.tags) & set(other.tags)
            tag_score = len(shared_tags) / max(len(concept.tags), 1) * 0.5

            # Signal 3: Content similarity
            content_score = compute_text_similarity(
                concept.definition[:200],
                other.definition[:200],
            ) * 0.4

            total_score = name_score + tag_score + content_score

            if total_score >= confidence_threshold and other.slug not in linked_slugs:
                linked_slugs.add(other.slug)

                # Detect relationship type
                rel_type = _detect_relationship_type(
                    combined_text, concept.name, other.name,
                )

                concept.links.append(other.name)
                concept.relationships.append(Relationship(
                    source=concept.slug,
                    target=other.slug,
                    rel_type=rel_type,
                    confidence=min(total_score, 1.0),
                ))

        # Deduplicate links
        concept.links = list(dict.fromkeys(concept.links))


def insert_wiki_links(text: str, concept_names: list[str]) -> str:
    """Insert wiki-style links into text for known concept names.

    Replaces the first occurrence of each concept name with [[Name]].
    Only links names that aren't already linked.
    """
    result = text
    for name in concept_names:
        # Don't double-link
        if f"[[{name}]]" in result:
            continue
        # Replace first occurrence (case-insensitive) that isn't already linked
        pattern = re.compile(
            rf"(?<!\[\[)\b({re.escape(name)})\b(?!\]\])",
            re.IGNORECASE,
        )
        result = pattern.sub(rf"[[\1]]", result, count=1)
    return result
