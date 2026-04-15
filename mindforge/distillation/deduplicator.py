"""Concept deduplication: merges similar or overlapping concepts.

Uses text similarity to detect near-duplicates and merges them,
preserving the best content from each.
"""

from __future__ import annotations

from mindforge.ingestion.extractor import RawConcept
from mindforge.utils.text import compute_text_similarity, slugify


def deduplicate_concepts(
    concepts: list[RawConcept],
    similarity_threshold: float = 0.75,
) -> list[RawConcept]:
    """Deduplicate raw concepts by merging similar ones.

    Two concepts are considered duplicates if:
    1. Their names produce the same slug, OR
    2. Their content similarity exceeds the threshold

    When merging, the higher-confidence concept is kept as the primary,
    and content from the other is appended.
    """
    if not concepts:
        return []

    # Group by slug first (exact name matches)
    slug_groups: dict[str, list[RawConcept]] = {}
    for concept in concepts:
        slug = slugify(concept.name)
        slug_groups.setdefault(slug, []).append(concept)

    # Merge within slug groups
    merged: list[RawConcept] = []
    for slug, group in slug_groups.items():
        primary = max(group, key=lambda c: (c.confidence, len(c.raw_content)))
        for other in group:
            if other is not primary:
                primary = _merge_raw(primary, other)
        merged.append(primary)

    # Cross-group similarity check
    result: list[RawConcept] = []
    consumed: set[int] = set()

    for i, concept_a in enumerate(merged):
        if i in consumed:
            continue

        current = concept_a
        for j in range(i + 1, len(merged)):
            if j in consumed:
                continue
            concept_b = merged[j]

            # Check name similarity
            name_sim = compute_text_similarity(current.name, concept_b.name)
            # Check content similarity
            content_sim = compute_text_similarity(
                current.raw_content[:500],
                concept_b.raw_content[:500],
            )

            if name_sim > 0.6 or content_sim > similarity_threshold:
                current = _merge_raw(current, concept_b)
                consumed.add(j)

        result.append(current)

    return result


def _merge_raw(primary: RawConcept, secondary: RawConcept) -> RawConcept:
    """Merge two raw concepts, keeping the primary's name and best content."""
    # Combine content, deduplicating paragraphs
    primary_paragraphs = set(
        p.strip().lower() for p in primary.raw_content.split("\n\n") if p.strip()
    )
    new_paragraphs = [
        p.strip() for p in secondary.raw_content.split("\n\n")
        if p.strip() and p.strip().lower() not in primary_paragraphs
    ]
    if new_paragraphs:
        combined = primary.raw_content + "\n\n" + "\n\n".join(new_paragraphs)
    else:
        combined = primary.raw_content

    return RawConcept(
        name=primary.name,
        raw_content=combined[:5000],  # Cap total length
        source_chunks=list(dict.fromkeys(
            primary.source_chunks + secondary.source_chunks
        )),
        source_files=list(dict.fromkeys(
            primary.source_files + secondary.source_files
        )),
        extraction_method=primary.extraction_method,
        confidence=max(primary.confidence, secondary.confidence),
    )
