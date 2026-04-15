"""LLM-aware concept distiller: enhanced distillation for LLM-extracted concepts.

When concepts are extracted by the LLM, the raw_content contains structured
metadata markers (relationship tags, tag annotations) that this distiller
understands and processes into proper Concept fields.

For concepts NOT extracted by the LLM, falls back to the standard distiller.
"""

from __future__ import annotations

import re

from mindforge.distillation.concept import Concept, Relationship, RelationshipType
from mindforge.distillation.distiller import distill_concept as heuristic_distill
from mindforge.ingestion.extractor import RawConcept
from mindforge.utils.text import extract_keywords, extract_sentences, normalize_whitespace


# Map LLM relationship type strings to our enum
_REL_TYPE_MAP = {
    "uses": RelationshipType.USES,
    "improves": RelationshipType.IMPROVES,
    "depends_on": RelationshipType.DEPENDS_ON,
    "related_to": RelationshipType.RELATED_TO,
    "part_of": RelationshipType.PART_OF,
    "example_of": RelationshipType.EXAMPLE_OF,
    "contrasts_with": RelationshipType.CONTRASTS_WITH,
    "enables": RelationshipType.ENABLES,
}

# Pattern to match embedded relationship markers
_REL_PATTERN = re.compile(r"\[\[rel:(\w+):(.+?)\]\]")
# Pattern to match embedded tag markers
_TAG_PATTERN = re.compile(r"\[\[tags:(.+?)\]\]")


def _extract_embedded_relationships(text: str, source_slug: str) -> list[Relationship]:
    """Extract relationship markers embedded by the LLM extractor."""
    relationships = []
    for match in _REL_PATTERN.finditer(text):
        rel_type_str = match.group(1)
        target_name = match.group(2).strip()
        rel_type = _REL_TYPE_MAP.get(rel_type_str, RelationshipType.RELATED_TO)

        # Slugify the target for the relationship
        from mindforge.utils.text import slugify
        target_slug = slugify(target_name)

        relationships.append(Relationship(
            source=source_slug,
            target=target_slug,
            rel_type=rel_type,
            confidence=0.85,
        ))

    return relationships


def _extract_embedded_tags(text: str) -> list[str]:
    """Extract tag markers embedded by the LLM extractor."""
    match = _TAG_PATTERN.search(text)
    if match:
        return [t.strip() for t in match.group(1).split(",") if t.strip()]
    return []


def _clean_markers(text: str) -> str:
    """Remove embedded metadata markers from text."""
    text = _REL_PATTERN.sub("", text)
    text = _TAG_PATTERN.sub("", text)
    return text.strip()


def distill_llm_concept(raw: RawConcept) -> Concept:
    """Distill a concept that was extracted by the LLM.

    The LLM extractor embeds structured metadata in the raw_content:
    - [[rel:type:Target Name]] for relationships
    - [[tags:tag1,tag2]] for tags

    This distiller extracts that metadata and produces a clean Concept
    with proper relationships and tags, while also using the LLM's
    higher-quality definition and explanation directly.
    """
    from mindforge.utils.text import slugify

    # Extract embedded metadata before cleaning
    source_slug = slugify(raw.name)
    relationships = _extract_embedded_relationships(raw.raw_content, source_slug)
    embedded_tags = _extract_embedded_tags(raw.raw_content)
    link_targets = list(dict.fromkeys(
        r.target for r in relationships
    ))
    link_names = []
    for match in _REL_PATTERN.finditer(raw.raw_content):
        link_names.append(match.group(2).strip())
    link_names = list(dict.fromkeys(link_names))

    # Clean the raw content of metadata markers
    cleaned = _clean_markers(raw.raw_content)

    # The LLM already structured the content well, so we can parse it directly
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]

    # First non-list paragraph is the definition
    definition = ""
    explanation_parts = []
    insights = []
    examples = []

    for para in paragraphs:
        if para.startswith("Examples:"):
            # Parse examples section
            example_lines = para.replace("Examples:", "").strip().split("\n")
            for line in example_lines:
                line = line.strip().lstrip("- ").strip()
                if line:
                    examples.append(line)
        elif para.startswith("- "):
            # Bullet points are insights
            for line in para.split("\n"):
                line = line.strip().lstrip("- ").strip()
                if line and len(line) > 10:
                    insights.append(line)
        elif not definition:
            definition = normalize_whitespace(para)
        else:
            explanation_parts.append(normalize_whitespace(para))

    explanation = " ".join(explanation_parts) if explanation_parts else ""

    # Use embedded tags, falling back to keyword extraction
    tags = embedded_tags if embedded_tags else extract_keywords(cleaned, top_n=5)

    return Concept(
        name=raw.name.strip(),
        definition=definition,
        explanation=explanation,
        insights=insights,
        examples=examples,
        tags=tags,
        source_files=raw.source_files,
        confidence=raw.confidence,
        links=link_names,
        relationships=relationships,
    )


def distill_concept_smart(raw: RawConcept) -> Concept:
    """Distill a concept using the appropriate method based on extraction source.

    LLM-extracted concepts get the LLM-aware distiller.
    Heuristic-extracted concepts get the standard distiller.
    """
    if raw.extraction_method == "llm":
        return distill_llm_concept(raw)
    return heuristic_distill(raw)


def distill_all_smart(raws: list[RawConcept]) -> list[Concept]:
    """Distill all raw concepts, using the appropriate distiller for each."""
    return [distill_concept_smart(raw) for raw in raws]
