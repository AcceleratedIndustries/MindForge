"""Concept distiller: transforms raw extracted concepts into clean, structured Concepts.

This is the core intelligence of MindForge. It takes noisy, raw extractions
and produces clean, atomic, human-readable concept entries.
"""

from __future__ import annotations

import re

from mindforge.distillation.concept import Concept
from mindforge.ingestion.extractor import RawConcept
from mindforge.utils.text import (
    compute_text_similarity,
    extract_keywords,
    extract_sentences,
    normalize_whitespace,
)


def distill_concept(raw: RawConcept) -> Concept:
    """Distill a raw concept into a clean, structured Concept.

    Performs:
    1. Clean the content (remove conversational fluff)
    2. Extract/refine the definition
    3. Build the explanation
    4. Extract key insights as bullet points
    5. Identify any examples
    """
    cleaned = _clean_content(raw.raw_content)
    sentences = extract_sentences(cleaned)

    definition = _build_definition(raw.name, sentences, raw.raw_content)
    explanation = _build_explanation(sentences, definition)
    insights = _extract_insights(cleaned, definition)
    examples = _extract_examples(raw.raw_content)
    tags = extract_keywords(cleaned, top_n=5)

    return Concept(
        name=raw.name.strip(),
        definition=definition,
        explanation=explanation,
        insights=insights,
        examples=examples,
        tags=tags,
        source_files=raw.source_files,
        confidence=raw.confidence,
    )


def _clean_content(text: str) -> str:
    """Remove conversational artifacts from extracted text."""
    # Remove references to "the conversation", "I mentioned", "as we discussed"
    fluff_patterns = [
        r"(?i)\b(?:as (?:I|we) (?:mentioned|discussed|said|noted))(?:\s+(?:earlier|before|above))?\b[,.]?\s*",
        r"(?i)\b(?:in (?:our|the|this) (?:conversation|discussion|chat))\b[,.]?\s*",
        r"(?i)\b(?:let me explain|I(?:'ll| will) explain|here's (?:the|an?) explanation)\b[,.]?\s*",
        r"(?i)\b(?:great question|good question|that's a good point)\b[!,.]?\s*",
        r"(?i)\b(?:sure|absolutely|definitely|of course)[!,.]?\s*",
        r"(?i)\b(?:hope (?:this|that) helps)\b[!,.]?\s*",
        r"(?i)\b(?:you're right|exactly|precisely)\b[!,.]?\s*",
    ]

    result = text
    for pattern in fluff_patterns:
        result = re.sub(pattern, "", result)

    # Clean up resulting whitespace
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _build_definition(name: str, sentences: list[str], raw: str) -> str:
    """Build a 2-3 sentence definition for the concept."""
    # Try to find a sentence that defines the concept
    name_lower = name.lower()
    defining_sentences: list[str] = []

    for sentence in sentences:
        s_lower = sentence.lower()
        # Look for definitional patterns
        if any(
            p in s_lower
            for p in [
                f"{name_lower} is",
                f"{name_lower} are",
                f"{name_lower} refers",
                f"{name_lower} means",
                f"{name_lower} describes",
                f"{name_lower} represents",
                f"{name_lower} provides",
                f"{name_lower} enables",
            ]
        ):
            defining_sentences.append(sentence)
        elif name_lower in s_lower and len(defining_sentences) < 3:
            defining_sentences.append(sentence)

    if defining_sentences:
        return normalize_whitespace(" ".join(defining_sentences[:3]))

    # Fallback: use the first 2-3 sentences
    fallback = " ".join(sentences[:3]) if sentences else raw[:300]
    return normalize_whitespace(fallback)


def _build_explanation(sentences: list[str], definition: str) -> str:
    """Build an expanded explanation from remaining sentences."""
    def_lower = normalize_whitespace(definition).lower()

    # Deduplicate sentences and skip those already in the definition
    seen: set[str] = set()
    remaining: list[str] = []
    for s in sentences:
        s_norm = normalize_whitespace(s).lower()
        if s_norm in seen:
            continue
        seen.add(s_norm)
        # Skip if this sentence is part of the definition or vice versa
        if s_norm in def_lower or def_lower in s_norm:
            continue
        # Also skip if high overlap with definition
        if compute_text_similarity(s_norm, def_lower) > 0.7:
            continue
        remaining.append(s)

    if not remaining:
        return ""

    # Take up to 8 sentences for the explanation
    explanation_text = " ".join(remaining[:8])
    return normalize_whitespace(explanation_text)


def _extract_insights(text: str, definition: str = "") -> list[str]:
    """Extract key insights as bullet points.

    Looks for:
    - Existing bullet points / list items
    - Key takeaway patterns
    - Important qualifications
    """
    insights: list[str] = []
    seen: set[str] = set()
    def_lower = normalize_whitespace(definition).lower()

    # Extract existing list items
    list_pattern = re.compile(r"^\s*[-*]\s+(.{10,200})$", re.MULTILINE)
    for match in list_pattern.finditer(text):
        item = normalize_whitespace(match.group(1))
        item_lower = item.lower()
        if item_lower not in seen and len(item) > 10 and item_lower not in def_lower:
            seen.add(item_lower)
            insights.append(item)

    # Extract numbered list items
    numbered_pattern = re.compile(r"^\s*\d+[.)]\s+(.{10,200})$", re.MULTILINE)
    for match in numbered_pattern.finditer(text):
        item = normalize_whitespace(match.group(1))
        item_lower = item.lower()
        if item_lower not in seen and len(item) > 10 and item_lower not in def_lower:
            seen.add(item_lower)
            insights.append(item)

    # Extract sentences with key phrases
    key_phrases = [
        "important", "key", "critical", "essential", "note that",
        "remember", "crucial", "significant", "advantage", "benefit",
        "drawback", "limitation", "trade-off", "tradeoff",
    ]
    for sentence in extract_sentences(text):
        s_lower = sentence.lower()
        if any(phrase in s_lower for phrase in key_phrases):
            clean = normalize_whitespace(sentence)
            if clean.lower() not in seen and len(clean) > 15:
                seen.add(clean.lower())
                insights.append(clean)

    return insights[:10]  # Cap at 10 insights


def _extract_examples(text: str) -> list[str]:
    """Extract examples from text."""
    examples: list[str] = []

    # Look for "for example", "e.g.", "such as" patterns
    example_pattern = re.compile(
        r"(?:(?:for example|e\.g\.|such as|for instance)[,:]?\s*)(.{15,300}?)[.!]",
        re.IGNORECASE,
    )
    for match in example_pattern.finditer(text):
        ex = normalize_whitespace(match.group(1))
        if ex not in examples:
            examples.append(ex)

    # Extract code blocks as examples
    code_pattern = re.compile(r"```\w*\n([\s\S]*?)```")
    for match in code_pattern.finditer(text):
        code = match.group(1).strip()
        if len(code) > 10 and code not in examples:
            examples.append(f"```\n{code}\n```")

    return examples[:5]  # Cap at 5 examples


def distill_all(raws: list[RawConcept]) -> list[Concept]:
    """Distill all raw concepts into clean Concepts."""
    return [distill_concept(raw) for raw in raws]
