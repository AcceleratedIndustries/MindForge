"""Concept extractor: identifies candidate concepts from text chunks.

Uses heuristic and statistical methods (no LLM required):
- Pattern matching for definitions and explanations
- Keyword frequency analysis
- Technical term detection
- Section heading extraction
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mindforge.ingestion.chunker import Chunk
from mindforge.utils.text import extract_keywords, normalize_whitespace


@dataclass
class RawConcept:
    """A candidate concept before distillation."""
    name: str
    raw_content: str
    source_chunks: list[str] = field(default_factory=list)  # chunk IDs
    source_files: list[str] = field(default_factory=list)
    extraction_method: str = "unknown"
    confidence: float = 0.5


# Patterns that indicate a definition or explanation
_DEFINITION_PATTERNS = [
    # "X is a/an ..."
    re.compile(
        r"(?:^|\.\s+)(?P<term>[A-Z][\w\s]{1,40}?)\s+(?:is|are)\s+(?:a|an|the)\s+(?P<def>.{20,300}?)[.!]",
        re.MULTILINE,
    ),
    # "X refers to ..."
    re.compile(
        r"(?:^|\.\s+)(?P<term>[A-Z][\w\s]{1,40}?)\s+(?:refers?\s+to|means?|describes?|represents?)\s+(?P<def>.{20,300}?)[.!]",
        re.MULTILINE,
    ),
    # "X: description"  (heading-like)
    re.compile(
        r"^(?:#+\s+)?(?P<term>[A-Z][\w\s/\-]{2,40})\s*[-:]\s*(?P<def>.{20,300}?)$",
        re.MULTILINE,
    ),
    # "**X** - description" (bold term with explanation)
    re.compile(
        r"\*\*(?P<term>[A-Za-z][\w\s/\-]{2,40}?)\*\*\s*[-:–]\s*(?P<def>.{20,300}?)[.!]",
    ),
]

# Patterns for headings that name concepts
_HEADING_PATTERN = re.compile(r"^#{1,4}\s+(.{3,60})$", re.MULTILINE)

# Technical term patterns (CamelCase, acronyms, compound terms)
_TECH_TERM_PATTERNS = [
    re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b"),  # CamelCase
    re.compile(r"\b([A-Z]{2,6})\b"),  # Acronyms
]

# Generic words that should never be concepts on their own
_BLOCKED_NAMES = {
    "this", "that", "these", "those", "here", "there", "what", "which",
    "how", "why", "when", "where", "who", "the", "key", "critical",
    "important", "note", "meaning", "definition", "example", "examples",
    "overview", "summary", "conclusion", "introduction", "section",
    "part", "step", "result", "value", "type", "data", "system",
    "method", "approach", "process", "model", "query", "search",
    "token", "vector", "chunk", "generation", "retrieval", "semantic",
    "attention", "it", "they", "its", "their", "for each new token generated",
    "applications", "comparison", "popular options", "performance impact",
    "memory considerations", "key insight", "advanced", "patterns",
    "how it works", "how they are created", "why it matters",
    "why embeddings matter", "why rag matters", "rag architecture",
    "comparison with keyword search", "applications semantic search",
    "advanced rag patterns", "ingestion pipeline", "query pipeline",
    "indexing phase", "query phase", "cache", "documents",
    "applications", "performance impact",
}


def _is_valid_concept_name(name: str) -> bool:
    """Check if a concept name is specific enough to be useful."""
    name_lower = name.lower().strip()

    # Blocked generic names
    if name_lower in _BLOCKED_NAMES:
        return False

    # Too short (single word under 4 chars)
    words = name_lower.split()
    if len(words) == 1 and len(name_lower) < 4:
        return False

    # Starts with articles or pronouns
    if words[0] in ("the", "a", "an", "this", "that", "it", "its"):
        # Allow if the rest is meaningful and multi-word (e.g., "The KV Cache")
        if len(words) < 3:
            return False

    # Sentences (too long to be a concept name)
    if len(words) > 6:
        return False

    # Must contain at least one alphabetic word > 2 chars
    if not any(len(w) > 2 and w.isalpha() for w in words):
        return False

    # Names starting with verbs/prepositions are usually sentence fragments
    fragment_starters = {
        "it", "this", "that", "there", "here", "with", "without",
        "for", "from", "into", "about", "also", "each", "every",
        "some", "any", "all", "both", "most", "more", "less",
        "llms", "using", "based",
    }
    if words[0] in fragment_starters:
        return False

    # Reject if name looks like a phrase/clause (contains common verbs)
    verb_indicators = {"is", "are", "was", "were", "has", "have", "uses", "does"}
    if len(words) > 2 and any(w in verb_indicators for w in words[1:]):
        return False

    return True


def _extract_definitions(text: str) -> list[RawConcept]:
    """Extract concepts from definition patterns."""
    concepts = []
    seen_names: set[str] = set()

    for pattern in _DEFINITION_PATTERNS:
        for match in pattern.finditer(text):
            name = normalize_whitespace(match.group("term"))
            definition = normalize_whitespace(match.group("def"))

            # Skip if too short, too generic, or already seen
            name_lower = name.lower()
            if len(name) < 3 or name_lower in seen_names:
                continue
            if not _is_valid_concept_name(name):
                continue
            seen_names.add(name_lower)

            # Capture surrounding context (up to 500 chars after the match)
            context_end = min(match.end() + 500, len(text))
            after_context = text[match.end():context_end].strip()

            # Build raw content: the full match sentence + surrounding context
            full_sentence = normalize_whitespace(match.group(0))
            full_content = full_sentence
            if after_context:
                full_content += "\n\n" + after_context

            concepts.append(RawConcept(
                name=name,
                raw_content=full_content[:2000],
                extraction_method="definition_pattern",
                confidence=0.8,
            ))

    return concepts


def _extract_from_headings(text: str, full_text: str) -> list[RawConcept]:
    """Extract concepts from markdown headings and their content."""
    concepts = []
    seen: set[str] = set()

    headings = _HEADING_PATTERN.findall(text)
    for heading in headings:
        name = heading.strip().strip("*#").strip()
        name_lower = name.lower()
        if name_lower in seen or len(name) < 3:
            continue
        if not _is_valid_concept_name(name):
            continue
        seen.add(name_lower)

        # Find content after this heading (up to next heading or end)
        escaped = re.escape(heading)
        pattern = re.compile(
            r"#{{1,4}}\s+{}\s*\n([\s\S]*?)(?=\n#{{1,4}}\s|\Z)".format(escaped),
            re.MULTILINE,
        )
        match = pattern.search(full_text)
        content = match.group(1).strip() if match else ""

        if len(content) > 20:
            concepts.append(RawConcept(
                name=name,
                raw_content=content[:2000],
                extraction_method="heading",
                confidence=0.7,
            ))

    return concepts


def _extract_keyword_concepts(chunks: list[Chunk]) -> list[RawConcept]:
    """Extract concepts based on keyword frequency across chunks.

    Groups chunks by dominant keywords to identify recurring topics
    that may not have explicit definitions.
    """
    # Collect keywords across all chunks
    keyword_chunks: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        keywords = extract_keywords(chunk.content, top_n=5)
        for kw in keywords:
            keyword_chunks.setdefault(kw, []).append(chunk)

    concepts = []
    # Only create concepts for keywords that appear in multiple chunks
    for keyword, kw_chunks in keyword_chunks.items():
        if len(kw_chunks) < 3:  # require 3+ mentions for keyword concepts
            continue

        name = keyword.title()
        if not _is_valid_concept_name(name):
            continue

        # Find the chunk with the most content about this keyword
        best_chunk = max(
            kw_chunks,
            key=lambda c: c.content.lower().count(keyword),
        )

        concepts.append(RawConcept(
            name=name,
            raw_content=best_chunk.content[:2000],
            source_chunks=[c.id for c in kw_chunks],
            source_files=list({c.source_file for c in kw_chunks}),
            extraction_method="keyword_frequency",
            confidence=min(0.4 + len(kw_chunks) * 0.1, 0.8),
        ))

    return concepts


def extract_concepts(chunks: list[Chunk]) -> list[RawConcept]:
    """Extract all candidate concepts from a list of chunks.

    Combines multiple extraction strategies:
    1. Definition pattern matching (highest confidence)
    2. Heading extraction (medium confidence)
    3. Keyword frequency analysis (lower confidence)
    """
    all_concepts: list[RawConcept] = []
    seen_names: set[str] = set()

    # Combine all chunk text for full-document analysis
    full_text = "\n\n".join(c.content for c in chunks)

    # Strategy 1: Definition patterns
    for chunk in chunks:
        for concept in _extract_definitions(chunk.content):
            name_lower = concept.name.lower()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                concept.source_chunks = [chunk.id]
                concept.source_files = [chunk.source_file]
                all_concepts.append(concept)

    # Strategy 2: Heading extraction
    for concept in _extract_from_headings(full_text, full_text):
        name_lower = concept.name.lower()
        if name_lower not in seen_names:
            seen_names.add(name_lower)
            all_concepts.append(concept)

    # Strategy 3: Keyword frequency
    for concept in _extract_keyword_concepts(chunks):
        name_lower = concept.name.lower()
        if name_lower not in seen_names:
            seen_names.add(name_lower)
            all_concepts.append(concept)

    return all_concepts
