"""LLM-based concept extractor: uses an LLM to identify and extract concepts from text.

This replaces or augments the heuristic extractor with dramatically better
concept identification, definition extraction, and relationship detection.

The LLM is prompted to return structured JSON, which is parsed into the same
RawConcept objects used by the heuristic extractor — fully compatible with
the existing deduplication and distillation pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from mindforge.ingestion.chunker import Chunk
from mindforge.ingestion.extractor import RawConcept
from mindforge.llm.client import LLMClient

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """\
You are a knowledge extraction engine. Your job is to identify distinct \
technical concepts from text and extract them as structured data.

Rules:
- Each concept must be atomic: one idea per concept.
- Concept names should be proper nouns or technical terms (e.g., "KV Cache", \
"Vector Embeddings", "Retrieval-Augmented Generation").
- Do NOT extract generic words like "performance", "system", "method".
- Do NOT extract section headings that are not concepts (e.g., "How It Works", \
"Applications").
- Definitions should be 1-3 clear sentences, written in a neutral encyclopedic tone.
- Remove all conversational language ("As I mentioned", "Great question", etc.).
- Identify relationships between concepts you extract.

You MUST respond with valid JSON only. No markdown fences, no explanations."""

EXTRACTION_USER_PROMPT = """\
Extract all distinct technical concepts from the following text. \
For each concept, provide its name, a clean definition, key insights, \
and any relationships to other concepts.

TEXT:
{text}

Respond with this exact JSON structure:
{{
  "concepts": [
    {{
      "name": "Concept Name",
      "definition": "A clear 1-3 sentence definition.",
      "explanation": "An expanded explanation with more detail.",
      "insights": ["Key insight 1", "Key insight 2"],
      "examples": ["Example if present"],
      "tags": ["tag1", "tag2"],
      "relationships": [
        {{"target": "Other Concept Name", "type": "uses|depends_on|related_to|enables|improves|part_of|contrasts_with|example_of"}}
      ]
    }}
  ]
}}"""


@dataclass
class ExtractionStats:
    """Track LLM extraction statistics."""
    chunks_processed: int = 0
    llm_calls: int = 0
    concepts_extracted: int = 0
    parse_failures: int = 0
    fallbacks_to_heuristic: int = 0


def _extract_json_from_response(text: str) -> dict | None:
    """Parse JSON from LLM response, handling common formatting issues."""
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the response
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _parse_llm_concepts(
    response_data: dict,
    source_chunks: list[str],
    source_files: list[str],
) -> list[RawConcept]:
    """Parse LLM JSON response into RawConcept objects."""
    concepts = []
    raw_concepts = response_data.get("concepts", [])

    for item in raw_concepts:
        name = item.get("name", "").strip()
        if not name or len(name) < 3:
            continue

        definition = item.get("definition", "").strip()
        explanation = item.get("explanation", "").strip()
        insights = item.get("insights", [])
        examples = item.get("examples", [])
        tags = item.get("tags", [])
        relationships = item.get("relationships", [])

        # Build rich raw content from all LLM-extracted fields
        content_parts = []
        if definition:
            content_parts.append(definition)
        if explanation and explanation != definition:
            content_parts.append(explanation)
        if insights:
            content_parts.append("\n".join(f"- {i}" for i in insights))
        if examples:
            content_parts.append("Examples:\n" + "\n".join(f"- {e}" for e in examples))

        # Encode relationships and metadata as structured content
        # (the distiller and linker will process these)
        if relationships:
            rel_lines = []
            for rel in relationships:
                target = rel.get("target", "")
                rel_type = rel.get("type", "related_to")
                if target:
                    rel_lines.append(f"[[rel:{rel_type}:{target}]]")
            if rel_lines:
                content_parts.append("\n".join(rel_lines))

        if tags:
            content_parts.append(f"[[tags:{','.join(tags)}]]")

        raw_content = "\n\n".join(content_parts)

        concepts.append(RawConcept(
            name=name,
            raw_content=raw_content[:5000],
            source_chunks=source_chunks,
            source_files=source_files,
            extraction_method="llm",
            confidence=0.9,  # LLM extraction gets high confidence
        ))

    return concepts


def _batch_chunks(chunks: list[Chunk], max_chars: int = 6000) -> list[list[Chunk]]:
    """Group chunks into batches that fit within the LLM context window.

    Keeps chunks together up to max_chars total, so each LLM call
    processes a coherent block of text.
    """
    batches: list[list[Chunk]] = []
    current_batch: list[Chunk] = []
    current_size = 0

    for chunk in chunks:
        chunk_size = len(chunk.content)
        if current_batch and current_size + chunk_size > max_chars:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append(chunk)
        current_size += chunk_size

    if current_batch:
        batches.append(current_batch)

    return batches


def extract_concepts_llm(
    chunks: list[Chunk],
    client: LLMClient,
    max_chars_per_call: int = 6000,
) -> tuple[list[RawConcept], ExtractionStats]:
    """Extract concepts from chunks using an LLM.

    Batches chunks to fit within context limits, sends each batch
    to the LLM for structured extraction, and parses the results.

    Returns:
        Tuple of (extracted concepts, extraction statistics)
    """
    stats = ExtractionStats()
    all_concepts: list[RawConcept] = []
    seen_names: set[str] = set()

    batches = _batch_chunks(chunks, max_chars=max_chars_per_call)

    for i, batch in enumerate(batches):
        batch_text = "\n\n".join(c.content for c in batch)
        source_chunks = [c.id for c in batch]
        source_files = list({c.source_file for c in batch})

        stats.chunks_processed += len(batch)

        prompt = EXTRACTION_USER_PROMPT.format(text=batch_text[:max_chars_per_call])

        logger.info("LLM extraction batch %d/%d (%d chunks)", i + 1, len(batches), len(batch))
        response = client.generate(prompt, system=EXTRACTION_SYSTEM_PROMPT)
        stats.llm_calls += 1

        if not response.success:
            logger.warning("LLM call failed: %s", response.error)
            stats.parse_failures += 1
            continue

        data = _extract_json_from_response(response.content)
        if data is None:
            logger.warning("Failed to parse LLM response as JSON")
            logger.debug("Raw response: %s", response.content[:500])
            stats.parse_failures += 1
            continue

        concepts = _parse_llm_concepts(data, source_chunks, source_files)

        for concept in concepts:
            name_lower = concept.name.lower()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                all_concepts.append(concept)

        stats.concepts_extracted += len(concepts)

    return all_concepts, stats
