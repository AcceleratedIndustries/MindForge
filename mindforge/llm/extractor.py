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

from mindforge.distillation.raw import RawConcept
from mindforge.ingestion.chunker import Chunk
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
    rejected_by_grounding: int = 0


def _name_in_text(name: str, text: str) -> bool:
    """Return True iff ``name`` appears in ``text`` as a token-bounded substring.

    The grounding filter's primary purpose is catching stock-AI hallucinations
    (KV Cache, Vector Embeddings, RAG, etc.) the LLM emits unprompted in
    projects whose source text doesn't mention them. Match rules:

    - case-insensitive
    - alphanumeric word boundaries on both sides — so "RAG" doesn't match
      inside "storage" / "coverage" / "paragraph" / "drag"
    - simple plural fallback: if the literal name doesn't match, try
      stripping a trailing 's' from the last word ("Vector Embeddings"
      grounded by source "vector embedding")
    """
    if not name:
        return False
    text_lower = text.lower()
    candidates = [name.lower()]
    if name.lower().endswith("s") and len(name) > 3:
        candidates.append(name.lower()[:-1])
    for cand in candidates:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(cand) + r"(?![A-Za-z0-9])"
        if re.search(pattern, text_lower):
            return True
    return False


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

        concepts.append(
            RawConcept(
                name=name,
                raw_content=raw_content[:5000],
                source_chunks=source_chunks,
                source_files=source_files,
                extraction_method="llm",
                confidence=0.9,  # LLM extraction gets high confidence
            )
        )

    return concepts


def _batch_chunks(chunks: list[Chunk], max_chars: int = 6000) -> list[list[Chunk]]:
    """Group chunks into batches that fit within the LLM context window.

    Keeps chunks together up to max_chars total, so each LLM call
    processes a coherent block of text. Never mixes chunks from
    different source files: every concept extracted from a batch is
    attributed to the batch's source files, so cross-file batches
    poison per-file provenance and break incremental soft-delete on
    file removal.
    """
    batches: list[list[Chunk]] = []
    current_batch: list[Chunk] = []
    current_size = 0
    current_source: str | None = None

    for chunk in chunks:
        chunk_size = len(chunk.content)
        crosses_file = current_batch and chunk.source_file != current_source
        too_big = current_batch and current_size + chunk_size > max_chars
        if crosses_file or too_big:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
            current_source = None
        current_batch.append(chunk)
        current_size += chunk_size
        current_source = chunk.source_file

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
        response = client.generate(prompt, system=EXTRACTION_SYSTEM_PROMPT, response_format="json")
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

        # Grounding filter: reject concepts whose name doesn't appear in
        # the source text the LLM saw. Catches stock-AI hallucinations
        # (KV Cache, Vector Embeddings, RAG) the model emits unprompted
        # in projects unrelated to those topics. For surviving concepts,
        # narrow source_chunks to only the chunks whose content actually
        # contains the name — produces accurate per-concept provenance
        # so `mindforge show <slug> --sources` returns the supporting
        # span instead of the whole batch. (The distiller's
        # `_build_source_refs` derives turn_indices, chunk_id, and
        # snippet from source_chunks, so this narrowing flows downstream
        # without any distiller changes.)
        grounded = []
        for concept in concepts:
            if not _name_in_text(concept.name, batch_text):
                stats.rejected_by_grounding += 1
                logger.info("grounding filter rejected '%s' (not in source text)", concept.name)
                continue
            supporting_chunks = [c for c in batch if _name_in_text(concept.name, c.content)]
            concept.source_chunks = [c.id for c in supporting_chunks]
            grounded.append(concept)

        for concept in grounded:
            name_lower = concept.name.lower()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                all_concepts.append(concept)

        stats.concepts_extracted += len(grounded)

    return all_concepts, stats
