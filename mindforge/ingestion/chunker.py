"""Semantic chunker: splits transcript turns into coherent knowledge chunks.

Unlike fixed-size chunking, this respects semantic boundaries:
- Paragraph boundaries
- Topic shifts (detected by heading patterns and keyword changes)
- Code block boundaries
- List item groupings
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mindforge.ingestion.parser import ConversationTurn


@dataclass
class Chunk:
    """A semantically coherent piece of text from a transcript."""
    content: str
    source_file: str
    turn_index: int
    chunk_index: int
    chunk_type: str  # "prose", "code", "list", "definition"

    @property
    def id(self) -> str:
        return f"{self.source_file}:t{self.turn_index}:c{self.chunk_index}"


def _classify_block(text: str) -> str:
    """Classify a text block by its dominant structure."""
    stripped = text.strip()
    if stripped.startswith("```") or stripped.startswith("    ") and "\n    " in stripped:
        return "code"
    if re.match(r"^[\s]*[-*]\s", stripped) or re.match(r"^[\s]*\d+\.\s", stripped):
        return "list"
    # Definition patterns: "X is...", "X refers to...", "X: ..."
    if re.match(r"^[A-Z][\w\s]{2,30}(?:is|refers to|means|describes|represents)\b", stripped):
        return "definition"
    return "prose"


def _split_by_headings(text: str) -> list[str]:
    """Split text at markdown heading boundaries."""
    # Split at lines starting with # (but keep the heading with its section)
    sections = re.split(r"\n(?=#{1,4}\s)", text)
    return [s.strip() for s in sections if s.strip()]


def _split_by_paragraphs(text: str) -> list[str]:
    """Split text at double-newline paragraph boundaries."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_code_blocks(text: str) -> list[str]:
    """Separate code blocks from surrounding prose."""
    parts: list[str] = []
    # Match fenced code blocks
    pattern = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)
    last_end = 0
    for match in pattern.finditer(text):
        before = text[last_end:match.start()].strip()
        if before:
            parts.append(before)
        parts.append(match.group(0))
        last_end = match.end()
    after = text[last_end:].strip()
    if after:
        parts.append(after)
    return parts if parts else [text]


def _merge_small_chunks(chunks: list[str], min_length: int = 80) -> list[str]:
    """Merge very small chunks with their neighbors.

    Code blocks (starting with ```) are never merged with other chunks.
    """
    if not chunks:
        return chunks

    merged: list[str] = []
    buffer = ""

    for chunk in chunks:
        is_code = chunk.strip().startswith("```")

        if is_code:
            # Flush buffer before code block
            if buffer:
                if merged:
                    merged[-1] = merged[-1] + "\n\n" + buffer
                else:
                    merged.append(buffer)
                buffer = ""
            merged.append(chunk)
        elif buffer:
            buffer = buffer + "\n\n" + chunk
            if len(buffer) >= min_length:
                merged.append(buffer)
                buffer = ""
        elif len(chunk) < min_length:
            buffer = chunk
        else:
            merged.append(chunk)

    if buffer:
        if merged:
            merged[-1] = merged[-1] + "\n\n" + buffer
        else:
            merged.append(buffer)

    return merged


def chunk_turn(turn: ConversationTurn) -> list[Chunk]:
    """Split a conversation turn into semantic chunks."""
    text = turn.content.strip()
    if not text:
        return []

    # Step 1: separate code blocks from prose
    code_separated = _split_code_blocks(text)

    raw_chunks: list[str] = []
    for block in code_separated:
        if block.startswith("```"):
            # Keep code blocks intact
            raw_chunks.append(block)
        else:
            # Step 2: split prose by headings
            sections = _split_by_headings(block)
            for section in sections:
                # Step 3: split sections by paragraphs
                paragraphs = _split_by_paragraphs(section)
                raw_chunks.extend(paragraphs)

    # Step 4: merge fragments that are too small
    raw_chunks = _merge_small_chunks(raw_chunks)

    # Build typed Chunk objects
    chunks = []
    for i, content in enumerate(raw_chunks):
        chunks.append(Chunk(
            content=content,
            source_file=turn.source_file,
            turn_index=turn.index,
            chunk_index=i,
            chunk_type=_classify_block(content),
        ))

    return chunks


def chunk_turns(turns: list[ConversationTurn]) -> list[Chunk]:
    """Chunk all turns, returning a flat list of chunks."""
    chunks = []
    for turn in turns:
        chunks.extend(chunk_turn(turn))
    return chunks
