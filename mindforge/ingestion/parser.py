"""Transcript parser: reads raw conversation files and extracts structured turns."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str        # "human" | "assistant" | "system" | "unknown"
    content: str
    index: int       # position in the conversation
    source_file: str


@dataclass
class Transcript:
    """A parsed conversation transcript."""
    source_file: str
    turns: list[ConversationTurn]

    @property
    def assistant_turns(self) -> list[ConversationTurn]:
        return [t for t in self.turns if t.role == "assistant"]

    @property
    def full_text(self) -> str:
        return "\n\n".join(t.content for t in self.turns)


# Patterns for detecting speaker roles in transcripts
_ROLE_PATTERNS = [
    # "Human:", "User:", "Assistant:", "AI:", "Claude:", "System:" prefixes
    (re.compile(r"^(?:Human|User|Me)\s*:", re.IGNORECASE), "human"),
    (re.compile(r"^(?:Assistant|AI|Claude|Bot|GPT|ChatGPT)\s*:", re.IGNORECASE), "assistant"),
    (re.compile(r"^System\s*:", re.IGNORECASE), "system"),
    # Markdown heading style: "## Human", "### Assistant"
    (re.compile(r"^#{1,4}\s*(?:Human|User|Me)\b", re.IGNORECASE), "human"),
    (re.compile(r"^#{1,4}\s*(?:Assistant|AI|Claude|Bot)\b", re.IGNORECASE), "assistant"),
]


def _detect_role(line: str) -> tuple[str, str] | None:
    """Detect if a line is a role marker. Returns (role, remaining_content) or None."""
    stripped = line.strip()
    for pattern, role in _ROLE_PATTERNS:
        match = pattern.match(stripped)
        if match:
            # Extract content after the role marker
            remaining = stripped[match.end():].strip().lstrip(":").strip()
            return role, remaining
    return None


def parse_transcript(path: Path) -> Transcript:
    """Parse a transcript file into structured turns.

    Supports multiple formats:
    - Role-prefixed lines: "Human: ...", "Assistant: ..."
    - Heading-prefixed: "## Human", "### Assistant"
    - Separator-based: "---" between turns (alternating human/assistant)
    """
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    turns: list[ConversationTurn] = []
    current_role: str | None = None
    current_lines: list[str] = []
    source = str(path)

    def _flush() -> None:
        nonlocal current_role, current_lines
        if current_role and current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                turns.append(ConversationTurn(
                    role=current_role,
                    content=content,
                    index=len(turns),
                    source_file=source,
                ))
        current_lines = []

    for line in lines:
        detected = _detect_role(line)
        if detected is not None:
            _flush()
            current_role, remaining = detected
            if remaining:
                current_lines.append(remaining)
        elif line.strip() == "---" and current_role:
            # Separator: flush current turn
            _flush()
            # Alternate role for separator-based formats
            if current_role == "human":
                current_role = "assistant"
            elif current_role == "assistant":
                current_role = "human"
        else:
            current_lines.append(line)

    _flush()

    # If no roles detected, treat entire file as a single assistant turn
    # (common for knowledge dump files)
    if not turns:
        turns.append(ConversationTurn(
            role="assistant",
            content=text.strip(),
            index=0,
            source_file=source,
        ))

    return Transcript(source_file=source, turns=turns)


def parse_all_transcripts(directory: Path) -> list[Transcript]:
    """Parse all .md and .txt files in a directory."""
    transcripts = []
    for ext in ("*.md", "*.txt"):
        for path in sorted(directory.glob(ext)):
            transcripts.append(parse_transcript(path))
    return transcripts
