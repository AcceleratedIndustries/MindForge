"""SourceRef: a citation pointer from a concept to its originating turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SNIPPET_MAX_CHARS = 500


@dataclass
class SourceRef:
    transcript_path: str
    transcript_hash: str
    turn_indices: list[int]
    extracted_at: str  # ISO 8601 UTC
    chunk_id: str | None = None
    snippet: str | None = None

    def __post_init__(self) -> None:
        if self.snippet is not None and len(self.snippet) > SNIPPET_MAX_CHARS:
            self.snippet = self.snippet[:SNIPPET_MAX_CHARS]

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcript_path": self.transcript_path,
            "transcript_hash": self.transcript_hash,
            "turn_indices": list(self.turn_indices),
            "extracted_at": self.extracted_at,
            "chunk_id": self.chunk_id,
            "snippet": self.snippet,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceRef:
        return cls(
            transcript_path=data["transcript_path"],
            transcript_hash=data["transcript_hash"],
            turn_indices=list(data["turn_indices"]),
            extracted_at=data["extracted_at"],
            chunk_id=data.get("chunk_id"),
            snippet=data.get("snippet"),
        )
