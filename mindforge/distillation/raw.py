"""RawConcept: candidate concept produced by an extractor before distillation.

Lives here (not in extractors) because it's the input shape the distillation
pipeline consumes. Both the LLM extractor and the mock LLM client produce
RawConcept instances; the distiller is the consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RawConcept:
    """A candidate concept before distillation."""

    name: str
    raw_content: str
    source_chunks: list[str] = field(default_factory=list)  # chunk IDs
    source_files: list[str] = field(default_factory=list)
    extraction_method: str = "unknown"
    confidence: float = 0.5
    source_hash: str = ""  # Content hash for tracking modifications (incremental ingest)
