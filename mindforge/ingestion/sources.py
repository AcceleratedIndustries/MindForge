"""SourceAdapter protocol.

The pipeline ingests from multiple source types: markdown transcripts today;
Claude Code project JSONL, ChatGPT exports, Cursor logs, Hermes transcripts
tomorrow (Phase 4). All of them boil down to a list of turns.

This module defines the adapter protocol. One adapter ships today:
MarkdownSourceAdapter (see parser.py). Future adapters register themselves
via register_adapter().
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from mindforge.ingestion.parser import Transcript


@runtime_checkable
class SourceAdapter(Protocol):
    """A source adapter produces a Transcript from a path."""

    def parse(self, path: Path) -> Transcript: ...


# extension (lowercase, includes the dot) -> adapter factory
_ADAPTERS: dict[str, Callable[[], SourceAdapter]] = {}


def register_adapter(extension: str, factory: Callable[[], SourceAdapter]) -> None:
    """Register an adapter factory for a file extension (e.g. ``.md``)."""
    _ADAPTERS[extension.lower()] = factory


def get_adapter_for(path: Path) -> SourceAdapter:
    """Return a configured adapter for the given path (dispatch by extension)."""
    ext = path.suffix.lower()
    factory = _ADAPTERS.get(ext)
    if factory is None:
        raise ValueError(f"No SourceAdapter registered for extension: {ext!r}")
    return factory()


def registered_extensions() -> list[str]:
    """Return a sorted list of registered extensions."""
    return sorted(_ADAPTERS.keys())
