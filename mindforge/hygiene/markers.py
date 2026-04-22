"""Dataclasses for conflict markers and variants."""

from __future__ import annotations

from dataclasses import dataclass, field

from mindforge.distillation.source_ref import SourceRef


@dataclass
class ConflictVariant:
    source: SourceRef
    text: str

    def to_dict(self) -> dict:
        return {"source": self.source.to_dict(), "text": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> ConflictVariant:
        return cls(source=SourceRef.from_dict(d["source"]), text=d["text"])


@dataclass
class ConflictMarker:
    field: str  # "definition" | "insights" | "tags"
    variants: list[ConflictVariant] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "variants": [v.to_dict() for v in self.variants],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ConflictMarker:
        return cls(
            field=d["field"],
            variants=[ConflictVariant.from_dict(v) for v in d["variants"]],
        )
