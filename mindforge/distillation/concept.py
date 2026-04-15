"""Core data models for MindForge concepts and relationships."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from mindforge.utils.text import slugify, content_hash


class RelationshipType(str, Enum):
    """Types of relationships between concepts."""
    USES = "uses"
    IMPROVES = "improves"
    DEPENDS_ON = "depends_on"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    EXAMPLE_OF = "example_of"
    CONTRASTS_WITH = "contrasts_with"
    ENABLES = "enables"


@dataclass
class Relationship:
    """A directed relationship between two concepts."""
    source: str  # concept slug
    target: str  # concept slug
    rel_type: RelationshipType
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.rel_type.value,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Relationship:
        return cls(
            source=data["source"],
            target=data["target"],
            rel_type=RelationshipType(data["type"]),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class Concept:
    """An atomic knowledge concept extracted from transcripts."""
    name: str
    definition: str  # 2-3 sentence definition
    explanation: str  # Expanded explanation
    insights: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    confidence: float = 1.0
    links: list[str] = field(default_factory=list)  # wiki-link targets (slugs)
    relationships: list[Relationship] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return slugify(self.name)

    @property
    def hash(self) -> str:
        return content_hash(f"{self.name}:{self.definition}")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "slug": self.slug,
            "definition": self.definition,
            "explanation": self.explanation,
            "insights": self.insights,
            "examples": self.examples,
            "tags": self.tags,
            "source_files": self.source_files,
            "confidence": self.confidence,
            "links": self.links,
            "relationships": [r.to_dict() for r in self.relationships],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Concept:
        rels = [Relationship.from_dict(r) for r in data.get("relationships", [])]
        return cls(
            name=data["name"],
            definition=data["definition"],
            explanation=data["explanation"],
            insights=data.get("insights", []),
            examples=data.get("examples", []),
            tags=data.get("tags", []),
            source_files=data.get("source_files", []),
            confidence=data.get("confidence", 1.0),
            links=data.get("links", []),
            relationships=rels,
        )

    def merge_with(self, other: Concept) -> Concept:
        """Merge another concept into this one, combining insights and sources."""
        merged_insights = list(dict.fromkeys(self.insights + other.insights))
        merged_examples = list(dict.fromkeys(self.examples + other.examples))
        merged_tags = list(dict.fromkeys(self.tags + other.tags))
        merged_sources = list(dict.fromkeys(self.source_files + other.source_files))

        # Keep the longer/better explanation
        explanation = self.explanation if len(self.explanation) >= len(other.explanation) else other.explanation
        definition = self.definition if len(self.definition) >= len(other.definition) else other.definition

        return Concept(
            name=self.name,
            definition=definition,
            explanation=explanation,
            insights=merged_insights,
            examples=merged_examples,
            tags=merged_tags,
            source_files=merged_sources,
            confidence=max(self.confidence, other.confidence),
            links=list(dict.fromkeys(self.links + other.links)),
            relationships=self.relationships + other.relationships,
        )


@dataclass
class ConceptStore:
    """In-memory store for all concepts, supporting lookup and persistence."""
    concepts: dict[str, Concept] = field(default_factory=dict)  # slug -> Concept

    def add(self, concept: Concept) -> None:
        slug = concept.slug
        if slug in self.concepts:
            self.concepts[slug] = self.concepts[slug].merge_with(concept)
        else:
            self.concepts[slug] = concept

    def get(self, slug: str) -> Concept | None:
        return self.concepts.get(slug)

    def all(self) -> list[Concept]:
        return list(self.concepts.values())

    def slugs(self) -> list[str]:
        return list(self.concepts.keys())

    def save(self, path: Path) -> None:
        """Save all concepts to a JSON manifest."""
        data = {slug: c.to_dict() for slug, c in self.concepts.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> ConceptStore:
        """Load concepts from a JSON manifest."""
        store = cls()
        if path.exists():
            data = json.loads(path.read_text())
            for slug, cdata in data.items():
                store.concepts[slug] = Concept.from_dict(cdata)
        return store
