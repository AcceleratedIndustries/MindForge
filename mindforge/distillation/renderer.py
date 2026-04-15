"""Markdown renderer: writes Concept objects as clean Markdown files.

Each concept becomes a single Markdown file with:
- YAML frontmatter (tags, confidence, sources)
- Title
- Definition
- Explanation
- Key Insights
- Examples
- Related concepts (wiki-links)
"""

from __future__ import annotations

from pathlib import Path

from mindforge.distillation.concept import Concept


def render_concept(concept: Concept) -> str:
    """Render a Concept as a Markdown string."""
    lines: list[str] = []

    # YAML frontmatter
    lines.append("---")
    lines.append(f"title: \"{concept.name}\"")
    lines.append(f"slug: \"{concept.slug}\"")
    if concept.tags:
        lines.append(f"tags: [{', '.join(concept.tags)}]")
    lines.append(f"confidence: {concept.confidence:.2f}")
    if concept.source_files:
        lines.append("sources:")
        for src in concept.source_files:
            lines.append(f"  - \"{src}\"")
    lines.append("---")
    lines.append("")

    # Title
    lines.append(f"# {concept.name}")
    lines.append("")

    # Definition
    lines.append("## Definition")
    lines.append("")
    lines.append(concept.definition)
    lines.append("")

    # Explanation
    if concept.explanation and concept.explanation != concept.definition:
        lines.append("## Explanation")
        lines.append("")
        lines.append(concept.explanation)
        lines.append("")

    # Key Insights
    if concept.insights:
        lines.append("## Key Insights")
        lines.append("")
        for insight in concept.insights:
            lines.append(f"- {insight}")
        lines.append("")

    # Examples
    if concept.examples:
        lines.append("## Examples")
        lines.append("")
        for example in concept.examples:
            if example.startswith("```"):
                lines.append(example)
            else:
                lines.append(f"- {example}")
        lines.append("")

    # Related Concepts (wiki-links)
    if concept.links:
        lines.append("## Related Concepts")
        lines.append("")
        for link in concept.links:
            lines.append(f"- [[{link}]]")
        lines.append("")

    # Relationships (typed links)
    if concept.relationships:
        lines.append("## Relationships")
        lines.append("")
        for rel in concept.relationships:
            lines.append(f"- **{rel.rel_type.value}**: [[{rel.target}]]")
        lines.append("")

    return "\n".join(lines)


def write_concept(concept: Concept, output_dir: Path) -> Path:
    """Write a concept to a Markdown file and return the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{concept.slug}.md"
    path = output_dir / filename
    path.write_text(render_concept(concept), encoding="utf-8")
    return path


def write_all_concepts(concepts: list[Concept], output_dir: Path) -> list[Path]:
    """Write all concepts to Markdown files."""
    return [write_concept(c, output_dir) for c in concepts]
