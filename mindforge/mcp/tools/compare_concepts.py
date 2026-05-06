"""MCP tool: ``compare_concepts`` — synthesized comparison between concepts.

Resolves each input slug-or-name, gathers the relationships between them
that already exist in the graph, sends a comparison prompt to the LLM,
and returns prose + relationship_types + concepts_consulted wrapped in
the indirect-prompt-injection-safe content tag.
"""

from __future__ import annotations

import json

from mindforge.distillation.concept import Concept, ConceptStore
from mindforge.graph.builder import KnowledgeGraph
from mindforge.llm.client import LLMClient
from mindforge.mcp.safety import wrap_retrieved_content

_PROMPT = """\
Write a {aspect}prose comparison of the following concepts. 2-4 sentences.
Use the supplied definitions and the relationships between them. Be neutral and grounded.

CONCEPTS:
{concept_blocks}

RELATIONSHIPS BETWEEN THESE CONCEPTS:
{relationships}
"""


def _resolve(store: ConceptStore, name_or_slug: str) -> Concept | None:
    if name_or_slug in store.concepts:
        return store.concepts[name_or_slug]
    target = name_or_slug.lower()
    return next(
        (c for c in store.concepts.values() if c.name.lower() == target or c.slug == target),
        None,
    )


def handle_compare_concepts(
    *,
    store: ConceptStore,
    graph: KnowledgeGraph,
    llm_client: LLMClient,
    concepts: list[str],
    aspect: str = "",
) -> str:
    if not llm_client.available:
        return wrap_retrieved_content(
            json.dumps(
                {
                    "error": "synthesis_backend_unavailable",
                    "message": "LLM unreachable.",
                }
            )
        )

    resolved: list[Concept] = []
    for name_or_slug in concepts:
        match = _resolve(store, name_or_slug)
        if match is not None:
            resolved.append(match)

    if len(resolved) < 2:
        return wrap_retrieved_content(
            json.dumps(
                {
                    "error": "insufficient_concepts",
                    "message": f"Need at least 2 known concepts; got {len(resolved)}.",
                }
            )
        )

    rel_types: set[str] = set()
    rel_lines: list[str] = []
    slugs_in_pack = {c.slug for c in resolved}
    for c in resolved:
        for r in c.relationships:
            if r.target in slugs_in_pack:
                rel_types.add(r.rel_type.value)
                rel_lines.append(f"- {c.slug} {r.rel_type.value} {r.target}")

    aspect_clause = f"focused on {aspect} " if aspect else ""
    concept_blocks = "\n\n".join(f"## {c.name}\n{c.definition}" for c in resolved)
    rel_block = "\n".join(rel_lines) if rel_lines else "(no direct relationships in graph)"

    prompt = _PROMPT.format(
        aspect=aspect_clause,
        concept_blocks=concept_blocks,
        relationships=rel_block,
    )
    response = llm_client.generate(prompt)
    if not response.success:
        return wrap_retrieved_content(
            json.dumps({"error": "synthesis_failed", "message": response.error})
        )

    return wrap_retrieved_content(
        json.dumps(
            {
                "comparison": response.content.strip(),
                "relationship_types": sorted(rel_types),
                "concepts_consulted": [c.slug for c in resolved],
            }
        )
    )
