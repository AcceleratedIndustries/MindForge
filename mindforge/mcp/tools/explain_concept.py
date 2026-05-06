"""MCP tool: ``explain_concept`` — compressed concept explanation.

Brief mode is the no-LLM fallback (returns the stored definition + first
insight, trimmed to a budget). Standard and detailed modes call the LLM to
paraphrase the stored explanation into encyclopedic prose. All modes wrap
the JSON payload in the indirect-prompt-injection-safe content tag.
"""

from __future__ import annotations

import json

from mindforge.distillation.concept import ConceptStore
from mindforge.graph.builder import KnowledgeGraph
from mindforge.llm.client import LLMClient
from mindforge.mcp.safety import wrap_retrieved_content

_BRIEF_MAX_CHARS = 400

_PARAPHRASE_PROMPT = """\
Restate the following concept in clear, neutral encyclopedic prose.
- Length target: {target_words} words.
- No conversational language, no markdown headers.
- Stay grounded in the supplied content.

CONCEPT: {name}
DEFINITION: {definition}
EXPLANATION: {explanation}
KEY INSIGHTS: {insights}
"""


def _resolve_slug(store: ConceptStore, name_or_slug: str) -> str | None:
    if name_or_slug in store.concepts:
        return name_or_slug
    target = name_or_slug.lower()
    for slug, c in store.concepts.items():
        if c.name.lower() == target or c.slug == target:
            return slug
    return None


def handle_explain_concept(
    *,
    store: ConceptStore,
    graph: KnowledgeGraph,
    llm_client: LLMClient,
    concept: str,
    depth: str = "standard",
) -> str:
    slug = _resolve_slug(store, concept)
    if slug is None:
        return wrap_retrieved_content(
            json.dumps(
                {
                    "error": "concept_not_found",
                    "message": f"No concept matching '{concept}'.",
                }
            )
        )
    c = store.concepts[slug]
    related = list(graph.neighbors(slug))[:5]

    if depth == "brief":
        explanation = c.definition
        if c.insights:
            explanation = f"{explanation} {c.insights[0]}"
        explanation = explanation[:_BRIEF_MAX_CHARS]
    else:
        if not llm_client.available:
            return wrap_retrieved_content(
                json.dumps(
                    {
                        "error": "synthesis_backend_unavailable",
                        "message": (
                            "LLM unreachable; use depth='brief' or `get_concept` for raw content."
                        ),
                    }
                )
            )
        target_words = 80 if depth == "standard" else 200
        prompt = _PARAPHRASE_PROMPT.format(
            target_words=target_words,
            name=c.name,
            definition=c.definition,
            explanation=c.explanation,
            insights="; ".join(c.insights) or "(none)",
        )
        response = llm_client.generate(prompt)
        if not response.success:
            return wrap_retrieved_content(
                json.dumps({"error": "synthesis_failed", "message": response.error})
            )
        explanation = response.content.strip()

    return wrap_retrieved_content(
        json.dumps(
            {
                "slug": slug,
                "explanation": explanation,
                "related_slugs": related,
            }
        )
    )
