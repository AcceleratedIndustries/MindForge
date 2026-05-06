"""MCP tool: ``summarize_query`` — server-side LLM synthesis of retrieved knowledge.

Tier-3 (preferred-for-open-ended-questions) handler. Runs hybrid retrieval +
1-hop graph traversal via ``compose_context_pack``, sends a synthesis prompt
to the configured LLM, returns prose answer + concepts_consulted +
suggested_followup wrapped in the indirect-prompt-injection-safe content tag.
"""

from __future__ import annotations

import json

from mindforge.distillation.concept import ConceptStore
from mindforge.graph.builder import KnowledgeGraph
from mindforge.llm.client import LLMClient
from mindforge.mcp.safety import wrap_retrieved_content
from mindforge.query.context_pack import compose_context_pack

_SYSTEM_PROMPT = """\
You are a knowledge synthesizer. Given retrieved concepts and their relationships,
write a short prose answer (2-5 sentences) to the user's question.

Rules:
- Use only the supplied facts; do not invent.
- If the supplied facts don't answer the question, say so plainly.
- Output prose only - no bullet lists, no JSON wrapping, no markdown headers.
"""

_USER_TEMPLATE = """\
QUESTION: {question}

RETRIEVED CONCEPTS:
{concept_blocks}

RELATIONSHIPS BETWEEN THESE CONCEPTS:
{relationship_block}

Write the prose answer now."""


def handle_summarize_query(
    *,
    store: ConceptStore,
    graph: KnowledgeGraph,
    llm_client: LLMClient,
    question: str,
    top_k: int = 5,
    max_hops: int = 2,
    max_concepts: int = 5,
    focus_tags: list[str] | None = None,
    include_provenance: bool = False,
) -> str:
    if not llm_client.available:
        return wrap_retrieved_content(
            json.dumps(
                {
                    "error": "synthesis_backend_unavailable",
                    "message": (
                        "LLM endpoint unreachable; use raw `get_concept` or `get_subgraph` instead."
                    ),
                }
            )
        )

    pack = compose_context_pack(
        store=store,
        graph=graph,
        query=question,
        top_k=min(top_k, max_concepts),
        max_hops=max_hops,
    )

    concept_blocks = (
        "\n\n".join(f"## {c.name}\n{c.definition}" for c in pack.concepts)
        or "(no relevant concepts found)"
    )
    relationship_block = (
        "\n".join(f"- {r.source} {r.rel_type} {r.target}" for r in pack.relationships) or "(none)"
    )

    user_prompt = _USER_TEMPLATE.format(
        question=question,
        concept_blocks=concept_blocks,
        relationship_block=relationship_block,
    )
    response = llm_client.generate(user_prompt, system=_SYSTEM_PROMPT)
    if not response.success:
        return wrap_retrieved_content(
            json.dumps({"error": "synthesis_failed", "message": response.error})
        )

    answer = response.content.strip()
    payload: dict[str, object] = {
        "answer": answer,
        "concepts_consulted": [c.slug for c in pack.concepts],
        "confidence": round(pack.confidence, 3),
        "suggested_followup": pack.neighbor_slugs[:5],
    }
    if include_provenance:
        provenance: list[dict[str, object]] = []
        for c in pack.concepts:
            for src in c.sources:
                provenance.append(
                    {
                        "slug": c.slug,
                        "transcript": getattr(src, "transcript_path", ""),
                        "turns": list(getattr(src, "turn_indices", [])),
                    }
                )
        payload["provenance"] = provenance

    # focus_tags is reserved for future filtering; suppress unused-arg lint.
    _ = focus_tags

    return wrap_retrieved_content(json.dumps(payload))
