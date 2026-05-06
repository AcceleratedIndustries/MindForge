"""MCP tool: ``path_between`` — graph shortest path with optional LLM narration.

Returns the shortest concept chain between two slugs (treating edges as
undirected), the relationship types along the chain, and an LLM-generated
prose narrative. The path is computed regardless of LLM availability; only
the narrative degrades to an empty string when the synthesis backend is
unreachable.
"""

from __future__ import annotations

import json

from mindforge.distillation.concept import ConceptStore
from mindforge.graph.builder import KnowledgeGraph
from mindforge.llm.client import LLMClient
from mindforge.mcp.safety import wrap_retrieved_content

_NARRATE_PROMPT = """\
Given the following ordered chain of concepts and the relationship between each
adjacent pair, write a single neutral prose sentence (no markdown, no list)
describing the conceptual chain.

CHAIN: {chain}
RELATIONSHIPS: {edges}
"""


def handle_path_between(
    *,
    store: ConceptStore,
    graph: KnowledgeGraph,
    llm_client: LLMClient,
    from_concept: str,
    to_concept: str,
    max_hops: int = 4,
) -> str:
    paths = graph.shortest_paths(from_concept, to_concept, max_length=max_hops, max_paths=1)
    if not paths:
        return wrap_retrieved_content(
            json.dumps(
                {
                    "found": False,
                    "path": [],
                    "narrative": "",
                    "edge_types": [],
                }
            )
        )
    path = paths[0]

    edges: list[str] = []
    edge_types: list[str] = []
    for a, b in zip(path, path[1:], strict=False):
        ca = store.concepts.get(a)
        cb = store.concepts.get(b)
        # Edge may live on either endpoint since shortest_paths uses an
        # undirected projection of the graph.
        rel_type: str | None = None
        if ca is not None:
            for rel in ca.relationships:
                if rel.target == b:
                    rel_type = rel.rel_type.value
                    break
        if rel_type is None and cb is not None:
            for rel in cb.relationships:
                if rel.target == a:
                    rel_type = rel.rel_type.value
                    break
        if rel_type is not None:
            edge_types.append(rel_type)
            edges.append(f"{a} {rel_type} {b}")

    narrative = ""
    if llm_client.available:
        prompt = _NARRATE_PROMPT.format(chain=" -> ".join(path), edges="; ".join(edges))
        response = llm_client.generate(prompt)
        if response.success:
            narrative = response.content.strip()

    return wrap_retrieved_content(
        json.dumps(
            {
                "found": True,
                "path": path,
                "narrative": narrative,
                "edge_types": edge_types,
            }
        )
    )
