"""MCP tool: ``get_subgraph`` — raw graph delivery (Tier 4).

Returns the n-hop subgraph centered on a concept as both a structured
JSON payload (``{nodes, edges}``) and a markdown rendering. The markdown
is convenient for agents that want a single-blob view; the JSON is for
programmatic consumption. The full payload is wrapped in the
indirect-prompt-injection-safe content tag.
"""

from __future__ import annotations

import json
from typing import Any

from mindforge.distillation.concept import ConceptStore
from mindforge.graph.builder import KnowledgeGraph
from mindforge.mcp.safety import wrap_retrieved_content


def _render_markdown(center: str, depth: int, sub: dict[str, list[dict[str, Any]]]) -> str:
    lines = [f"# Subgraph centered on `{center}` (depth={depth})", "", "## Nodes"]
    lines += [f"- `{n['id']}` — {n['label']}" for n in sub["nodes"]]
    lines.append("")
    lines.append("## Edges")
    if sub["edges"]:
        lines += [f"- `{e['source']}` **{e['type']}** `{e['target']}`" for e in sub["edges"]]
    else:
        lines.append("(none)")
    return "\n".join(lines)


def handle_get_subgraph(
    *,
    store: ConceptStore,
    graph: KnowledgeGraph,
    center: str,
    depth: int = 1,
    edge_types: list[str] | None = None,
) -> str:
    sub = graph.subgraph(center, depth=depth, edge_types=edge_types)
    md = _render_markdown(center, depth, sub)
    # store is unused today but kept in the signature so future revisions can
    # inject (e.g.) per-concept tag filtering without changing call sites.
    _ = store
    return wrap_retrieved_content(json.dumps({"json": sub, "markdown": md}))
