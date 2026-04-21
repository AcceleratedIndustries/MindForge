# Feature: MCP Server Extensions

**Phase:** 3.3
**Depends on:** hybrid retrieval (3.4), provenance (1.1)
**Unblocks:** richer agent workflows, differentiated positioning vs. vector-only memory tools

---

## Motivation

The MCP server today exposes basic lookups: `search`, `get_concept`, `list_concepts`, `get_neighbors`, `get_stats`.

Agents want **graph-shaped** queries: "what does X depend on?", "how are A and B connected?", "give me everything relevant to question Q as one blob."

This is the feature that differentiates MindForge from vector-database memory tools, which can't answer any of these.

---

## New MCP tools

### `get_subgraph`

```json
{
  "name": "get_subgraph",
  "description": "Get a subgraph centered on a concept, with configurable depth and edge type filters.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "center": {"type": "string", "description": "Concept slug or name"},
      "depth": {"type": "integer", "default": 1, "maximum": 3},
      "edge_types": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Filter to these edge types (e.g., [\"uses\", \"depends_on\"])"
      }
    },
    "required": ["center"]
  }
}
```

Returns nodes + edges as JSON, plus a markdown rendering for agent consumption.

### `find_path`

```json
{
  "name": "find_path",
  "description": "Find the shortest paths between two concepts via the knowledge graph.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "from": {"type": "string"},
      "to": {"type": "string"},
      "max_length": {"type": "integer", "default": 4},
      "max_paths": {"type": "integer", "default": 3}
    },
    "required": ["from", "to"]
  }
}
```

Returns the top-N shortest paths with the concepts and edge types along each.

### `explain_relationship`

```json
{
  "name": "explain_relationship",
  "description": "Given two concepts, return the direct edge (if any), relevant shared neighbors, and the provenance snippets that established the relationship.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "a": {"type": "string"},
      "b": {"type": "string"}
    },
    "required": ["a", "b"]
  }
}
```

Pulls the edge type, confidence, and the original source snippets that grounded the relationship.

### `get_context_pack`

```json
{
  "name": "get_context_pack",
  "description": "Given a question, return a single prompt-ready markdown blob containing the top-k concepts, their definitions, and their key relationships. Designed to be inserted into an agent's context.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "top_k": {"type": "integer", "default": 5},
      "max_chars": {"type": "integer", "default": 8000},
      "include_graph": {"type": "boolean", "default": true}
    },
    "required": ["query"]
  }
}
```

This is the **single most-used tool in a real agent workflow**. Returns something like:

```markdown
# Context for: "how does RAG use vector search?"

## RAG
RAG is a pattern combining retrieval with generation, where a model retrieves
relevant documents and grounds its output in them.

Related: [[Semantic Search]], [[Vector Embeddings]]

## Semantic Search
Semantic Search uses embeddings to find conceptually similar content...

## Relationships
- RAG uses Semantic Search
- Semantic Search depends on Vector Embeddings
- Vector Embeddings enables Similarity Search
```

Trimmed to `max_chars`, prioritizing direct hits first, then one-hop neighbors.

### `list_review_queue` (Phase 1.3 dependent)

```json
{
  "name": "list_review_queue",
  "description": "List concepts needing review (conflicted, stale, or orphaned).",
  "inputSchema": {
    "type": "object",
    "properties": {
      "status": {"type": "string", "enum": ["conflicted", "stale", "orphaned", "any"]},
      "limit": {"type": "integer", "default": 20}
    }
  }
}
```

Lets an agent act as a human-in-the-loop reviewer: surface conflicts, propose resolutions, the human approves.

---

## Files touched

### New
- `mindforge/mcp/tools/subgraph.py`
- `mindforge/mcp/tools/path.py`
- `mindforge/mcp/tools/explain.py`
- `mindforge/mcp/tools/context_pack.py`
- `mindforge/mcp/tools/review.py`

### Modified
- `mindforge/mcp/server.py` — register new tools, route calls
- `mindforge/graph/builder.py` — add `subgraph()`, `shortest_paths()` methods if not present
- `mindforge/query/engine.py` — add `context_pack()` method (composes retrieval + graph walk)

---

## Testing

- `tests/test_mcp_tools.py` — one test per tool, using the MCP JSON-RPC framing
- `tests/test_context_pack.py` — end-to-end: query, pack, assert the pack is well-formed markdown and under `max_chars`

---

## Open questions

- **Context pack templating:** should the output template be user-configurable? **Proposed:** ship a sensible default, allow `--template` pointer to a Jinja-like template file. Low priority for v1.
- **Path-finding edge weights:** unweighted shortest-path is fine for v1. Weighted (by edge confidence) is better but adds tuning surface. **Proposed:** unweighted, revisit if paths feel irrelevant in practice.
- **Write tools:** should the MCP server expose `create_concept`, `update_concept`, `delete_concept`? Risky — agents that hallucinate would corrupt the KB. **Proposed:** no write tools in v1. If added later, gate behind explicit `--allow-writes` flag.
