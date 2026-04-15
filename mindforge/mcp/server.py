"""MCP (Model Context Protocol) server for MindForge.

Exposes the MindForge knowledge base as tools over the MCP protocol,
allowing external AI agents to search, browse, and explore concepts.

Protocol: JSON-RPC 2.0 over stdio (stdin/stdout).
Zero external dependencies — uses only stdlib.

Usage:
    mindforge mcp --output path/to/knowledge-base
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from mindforge.config import MindForgeConfig
from mindforge.distillation.concept import ConceptStore
from mindforge.graph.builder import KnowledgeGraph
from mindforge.query.engine import QueryEngine

logger = logging.getLogger(__name__)

# MCP protocol version
PROTOCOL_VERSION = "2024-11-05"

# Server identity
SERVER_NAME = "mindforge"
SERVER_VERSION = "0.1.0"


# ── Tool definitions ─────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search",
        "description": (
            "Search the MindForge knowledge base with a natural language query. "
            "Returns the top matching concepts with definitions and related concepts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_concept",
        "description": (
            "Get full details of a specific concept by name or slug. "
            "Returns definition, explanation, insights, examples, tags, "
            "and relationships."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Concept name or slug (e.g., 'KV Cache' or 'kv-cache')",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "list_concepts",
        "description": (
            "List all concepts in the knowledge base, optionally filtered by tag. "
            "Returns concept names, slugs, confidence scores, and tags."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tag": {
                    "type": "string",
                    "description": "Filter by tag (optional)",
                },
            },
        },
    },
    {
        "name": "get_neighbors",
        "description": (
            "Get concepts related to a given concept via the knowledge graph. "
            "Returns directly connected concepts and their relationship types."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Concept name or slug",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_stats",
        "description": (
            "Get knowledge base statistics: concept count, edge count, "
            "clusters, most central concepts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ── Tool handlers ────────────────────────────────────────────────────────────

class ToolHandlers:
    """Implements the MCP tool call handlers backed by MindForge state."""

    def __init__(self, config: MindForgeConfig) -> None:
        self.config = config
        self.store = ConceptStore()
        self.graph: KnowledgeGraph | None = None
        self.query_engine: QueryEngine | None = None
        self._load()

    def _load(self) -> None:
        """Load knowledge base state from disk."""
        manifest = self.config.output_dir / "concepts.json"
        graph_file = self.config.graph_dir / "knowledge_graph.json"

        if manifest.exists():
            self.store = ConceptStore.load(manifest)

        if graph_file.exists():
            self.graph = KnowledgeGraph.load(graph_file)

        self.query_engine = QueryEngine(self.store, self.graph, None)

    def _resolve_slug(self, name: str) -> str:
        """Resolve a concept name or slug to a slug."""
        from mindforge.utils.text import slugify

        # Try direct slug lookup first
        slug = slugify(name)
        if self.store.get(slug):
            return slug

        # Try case-insensitive name match
        for concept in self.store.all():
            if concept.name.lower() == name.lower():
                return concept.slug

        return slug  # Return computed slug even if not found

    def handle_search(self, arguments: dict) -> list[dict]:
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 5)

        if not self.query_engine:
            return [{"type": "text", "text": "Knowledge base not loaded."}]

        results = self.query_engine.search(query, top_k=top_k)

        if not results:
            return [{"type": "text", "text": f"No results found for: {query}"}]

        output = []
        for r in results:
            entry = {
                "name": r.concept.name,
                "slug": r.concept.slug,
                "score": round(r.score, 3),
                "definition": r.concept.definition,
                "tags": r.concept.tags,
                "related": r.neighbors[:5],
            }
            output.append(entry)

        return [{"type": "text", "text": json.dumps(output, indent=2)}]

    def handle_get_concept(self, arguments: dict) -> list[dict]:
        name = arguments.get("name", "")
        slug = self._resolve_slug(name)
        concept = self.store.get(slug)

        if not concept:
            return [{"type": "text", "text": f"Concept not found: {name}"}]

        data = {
            "name": concept.name,
            "slug": concept.slug,
            "definition": concept.definition,
            "explanation": concept.explanation,
            "insights": concept.insights,
            "examples": concept.examples,
            "tags": concept.tags,
            "confidence": concept.confidence,
            "links": concept.links,
            "relationships": [
                {"target": r.target, "type": r.rel_type.value}
                for r in concept.relationships
            ],
        }
        return [{"type": "text", "text": json.dumps(data, indent=2)}]

    def handle_list_concepts(self, arguments: dict) -> list[dict]:
        tag_filter = arguments.get("tag", "")
        concepts = self.store.all()

        if tag_filter:
            tag_lower = tag_filter.lower()
            concepts = [c for c in concepts if tag_lower in [t.lower() for t in c.tags]]

        entries = []
        for c in sorted(concepts, key=lambda x: x.confidence, reverse=True):
            entries.append({
                "name": c.name,
                "slug": c.slug,
                "confidence": c.confidence,
                "tags": c.tags,
            })

        return [{"type": "text", "text": json.dumps(entries, indent=2)}]

    def handle_get_neighbors(self, arguments: dict) -> list[dict]:
        name = arguments.get("name", "")
        slug = self._resolve_slug(name)
        concept = self.store.get(slug)

        if not concept:
            return [{"type": "text", "text": f"Concept not found: {name}"}]

        neighbors = []
        if self.graph:
            neighbor_slugs = self.graph.neighbors(slug)
            for ns in neighbor_slugs:
                nc = self.store.get(ns)
                if nc:
                    neighbors.append({
                        "name": nc.name,
                        "slug": nc.slug,
                        "definition": nc.definition[:150],
                    })

        # Also include typed relationships from the concept itself
        relationships = [
            {"target": r.target, "type": r.rel_type.value, "confidence": r.confidence}
            for r in concept.relationships
        ]

        data = {
            "concept": concept.name,
            "neighbors": neighbors,
            "relationships": relationships,
        }
        return [{"type": "text", "text": json.dumps(data, indent=2)}]

    def handle_get_stats(self, arguments: dict) -> list[dict]:
        concepts = self.store.all()
        stats: dict[str, Any] = {
            "total_concepts": len(concepts),
        }

        if concepts:
            stats["avg_confidence"] = round(
                sum(c.confidence for c in concepts) / len(concepts), 2
            )
            stats["total_insights"] = sum(len(c.insights) for c in concepts)
            stats["total_links"] = sum(len(c.links) for c in concepts)

        if self.graph:
            graph_stats = self.graph.stats()
            stats["graph_nodes"] = graph_stats["nodes"]
            stats["graph_edges"] = graph_stats["edges"]
            stats["graph_clusters"] = graph_stats["clusters"]
            if "density" in graph_stats:
                stats["graph_density"] = graph_stats["density"]

            top = self.graph.central_concepts(top_n=5)
            if top:
                central = []
                for slug, centrality in top:
                    c = self.store.get(slug)
                    central.append({
                        "name": c.name if c else slug,
                        "centrality": round(centrality, 3),
                    })
                stats["most_central"] = central

        return [{"type": "text", "text": json.dumps(stats, indent=2)}]

    def dispatch(self, tool_name: str, arguments: dict) -> list[dict]:
        """Route a tool call to the appropriate handler."""
        handlers = {
            "search": self.handle_search,
            "get_concept": self.handle_get_concept,
            "list_concepts": self.handle_list_concepts,
            "get_neighbors": self.handle_get_neighbors,
            "get_stats": self.handle_get_stats,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return [{"type": "text", "text": f"Unknown tool: {tool_name}"}]
        return handler(arguments)


# ── MCP JSON-RPC server ──────────────────────────────────────────────────────

class MCPServer:
    """MCP server implementing JSON-RPC 2.0 over stdio.

    Reads newline-delimited JSON-RPC messages from stdin,
    dispatches them, and writes responses to stdout.
    """

    def __init__(self, handlers: ToolHandlers) -> None:
        self.handlers = handlers
        self._initialized = False

    def run(self) -> None:
        """Main server loop: read stdin, process, write stdout."""
        logger.info("MindForge MCP server starting...")

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._write_error(None, -32700, "Parse error")
                continue

            response = self._handle_message(message)
            if response is not None:
                self._write(response)

    def _handle_message(self, message: dict) -> dict | None:
        """Route a JSON-RPC message to the appropriate handler."""
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params", {})

        # Notifications (no id) don't get responses
        if msg_id is None and method == "notifications/initialized":
            return None

        if method == "initialize":
            return self._handle_initialize(msg_id, params)
        elif method == "tools/list":
            return self._handle_tools_list(msg_id)
        elif method == "tools/call":
            return self._handle_tools_call(msg_id, params)
        elif method == "ping":
            return self._result(msg_id, {})
        else:
            return self._error(msg_id, -32601, f"Method not found: {method}")

    def _handle_initialize(self, msg_id: Any, params: dict) -> dict:
        self._initialized = True
        return self._result(msg_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        })

    def _handle_tools_list(self, msg_id: Any) -> dict:
        return self._result(msg_id, {"tools": TOOLS})

    def _handle_tools_call(self, msg_id: Any, params: dict) -> dict:
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        try:
            content = self.handlers.dispatch(tool_name, arguments)
            return self._result(msg_id, {"content": content})
        except Exception as e:
            logger.exception("Tool call failed: %s", tool_name)
            return self._result(msg_id, {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            })

    @staticmethod
    def _result(msg_id: Any, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _error(msg_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}

    def _write(self, message: dict) -> None:
        """Write a JSON-RPC message to stdout."""
        text = json.dumps(message)
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    def _write_error(self, msg_id: Any, code: int, message: str) -> None:
        self._write(self._error(msg_id, code, message))


def create_server(config: MindForgeConfig) -> MCPServer:
    """Create an MCP server backed by a MindForge knowledge base."""
    handlers = ToolHandlers(config)
    return MCPServer(handlers)
