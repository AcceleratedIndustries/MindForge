#!/usr/bin/env python3
"""MCP server for MindForge using the official MCP SDK.

This exposes MindForge knowledge base tools via the Model Context Protocol.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add MindForge to path
sys.path.insert(0, os.environ.get("PYTHONPATH", "/home/will/MindForge"))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

from mindforge.config import MindForgeConfig
from mindforge.distillation.concept import ConceptStore
from mindforge.graph.builder import KnowledgeGraph
from mindforge.query.engine import QueryEngine
from mindforge.utils.text import slugify


# Global server instance
app = Server("mindforge")


class MindForgeState:
    """Holds the loaded knowledge base state."""
    
    def __init__(self) -> None:
        # Get output directory from environment or use default
        output_dir = Path(os.environ.get("MINDFORGE_OUTPUT", "/home/will/knowledge-base"))
        self.config = MindForgeConfig(output_dir=output_dir)
        self.store = ConceptStore()
        self.graph: KnowledgeGraph | None = None
        self.query_engine: QueryEngine | None = None
        self._load()
    
    def _load(self) -> None:
        """Load knowledge base from disk if available."""
        manifest = self.config.output_dir / "concepts.json"
        if manifest.exists():
            try:
                self.store = ConceptStore.load(manifest)
            except Exception:
                pass  # Keep empty store
        
        graph_file = self.config.graph_dir / "knowledge_graph.json"
        if graph_file.exists():
            try:
                self.graph = KnowledgeGraph.load(graph_file)
            except Exception:
                pass  # No graph
        
        self.query_engine = QueryEngine(self.store, self.graph, None)
    
    def _resolve_slug(self, name: str) -> str:
        """Resolve a concept name or slug to a slug."""
        # Try direct slug lookup
        slug = slugify(name)
        if self.store.get(slug):
            return slug
        
        # Try case-insensitive name match
        for concept in self.store.all():
            if concept.name.lower() == name.lower():
                return concept.slug
        
        return slug


# Global state (initialized when module loads)
_state = MindForgeState()


# --- Tool Definitions ---

TOOLS: list[Tool] = [
    Tool(
        name="search",
        description="Search the MindForge knowledge base with a natural language query. Returns the top matching concepts with definitions and related concepts.",
        inputSchema={
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
    ),
    Tool(
        name="get_concept",
        description="Get full details of a specific concept by name or slug. Returns definition, explanation, insights, examples, tags, and relationships.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Concept name or slug (e.g., 'KV Cache' or 'kv-cache')",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="list_concepts",
        description="List all concepts in the knowledge base, optionally filtered by tag. Returns concept names, slugs, confidence scores, and tags.",
        inputSchema={
            "type": "object",
            "properties": {
                "tag": {
                    "type": "string",
                    "description": "Filter by tag (optional)",
                },
            },
        },
    ),
    Tool(
        name="get_neighbors",
        description="Get concepts related to a given concept via the knowledge graph. Returns directly connected concepts and their relationship types.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Concept name or slug",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="get_stats",
        description="Get knowledge base statistics: concept count, edge count, clusters, most central concepts.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]


# --- Tool Handlers ---

@app.call_tool()
async def handle_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle MCP tool calls."""
    import json
    
    if name == "search":
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 5)
        
        if not _state.query_engine:
            return [TextContent(type="text", text="Knowledge base not loaded.")]
        
        results = _state.query_engine.search(query, top_k=top_k)
        
        if not results:
            return [TextContent(type="text", text=f"No results found for: {query}")]
        
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
        
        return [TextContent(type="text", text=json.dumps(output, indent=2))]
    
    elif name == "get_concept":
        name_arg = arguments.get("name", "")
        slug = _state._resolve_slug(name_arg)
        concept = _state.store.get(slug)
        
        if not concept:
            return [TextContent(type="text", text=f"Concept not found: {name_arg}")]
        
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
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    
    elif name == "list_concepts":
        tag_filter = arguments.get("tag", "")
        concepts = _state.store.all()
        
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
        
        return [TextContent(type="text", text=json.dumps(entries, indent=2))]
    
    elif name == "get_neighbors":
        name_arg = arguments.get("name", "")
        slug = _state._resolve_slug(name_arg)
        concept = _state.store.get(slug)
        
        if not concept:
            return [TextContent(type="text", text=f"Concept not found: {name_arg}")]
        
        neighbors = []
        if _state.graph:
            neighbor_slugs = _state.graph.neighbors(slug)
            for ns in neighbor_slugs:
                nc = _state.store.get(ns)
                if nc:
                    neighbors.append({
                        "name": nc.name,
                        "slug": nc.slug,
                        "definition": nc.definition[:150],
                    })
        
        relationships = [
            {"target": r.target, "type": r.rel_type.value, "confidence": r.confidence}
            for r in concept.relationships
        ]
        
        data = {
            "concept": concept.name,
            "neighbors": neighbors,
            "relationships": relationships,
        }
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    
    elif name == "get_stats":
        import typing
        concepts = _state.store.all()
        stats: dict[str, typing.Any] = {
            "total_concepts": len(concepts),
        }
        
        if concepts:
            stats["avg_confidence"] = round(
                sum(c.confidence for c in concepts) / len(concepts), 2
            )
            stats["total_insights"] = sum(len(c.insights) for c in concepts)
            stats["total_links"] = sum(len(c.links) for c in concepts)
        
        if _state.graph:
            graph_stats = _state.graph.stats()
            stats["graph_nodes"] = graph_stats["nodes"]
            stats["graph_edges"] = graph_stats["edges"]
            stats["graph_clusters"] = graph_stats["clusters"]
            if "density" in graph_stats:
                stats["graph_density"] = graph_stats["density"]
            
            top = _state.graph.central_concepts(top_n=5)
            if top:
                central = []
                for slug, centrality in top:
                    c = _state.store.get(slug)
                    central.append({
                        "name": c.name if c else slug,
                        "centrality": round(centrality, 3),
                    })
                stats["most_central"] = central
        
        return [TextContent(type="text", text=json.dumps(stats, indent=2))]
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of available tools."""
    return TOOLS


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
