#!/usr/bin/env python3
"""MCP server for MindForge with Multi-Knowledgebase support.

This exposes MindForge knowledge base tools via the Model Context Protocol,
with support for multiple topic-based knowledge bases.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

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
from mindforge.mcp.adapter import get_adapter
from mindforge.paths import MindForgePaths
from mindforge.query.engine import QueryEngine
from mindforge.utils.text import slugify


# Client adapter: resolved once per process. DefaultAdapter is pass-through;
# future per-client quirks plug in via MINDFORGE_MCP_ADAPTER or register_adapter().
_ADAPTER = get_adapter()


# Base paths — resolved from MINDFORGE_ROOT or the default ~/.mindforge.
# Hermes users opt in by setting MINDFORGE_ROOT=~/.hermes/mindforge.
_PATHS = MindForgePaths.resolve()
MINDFORGE_ROOT = _PATHS.root
KBS_DIR = _PATHS.kbs_dir
TRASH_DIR = _PATHS.trash_dir
REGISTRY_FILE = _PATHS.registry_file


def ensure_structure():
    """Ensure the multi-KB directory structure exists."""
    KBS_DIR.mkdir(parents=True, exist_ok=True)
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_FILE.exists():
        registry = {
            "version": "1.0",
            "kbs": {},
            "trash": {},
            "settings": {
                "default_kb": None,
                "naming_convention": "kebab-case"
            }
        }
        REGISTRY_FILE.write_text(json.dumps(registry, indent=2))


def load_registry() -> dict:
    """Load the KB registry."""
    ensure_structure()
    return json.loads(REGISTRY_FILE.read_text())


def save_registry(registry: dict):
    """Save the KB registry."""
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2))


def kebab_case(name: str) -> str:
    """Convert a name to kebab-case."""
    return slugify(name)


class KnowledgeBase:
    """Wrapper for a single Knowledge Base."""
    
    def __init__(self, kb_id: str, path: Path):
        self.kb_id = kb_id
        self.path = path
        self.config = MindForgeConfig(output_dir=path)
        self.store: Optional[ConceptStore] = None
        self.graph: Optional[KnowledgeGraph] = None
        self.query_engine: Optional[QueryEngine] = None
        self._loaded = False
    
    def load(self) -> bool:
        """Load this KB from disk."""
        if self._loaded:
            return True
            
        manifest = self.path / "concepts.json"
        if manifest.exists():
            try:
                self.store = ConceptStore.load(manifest)
            except Exception as e:
                print(f"Error loading concepts: {e}", file=sys.stderr)
                self.store = ConceptStore()
        else:
            self.store = ConceptStore()
        
        graph_file = self.path / "graph.json"
        if graph_file.exists():
            try:
                self.graph = KnowledgeGraph.load(graph_file)
            except Exception as e:
                print(f"Error loading graph: {e}", file=sys.stderr)
        
        self.query_engine = QueryEngine(self.store, self.graph, None)
        self._loaded = True
        return True
    
    def get_stats(self) -> dict:
        """Get statistics for this KB."""
        self.load()
        concepts = self.store.all() if self.store else []
        stats = {
            "concept_count": len(concepts),
        }
        if concepts:
            stats["avg_confidence"] = round(sum(c.confidence for c in concepts) / len(concepts), 2)
        if self.graph:
            graph_stats = self.graph.stats()
            stats["graph_nodes"] = graph_stats.get("nodes", 0)
            stats["graph_edges"] = graph_stats.get("edges", 0)
        return stats


class MultiKBManager:
    """Manages multiple knowledge bases and the active selection."""
    
    def __init__(self):
        ensure_structure()
        self.registry = load_registry()
        self.active_kb_id: Optional[str] = None
        self.active_kb: Optional[KnowledgeBase] = None
        
        # Try to load last active from environment or session
        self.active_kb_id = os.environ.get("MINDFORGE_ACTIVE_KB")
        if self.active_kb_id and self.active_kb_id in self.registry["kbs"]:
            self._load_active()
    
    def _load_active(self):
        """Load the active KB."""
        if not self.active_kb_id:
            return
        kb_info = self.registry["kbs"].get(self.active_kb_id)
        if kb_info:
            self.active_kb = KnowledgeBase(self.active_kb_id, KBS_DIR / self.active_kb_id)
            self.active_kb.load()
    
    def list_kbs(self) -> list[dict]:
        """List all available KBs."""
        result = []
        for kb_id, info in self.registry["kbs"].items():
            kb = KnowledgeBase(kb_id, KBS_DIR / kb_id)
            stats = kb.get_stats()
            entry = {
                "id": kb_id,
                "name": info.get("name", kb_id),
                "description": info.get("description", ""),
                "author": info.get("author", ""),
                "created_at": info.get("created_at", ""),
                "updated_at": info.get("updated_at", ""),
                "concept_count": stats["concept_count"],
                "is_active": kb_id == self.active_kb_id
            }
            result.append(entry)
        return result
    
    def create_kb(self, name: str, description: str = "", author: str = "") -> dict:
        """Create a new knowledge base."""
        kb_id = kebab_case(name)
        
        if kb_id in self.registry["kbs"]:
            return {"success": False, "error": f"KB '{kb_id}' already exists"}
        
        # Create directory structure
        kb_path = KBS_DIR / kb_id
        kb_path.mkdir(parents=True, exist_ok=True)
        (kb_path / "concepts").mkdir(exist_ok=True)
        
        # Initialize empty files
        empty_store = ConceptStore()
        empty_store.save(kb_path / "concepts.json")
        open(kb_path / "graph.json", "w").write('{"nodes": [], "edges": []}')
        
        # Add to registry
        now = datetime.now().isoformat()
        self.registry["kbs"][kb_id] = {
            "name": name,
            "description": description,
            "author": author,
            "path": f"kbs/{kb_id}",
            "created_at": now,
            "updated_at": now
        }
        save_registry(self.registry)
        
        return {"success": True, "id": kb_id, "path": str(kb_path)}
    
    def select_kb(self, kb_id: str) -> dict:
        """Select (activate) a knowledge base."""
        if kb_id not in self.registry["kbs"]:
            return {"success": False, "error": f"KB '{kb_id}' not found"}
        
        self.active_kb_id = kb_id
        self._load_active()
        
        # Update process env for persistence
        os.environ["MINDFORGE_ACTIVE_KB"] = kb_id
        
        if self.active_kb:
            stats = self.active_kb.get_stats()
            return {
                "success": True,
                "id": kb_id,
                "name": self.registry["kbs"][kb_id]["name"],
                "concept_count": stats["concept_count"]
            }
        return {"success": False, "error": "Failed to load KB"}
    
    def delete_kb(self, kb_id: str) -> dict:
        """Move a KB to trash."""
        if kb_id not in self.registry["kbs"]:
            return {"success": False, "error": f"KB '{kb_id}' not found"}
        
        kb_info = self.registry["kbs"][kb_id]
        kb_path = KBS_DIR / kb_id
        
        # Move to trash with timestamp
        trashed_name = f"{kb_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trashed_path = TRASH_DIR / trashed_name
        
        try:
            shutil.move(str(kb_path), str(trashed_path))
            
            # Update registry
            kb_info["deleted_at"] = datetime.now().isoformat()
            self.registry["trash"][trashed_name] = kb_info
            del self.registry["kbs"][kb_id]
            
            # Clear active if needed
            if self.active_kb_id == kb_id:
                self.active_kb = None
                self.active_kb_id = None
                os.environ.pop("MINDFORGE_ACTIVE_KB", None)
            
            save_registry(self.registry)
            return {"success": True, "trashed": trashed_name}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def rename_kb(self, old_id: str, new_name: str) -> dict:
        """Rename a knowledge base."""
        if old_id not in self.registry["kbs"]:
            return {"success": False, "error": f"KB '{old_id}' not found"}
        
        new_id = kebab_case(new_name)
        if new_id != old_id and new_id in self.registry["kbs"]:
            return {"success": False, "error": f"KB '{new_id}' already exists"}
        
        old_path = KBS_DIR / old_id
        new_path = KBS_DIR / new_id
        
        try:
            shutil.move(str(old_path), str(new_path))
            
            # Update registry
            kb_info = self.registry["kbs"][old_id]
            kb_info["name"] = new_name
            kb_info["path"] = f"kbs/{new_id}"
            kb_info["updated_at"] = datetime.now().isoformat()
            
            del self.registry["kbs"][old_id]
            self.registry["kbs"][new_id] = kb_info
            
            # Update active if needed
            if self.active_kb_id == old_id:
                self.active_kb_id = new_id
                os.environ["MINDFORGE_ACTIVE_KB"] = new_id
            
            save_registry(self.registry)
            return {"success": True, "old_id": old_id, "new_id": new_id}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_current(self) -> Optional[dict]:
        """Get info about the current active KB."""
        if not self.active_kb_id or not self.active_kb:
            return None
        
        info = self.registry["kbs"].get(self.active_kb_id, {})
        stats = self.active_kb.get_stats()
        return {
            "id": self.active_kb_id,
            "name": info.get("name", self.active_kb_id),
            "description": info.get("description", ""),
            "concept_count": stats["concept_count"],
            "graph_nodes": stats.get("graph_nodes", 0),
            "graph_edges": stats.get("graph_edges", 0)
        }
    
    def require_active(self) -> KnowledgeBase:
        """Get active KB or raise error."""
        if not self.active_kb:
            raise RuntimeError("No knowledge base selected. Use 'select_kb' first.")
        return self.active_kb


# Global server instance
app = Server("mindforge")

# Global multi-KB manager
_manager: Optional[MultiKBManager] = None

def get_manager() -> MultiKBManager:
    global _manager
    if _manager is None:
        _manager = MultiKBManager()
    return _manager


# --- Tool Definitions ---

KB_TOOLS: list[Tool] = [
    Tool(
        name="kb_list",
        description="List all available knowledge bases with their metadata and concept counts.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="kb_create",
        description="Create a new knowledge base with a given name.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name for the new KB (will be converted to kebab-case)",
                },
                "description": {
                    "type": "string",
                    "description": "Optional description of the KB's purpose",
                    "default": "",
                },
                "author": {
                    "type": "string",
                    "description": "Optional author name",
                    "default": "",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="kb_select",
        description="Select (activate) a knowledge base by ID. All search operations will target this KB until another is selected.",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "KB ID (kebab-case identifier)",
                },
            },
            "required": ["id"],
        },
    ),
    Tool(
        name="kb_delete",
        description="Delete a knowledge base (moves to trash).",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "KB ID to delete",
                },
            },
            "required": ["id"],
        },
    ),
    Tool(
        name="kb_rename",
        description="Rename a knowledge base.",
        inputSchema={
            "type": "object",
            "properties": {
                "old_id": {
                    "type": "string",
                    "description": "Current KB ID",
                },
                "new_name": {
                    "type": "string",
                    "description": "New display name (will be converted to kebab-case ID)",
                },
            },
            "required": ["old_id", "new_name"],
        },
    ),
    Tool(
        name="kb_get_current",
        description="Get information about the currently selected knowledge base.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]

SEARCH_TOOLS: list[Tool] = [
    Tool(
        name="search",
        description="Search the ACTIVE knowledge base with a natural language query. Returns top matching concepts.",
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
        name="search_all",
        description="Search ALL knowledge bases with a natural language query. Returns results with KB source indicated.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Results per KB (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="search_selected",
        description="Search SELECTED knowledge bases by ID. Provide a list of KB IDs to search across.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "kb_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of KB IDs to search",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Results per KB (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query", "kb_ids"],
        },
    ),
]

CONCEPT_TOOLS: list[Tool] = [
    Tool(
        name="get_concept",
        description="Get full details of a specific concept by name or slug from the ACTIVE KB.",
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
        name="list_concepts",
        description="List all concepts in the ACTIVE KB, optionally filtered by tag.",
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
        description="Get concepts related to a given concept via the knowledge graph (from ACTIVE KB).",
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
        description="Get statistics for the ACTIVE knowledge base.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]

ALL_TOOLS = KB_TOOLS + SEARCH_TOOLS + CONCEPT_TOOLS


# --- Helper Functions ---

def _resolve_slug(store, name: str) -> str:
    """Resolve a concept name or slug."""
    slug = slugify(name)
    if store.get(slug):
        return slug
    for concept in store.all():
        if concept.name.lower() == name.lower():
            return concept.slug
    return slug


# --- Tool Handlers ---

@app.call_tool()
async def handle_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle MCP tool calls."""
    manager = get_manager()
    
    # --- KB Management Tools ---
    
    if name == "kb_list":
        kbs = manager.list_kbs()
        return [TextContent(type="text", text=json.dumps(kbs, indent=2))]
    
    elif name == "kb_create":
        result = manager.create_kb(
            name=arguments.get("name", ""),
            description=arguments.get("description", ""),
            author=arguments.get("author", "")
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "kb_select":
        result = manager.select_kb(arguments.get("id", ""))
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "kb_delete":
        result = manager.delete_kb(arguments.get("id", ""))
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "kb_rename":
        result = manager.rename_kb(
            arguments.get("old_id", ""),
            arguments.get("new_name", "")
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "kb_get_current":
        current = manager.get_current()
        if current is None:
            return [TextContent(type="text", text="No knowledge base selected.")]
        return [TextContent(type="text", text=json.dumps(current, indent=2))]
    
    # --- Search Tools ---
    
    elif name == "search":
        try:
            kb = manager.require_active()
        except RuntimeError as e:
            return [TextContent(type="text", text=str(e))]
        
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 5)
        
        if not kb.query_engine:
            return [TextContent(type="text", text="Query engine not available.")]
        
        results = kb.query_engine.search(query, top_k=top_k)
        if not results:
            return [TextContent(type="text", text=f"No results found in active KB for: {query}")]
        
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
    
    elif name == "search_all":
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 5)
        
        all_results = []
        for kb_id in manager.registry["kbs"].keys():
            kb = KnowledgeBase(kb_id, KBS_DIR / kb_id)
            kb.load()
            
            if kb.query_engine:
                results = kb.query_engine.search(query, top_k=top_k)
                for r in results:
                    entry = {
                        "kb_id": kb_id,
                        "kb_name": manager.registry["kbs"][kb_id]["name"],
                        "name": r.concept.name,
                        "slug": r.concept.slug,
                        "score": round(r.score, 3),
                        "definition": r.concept.definition[:200] + "..." if len(r.concept.definition) > 200 else r.concept.definition,
                    }
                    all_results.append(entry)
        
        if not all_results:
            return [TextContent(type="text", text=f"No results found in any KB for: {query}")]
        
        # Sort by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return [TextContent(type="text", text=json.dumps(all_results[:20], indent=2))]
    
    elif name == "search_selected":
        query = arguments.get("query", "")
        kb_ids = arguments.get("kb_ids", [])
        top_k = arguments.get("top_k", 5)
        
        all_results = []
        for kb_id in kb_ids:
            if kb_id not in manager.registry["kbs"]:
                continue
                
            kb = KnowledgeBase(kb_id, KBS_DIR / kb_id)
            kb.load()
            
            if kb.query_engine:
                results = kb.query_engine.search(query, top_k=top_k)
                for r in results:
                    entry = {
                        "kb_id": kb_id,
                        "kb_name": manager.registry["kbs"][kb_id]["name"],
                        "name": r.concept.name,
                        "slug": r.concept.slug,
                        "score": round(r.score, 3),
                        "definition": r.concept.definition[:200] + "..." if len(r.concept.definition) > 200 else r.concept.definition,
                    }
                    all_results.append(entry)
        
        if not all_results:
            return [TextContent(type="text", text=f"No results found in selected KBs for: {query}")]
        
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return [TextContent(type="text", text=json.dumps(all_results[:20], indent=2))]
    
    # --- Concept Tools (require active KB) ---
    
    elif name == "get_concept":
        try:
            kb = manager.require_active()
        except RuntimeError as e:
            return [TextContent(type="text", text=str(e))]
        
        name_arg = arguments.get("name", "")
        slug = _resolve_slug(kb.store, name_arg) if kb.store else name_arg
        concept = kb.store.get(slug) if kb.store else None
        
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
        try:
            kb = manager.require_active()
        except RuntimeError as e:
            return [TextContent(type="text", text=str(e))]
        
        tag_filter = arguments.get("tag", "")
        concepts = kb.store.all() if kb.store else []
        
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
        try:
            kb = manager.require_active()
        except RuntimeError as e:
            return [TextContent(type="text", text=str(e))]
        
        name_arg = arguments.get("name", "")
        slug = _resolve_slug(kb.store, name_arg) if kb.store else name_arg
        concept = kb.store.get(slug) if kb.store else None
        
        if not concept:
            return [TextContent(type="text", text=f"Concept not found: {name_arg}")]
        
        neighbors = []
        if kb.graph:
            neighbor_slugs = kb.graph.neighbors(slug)
            for ns in neighbor_slugs:
                nc = kb.store.get(ns) if kb.store else None
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
        try:
            kb = manager.require_active()
        except RuntimeError as e:
            return [TextContent(type="text", text=str(e))]
        
        current = manager.get_current()
        concepts = kb.store.all() if kb.store else []
        
        stats: dict = {
            "kb_id": current["id"] if current else None,
            "kb_name": current["name"] if current else None,
            "total_concepts": len(concepts),
        }
        
        if concepts:
            stats["avg_confidence"] = round(sum(c.confidence for c in concepts) / len(concepts), 2)
            stats["total_insights"] = sum(len(c.insights) for c in concepts)
            stats["total_links"] = sum(len(c.links) for c in concepts)
        
        if kb.graph:
            graph_stats = kb.graph.stats()
            stats["graph_nodes"] = graph_stats.get("nodes", 0)
            stats["graph_edges"] = graph_stats.get("edges", 0)
            stats["graph_clusters"] = graph_stats.get("clusters", 0)
            if "density" in graph_stats:
                stats["graph_density"] = graph_stats["density"]
            
            top = kb.graph.central_concepts(top_n=5)
            if top:
                central = []
                for s, centrality in top:
                    c = kb.store.get(s) if kb.store else None
                    central.append({
                        "name": c.name if c else s,
                        "centrality": round(centrality, 3),
                    })
                stats["most_central"] = central
        
        return [TextContent(type="text", text=json.dumps(stats, indent=2))]
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of available tools."""
    return ALL_TOOLS


async def main():
    """Run the MCP server (async)."""
    get_manager()
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


class _ServerShim:
    """Sync wrapper returned by create_server() for CLI callers."""

    def __init__(self, config: Optional["MindForgeConfig"] = None) -> None:
        self.config = config

    def run(self) -> None:
        import asyncio
        asyncio.run(main())


def create_server(config: Optional["MindForgeConfig"] = None) -> _ServerShim:
    """Return a runnable server object. ``config`` is currently unused;
    the multi-KB server reads MINDFORGE_ROOT from the environment.
    """
    return _ServerShim(config)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
