"""Tests for the MindForge MCP server.

Tests cover:
- Tool handler logic (search, get_concept, list, neighbors, stats)
- JSON-RPC message routing
- Protocol responses (initialize, tools/list, tools/call)
- Error handling
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from mindforge.config import MindForgeConfig
from mindforge.distillation.concept import Concept, ConceptStore, Relationship, RelationshipType
from mindforge.graph.builder import KnowledgeGraph
from mindforge.mcp.server import (
    MCPServer,
    ToolHandlers,
    TOOLS,
    PROTOCOL_VERSION,
    SERVER_NAME,
    create_server,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_store() -> ConceptStore:
    """Create a test ConceptStore with sample concepts."""
    store = ConceptStore()
    store.add(Concept(
        name="Vector Embeddings",
        definition="Dense numerical representations of data in vector space.",
        explanation="Maps text, images, or other data to fixed-length arrays.",
        insights=["Capture semantic meaning", "Enable similarity search"],
        examples=["Word2Vec", "Sentence Transformers"],
        tags=["ml", "vectors", "nlp"],
        confidence=0.9,
        links=["Semantic Search"],
        relationships=[
            Relationship("vector-embeddings", "semantic-search", RelationshipType.ENABLES),
        ],
    ))
    store.add(Concept(
        name="Semantic Search",
        definition="Search that understands meaning rather than matching keywords.",
        explanation="Uses vector similarity to find relevant documents.",
        insights=["Handles synonyms", "Requires embedding models"],
        tags=["search", "nlp"],
        confidence=0.85,
        links=["Vector Embeddings", "Vector Database"],
        relationships=[
            Relationship("semantic-search", "vector-embeddings", RelationshipType.USES),
        ],
    ))
    store.add(Concept(
        name="Vector Database",
        definition="Specialized database for high-dimensional vector storage and search.",
        explanation="Provides fast approximate nearest neighbor search.",
        tags=["database", "vectors"],
        confidence=0.8,
    ))
    return store


def _make_handlers(tmp_path: Path) -> ToolHandlers:
    """Create ToolHandlers backed by a test knowledge base on disk."""
    store = _make_store()
    graph = KnowledgeGraph.from_store(store)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "concepts").mkdir()
    (output_dir / "graph").mkdir()
    (output_dir / "embeddings").mkdir()

    store.save(output_dir / "concepts.json")
    graph.save(output_dir / "graph" / "knowledge_graph.json")

    config = MindForgeConfig(output_dir=output_dir)
    return ToolHandlers(config)


# ── Tool handler tests ───────────────────────────────────────────────────────

class TestToolHandlers:
    def test_search(self, tmp_path):
        handlers = _make_handlers(tmp_path)
        result = handlers.handle_search({"query": "embeddings"})
        assert len(result) == 1
        data = json.loads(result[0]["text"])
        assert len(data) > 0
        assert any(r["name"] == "Vector Embeddings" for r in data)

    def test_search_no_results(self, tmp_path):
        handlers = _make_handlers(tmp_path)
        result = handlers.handle_search({"query": "quantum blockchain"})
        assert len(result) == 1
        text = result[0]["text"]
        # Either empty list or "no results" message
        assert "No results" in text or "[]" in text

    def test_get_concept_by_name(self, tmp_path):
        handlers = _make_handlers(tmp_path)
        result = handlers.handle_get_concept({"name": "Vector Embeddings"})
        data = json.loads(result[0]["text"])
        assert data["name"] == "Vector Embeddings"
        assert "Dense numerical" in data["definition"]
        assert "ml" in data["tags"]

    def test_get_concept_by_slug(self, tmp_path):
        handlers = _make_handlers(tmp_path)
        result = handlers.handle_get_concept({"name": "vector-embeddings"})
        data = json.loads(result[0]["text"])
        assert data["name"] == "Vector Embeddings"

    def test_get_concept_not_found(self, tmp_path):
        handlers = _make_handlers(tmp_path)
        result = handlers.handle_get_concept({"name": "nonexistent"})
        assert "not found" in result[0]["text"].lower()

    def test_list_concepts(self, tmp_path):
        handlers = _make_handlers(tmp_path)
        result = handlers.handle_list_concepts({})
        data = json.loads(result[0]["text"])
        assert len(data) == 3
        # Should be sorted by confidence (descending)
        assert data[0]["confidence"] >= data[1]["confidence"]

    def test_list_concepts_filter_by_tag(self, tmp_path):
        handlers = _make_handlers(tmp_path)
        result = handlers.handle_list_concepts({"tag": "vectors"})
        data = json.loads(result[0]["text"])
        assert len(data) == 2  # Vector Embeddings and Vector Database
        names = [c["name"] for c in data]
        assert "Vector Embeddings" in names
        assert "Vector Database" in names

    def test_list_concepts_filter_no_match(self, tmp_path):
        handlers = _make_handlers(tmp_path)
        result = handlers.handle_list_concepts({"tag": "quantum"})
        data = json.loads(result[0]["text"])
        assert len(data) == 0

    def test_get_neighbors(self, tmp_path):
        handlers = _make_handlers(tmp_path)
        result = handlers.handle_get_neighbors({"name": "Vector Embeddings"})
        data = json.loads(result[0]["text"])
        assert data["concept"] == "Vector Embeddings"
        assert len(data["relationships"]) > 0
        assert data["relationships"][0]["type"] == "enables"

    def test_get_neighbors_not_found(self, tmp_path):
        handlers = _make_handlers(tmp_path)
        result = handlers.handle_get_neighbors({"name": "nonexistent"})
        assert "not found" in result[0]["text"].lower()

    def test_get_stats(self, tmp_path):
        handlers = _make_handlers(tmp_path)
        result = handlers.handle_get_stats({})
        data = json.loads(result[0]["text"])
        assert data["total_concepts"] == 3
        assert data["graph_nodes"] == 3
        assert data["graph_edges"] >= 1
        assert "avg_confidence" in data

    def test_dispatch_routes_correctly(self, tmp_path):
        handlers = _make_handlers(tmp_path)

        result = handlers.dispatch("get_stats", {})
        data = json.loads(result[0]["text"])
        assert "total_concepts" in data

    def test_dispatch_unknown_tool(self, tmp_path):
        handlers = _make_handlers(tmp_path)
        result = handlers.dispatch("nonexistent_tool", {})
        assert "Unknown tool" in result[0]["text"]


# ── MCP protocol tests ──────────────────────────────────────────────────────

class TestMCPServer:
    def _make_server(self, tmp_path: Path) -> MCPServer:
        handlers = _make_handlers(tmp_path)
        return MCPServer(handlers)

    def test_handle_initialize(self, tmp_path):
        server = self._make_server(tmp_path)
        response = server._handle_message({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        })
        assert response["id"] == 1
        assert response["result"]["protocolVersion"] == PROTOCOL_VERSION
        assert response["result"]["serverInfo"]["name"] == SERVER_NAME
        assert "tools" in response["result"]["capabilities"]

    def test_handle_tools_list(self, tmp_path):
        server = self._make_server(tmp_path)
        response = server._handle_message({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        })
        tools = response["result"]["tools"]
        assert len(tools) == 5
        tool_names = {t["name"] for t in tools}
        assert tool_names == {"search", "get_concept", "list_concepts", "get_neighbors", "get_stats"}

    def test_handle_tools_call(self, tmp_path):
        server = self._make_server(tmp_path)
        response = server._handle_message({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_stats",
                "arguments": {},
            },
        })
        assert "error" not in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert data["total_concepts"] == 3

    def test_handle_tools_call_search(self, tmp_path):
        server = self._make_server(tmp_path)
        response = server._handle_message({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {"query": "vector database", "top_k": 2},
            },
        })
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert len(data) <= 2

    def test_handle_ping(self, tmp_path):
        server = self._make_server(tmp_path)
        response = server._handle_message({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "ping",
        })
        assert response["id"] == 5
        assert response["result"] == {}

    def test_handle_unknown_method(self, tmp_path):
        server = self._make_server(tmp_path)
        response = server._handle_message({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "unknown/method",
        })
        assert "error" in response
        assert response["error"]["code"] == -32601

    def test_notification_no_response(self, tmp_path):
        server = self._make_server(tmp_path)
        response = server._handle_message({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        assert response is None

    def test_tools_have_valid_schemas(self):
        """Verify all tool definitions have required MCP fields."""
        for tool in TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"
            assert "properties" in tool["inputSchema"]


# ── Integration test ─────────────────────────────────────────────────────────

class TestMCPIntegration:
    def test_full_conversation(self, tmp_path):
        """Simulate a full MCP conversation: init -> list tools -> call tool."""
        server = self._make_server(tmp_path)

        # 1. Initialize
        r1 = server._handle_message({
            "jsonrpc": "2.0", "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": PROTOCOL_VERSION, "clientInfo": {"name": "test"}},
        })
        assert r1["result"]["protocolVersion"] == PROTOCOL_VERSION

        # 2. Notification (no response expected)
        r_notif = server._handle_message({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        assert r_notif is None

        # 3. List tools
        r2 = server._handle_message({
            "jsonrpc": "2.0", "id": 2,
            "method": "tools/list",
        })
        assert len(r2["result"]["tools"]) == 5

        # 4. Search
        r3 = server._handle_message({
            "jsonrpc": "2.0", "id": 3,
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "embeddings"}},
        })
        results = json.loads(r3["result"]["content"][0]["text"])
        assert len(results) > 0

        # 5. Get details of first result
        slug = results[0]["slug"]
        r4 = server._handle_message({
            "jsonrpc": "2.0", "id": 4,
            "method": "tools/call",
            "params": {"name": "get_concept", "arguments": {"name": slug}},
        })
        concept = json.loads(r4["result"]["content"][0]["text"])
        assert "definition" in concept

        # 6. Get neighbors
        r5 = server._handle_message({
            "jsonrpc": "2.0", "id": 5,
            "method": "tools/call",
            "params": {"name": "get_neighbors", "arguments": {"name": slug}},
        })
        neighbors = json.loads(r5["result"]["content"][0]["text"])
        assert "neighbors" in neighbors

    def _make_server(self, tmp_path: Path) -> MCPServer:
        handlers = _make_handlers(tmp_path)
        return MCPServer(handlers)
