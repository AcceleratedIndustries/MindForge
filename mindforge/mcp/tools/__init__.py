"""MCP synthesis tool handlers.

Each tool's ``handle_*`` function accepts a ConceptStore, a KnowledgeGraph,
an LLMClient (or None for non-synthesis brief modes), plus tool-specific
arguments. It returns the wrapped agent-facing payload as a string. The
server (mindforge/mcp/server.py) wraps each handler's return in a
``TextContent`` and routes the call.
"""
