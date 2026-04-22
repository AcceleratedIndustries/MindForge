# Integrating MindForge with a Generic MCP Client

Any client that speaks MCP stdio JSON-RPC can drive MindForge.

## Minimum viable integration

The MCP server binary is:

```
python -m mindforge.mcp.server
```

- **Transport:** stdio
- **Protocol:** JSON-RPC 2.0 with MCP framing
- **Required env:** `MINDFORGE_ROOT` (defaults to `~/.mindforge`)

Your client must:

1. Spawn the command as a subprocess.
2. Write JSON-RPC requests to stdin, read responses from stdout (one JSON object per line).
3. Start with an `initialize` request; follow the MCP handshake.
4. Call `tools/list` to discover tools.
5. Call `tools/call` with `{"name": "<tool>", "arguments": {...}}` to invoke.

Any stderr output is the server's log — route it somewhere readable during development.

## Tool surface

- **KB management:** `kb_list`, `kb_create`, `kb_select`, `kb_get_current`, `kb_rename`, `kb_delete`
- **Search:** `search`, `search_all`, `search_selected`
- **Concepts:** `get_concept`, `list_concepts`, `get_neighbors`, `get_stats`

Get the authoritative input schema for each tool via `tools/list`.

## Adapting for non-compliant clients

MindForge's MCP server supports a pluggable `ClientAdapter` (see `mindforge/mcp/adapter.py`) for per-client quirks — for example, truncating tool descriptions for clients that reject long strings, or rewriting response shapes.

Select an adapter by setting:

```
MINDFORGE_MCP_ADAPTER=<name>
```

To add one, subclass `ClientAdapter`, call `register_adapter("<name>", YourAdapter)` on import, and launch with the env var set.

## Known limitations

- MindForge does not currently support SSE or HTTP MCP transport — only stdio. (HTTP support arrives with the FastAPI surface in Phase 3.)
- Long-running tool calls (e.g. large KB scans) have no streaming. The whole response returns when the tool completes.
