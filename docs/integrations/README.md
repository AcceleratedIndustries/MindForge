# MindForge Integrations

MindForge speaks the Model Context Protocol (MCP) over stdio JSON-RPC. Any MCP-compatible harness can drive it. Common integrations are documented here.

## Compatibility matrix

| Harness | Install method | Config path | MCP stdio | Status |
|---|---|---|---|---|
| [Claude Code](claude-code.md) | Anthropic CLI | `~/.claude/mcp_servers.json` or project `.mcp.json` | yes | Supported |
| [Claude Desktop](claude-desktop.md) | macOS/Windows app | `~/Library/Application Support/Claude/claude_desktop_config.json` | yes | Supported |
| [Hermes Agent](hermes-agent.md) | Self-hosted | `~/.hermes/config.yaml` | yes | Supported |
| [OpenClaw](openclaw.md) | `github.com/openclaw/openclaw` | Project `.openclaw/config.yaml` | yes | Community-supported |
| [Codex CLI](codex-cli.md) | OpenAI Codex | `~/.codex/config.toml` | yes | Supported |
| [OpenAI Agents SDK](openai-agents-sdk.md) | Python library | Programmatic | yes | Supported |
| [Generic MCP client](generic-mcp.md) | Any stdio JSON-RPC MCP client | — | yes | — |

## Common environment

Every integration sets one env var:

```
MINDFORGE_ROOT=<path to your KB root>
```

Default: `~/.mindforge`. Hermes-style installs may prefer `~/.hermes/mindforge`.

## Command

The MCP server runs as:

```
python -m mindforge.mcp.server
```

No args. The server reads `MINDFORGE_ROOT` from the environment and manages one or more knowledge bases under `<root>/kbs/`.

## Tool surface

Every supported harness exposes the same multi-KB tool set:

- **KB management:** `kb_list`, `kb_create`, `kb_select`, `kb_get_current`, `kb_rename`, `kb_delete`
- **Search:** `search`, `search_all`, `search_selected`
- **Concepts:** `get_concept`, `list_concepts`, `get_neighbors`, `get_stats`

See each harness's guide for the exact config snippet to drop in.

## Extending to new harnesses

If your harness speaks MCP stdio JSON-RPC, it should work today. If it has quirks (strict JSON Schema, tool-description length limits, custom response shapes), see `mindforge/mcp/adapter.py` — the `ClientAdapter` seam is where per-client fixes go. Set `MINDFORGE_MCP_ADAPTER=<name>` to select a non-default adapter.
