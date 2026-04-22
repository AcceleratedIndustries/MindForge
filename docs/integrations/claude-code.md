# Integrating MindForge with Claude Code

Claude Code is Anthropic's official CLI. It natively speaks MCP stdio.

## Prerequisites

- Python 3.10+
- `pip install -e .` from the MindForge checkout (or `pip install mindforge` once published)
- Claude Code installed (`npm install -g @anthropic-ai/claude-code` or equivalent)

## Configuration

Add MindForge to Claude Code's MCP server list. Two places work:

**Project-scoped** — `.mcp.json` at the project root (shared via git):

```json
{
  "mcpServers": {
    "mindforge": {
      "command": "python",
      "args": ["-m", "mindforge.mcp.server"],
      "env": {
        "MINDFORGE_ROOT": "${HOME}/.mindforge"
      }
    }
  }
}
```

**User-scoped** — `~/.claude/mcp_servers.json` (applies to every project):

```json
{
  "mcpServers": {
    "mindforge": {
      "command": "python",
      "args": ["-m", "mindforge.mcp.server"],
      "env": {
        "MINDFORGE_ROOT": "${HOME}/.mindforge"
      }
    }
  }
}
```

For a different Python (e.g. a project venv), replace `"python"` with the absolute path to that interpreter.

## Verification

Start Claude Code in the project. Run:

```
/mcp
```

Expected output: a block showing `mindforge` as connected, and the full tool list (`kb_list`, `search`, `get_concept`, etc.).

Then try:

```
ask: What KBs do I have?
```

Claude Code should call `mcp__mindforge__kb_list` and return the list.

## Known limitations

None observed. Claude Code follows the MCP spec strictly; the `DefaultAdapter` is sufficient.
