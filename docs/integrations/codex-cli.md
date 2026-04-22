# Integrating MindForge with Codex CLI

Codex CLI is OpenAI's official agent CLI. It supports MCP servers via `~/.codex/config.toml`.

## Prerequisites

- Python 3.10+
- `pip install -e .` from the MindForge checkout
- Codex CLI installed (`npm install -g @openai/codex` or equivalent)

## Configuration

Edit `~/.codex/config.toml`:

```toml
[mcp_servers.mindforge]
command = "python"
args = ["-m", "mindforge.mcp.server"]

[mcp_servers.mindforge.env]
MINDFORGE_ROOT = "${HOME}/.mindforge"
```

Use an absolute path to your Python interpreter if the system default isn't correct:

```toml
[mcp_servers.mindforge]
command = "/Users/you/.venvs/mindforge/bin/python"
args = ["-m", "mindforge.mcp.server"]
```

## Verification

Start Codex in a project and ask: *"What MindForge KBs do I have?"* — it should call `kb_list`.

To inspect tools directly:

```
codex mcp list
```

Expected: `mindforge` listed with its tool set.

## Known limitations

None observed. Codex CLI follows the MCP spec strictly.
