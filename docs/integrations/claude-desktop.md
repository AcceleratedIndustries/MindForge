# Integrating MindForge with Claude Desktop

Claude Desktop is Anthropic's macOS/Windows app. It reads MCP server config from a JSON file.

## Prerequisites

- Python 3.10+ (absolute path to the interpreter you want Claude Desktop to use)
- `pip install -e .` from the MindForge checkout
- Claude Desktop installed

## Configuration

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows). If the file does not exist, create it.

```json
{
  "mcpServers": {
    "mindforge": {
      "command": "/absolute/path/to/python",
      "args": ["-m", "mindforge.mcp.server"],
      "env": {
        "MINDFORGE_ROOT": "/Users/you/.mindforge"
      }
    }
  }
}
```

**Important:** Claude Desktop does not inherit your shell's PATH. Use absolute paths for both the Python interpreter and `MINDFORGE_ROOT`. Run `which python` in Terminal to find your interpreter.

After editing, fully quit and relaunch Claude Desktop (Cmd-Q / File > Exit).

## Verification

1. Open Claude Desktop.
2. Click the paperclip icon. You should see a "mindforge" entry under "MCP servers."
3. In a chat, ask: *"What MindForge KBs do I have?"* — Claude will call `kb_list` and reply.

## Known limitations

- Env vars like `${HOME}` are NOT expanded by Claude Desktop. Hardcode absolute paths.
- The app silently ignores malformed JSON. Validate the file with `python -m json.tool <file>` after editing.
