# Integrating MindForge with OpenClaw

OpenClaw (`github.com/openclaw/openclaw`) is an open-source agent harness that speaks MCP.

## Prerequisites

- Python 3.10+
- `pip install -e .` from the MindForge checkout
- OpenClaw installed per its README

## Configuration

OpenClaw reads MCP server definitions from a project-scoped config file. Create or edit `.openclaw/config.yaml` at the project root:

```yaml
mcp_servers:
  - name: mindforge
    command: python
    args:
      - -m
      - mindforge.mcp.server
    env:
      MINDFORGE_ROOT: ${HOME}/.mindforge
```

If OpenClaw on your platform uses a different config location or key name, consult its README — the `command`/`args`/`env` shape matches every stdio MCP client we've seen.

## Verification

Start OpenClaw in the project. The MindForge tools (`kb_list`, `search`, `get_concept`, …) should appear in its tool inspector.

## Known limitations

- Community-supported. If you hit a protocol quirk, set `MINDFORGE_MCP_ADAPTER` to a custom adapter and subclass `ClientAdapter` in `mindforge/mcp/adapter.py`.
- OpenClaw's config path and env-expansion behavior vary by version; if `${HOME}` does not expand, hardcode the absolute path.
