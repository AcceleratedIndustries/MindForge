# Integrating MindForge with Hermes Agent

Hermes Agent is a self-hosted multimodal agent. It supports MCP servers through its `config.yaml`.

## Prerequisites

- Python 3.10+
- Hermes Agent installed (see its README)
- `pip install -e .` from the MindForge checkout into Hermes's venv

## Configuration

Edit `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  mindforge:
    command: /home/you/.hermes/hermes-agent/venv/bin/python
    args:
      - -m
      - mindforge.mcp.server
    env:
      PYTHONPATH: /home/you/MindForge
      MINDFORGE_ROOT: /home/you/.hermes/mindforge
    timeout: 60
    connect_timeout: 30
```

The `PYTHONPATH` line is only needed if you have not `pip install -e .` into Hermes's venv. If you have, drop it.

Hermes stores KBs under whatever `MINDFORGE_ROOT` you set. The historical Hermes default was `~/.hermes/mindforge`; keep that if you already have data there. New installs can use `~/.mindforge` instead.

## Verification

Start Hermes and run:

```
User: What KBs do I have?
```

Hermes should call `mcp_mindforge_kb_list` and return the list.

## Workflow examples

**Research project setup:**

```
User: Create a KB for "Project Alpha Research"
→ Hermes calls: mcp_mindforge_kb_create(name="Project Alpha Research")
→ Result: created kbs/project-alpha-research/
```

**Switch KB:**

```
User: Switch to project-alpha-research
→ Hermes calls: mcp_mindforge_kb_select(id="project-alpha-research")
```

**Cross-KB search:**

```
User: Has "attention mechanism" appeared in any of my KBs?
→ Hermes calls: mcp_mindforge_search_all(query="attention mechanism")
```

## Known limitations

- The Telegram gateway caches MCP connections. After creating a new KB, if Telegram doesn't see it, restart the gateway and reload config.
- Hermes reloads `config.yaml` only on startup. Restart Hermes after editing the MCP config.

## Hermes skill entry

A short skill entry lives at `skills/SKILL.md` for Hermes's skill system. It points back to this doc. Keep both in sync when adding new tools or workflows.
