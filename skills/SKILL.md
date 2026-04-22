# MindForge Skill (Hermes Agent)

> The full integration guide lives at `docs/integrations/hermes-agent.md`. This file is a short Hermes skill entry for the Hermes skill system.

**Repository:** https://github.com/AcceleratedIndustries/MindForgeUniversal

See `docs/integrations/README.md` for all supported harnesses.

## Quick Hermes config

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  mindforge:
    command: python
    args: ["-m", "mindforge.mcp.server"]
    env:
      MINDFORGE_ROOT: /home/you/.hermes/mindforge
    timeout: 60
```

## Tool surface

- **KB management:** `mcp_mindforge_kb_list`, `kb_create`, `kb_select`, `kb_get_current`, `kb_rename`, `kb_delete`
- **Search:** `mcp_mindforge_search`, `search_all`, `search_selected`
- **Concepts:** `mcp_mindforge_get_concept`, `list_concepts`, `get_neighbors`, `get_stats`

See `docs/integrations/hermes-agent.md` for the full workflow reference.

## Storage

KBs live under `$MINDFORGE_ROOT/kbs/`. Soft-deletes land in `$MINDFORGE_ROOT/trash/`. The registry index is `$MINDFORGE_ROOT/registry.json`.
