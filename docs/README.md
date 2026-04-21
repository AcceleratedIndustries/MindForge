# MindForge Docs

Planning and design documents for MindForge. Start here:

1. **[ROADMAP.md](ROADMAP.md)** — phased plan, status, and dependencies
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** — target architecture (Python core + JS UI)
3. **[product-strategy.md](product-strategy.md)** — SaaS exploration and commercialization options

## Feature design docs

Each feature has a standalone design doc under `features/`. Every doc follows the same structure:

- Motivation
- User-facing behavior
- Design
- Files touched
- Testing strategy
- Open questions

| Phase | Feature | Doc |
|---|---|---|
| 1.1 | Concept provenance | [features/provenance.md](features/provenance.md) |
| 1.2 | Evaluation harness | [features/evaluation-harness.md](features/evaluation-harness.md) |
| 1.3 | Knowledge hygiene (conflicts, decay, review queue) | [features/knowledge-hygiene.md](features/knowledge-hygiene.md) |
| 2.1 | Distribution (pipx, Homebrew, PyInstaller) | [features/distribution.md](features/distribution.md) |
| 2.2 | CLI polish (dry-run, diff, filters) | [features/cli-polish.md](features/cli-polish.md) |
| 3.1/3.2 | Local HTTP API + web UI | [features/http-api-and-web-ui.md](features/http-api-and-web-ui.md) |
| 3.3 | MCP server extensions | [features/mcp-extensions.md](features/mcp-extensions.md) |
| 3.4 | Hybrid retrieval | [features/hybrid-retrieval.md](features/hybrid-retrieval.md) |
| 4.1 | Auto-ingest daemon | [features/auto-ingest.md](features/auto-ingest.md) |
| 5.1 | Obsidian plugin | [features/obsidian-plugin.md](features/obsidian-plugin.md) |
| 5.2 | Export formats | [features/export-formats.md](features/export-formats.md) |

## For Claude Code sessions

When implementing a feature from this plan:

1. Read `ARCHITECTURE.md` first.
2. Read the feature's design doc end-to-end.
3. Resolve any **Open questions** at the bottom of the doc before writing code — confirm with a human if needed.
4. Check for a section called "Files touched" — that's the blast radius.
5. Write tests alongside the implementation, using the patterns in the "Testing" section.
