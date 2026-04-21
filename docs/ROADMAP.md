# MindForge Roadmap

Sequenced plan for taking MindForge from "working pipeline" to "sticky product."

Each phase is ordered by dependency: later phases assume earlier ones are in place. Every feature has a detailed design doc under `docs/features/`.

---

## Status

| Feature | Status |
|---------|--------|
| Core ingestion pipeline | Shipped |
| MCP server | Shipped |
| LLM-assisted extraction | Shipped |
| **Incremental ingestion (content hashing)** | **Shipped** (Hermes, `dade9ab`) |

---

## Guiding principles

1. **Trust before polish.** Provenance and evaluation come before UI. A pretty graph over untrustworthy distillations is worse than no graph.
2. **Distribute before you decorate.** An extra `pipx install` path reaches more users than a new feature.
3. **Split the stack early.** Python for the pipeline, JS for the UI, HTTP between them. Don't let one language constrain both surfaces.
4. **Every feature ships with an eval hook.** Extraction quality regresses silently otherwise.

---

## Phase 1 — Trust (weeks 1-2)

Goal: make the output of the pipeline verifiable and testable. Without this, every later feature is built on sand.

| Order | Feature | Design doc | Depends on |
|-------|---------|-----------|------------|
| 1.1 | **Concept provenance** — every concept cites the transcript + turn(s) it came from | [provenance.md](features/provenance.md) | incremental ingestion |
| 1.2 | **Evaluation harness** — regression-testable extraction quality on a fixed corpus | [evaluation-harness.md](features/evaluation-harness.md) | — |
| 1.3 | **Knowledge hygiene** — conflict detection, confidence decay, review queue | [knowledge-hygiene.md](features/knowledge-hygiene.md) | provenance |

**Exit criteria:** every concept links to its sources; a prompt or code change in the extractor shows a measurable diff in the eval report; low-confidence and conflicting concepts are surfaced to the user.

---

## Phase 2 — Distribution (week 3)

Goal: remove the "how do I install this" friction. Reach 10x more users with zero new features.

| Order | Feature | Design doc |
|-------|---------|-----------|
| 2.1 | **Distribution** — `pipx`, `uv tool install`, Homebrew, PyInstaller single-file | [distribution.md](features/distribution.md) |
| 2.2 | **CLI polish** — dry-run, `mindforge diff`, tag/date filters, CLAUDE.md | [cli-polish.md](features/cli-polish.md) |

**Exit criteria:** `brew install mindforge` works; a zero-Python-knowledge user can install and run it; diff/filter commands exist for daily workflows.

---

## Phase 3 — Product layer (weeks 4-6)

Goal: turn markdown files into a place people *visit*. This is the single biggest adoption lever.

| Order | Feature | Design doc | Depends on |
|-------|---------|-----------|------------|
| 3.1 | **Local HTTP API** — FastAPI-based service layer, foundation for UI + plugins | [http-api-and-web-ui.md](features/http-api-and-web-ui.md) | — |
| 3.2 | **Web UI** — graph visualization, concept browser, search box | [http-api-and-web-ui.md](features/http-api-and-web-ui.md) | HTTP API |
| 3.3 | **MCP extensions** — `get_subgraph`, `find_path`, `explain_relationship` | [mcp-extensions.md](features/mcp-extensions.md) | — |
| 3.4 | **Hybrid retrieval** — keyword + vector + graph-walk rerank, default for all queries | [hybrid-retrieval.md](features/hybrid-retrieval.md) | — |

**Exit criteria:** `mindforge serve` opens a browser showing the graph; the MCP server exposes graph-shaped queries an agent can actually use; a single query hits all three retrieval signals.

---

## Phase 4 — Living system (weeks 7-8)

Goal: the "second brain that organizes itself" pitch only works if it keeps up without manual effort.

| Order | Feature | Design doc | Depends on |
|-------|---------|-----------|------------|
| 4.1 | **Auto-ingest daemon** — watch `~/.claude/projects`, ChatGPT exports, Claude Desktop, Cursor logs | [auto-ingest.md](features/auto-ingest.md) | incremental ingestion |

**Exit criteria:** user installs once, and the KB stays current as they use Claude/ChatGPT/Cursor. No manual `ingest` commands in daily use.

---

## Phase 5 — Integration surface (weeks 9-10)

Goal: meet users where they already are.

| Order | Feature | Design doc | Depends on |
|-------|---------|-----------|------------|
| 5.1 | **Obsidian plugin** — TypeScript plugin talking to the local HTTP API | [obsidian-plugin.md](features/obsidian-plugin.md) | HTTP API |
| 5.2 | **Export formats** — JSON-LD, RDF, agent-ready context packs | [export-formats.md](features/export-formats.md) | — |

**Exit criteria:** Obsidian users can drop in the plugin and browse their MindForge KB natively; agents can fetch a single prompt-ready context blob for a question.

---

## Parking lot (not scheduled)

- Mobile/desktop app (Tauri) — only if core usage justifies it
- Rust/Go port of the daemon — only if startup time or memory becomes a real complaint
- Collaborative / multi-user KBs — see `product-strategy.md`

---

## Dependencies at a glance

```
incremental ingestion (done)
        │
        ├──► provenance ──► knowledge hygiene
        │                         │
        │                         └──► (UI surfaces review queue)
        │
        └──► auto-ingest daemon

evaluation harness (parallel)

distribution + CLI polish (parallel, independent)

HTTP API ──► web UI
         └─► Obsidian plugin

MCP extensions (parallel)
hybrid retrieval (parallel, feeds UI + MCP)
```

---

## How to use this roadmap

- Each feature doc under `docs/features/` is scoped to be implementable in isolation.
- Every doc includes: motivation, user-facing behavior, design, files touched, testing strategy, open questions.
- Claude Code sessions working from these docs should start by reading the relevant feature doc + `ARCHITECTURE.md`.
- Open questions in feature docs are flagged with `**Open:**` — resolve them before implementing, not during.
