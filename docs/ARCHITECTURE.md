# MindForge Architecture

Target architecture after Phases 1-3. Captures the decisions that otherwise get made by accident.

---

## The split

MindForge is a **Python core** with a **JavaScript UI layer**, connected by a **local HTTP API**. No ports, no rewrites — just a clean interface between languages.

```
┌──────────────────────────────────────────────────────────────────┐
│                         Surfaces                                 │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  CLI         │  Web UI      │  Obsidian    │  External agents   │
│  (Python)    │  (JS/TS)     │  plugin      │  (via MCP)         │
│              │              │  (JS/TS)     │                    │
└──────┬───────┴──────┬───────┴──────┬───────┴─────────┬──────────┘
       │              │              │                 │
       │              │  HTTP        │  HTTP           │  stdio
       │              ▼              ▼                 │  JSON-RPC
       │       ┌─────────────────────────┐             │
       │       │  Local HTTP API         │             │
       │       │  (FastAPI, stdlib OK)   │             │
       │       └────────────┬────────────┘             │
       │                    │                          │
       └────────────┬───────┴──────────────────────────┘
                   ▼
       ┌──────────────────────────────────┐
       │     Pipeline orchestrator        │
       │     (mindforge.pipeline)         │
       └──────────────┬───────────────────┘
                      │
       ┌──────────────┴───────────────────────────────┐
       │                                              │
       ▼                                              ▼
┌─────────────────────┐                   ┌──────────────────────┐
│  Core stages        │                   │  Optional services   │
│                     │                   │                      │
│  ingestion/         │                   │  llm/                │
│  distillation/      │                   │  embeddings/         │
│  linking/           │                   │  auto-ingest daemon  │
│  graph/             │                   │  evaluation harness  │
│  query/             │                   │                      │
└──────────┬──────────┘                   └──────────┬───────────┘
           │                                         │
           └──────────────────┬──────────────────────┘
                              ▼
                  ┌────────────────────────┐
                  │  Storage               │
                  │                        │
                  │  output/concepts/*.md  │
                  │  output/concepts.json  │
                  │  output/graph/*.json   │
                  │  output/embeddings/    │
                  │  output/provenance/    │
                  └────────────────────────┘
```

---

## Why this shape

- **Python stays** — the ML ecosystem (sentence-transformers, FAISS, tokenizers, LLM SDKs) is where the pipeline's value lives. Porting to Rust/Go would rebuild months of integration for no user-visible win.
- **UI is JS** — Obsidian plugins must be TypeScript; graph visualization libraries (Cytoscape, Sigma, D3) are JS-native. There's no path where a Python-rendered UI competes.
- **HTTP in between** — a boring, stable interface. Same API serves the web UI, the Obsidian plugin, and any future integrations. Local by default, cloud-ready if ever needed.
- **MCP stays stdio** — that's what the MCP spec expects. It's a separate surface for agents, not a replacement for the HTTP API.

---

## Layering rules

1. **Surfaces never touch storage directly.** CLI, UI, MCP server all go through the pipeline orchestrator or the HTTP API.
2. **Stages don't call each other.** The pipeline orchestrates them. Stages consume `ConceptStore` and return new concepts/edges.
3. **Storage is the source of truth.** Every derived artifact (graph, embeddings index) can be rebuilt from `concepts/*.md` + `concepts.json`. No hidden state.
4. **Optional services fail soft.** LLM unreachable → heuristic fallback. Embeddings not installed → keyword-only search. Nothing hard-requires a non-core dependency.

---

## Storage schema (v1, post-Phase 1)

```
output/
├── concepts/                    # Human-readable concept files
│   └── <slug>.md                # YAML frontmatter + markdown body
├── concepts.json                # Manifest: all concepts, links, metadata
├── graph/
│   └── knowledge_graph.json     # NetworkX JSON export
├── embeddings/                  # (optional) FAISS index + metadata
│   ├── index.faiss
│   └── concepts.jsonl
├── provenance/                  # NEW in Phase 1
│   └── <slug>.json              # Per-concept: source transcript paths + turn ranges
└── manifest.json                # NEW: pipeline version, last-run hash, ingestion history
```

**Design principle:** everything under `output/` is regenerable from `transcripts/` + the `manifest.json`. Delete `output/` and a rerun reproduces it.

---

## Process model

### Today
- `mindforge ingest` — one-shot batch process
- `mindforge mcp` — long-running stdio server
- `mindforge query` — one-shot load + query

### After Phase 3
- `mindforge serve` — long-running HTTP server (FastAPI) + web UI
- `mindforge daemon` — long-running file watcher + incremental ingest (Phase 4)
- Everything else unchanged

### After Phase 4
- `mindforge daemon` and `mindforge serve` can run together. The daemon writes to storage; `serve` watches storage for changes and pushes events via SSE/WebSocket to the UI.

---

## Key interfaces

### Python: the pipeline orchestrator

```python
from mindforge.pipeline import MindForgePipeline
from mindforge.config import MindForgeConfig

pipeline = MindForgePipeline(config)
result = pipeline.run()                      # batch ingest
result = pipeline.run_incremental()          # only changed transcripts (Phase 0, done)
results = pipeline.query("how does X work?") # hybrid retrieval (Phase 3)
```

### HTTP: the API (Phase 3)

```
GET  /api/concepts                     # list, with optional ?tag= &min_confidence=
GET  /api/concepts/{slug}              # full concept + provenance + neighbors
GET  /api/graph                        # full graph JSON
GET  /api/graph/subgraph?center=&depth=
GET  /api/search?q=&top_k=&mode=hybrid
POST /api/ingest                       # trigger ingestion (body: {input_dir})
GET  /api/events                       # SSE stream of KB changes (Phase 4)
```

### MCP: stdio JSON-RPC (Phases 1 + 3)

Existing: `search`, `get_concept`, `list_concepts`, `get_neighbors`, `get_stats`.
Added in Phase 3: `get_subgraph`, `find_path`, `explain_relationship`, `get_context_pack`.

---

## Dependency policy

| Category | Policy |
|---|---|
| **Core runtime** | `networkx`, `pyyaml`, stdlib only |
| **Optional (`[embeddings]`)** | `sentence-transformers`, `faiss-cpu`, `numpy` |
| **Optional (`[server]`) — new** | `fastapi`, `uvicorn`, `watchfiles` (Phase 3/4) |
| **Optional (`[eval]`) — new** | `pytest`, `jsonschema`, a small fixture corpus (Phase 1) |
| **UI (separate)** | TypeScript, Vite, Cytoscape.js — lives under `ui/`, not pip-installed |
| **Obsidian plugin (separate)** | TypeScript, Obsidian API — own repo, pinned to a MindForge HTTP API version |

Rule: **core install (`pip install mindforge`) never gets heavier than it is today.** All new features live behind extras.

---

## What changes under the hood

- **`pipeline.py`** gains `run_incremental()` (done) and `query()` that routes to hybrid retrieval (Phase 3).
- **`distillation/concept.py`** gains a `provenance` field: `list[SourceRef]` (Phase 1).
- **`query/engine.py`** becomes the single entry point for all retrieval, blending keyword/vector/graph (Phase 3).
- **new `mindforge/server/`** — FastAPI app + route handlers (Phase 3).
- **new `mindforge/daemon/`** — file watchers, source adapters (Phase 4).
- **new `mindforge/eval/`** — fixture corpus + scoring (Phase 1).

---

## What stays the same

- Module layout under `mindforge/`
- CLI top-level commands (`ingest`, `query`, `stats`, `mcp`)
- On-disk format of `concepts/*.md` (adds `provenance` to frontmatter, back-compat)
- MCP tool names already shipped

No breaking changes are planned. Additions only.
