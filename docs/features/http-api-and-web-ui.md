# Feature: Local HTTP API + Web UI

**Phase:** 3.1 + 3.2
**Depends on:** provenance (1.1), hybrid retrieval (3.4 — can ship in parallel)
**Unblocks:** Obsidian plugin (5.1), any future GUI surface, hosted SaaS

---

## Motivation

A CLI + markdown files is a developer tool, not a product. The single largest adoption lever is making the knowledge base a place users *visit*.

Two intertwined pieces:

- **The API** is the boring, stable interface that every UI (web, Obsidian, future) targets.
- **The web UI** is the first consumer: graph visualization + concept browser + search.

Designed together because they define each other.

---

## User-facing behavior

```bash
mindforge serve
# → MindForge serving at http://localhost:7823
# → API: http://localhost:7823/api
# → UI:  http://localhost:7823/
# → Opening browser...
```

The browser opens to:

- A graph view (left: node-link diagram, right: concept detail pane on click)
- A search bar (top, uses hybrid retrieval)
- A concept list (sidebar, filterable by tag/confidence/freshness)
- A "review queue" tab if Phase 1.3 has shipped

---

## API design

FastAPI, mounted at `/api`. Versioned via URL prefix: `/api/v1/...`.

### Endpoints

```
# Concepts
GET    /api/v1/concepts?tag=&min_confidence=&limit=&offset=
GET    /api/v1/concepts/{slug}
GET    /api/v1/concepts/{slug}/neighbors?depth=1
GET    /api/v1/concepts/{slug}/sources     # provenance
PATCH  /api/v1/concepts/{slug}             # edits (used by review flow)
DELETE /api/v1/concepts/{slug}

# Graph
GET    /api/v1/graph                       # full graph JSON
GET    /api/v1/graph/subgraph?center=&depth=&edge_types=
GET    /api/v1/graph/paths?from=&to=&max_length=
GET    /api/v1/graph/central?top_n=10

# Search
GET    /api/v1/search?q=&top_k=&mode=hybrid|keyword|semantic
GET    /api/v1/search/context-pack?q=&top_k=&max_chars=

# Pipeline control
POST   /api/v1/ingest                      # {input_dir: "..."} → returns job id
GET    /api/v1/jobs/{id}                   # status + progress
GET    /api/v1/events                      # SSE stream (changes, jobs)

# Review queue (Phase 1.3)
GET    /api/v1/review
POST   /api/v1/review/{slug}/action        # {action: "accept_a|merge|delete|..."}

# Stats
GET    /api/v1/stats
```

### Auth

Local-only by default: bind to `127.0.0.1`, no auth. Configurable:

```bash
mindforge serve --host 0.0.0.0 --token-file .mindforge-token
```

For non-localhost binds, require a bearer token. No password auth, no OAuth — keep it minimal.

### Versioning

Breaking changes → `/api/v2/...`. Non-breaking additions to `v1` freely.

---

## Web UI design

Separate subdirectory: `ui/`, built with Vite + TypeScript. Not Python-packaged; shipped as static assets served by FastAPI from `mindforge/server/static/` after build.

### Stack
- **Framework:** SvelteKit (static adapter) or plain Vite + React. Svelte chosen for size + simplicity.
- **Graph rendering:** Cytoscape.js (node-link) with `cola` or `cose-bilkent` layout.
- **Styling:** Tailwind CSS. No design system — use `shadcn/ui`-equivalent Svelte components (`shadcn-svelte`).

### Views
1. **Graph view** — interactive canvas, click to drill, hover to preview.
2. **Concept view** — rendered markdown, backlinks, provenance (with source snippets), neighbors list.
3. **Search view** — query bar, result list grouped by relevance mode (keyword / semantic / graph).
4. **Review view** (requires Phase 1.3) — queue of conflicted/stale concepts with resolution actions.

### Build & serve

- `cd ui && npm run build` emits static files into `mindforge/server/static/`.
- FastAPI serves `/` → `index.html`, everything else → the static files.
- Development: `npm run dev` proxies API calls to the Python server.

### What we don't build
- No account system
- No sharing / collaboration (that's a SaaS feature)
- No real-time collaboration
- No mobile-first responsive design (desktop-first, responsive as a nice-to-have)

---

## Files touched

### New (API)
- `mindforge/server/__init__.py`
- `mindforge/server/app.py` — FastAPI app factory
- `mindforge/server/routes/concepts.py`
- `mindforge/server/routes/graph.py`
- `mindforge/server/routes/search.py`
- `mindforge/server/routes/ingest.py`
- `mindforge/server/routes/review.py`
- `mindforge/server/routes/stats.py`
- `mindforge/server/events.py` — SSE broadcaster
- `mindforge/server/static/` — populated by UI build

### New (UI)
- `ui/` — separate directory, own `package.json`, `vite.config.ts`
- `ui/src/routes/+page.svelte` — graph view
- `ui/src/routes/concepts/[slug]/+page.svelte`
- `ui/src/routes/search/+page.svelte`
- `ui/src/routes/review/+page.svelte`
- `ui/src/lib/api.ts` — typed client for `/api/v1`
- `ui/README.md` — dev instructions

### Modified
- `mindforge/cli.py` — `mindforge serve [--host] [--port] [--no-open]`
- `pyproject.toml` — new `[server]` extra: `fastapi`, `uvicorn`, `sse-starlette`
- `.github/workflows/ui-build.yml` — build UI on release, attach to PyPI dist

---

## Testing

- `tests/test_server_concepts.py`, `test_server_graph.py`, etc. — one test file per route module, using FastAPI's `TestClient`
- UI gets a minimal Playwright smoke test: server boots, page loads, graph canvas renders
- API contract test: JSON schema for every response; UI's TypeScript types are generated from these

---

## Open questions

- **Auth story for remote-accessible servers:** bearer token is minimal but clunky. If real demand exists, add OAuth device flow later. **Proposed:** bearer only, with a loud warning if binding non-localhost.
- **Storage abstraction:** `ARCHITECTURE.md` flags this. The API layer is the right place to introduce a `Storage` protocol — in-process direct fs access today, pluggable for the hosted product later.
- **SSE vs WebSocket:** SSE is simpler, one-way, sufficient for KB-change notifications. WebSocket only if an interactive editing use case emerges. **Proposed:** SSE.
- **UI framework choice:** Svelte is proposed; React is safer for contributor familiarity. **Low-stakes; decide by team preference.**
