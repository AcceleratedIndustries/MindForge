# Phase 0 + Phase 1 Design

**Date:** 2026-04-21
**Scope:** Rename, universalization (paths + docs), structural seams, and Phase 1 "Trust" features (provenance, eval harness, knowledge hygiene).
**Out of scope:** Phase 2-5, active MCP compatibility hardening, new source adapters, auto-ingest watchers.

---

## Context and motivation

The roadmap at `docs/ROADMAP.md` sequences MindForge from "working pipeline" to "sticky product" across five phases. This spec covers the prep work (Phase 0, new) plus Phase 1 ("Trust"). Phase 0 exists because:

1. The GitHub repo is `MindForgeForHermes` but the product name is `MindForge`. The Hermes suffix is an artifact of a past fork and must go before the product reaches more users.
2. The product today carries Hermes-specific defaults (paths, configuration) that block adoption by users on Claude Code, Claude Desktop, OpenClaw, Codex CLI, OpenAI Agents SDK, and generic MCP clients.
3. Phase 4 ("Living system") will add source watchers for several harnesses. The current parser is single-purpose; adding a seam now costs little and unblocks Phase 4 cleanly.

Phase 1 is the first roadmap phase because provenance, evaluation, and hygiene are the foundation for trust — without them, every later feature is built on an unverifiable pipeline.

---

## Phase 0: rename + universalization + seams

### 0.1 Repo rename

The repo becomes `MindForgeUniversal` (the legacy `MindForge` name is already taken by an older repo). Mechanics:

- **GitHub:** `gh repo rename MindForgeUniversal` — gated behind explicit user confirmation at execution time. GitHub auto-redirects the old URL.
- **Local:** working directory stays `MindForgeForHermes/` to avoid disrupting the user's workspace. `git remote set-url origin` updates the push target.
- **Docs:** grep-replace `MindForgeForHermes` → `MindForgeUniversal` in URLs, and ensure the product name `MindForge` is consistent throughout.

Substantive references found: `docs/ROADMAP.md` (one line), `skills/SKILL.md` (two URLs).

### 0.2 Paths and configuration (`mindforge/paths.py`, new)

Centralize path resolution with this precedence:

1. Explicit CLI flag (`--output`, `--root`)
2. Env var `MINDFORGE_ROOT`
3. Config file at `$MINDFORGE_CONFIG` or `~/.mindforge/config.yaml`
4. Default: `~/.mindforge/` (KBs at `~/.mindforge/kbs/`, trash at `~/.mindforge/trash/`)

Hermes Agent users opt in by setting `MINDFORGE_ROOT=~/.hermes/mindforge` — no code changes for them. No `platformdirs` dependency; stdlib only (matches existing core dependency policy).

`mindforge/config.py` extends with a `MindForgePaths` accessor; `mindforge/cli.py` uses it for default location resolution. Existing `--output ./output` local mode continues to work.

### 0.3 MCP server consolidation + adapter seam

The repo currently carries three MCP server files: `server.py`, `server_multikb.py`, `server_original.py`. This is drift from past fork merges.

- **Consolidate:** one `server.py` that carries the multi-KB behavior (today's production surface). The other two files are deleted. No behavior change — covered by existing `tests/test_mcp.py`.
- **Client adapter seam:** add `mindforge/mcp/adapter.py` with a `ClientAdapter` base class and one `DefaultAdapter` implementation. The server routes all client-facing formatting (tool descriptions, response shapes) through the adapter. Future client-specific quirks plug in as adapter subclasses without touching the core server.

**This session does not add active compatibility work beyond today's behavior.** The seam exists so future sessions can.

### 0.4 Source adapter seam

The current parser (`mindforge/ingestion/parser.py`) accepts markdown transcripts. Phase 4 will add watchers for Claude Code projects, ChatGPT exports, Cursor logs, etc. To avoid rewriting later:

- **Introduce `mindforge/ingestion/sources.py`:** a `SourceAdapter` protocol with `parse(path_or_stream) -> Iterable[Turn]`.
- **Refactor current parser:** becomes `MarkdownSourceAdapter` implementing the protocol. Behavior unchanged.
- **No new adapters in this session.** No watchers.

Phase 4 can then add `ClaudeCodeProjectAdapter`, `ChatGPTExportAdapter`, `HermesTranscriptAdapter`, etc. as drop-in implementations.

### 0.5 Integration docs (`docs/integrations/`, new directory)

Replace the single Hermes-centric `skills/SKILL.md` framing with a harness-agnostic set of integration guides. `skills/SKILL.md` remains as a shortened Hermes skill entry that points to the full docs.

New files:

- `docs/integrations/README.md` — compatibility matrix (which tools, which features, which config paths)
- `docs/integrations/claude-code.md`
- `docs/integrations/claude-desktop.md`
- `docs/integrations/hermes-agent.md` (generalized from current skill)
- `docs/integrations/openclaw.md`
- `docs/integrations/codex-cli.md`
- `docs/integrations/openai-agents-sdk.md`
- `docs/integrations/generic-mcp.md`

### 0.6 Files touched (Phase 0)

**New:**
- `mindforge/paths.py`
- `mindforge/mcp/adapter.py`
- `mindforge/ingestion/sources.py`
- `docs/integrations/*.md` (8 files)
- `tests/test_paths.py`
- `tests/test_source_adapter.py`
- `tests/test_mcp_adapter.py`

**Modified:**
- `mindforge/config.py`
- `mindforge/cli.py`
- `mindforge/mcp/server.py` (absorbs multikb/original)
- `mindforge/ingestion/parser.py` (now implements `SourceAdapter`)
- `README.md`
- `docs/ROADMAP.md`
- `skills/SKILL.md`

**Deleted:**
- `mindforge/mcp/server_multikb.py`
- `mindforge/mcp/server_original.py`

---

## Phase 1: Trust

Each feature's detailed design already lives in `docs/features/`. This spec ratifies them, resolves open questions, and defines the cross-cutting commitments.

### 1.1 Concept provenance (`docs/features/provenance.md`)

Threads `SourceRef` through the pipeline so every concept cites the transcript(s) and turn(s) that produced it. Persisted in YAML frontmatter (summary) and `output/provenance/<slug>.json` (detailed, with snippets).

**Open-question resolutions:**
- **Snippet capture:** yes, 500-char cap per snippet (feature-doc default).
- **Migration:** missing `sources` in old KBs → default to `[]`, one-time stderr warning suggesting re-ingestion. No destructive migration.
- **Storage abstraction:** introduce a minimal `Storage` protocol here (the architecture doc recommends this; doing it now avoids Phase 3 inheriting filesystem hardcoding).

### 1.2 Evaluation harness (`docs/features/evaluation-harness.md`)

Fixture corpus at `eval/fixtures/` with sibling `.gt.yaml` ground-truth files. `mindforge eval` subcommand computes concept recall/precision, phrase grounding, relationship recall, and relationship-type accuracy. Markdown report to stdout + JSON to `eval/reports/<timestamp>.json`. CI workflow runs heuristic-only eval on PRs touching ingestion/distillation/llm/linking code.

**Open-question resolutions:**
- **LLM non-determinism:** CI gates on heuristic mode only. Local `mindforge eval --mode llm` runs N=3 and reports mean±stdev.
- **Ground truth maintenance:** convention — GT YAML edits land in the same PR as the extractor change that motivated them.
- **Corpus licensing:** 100% synthetic. ~12 fixture transcripts covering the format/length/topic/conflict dimensions specified in the feature doc.

### 1.3 Knowledge hygiene (`docs/features/knowledge-hygiene.md`)

Depends on 1.1. Three mechanisms: rule-based conflict detection, confidence decay, orphan detection. `mindforge review` TUI (stdlib input loop, `rich` optional) lets users accept/merge/edit/delete flagged concepts. MCP gains a `list_review_queue` tool. Stats output surfaces queue counts.

**Open-question resolutions:**
- **Decay half-life:** 62 days default; exposed as `MindForgeConfig.decay_half_life_days`.
- **LLM-assisted conflict detection:** rule-based only this session. Leave the seam for a future `--llm` pass but do not implement it (YAGNI; Phase 3 territory).
- **Auto-merge (`--auto`):** skip this session. Not in exit criteria.

---

## Execution plan

**Branch:** `claude/phase0-and-1` off `main`. All work lands on this branch. No push until the user reviews.

**Commit cadence:** one commit per logical unit (~15-25 commits total).

**Method:** TDD via the `superpowers:test-driven-development` skill. Per-feature implementation plans via `superpowers:writing-plans`. Execution via `superpowers:executing-plans`.

**Gates that interrupt auto mode:**
- Repo rename on GitHub (`gh repo rename`) — explicit confirmation required; shared-state side effects.
- Phase 1 exit review — push branch, summarize against exit criteria, pause for user review before PR or merge.

**Non-gates:** local file changes, test runs, per-feature commits, doc writes, seam introductions. These proceed autonomously.

---

## Phase 1 exit criteria (from `docs/ROADMAP.md`)

1. Every concept links to its sources. → delivered by 1.1.
2. A prompt or code change in the extractor shows a measurable diff in the eval report. → delivered by 1.2.
3. Low-confidence and conflicting concepts are surfaced to the user. → delivered by 1.3.

When these hold on the branch, Phase 1 is done.

---

## Deferred / out of scope

- Active MCP compatibility hardening for any specific client (explicit user decision this session).
- New source adapters beyond `MarkdownSourceAdapter` (Phase 4).
- Watchers / auto-ingest daemon (Phase 4).
- LLM-assisted conflict detection (Phase 3 territory).
- `mindforge review --auto` (not in exit criteria).
- Web UI, HTTP API, hybrid retrieval, MCP extensions (Phase 3).
- Obsidian plugin, export formats (Phase 5).
