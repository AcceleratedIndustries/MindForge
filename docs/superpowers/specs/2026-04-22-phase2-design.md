# Phase 2 Design — Repo Hygiene, Distribution, CLI Polish

**Date:** 2026-04-22
**Scope:** Phase 2.0 (new — CI/lint/type cleanup), Phase 2.1 (distribution), Phase 2.2 (CLI polish) from `docs/ROADMAP.md`.
**Out of scope:** Phase 3+; publishing to PyPI / Homebrew (external ops handed off to user after PR merges).

---

## Context

Phase 1 shipped in the previous session. Between then and now, `main` gained CI workflows (pytest matrix + ruff + mypy strict), a security workflow (Bandit + pip-audit + Gitleaks), Dependabot config, and a pre-commit config. Good additions. They surface a state Phase 1 didn't address: the codebase does not currently pass the new checks.

Root-cause summary:

1. `pyproject.toml` is not parseable — the new dev-deps block was appended, producing duplicate `[project.optional-dependencies]` and `[tool.pytest.ini_options]` headers. Nothing that reads the config works.
2. Once parseable, 123 ruff errors remain (74 auto-fixable). Two are real bugs: `sys.stderr` referenced in `mindforge/mcp/server.py` at lines 104 and 114 without `import sys` (I removed the import in Phase 0's MCP consolidation and missed these error paths).
3. 93 mypy strict errors across 15 files, mostly missing return annotations and a few `Optional[]` holes.
4. Pre-commit config is `.pre-commit-config.yml`; canonical is `.yaml`. Some pre-commit versions silently skip `.yml`.

Phase 2.0 addresses the above before Phase 2.1 / 2.2 ship, so new features land on a green CI baseline.

---

## Phase 2.0 — Repo hygiene (new; own PR, lands first)

### Fixes

- Merge the two `[project.optional-dependencies]` tables in `pyproject.toml` into one. Merge the two `[tool.pytest.ini_options]` tables. Keep the new stricter dev deps (`ruff>=0.8`, `mypy>=1.13`, `hypothesis>=6`, `bandit[toml]>=1.7`) and the new `addopts = "-ra"`.
- Add `import sys` to `mindforge/mcp/server.py` (real bug).
- Rename `.pre-commit-config.yml` → `.pre-commit-config.yaml`.

### Lint pass

- Run `ruff check --fix` — resolves ~74 issues mechanically (import order, unused imports, f-string placeholders, `UP` upgrades).
- Run `ruff format` — normalizes formatting across the repo.
- Hand-fix the remaining ~47 errors (B rules + SIM rules not auto-fixable): duplicate-value in `Enum`s, unused-loop-control-variable, `zip(..., strict=)`, `in dict.keys()` idioms, a couple of collapsible conditionals, one `open()` without context manager.

### Type pass

- Retrofit annotations until `mypy --strict files=mindforge` passes. ~93 errors, mostly:
  - Missing `-> None` on CLI handlers and fixture helpers.
  - Untyped local variables (`dict`, `list`, `set` literals that mypy infers as `dict[Any, Any]` etc.; add explicit types).
  - A handful of `Optional[...]` holes (e.g. `self.graph: Optional[KnowledgeGraph] = None`).
  - Two `Any`-leaks in `mindforge/mcp/server.py` where handler functions return `list[TextContent | ImageContent | EmbeddedResource]` but a branch returns something narrower — broaden or narrow as appropriate.

### Exit criteria (Phase 2.0)

- `pytest` — all tests pass (no regressions from the retrofit).
- `ruff check .` — zero errors.
- `ruff format --check .` — no diffs.
- `mypy mindforge` — zero errors.
- CI workflows run green on the PR.

---

## Phase 2.1 — Distribution

### PyPI naming and metadata

- **PyPI package name:** `mindforge-kb` (the bare `mindforge` name is taken by `mindforge.ai`'s client at v1.0.4).
- **CLI command:** stays `mindforge`.
- **Version bump:** `0.1.0` → `0.2.0` on the first publish-ready commit.
- **`pyproject.toml` metadata additions:** `readme = "README.md"`, `authors = [...]`, `keywords = [...]`, `classifiers = [...]`, `project.urls = { Homepage, Repository, Issues, Documentation }`.

### Release workflow (`.github/workflows/release.yml`)

Triggered on `v*` tag push. Builds sdist + wheel, publishes to PyPI via Trusted Publisher (OIDC — no API token stored in CI). Handoff: you configure PyPI Trusted Publisher pointing at this repo + workflow.

### Binary workflow (`.github/workflows/binaries.yml`)

Matrix: macOS-14 (arm64), ubuntu-latest (x64), windows-latest (x64). Builds via PyInstaller. macOS ad-hoc codesign (`codesign --force --sign -`). Attaches binaries to GitHub Release for the triggering tag. Excludes `sentence-transformers`/`faiss` from default to keep binary < 30MB.

### Install-smoke workflow (`.github/workflows/install-smoke.yml`)

Weekly cron + on release tag: `uv tool install` from built sdist in a clean environment, runs `mindforge --help` and `mindforge ingest` on the example transcripts. Fails if output is empty.

### PyInstaller spec (`packaging/mindforge.spec`)

Standard `analysis → pyz → exe` layout. Excludes `sentence-transformers`, `faiss`, `torch`, `numpy` from the default build. Hidden imports for any lazy-loaded mindforge modules.

### Dockerfile (`packaging/Dockerfile`)

`python:3.11-slim` base, `pip install mindforge-kb[embeddings]`, `ENTRYPOINT ["mindforge"]`. For hosted `serve` mode and CI integrations.

### Homebrew formula (`packaging/homebrew/mindforge.rb`)

Template in this repo. You create `AcceleratedIndustries/homebrew-mindforge` and copy the file in. Uses `Language::Python::Virtualenv`. Resources regenerated via `homebrew-pypi-poet` on each release (a CI step in the tap repo, out of scope for this session).

### README updates

Install section covers four paths: `pipx install mindforge-kb`, `uv tool install mindforge-kb`, `brew install mindforge` (once tap is set up), and the single-binary download.

### Handoff checklist (user, after PR merges)

1. Reserve `mindforge-kb` on PyPI. Configure Trusted Publisher pointing at this repo + `release.yml` workflow.
2. Create `AcceleratedIndustries/homebrew-mindforge` repo; copy `packaging/homebrew/mindforge.rb` in.
3. Tag `v0.2.0` and push — release workflow picks up from there.

---

## Phase 2.2 — CLI polish

### `mindforge ingest --dry-run`

Runs the full pipeline but swaps `write_all_concepts()` + manifest save for a summary print showing new/updated/unchanged/removed counts computed by diffing the new `ConceptStore` against the existing on-disk manifest.

### `mindforge diff` and `mindforge diff --since <iso-date>`

Adds a `history` list to `output/manifest.json`: `[{timestamp, slug_hashes: {<slug>: <content_hash>}}, ...]`. `diff` compares the current snapshot against the previous one (or against the snapshot nearest the `--since` timestamp). Output: added / modified / deleted concept slugs, grouped.

Back-compat: manifests without `history` default to an empty list.

### Filter flags

- `mindforge query "..." --tag rag --min-confidence 0.7`
- New subcommand: `mindforge list --tag transformers --since 2025-03-01 --limit 20`
- `query` adds two optional args to the underlying `QueryEngine.search(...)` signature.

### Enhanced `mindforge show`

Current command (shipped in Phase 1.1) supports `--sources`. Adds:
- `--neighbors` — lists graph-connected concepts.
- `--raw` — prints the markdown file as-is (no ANSI, no reformatting).
- Default rendering uses ANSI formatting for headers/bold when stdout is a TTY.

### `mindforge open <slug>` and `mindforge open --graph`

Opens the concept file in `$EDITOR` (fallback: `vi`). `--graph` opens the graph JSON in `$EDITOR` for now. Phase 3 will redirect `--graph` to the web UI when `mindforge serve` is running.

### Root `CLAUDE.md`

Development guide for future Claude Code sessions. Contents (per the feature doc):
- Architecture overview pointer → `docs/ARCHITECTURE.md`.
- Test command — `pytest`.
- Eval command — `mindforge eval --mode heuristic`.
- Dependency policy (core stays lean; extras for everything else).
- Style — no emojis in files, default to no comments, prefer extending existing modules.
- Pre-PR checklist — `pytest && mindforge eval` (latter only if extraction touched).

---

## Execution plan

**Two PRs:**

1. **PR A — Phase 2.0 repo hygiene.** Branch `claude/phase2.0-hygiene`. Single focused PR, fast to review, unblocks CI. Lands before PR B.
2. **PR B — Phase 2.1 + 2.2 together.** Branch `claude/phase2-distribution-and-polish`. Features on a green baseline.

**Subagent strategy:**

- Phase 2.0: mostly sequential main-context work. Retrofit of mypy annotations per-module could fan out if it gets large, but likely inline is fine.
- Phase 2.1: the CI workflow files, Dockerfile, PyInstaller spec, Homebrew formula are independent. Fan out to ~5 subagents writing one file each in parallel.
- Phase 2.2: sequential `cli.py` + `pipeline.py` + `query/engine.py` edits. Stay inline.

**Checkpoints:**
- Pause at PR A merge before starting PR B. Confirms CI baseline is actually green on main.
- Pause at PR B exit criteria for user review.

---

## Exit criteria

**Phase 2.0:** CI green on `main` after PR A merges. `pytest`, `ruff check`, `ruff format --check`, `mypy mindforge` all pass locally and in CI.

**Phase 2.1:** release.yml, binaries.yml, install-smoke.yml all authored and committed. `pyproject.toml` carries full PyPI metadata under `mindforge-kb`. `packaging/mindforge.spec`, `packaging/Dockerfile`, `packaging/homebrew/mindforge.rb` committed. Handoff checklist delivered in PR B description.

**Phase 2.2:** `mindforge ingest --dry-run`, `mindforge diff`, `mindforge list`, `mindforge open`, enhanced `mindforge show` all work against a real KB. Manifest history round-trips. Root `CLAUDE.md` committed.

---

## Deferred / out of scope

- Actual PyPI publish, Homebrew tap creation, first release tag — user's hands (external ops).
- `homebrew-pypi-poet` automation in the tap repo — Phase 2.1 only ships the template formula; resource-update automation is tap-repo work.
- `mindforge open --graph` pointing at a web UI — requires Phase 3 `serve`.
- Per-module mypy opt-outs for legacy code — rejected; we retrofit fully instead.
- Notarization of macOS binaries — beyond ad-hoc codesign. Later problem.
- Windows compatibility guarantees — smoke test covers it; no promises.
