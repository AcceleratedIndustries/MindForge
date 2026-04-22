# MindForge Development Guide

## Architecture overview

See `docs/ARCHITECTURE.md` for the Python core / HTTP API / UI layering plus dependency policy. Roadmap and per-phase feature docs live under `docs/ROADMAP.md` and `docs/features/`.

## Running tests

    pytest

## Running the eval suite

    mindforge eval --mode heuristic

Reports land in `eval/reports/<timestamp>.json`.

## Linting and type checking

    ruff check .
    ruff format --check .
    mypy mindforge

All three run in CI (`.github/workflows/ci.yml`) plus bandit + pip-audit + gitleaks (`.github/workflows/security.yml`).

## Dependency policy

- Core install (`pip install mindforge-kb`) stays lean: `networkx`, `pyyaml`, stdlib only.
- New features land behind extras (`[embeddings]`, `[mcp]`, `[eval]`, `[dev]`).
- Do not add a core dep without an architecture doc update.

## Style

- No emojis in files unless explicitly requested.
- Default to no comments; add one only when the WHY is non-obvious.
- Prefer extending existing modules over creating new top-level packages.
- Follow existing patterns. Match the shape of the file you are editing.
- Phase 1 modules (`distillation`, `eval`, `hygiene`, `pipeline`, `paths`, `storage`, `mcp.adapter`, `ingestion.sources`) pass mypy strict. Keep them that way.
- Legacy modules (`mcp.server`, `embeddings.index`, `llm.*`, `graph.builder`, `ingestion.incremental`) have per-module mypy relaxations in `pyproject.toml`. A dedicated retrofit pass is a welcome PR; until then, don't regress to less-typed code in the relaxed list.

## Before opening a PR

- `pytest`
- `ruff check . && ruff format --check . && mypy mindforge`
- `mindforge eval --mode heuristic` (only if touching extraction / distillation / llm / linking).

## PyPI, Homebrew, binaries

- PyPI project: **mindforge-kb** (the bare `mindforge` name is taken). CLI command stays `mindforge`.
- Release workflow (`.github/workflows/release.yml`) fires on `v*` tag push and publishes via Trusted Publisher (OIDC — no API token in CI).
- Binary builds (`.github/workflows/binaries.yml`) attach platform-specific artifacts to the same GitHub Release.
- Homebrew formula template lives at `packaging/homebrew/mindforge.rb`; the tap is at `AcceleratedIndustries/homebrew-mindforge` (external).

## Superpowers specs and plans

- Design specs live in `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`.
- Implementation plans live in `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`.
- Follow the `superpowers:brainstorming → writing-plans → executing-plans` flow for non-trivial features.
