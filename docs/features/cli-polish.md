# Feature: CLI Polish

**Phase:** 2.2
**Depends on:** incremental ingestion (shipped), provenance (1.1)

---

## Motivation

Small CLI affordances that compound into daily workflow quality. None of these are big features; collectively they're the difference between "a tool I ran once" and "a tool I use."

---

## What to add

### 1. Dry-run ingestion

```bash
mindforge ingest --input transcripts/ --dry-run

# → Preview — no files will be written
# → Would extract ~42 concepts from 8 transcripts.
# → New: 31.  Updated: 7.  Unchanged: 4.  Removed: 0.
# → Run without --dry-run to apply.
```

Implementation: runs the full pipeline but the final `write_all_concepts()` call is swapped for a summary print. The `ConceptStore` diff against the on-disk manifest gives new/updated/removed counts.

### 2. Diff between runs

```bash
mindforge diff
# → Shows changes since last ingest: added/modified/deleted concepts + edges.

mindforge diff --since 2025-04-01
# → Shows changes since a specific date (from manifest history).
```

Implementation: `manifest.json` tracks a timestamped snapshot of concept slugs + their content hashes after each run. `diff` compares two snapshots.

### 3. Query + list filters

```bash
mindforge query "vector databases" --tag rag --min-confidence 0.7

mindforge list --tag transformers --since 2025-03-01 --limit 20
```

### 4. `mindforge show`

```bash
mindforge show kv-cache               # render the concept file to stdout (with ANSI)
mindforge show kv-cache --sources     # add source refs
mindforge show kv-cache --neighbors   # show graph-connected concepts
mindforge show kv-cache --raw         # print the markdown file as-is
```

### 5. `mindforge open`

```bash
mindforge open kv-cache    # opens the concept file in $EDITOR
mindforge open --graph     # opens the knowledge graph JSON
```

### 6. CLAUDE.md at repo root

Conventions for future Claude Code sessions:

```markdown
# MindForge Development Guide

## Architecture overview
See docs/ARCHITECTURE.md

## Running tests
pytest

## Running the eval suite
mindforge eval --mode heuristic

## Dependency policy
- Core install (`pip install mindforge`) stays lean (networkx, pyyaml, stdlib)
- New features land behind extras unless genuinely core
- See docs/ARCHITECTURE.md § Dependency policy

## Style
- No emojis in files unless explicitly requested
- Default to no comments; only add comments when the WHY is non-obvious
- Prefer extending existing modules over creating new top-level packages

## Before opening a PR
- pytest
- mindforge eval --mode heuristic (if touching extraction/distillation)
```

---

## Files touched

### New
- `CLAUDE.md` (repo root)

### Modified
- `mindforge/cli.py` — `show`, `open`, `diff`, `list` subcommands; `--dry-run`, `--tag`, `--since`, `--min-confidence` flags
- `mindforge/pipeline.py` — `--dry-run` path; snapshot writing for `diff`
- `mindforge/query/engine.py` — filter arguments
- `output/manifest.json` schema — gains a `history` list of `{timestamp, snapshot_hash}`

---

## Testing

- `tests/test_cli_dryrun.py` — dry-run doesn't write any files
- `tests/test_cli_diff.py` — diff after a no-op run reports zero changes; diff after adding one concept reports one addition
- `tests/test_cli_filters.py` — tag and confidence filters work on fixture KB

---

## Open questions

- **`mindforge open --graph` behavior:** just opens the JSON file, or launches the web UI if Phase 3 has shipped? **Proposed:** if `mindforge serve` is already running, open the URL; otherwise fall back to opening the JSON.
- **`list` vs `stats`:** they overlap. **Proposed:** `stats` stays as the global overview; `list` is the filterable concept enumerator.
