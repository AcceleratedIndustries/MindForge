# Feature: Knowledge Hygiene

**Phase:** 1.3
**Depends on:** provenance (1.1)
**Combines:** conflict detection, confidence decay, review queue

---

## Motivation

A knowledge base that silently accumulates stale, contradictory, or low-confidence concepts becomes untrusted and then abandoned. The system needs to surface problems, not hide them.

Three related mechanisms, one feature:

1. **Conflict detection** — when two sources disagree about a concept, flag it.
2. **Confidence decay** — unreinforced concepts slowly fade and become review candidates.
3. **Review queue** — a `mindforge review` workflow for accepting, editing, or removing flagged concepts.

---

## User-facing behavior

### Conflicts flagged in frontmatter

```markdown
---
title: "Context Window"
slug: "context-window"
confidence: 0.62
status: "conflicted"
conflicts:
  - field: "definition"
    sources:
      - transcript: "2024-09-01.md" (turn 3)
        text: "Context window is measured in tokens."
      - transcript: "2025-02-15.md" (turn 12)
        text: "Context window is measured in characters in some older APIs."
---
```

### Review command (TUI)

```bash
mindforge review

# → MindForge Review Queue
# → ========================================
# →  3 conflicted  |  5 stale  |  2 orphaned
# →
# → [1/10] context-window (conflicted)
# →   Definition conflict across 2 sources:
# →     (a) "Context window is measured in tokens." [2024-09-01]
# →     (b) "...measured in characters in some older APIs." [2025-02-15]
# →
# →   [a] accept (a)   [b] accept (b)   [m] merge both
# →   [e] edit         [d] delete       [s] skip   [q] quit
```

### Stats surface

```bash
mindforge stats
# → ...existing output...
# →   Review queue:
# →     Conflicted:  3
# →     Stale:       5 (not reinforced in > 90 days)
# →     Orphaned:    2 (all sources deleted)
```

---

## Design

### 1. Conflict detection

During distillation, when multiple source chunks contribute to one concept, check for disagreements on key fields: `definition`, `tags`, and factual claims in `insights`.

Algorithm:
- For the `definition` field: if both LLM-extracted and pairwise cosine similarity < 0.7 between two source definitions, flag.
- For `insights`: if two insights contradict per a small rule-based check (same subject, different quantifiers: "always" vs "sometimes"; different units: "tokens" vs "characters"), flag.
- Tag/fact conflicts are lower priority; log them but don't always flag.

On conflict: set `status: "conflicted"` and populate the `conflicts` list with the competing source refs + texts.

### 2. Confidence decay

A concept's "freshness" is a function of (a) how recently it was last reinforced (appeared in a new ingested transcript) and (b) how many transcripts mention it.

```python
# Pseudo:
age_days = (now - last_reinforced).days
decay_factor = exp(-age_days / 90)        # half-life ≈ 62 days
reinforcement_boost = min(1.0, log2(1 + source_count) / 4)
adjusted_confidence = base_confidence * (0.5 + 0.5 * max(decay_factor, reinforcement_boost))
```

A concept with `adjusted_confidence < 0.3` and no reinforcement in 90 days enters the review queue as `stale`.

### 3. Orphan detection

A concept whose `sources` list becomes empty after a transcript deletion is `orphaned`. It enters the review queue rather than being deleted outright.

### 4. Review TUI

Minimal terminal UI (no curses dep — `rich` optional, or stdlib `input()` loop). Presents one concept at a time. Actions:

- `a` / `b` / `c`... — accept one of the conflicting variants (writes that variant's text to the definition)
- `m` — merge: opens `$EDITOR` with all variants concatenated; user edits final text
- `e` — edit: opens `$EDITOR` on the concept file
- `d` — delete the concept
- `s` — skip (remains in queue)
- `q` — quit

Review actions update `status` back to `active` and record a `last_reviewed_at` timestamp.

### 5. Data model

Add to `Concept`:

```python
@dataclass
class Concept:
    # ...existing fields...
    status: str = "active"                 # active | conflicted | stale | orphaned
    conflicts: list[ConflictMarker] = field(default_factory=list)
    last_reinforced_at: str | None = None  # ISO 8601
    last_reviewed_at: str | None = None    # ISO 8601

@dataclass
class ConflictMarker:
    field: str                             # "definition" | "insights" | "tags"
    variants: list[ConflictVariant]

@dataclass
class ConflictVariant:
    source: SourceRef
    text: str
```

---

## Files touched

### New
- `mindforge/hygiene/__init__.py`
- `mindforge/hygiene/conflict_detector.py`
- `mindforge/hygiene/decay.py`
- `mindforge/hygiene/review_queue.py`
- `mindforge/hygiene/tui.py`

### Modified
- `mindforge/distillation/concept.py` — add `status`, `conflicts`, timestamps
- `mindforge/distillation/distiller.py` — call `conflict_detector` before finalizing
- `mindforge/pipeline.py` — update `last_reinforced_at` on every run
- `mindforge/distillation/renderer.py` — emit new frontmatter fields
- `mindforge/cli.py` — add `review` subcommand
- `mindforge/mcp/server.py` — add `list_review_queue` tool

---

## Testing

- `tests/test_conflict_detector.py` — definition-level and insight-level conflicts
- `tests/test_decay.py` — decay math, stale threshold crossing
- `tests/test_review_queue.py` — orphan detection, queue filtering, action application
- `tests/test_review_tui.py` — TUI actions using scripted input

---

## Open questions

- **Decay half-life:** 62 days is a guess. Make configurable via `MindForgeConfig.decay_half_life_days`. Revisit with usage data.
- **LLM-assisted conflict detection:** much better than rule-based but requires an LLM call per concept. **Proposed:** rule-based in core, LLM-assisted as an opt-in pass (`mindforge review --llm`).
- **Auto-merge option:** should `mindforge review --auto` resolve low-stakes conflicts (tag differences, whitespace-only definition variants) without human input? **Proposed:** yes, with a dry-run default and an explicit `--apply` flag.
