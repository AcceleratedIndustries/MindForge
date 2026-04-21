# Feature: Auto-Ingest Daemon

**Phase:** 4.1
**Depends on:** incremental ingestion (shipped), provenance (1.1)
**Unblocks:** the "it just keeps up" product promise

---

## Motivation

"Point `ingest` at a folder" is friction. Users don't run it consistently, the KB goes stale, the promise fails.

The daemon watches known AI conversation sources and keeps the KB current with zero manual commands.

---

## User-facing behavior

```bash
# One-time setup
mindforge daemon install

# Start (foregrounded)
mindforge daemon

# Or install as a background service
mindforge daemon install --service

# Check status
mindforge daemon status
# → MindForge Daemon
# →   Status:     running (pid 41251)
# →   Uptime:     3d 12h
# →   Sources:
# →     claude-desktop  ~/Library/Application Support/Claude/       7 files tracked
# →     claude-code     ~/.claude/projects/                        41 files tracked
# →     chatgpt-export  ~/Downloads/chatgpt-export-*.zip           (no files yet)
# →   Last ingest: 2 minutes ago (3 new concepts)
```

A file appearing or changing in a watched location triggers an incremental ingest within seconds.

---

## Design

### Source adapters

Pluggable. Each adapter implements:

```python
class SourceAdapter(Protocol):
    name: str

    def discover(self) -> list[Path]:
        """Return absolute paths to all relevant files right now."""

    def watch(self) -> Iterator[SourceEvent]:
        """Yield SourceEvent(path=..., kind="added"|"modified"|"deleted")."""

    def to_transcript(self, path: Path) -> Transcript | None:
        """Parse the source-specific format into MindForge's Transcript model."""
```

### Initial adapters

| Adapter | Watches | Format |
|---|---|---|
| `ClaudeDesktopAdapter` | `~/Library/Application Support/Claude/` + `~/.config/Claude/` | Claude Desktop history JSON |
| `ClaudeCodeAdapter` | `~/.claude/projects/<project>/*.jsonl` | Claude Code session JSONL |
| `ChatGPTExportAdapter` | configurable dir, watches for `chatgpt-export-*.zip` | OpenAI's export format |
| `CursorAdapter` | Cursor's chat history location (platform-dependent) | Cursor's format |
| `FolderAdapter` | any user-specified directory of markdown | existing MindForge parser |

Each adapter gets its own test fixture with real (sanitized) sample files.

### File watching

Use `watchfiles` (Rust-based, cross-platform, good debouncing). Falls back to polling if unavailable.

Debounce: coalesce changes within a 3-second window before triggering ingest (prevents ingest-storming during active use).

### Ingest invocation

Each batch of changes triggers `pipeline.run_incremental(paths=[...])`. The incremental logic (already shipped) skips unchanged files via content hash.

### Daemon lifecycle

- **Foreground mode** (`mindforge daemon`): runs until Ctrl-C. Useful for debugging.
- **Service mode** (`mindforge daemon install --service`):
  - macOS: `launchctl` user agent at `~/Library/LaunchAgents/com.mindforge.daemon.plist`
  - Linux: `systemd --user` unit at `~/.config/systemd/user/mindforge-daemon.service`
  - Windows: deferred — likely Task Scheduler; ship in a follow-up.

### Config

```toml
# ~/.config/mindforge/daemon.toml
output_dir = "~/mindforge-kb"

[[sources]]
adapter = "claude-code"

[[sources]]
adapter = "claude-desktop"

[[sources]]
adapter = "folder"
path = "~/notes/transcripts"
```

### Event streaming

Daemon publishes change events to the HTTP API (`/api/v1/events` SSE stream, Phase 3) so the web UI can live-update without polling. If API isn't running, events are logged to `~/.local/share/mindforge/daemon.log`.

---

## Files touched

### New
- `mindforge/daemon/__init__.py`
- `mindforge/daemon/runner.py` — main daemon loop
- `mindforge/daemon/adapters/base.py` — `SourceAdapter` protocol
- `mindforge/daemon/adapters/claude_desktop.py`
- `mindforge/daemon/adapters/claude_code.py`
- `mindforge/daemon/adapters/chatgpt_export.py`
- `mindforge/daemon/adapters/cursor.py`
- `mindforge/daemon/adapters/folder.py`
- `mindforge/daemon/service_installer.py` — launchctl / systemd unit generators
- `mindforge/daemon/config.py`

### Modified
- `mindforge/cli.py` — `mindforge daemon {install,uninstall,status,start,stop}`
- `pyproject.toml` — add `watchfiles` to `[server]` extra (shared with HTTP API)
- `mindforge/server/events.py` — accept daemon-published events when both run together

---

## Testing

- Per-adapter unit tests with fixture files in their native formats
- `tests/test_daemon_runner.py` — mock adapters, assert debounce and ingest-trigger behavior
- Integration test (slow, opt-in): spin up a temp dir with a `folder` adapter, drop a file, assert KB updates within 5s
- Service installer tests: generate the plist / unit, lint them, don't actually install in CI

---

## Open questions

- **Sensitive content:** Claude Desktop history may contain PII or work secrets. The daemon must honor a user-configurable ignore-list (regex or glob) for source content. **Proposed:** default ignore list includes common secret patterns (AWS keys, API tokens, emails when `--redact-emails`).
- **Multi-device:** if a user runs the daemon on laptop and desktop with the same KB path (e.g., Dropbox), two daemons will race. **Proposed:** file-lock the KB; second daemon exits with a clear message. Cross-device sync is a SaaS feature.
- **Adapter versioning:** Claude Desktop's history format has changed before. **Proposed:** adapters include a format-version detector and emit a warning on unknown versions rather than crashing.
- **Resource limits:** LLM-assisted extraction during auto-ingest can be expensive. **Proposed:** default to heuristic-only in daemon mode; `--llm` opt-in with a daily token budget.
