# Feature: Obsidian Plugin

**Phase:** 5.1
**Depends on:** HTTP API (3.1)
**Repo:** separate TypeScript repository (`obsidian-mindforge`)

---

## Motivation

Obsidian users already live in a markdown-based knowledge tool. MindForge's output is markdown with wiki-links. The integration is obvious and valuable — meet users where they are rather than asking them to switch.

The plugin is a thin TypeScript wrapper over MindForge's local HTTP API. No pipeline logic is duplicated.

---

## User-facing behavior

After installing the plugin and pointing it at a running `mindforge serve`:

- **Command: "MindForge: Insert context for selection"** — select text in a note, run the command, and the plugin fetches a context pack via `/api/v1/search/context-pack` and inserts it inline.
- **Command: "MindForge: Show graph"** — opens a side panel with a live graph view centered on the current note's concept (if matched).
- **Sidebar panel:** lists concepts, filterable by tag, clicking opens the concept in a preview pane.
- **Link autocomplete:** typing `[[` surfaces MindForge concepts alongside Obsidian's own notes.
- **Sync-out option:** a command "MindForge: Mirror KB into vault" symlinks or copies `output/concepts/` into a vault folder, letting Obsidian render the MindForge KB natively.

No user data leaves localhost.

---

## Design

### Plugin shape

Standard Obsidian community plugin:
- `manifest.json`
- `main.ts` — registers commands, views, settings tab
- `src/api.ts` — typed client over `/api/v1`
- `src/views/graph-view.ts` — Cytoscape embedded in a ItemView
- `src/views/concept-panel.ts` — concept list + detail
- `src/commands/` — one file per command

### Settings

```ts
interface MindForgeSettings {
    apiUrl: string;           // http://localhost:7823/api/v1
    apiToken?: string;        // optional, for non-localhost
    contextPackTopK: number;  // default 5
    contextPackMaxChars: number; // default 8000
    mirrorVaultFolder?: string;  // where to sync-out
}
```

### Dependencies

Pin the minimum MindForge HTTP API version in the plugin's settings screen. If the server reports a lower version, show a banner linking to upgrade instructions.

### Distribution

Submit to Obsidian's community plugin directory. In parallel, offer a direct `.obsidian/plugins/` install for power users.

---

## Files touched

Everything lives in a **separate repo**: `acceleratedindustries/obsidian-mindforge`.

No changes in the main MindForge repo beyond a README pointer.

---

## Testing

- Obsidian plugins use Jest for unit tests and a manual test vault for integration
- CI: lint, build, run unit tests
- Manual QA checklist in the plugin's `CONTRIBUTING.md`

---

## Open questions

- **API contract stability:** the plugin pins an API version. Breaking changes require a major bump on both sides. **Proposed:** `/api/v1` is stable; additions only.
- **Mirror vs. symlink:** mirroring (copy on API notification) is safer cross-platform; symlinks are cheaper. **Proposed:** default copy, with symlink as an advanced option on POSIX.
- **Offline behavior:** if `mindforge serve` isn't running, the plugin should fail gracefully with a clear banner, not silently break. **Required for v1.**
