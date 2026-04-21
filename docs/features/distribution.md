# Feature: Distribution

**Phase:** 2.1
**Depends on:** —
**Unblocks:** real adoption by non-Python users

---

## Motivation

`pip install -e .` + venv management + "why did faiss-cpu fail to build" is the single largest adoption barrier. None of it is fixed by new features. It's fixed by better packaging.

Three install paths, ordered by target audience:

1. **Python developers** — `pipx` or `uv tool install`. Works today in principle; needs verification + docs.
2. **Mac power users** — `brew install mindforge`. Homebrew tap.
3. **Everyone else** — single-file executable via PyInstaller. Download + run.

---

## User-facing behavior

### Python developers

```bash
# Recommended
uv tool install mindforge

# Or
pipx install mindforge
pipx install "mindforge[embeddings]"   # with semantic search
```

### Mac / Linux via Homebrew

```bash
brew tap acceleratedindustries/mindforge
brew install mindforge
```

### Everyone else

```bash
# Download single binary from GitHub Releases
curl -L https://github.com/acceleratedindustries/mindforgeforhermes/releases/latest/download/mindforge-macos-arm64 -o mindforge
chmod +x mindforge
./mindforge --help
```

---

## Design

### 1. PyPI publishing

- Add GitHub Action `.github/workflows/release.yml` triggered on tag push (`v*`).
- Build sdist + wheel, publish to PyPI via Trusted Publisher (OIDC, no API token in CI).
- Version bump: hand-edit `pyproject.toml`, tag, push. Keep it simple.

### 2. `uv` and `pipx` compatibility

Nothing blocks this today *in principle*, but verify:

- `pyproject.toml` entry point (`mindforge = "mindforge.cli:main"`) — already set.
- Optional extras (`[embeddings]`) work when installed as a tool, not a dep.
- No hidden cwd assumptions that break when installed globally.

Add a CI matrix step: install via `uv tool install .` and run `mindforge --help`.

### 3. Homebrew formula

Create a separate tap repo: `acceleratedindustries/homebrew-mindforge`.

Formula approach: use `brew install --python` to get a clean Python 3.11, then `pip install mindforge` into an isolated prefix. This avoids the "Homebrew Python conflicts with system Python" trap.

Base the formula on `Language::Python::Virtualenv`:

```ruby
class Mindforge < Formula
  include Language::Python::Virtualenv
  desc "Semantic memory engine for AI conversation transcripts"
  homepage "https://github.com/acceleratedindustries/mindforgeforhermes"
  url "https://files.pythonhosted.org/packages/.../mindforge-0.2.0.tar.gz"
  sha256 "..."
  license "GPL-3.0-or-later"

  depends_on "python@3.11"

  resource "networkx" do
    url "..."
    sha256 "..."
  end
  resource "pyyaml" do
    url "..."
    sha256 "..."
  end

  def install
    virtualenv_install_with_resources
  end

  test do
    system bin/"mindforge", "--help"
  end
end
```

Automate resource sha generation with `homebrew-pypi-poet`. Update formula automatically on each PyPI release via a GitHub Action in the tap repo.

### 4. PyInstaller single-file builds

Build per-OS binaries in CI, attach to GitHub Releases.

- Matrix: macOS-13 (x86_64), macOS-14 (arm64), ubuntu-latest (x86_64), windows-latest (x86_64).
- PyInstaller spec excludes `sentence-transformers` / `faiss` from the default build (binary size explodes, users unlikely to want CPU embeddings in a binary). Ship `mindforge` (core) and `mindforge-full` (with embeddings) separately.
- macOS builds: ad-hoc code-sign (`codesign --force --sign -`) so the binary doesn't get Gatekeeper-quarantined silently. Notarization is a later problem.

### 5. Docker image (bonus)

Publish `ghcr.io/acceleratedindustries/mindforge:latest` for users who prefer containers. Useful for the hosted `serve` mode and for CI integrations.

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir mindforge[embeddings]
ENTRYPOINT ["mindforge"]
```

---

## Files touched

### New
- `.github/workflows/release.yml` — PyPI publish on tag
- `.github/workflows/binaries.yml` — PyInstaller matrix build
- `.github/workflows/install-smoke.yml` — weekly: does `uv tool install mindforge` still work?
- `packaging/mindforge.spec` — PyInstaller spec
- `packaging/Dockerfile` — Docker image
- (separate repo) `homebrew-mindforge/Formula/mindforge.rb`

### Modified
- `pyproject.toml` — ensure project metadata (URLs, classifiers) is complete for PyPI
- `README.md` — install instructions for all paths

---

## Testing

- Install smoke test in CI (weekly cron + on each release tag): `uv tool install .` from the built sdist, run `mindforge ingest` against the example transcripts, assert non-zero output.
- Binary smoke: each PyInstaller artifact runs `mindforge --version` and `mindforge ingest` on the example corpus in a clean VM.
- Homebrew formula has `test do` block that runs `mindforge --help`.

---

## Open questions

- **License compatibility:** GPL-3.0 is fine for Homebrew (allowed) and PyPI (allowed). Some enterprises refuse GPL tools. **Recommend:** keep GPL; if SaaS Shape 3 ships, the SaaS client SDK can be separately MIT/Apache-2.0 to remove friction.
- **Binary size:** full builds with sentence-transformers can easily hit 500MB+. **Recommend:** core-only binary (< 30MB) as default; full binary labeled and optional.
- **Windows support:** currently untested. Phase 2 adds smoke tests but no promises. **Recommend:** explicitly document "tested on macOS + Linux, Windows best-effort" until usage data justifies otherwise.
