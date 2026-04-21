# MindForge

**Transform messy AI conversations into a structured, queryable knowledge base.**

MindForge is a local-first semantic memory engine. Feed it raw conversation transcripts and it distills them into atomic, interlinked concepts -- complete with a knowledge graph, wiki-style links, and optional vector search. Think of it as a second brain that actually organizes itself.

---

## Why MindForge?

Every conversation with an AI produces knowledge. Most of it vanishes into scroll history.

MindForge captures that knowledge and turns it into something you can **navigate, query, and build on**:

- **Concepts, not conversations** -- Each idea becomes its own clean Markdown file
- **Relationships are explicit** -- Wiki-style `[[links]]` and typed edges (`uses`, `depends_on`, `enables`)
- **Knowledge graph included** -- Visualize how concepts connect
- **Semantic search ready** -- Find concepts by meaning, not just keywords
- **LLM-powered extraction** -- Optionally use Ollama or any OpenAI-compatible API for dramatically better results
- **100% local** -- No cloud required. Your knowledge stays yours.

---

## Quick Start

```bash
# Install
pip install -e .

# Run on your transcripts
mindforge ingest --input path/to/transcripts --output output

# Ask questions
mindforge query "How does semantic search work?"

# See what you've built
mindforge stats
```

**With LLM extraction** (recommended for best results):

```bash
# Using local Ollama
mindforge ingest --input transcripts/ --llm --llm-model llama3.2

# Using OpenAI
mindforge ingest --input transcripts/ --llm --llm-provider openai --llm-api-key sk-...

# Using any OpenAI-compatible endpoint (vLLM, LM Studio, Together, etc.)
mindforge ingest --input transcripts/ --llm --llm-provider openai \
    --llm-base-url http://localhost:8000 --llm-model my-model
```

---

## What It Produces

Given a directory of conversation transcripts, MindForge outputs:

### Concept Files (Markdown)

Each concept gets its own file with YAML frontmatter, a clean definition, explanation, key insights, and wiki-style links to related concepts:

```markdown
---
title: "KV Cache"
slug: "kv-cache"
tags: [transformers, inference, optimization]
confidence: 0.90
---

# KV Cache

## Definition

KV Cache is a mechanism that stores the Key and Value matrices from the
attention computation of previously processed tokens, avoiding redundant
recomputation during autoregressive generation.

## Key Insights

- Trades memory for computation -- critical trade-off in production LLM serving
- Techniques like Multi-Query Attention reduce KV cache size

## Related Concepts

- [[Vector Embeddings]]
- [[Attention Mechanism]]
```

### Knowledge Graph (JSON)

A NetworkX-powered directed graph with typed edges, exportable as JSON:

```json
{
  "nodes": [{"id": "kv-cache", "label": "KV Cache"}],
  "edges": [{"source": "rag", "target": "semantic-search", "type": "uses"}]
}
```

### Embeddings Index (Optional)

FAISS-backed vector index for semantic search over your concepts.

---

## The Pipeline

```
Transcripts --> Parse --> Chunk --> Extract --> Deduplicate --> Distill --> Link --> Graph
                                     |                                      |
                                     +-- Heuristic (patterns, headings)     +-- Wiki-links
                                     +-- LLM (Ollama / OpenAI) [optional]   +-- Typed relationships
```

| Stage | What it does |
|-------|-------------|
| **Parse** | Multi-format transcript parser (role-prefixed, heading-style, separators) |
| **Chunk** | Semantic chunking that respects paragraphs, headings, and code blocks |
| **Extract** | Identify concepts via definition patterns, heading analysis, keyword frequency -- or LLM |
| **Deduplicate** | Merge near-duplicates by slug matching and Jaccard similarity |
| **Distill** | Clean conversational fluff, extract insights, build structured definitions |
| **Link** | Detect relationships via co-occurrence, keyword overlap, and structural patterns |
| **Graph** | Build a NetworkX knowledge graph, export as JSON |
| **Index** | Optional FAISS + sentence-transformers for semantic queries |

---

## Project Structure

```
mindforge/
├── cli.py                  # CLI: ingest / query / stats
├── config.py               # Central configuration
├── pipeline.py             # 6-stage pipeline orchestrator
├── ingestion/
│   ├── parser.py           # Multi-format transcript parser
│   ├── chunker.py          # Semantic chunking
│   └── extractor.py        # Heuristic concept extraction
├── llm/
│   ├── client.py           # Ollama + OpenAI HTTP client (stdlib only)
│   ├── extractor.py        # LLM-based concept extraction
│   └── distiller.py        # LLM-aware concept distillation
├── mcp/
│   └── server.py           # MCP server (JSON-RPC over stdio)
├── distillation/
│   ├── concept.py          # Data models (Concept, Relationship, ConceptStore)
│   ├── deduplicator.py     # Similarity-based deduplication
│   ├── distiller.py        # Raw -> clean concept transformation
│   └── renderer.py         # Concept -> Markdown with frontmatter
├── linking/
│   └── linker.py           # Relationship detection + wiki-links
├── graph/
│   └── builder.py          # NetworkX graph + JSON export
├── embeddings/
│   └── index.py            # FAISS + sentence-transformers index
├── query/
│   └── engine.py           # Keyword + semantic search
└── utils/
    └── text.py             # Slugify, similarity, keyword extraction
```

---

## Installation

**Core** (no external dependencies beyond NetworkX):

```bash
pip install -e .
```

**With semantic search**:

```bash
pip install -e ".[embeddings]"
```

**For development**:

```bash
pip install -e ".[dev]"
pytest
```

---

## Transcript Formats

MindForge accepts Markdown or plain text files. Drop them in a directory and point `--input` at it.

Supported conversation formats:

- **Role-prefixed**: `User: ...` / `Assistant: ...`
- **Heading-style**: `## User` / `## Assistant`
- **Separator-based**: Turns separated by `---`
- **Plain text**: Treated as a single knowledge document

---

## Heuristic vs LLM Extraction

MindForge works in two modes:

| | Heuristic (default) | LLM-assisted (`--llm`) |
|---|---|---|
| **Speed** | Instant | Depends on model |
| **Dependencies** | None | Ollama or API |
| **Quality** | Good for well-structured transcripts | Excellent for any text |
| **Relationships** | Detected via patterns | Explicitly identified by the LLM |
| **Fallback** | N/A | Automatic fallback to heuristic |

When `--llm` is enabled, MindForge runs **both** extractors and merges the results. LLM-identified concepts take priority, and any unique heuristic findings are added in. If the LLM server is unreachable, it falls back to heuristic-only with a clear message.

---

## Relationship Types

MindForge tracks eight types of concept relationships:

| Type | Meaning | Example |
|------|---------|---------|
| `uses` | A uses B | RAG **uses** Semantic Search |
| `depends_on` | A requires B | Semantic Search **depends on** Embeddings |
| `enables` | A makes B possible | Embeddings **enables** Similarity Search |
| `improves` | A enhances B | Hybrid Search **improves** Recall |
| `part_of` | A is a component of B | HNSW **part of** Vector Database |
| `example_of` | A is an instance of B | Qdrant **example of** Vector Database |
| `contrasts_with` | A differs from B | Keyword Search **contrasts with** Semantic Search |
| `related_to` | General association | KV Cache **related to** Attention Mechanism |

---

## MCP Server (AI Agent Interface)

MindForge includes an MCP (Model Context Protocol) server that lets external AI agents query your knowledge base as a tool.

```bash
# Start the MCP server
mindforge mcp --output path/to/knowledge-base
```

**Available tools:**

| Tool | Description |
|------|-------------|
| `search` | Natural language search across all concepts |
| `get_concept` | Get full details of a concept by name or slug |
| `list_concepts` | List all concepts, optionally filtered by tag |
| `get_neighbors` | Get related concepts via the knowledge graph |
| `get_stats` | Knowledge base statistics and most central concepts |

**Claude Desktop configuration** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "mindforge": {
      "command": "mindforge",
      "args": ["mcp", "--output", "/path/to/your/output"]
    }
  }
}
```

---

## Roadmap

- [x] **MCP server interface** -- Expose the knowledge base as a tool server for external AI agents
- [ ] **Incremental ingestion** -- Content hashing to skip already-processed transcripts
- [ ] **Concept versioning** -- Track how concepts evolve across ingestion runs
- [ ] **Confidence decay** -- Unreinforced concepts fade over time, surfacing review candidates
- [ ] **Auto-refactoring** -- Merge or split concepts as the knowledge base grows
- [ ] **Graph visualization** -- Interactive web UI for exploring the knowledge graph

---

## License

Business Source License 1.1 (BUSL-1.1). See [LICENSE](LICENSE) for details.

In short: free for any use, including commercial use, except offering MindForge as a hosted or managed service with functionality substantially similar to a paid offering by Accelerated Industries. Each release auto-converts to the Apache License 2.0 on April 21, 2028.
