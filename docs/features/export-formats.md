# Feature: Export Formats

**Phase:** 5.2
**Depends on:** —

---

## Motivation

A knowledge base that can't be exported is a silo. Two target formats serve different consumers:

1. **JSON-LD / RDF** — for users who want to push MindForge data into a broader semantic-web or graph toolchain (Neo4j, Blazegraph, etc.).
2. **Context packs** — for agents and LLM workflows that want a single prompt-ready markdown blob.

---

## User-facing behavior

```bash
# JSON-LD export
mindforge export jsonld -o mindforge.jsonld

# RDF/Turtle
mindforge export rdf --format turtle -o mindforge.ttl

# Context pack for a specific query (matches the MCP tool)
mindforge export context-pack "how does RAG work?" -o rag-context.md

# Whole-KB markdown bundle (all concepts, one file, for pasting to an LLM)
mindforge export bundle --max-chars 40000 -o bundle.md
```

---

## Design

### JSON-LD schema

Use a lightweight context that maps MindForge's types onto Schema.org and SKOS where possible:

```json
{
  "@context": {
    "@vocab": "https://mindforge.dev/ns#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "schema": "https://schema.org/",
    "name": "schema:name",
    "definition": "skos:definition",
    "related": {"@id": "skos:related", "@type": "@id"}
  },
  "@graph": [
    {
      "@id": "mf:kv-cache",
      "@type": "Concept",
      "name": "KV Cache",
      "definition": "...",
      "tags": ["transformers", "inference"],
      "related": ["mf:attention-mechanism"],
      "uses": ["mf:autoregressive-generation"]
    }
  ]
}
```

Edge types map to custom predicates under `mindforge.dev/ns#`.

### RDF

Convert the JSON-LD via `rdflib` to Turtle / N-Triples / RDF/XML.

### Context pack (CLI mirror of MCP tool)

Same code path as `get_context_pack` MCP tool. Just a CLI entrypoint.

### Bundle

Serialize the full KB as one markdown document: TOC at top, then each concept as a section. Truncates at `max_chars`, preferring high-confidence and high-centrality concepts.

---

## Files touched

### New
- `mindforge/export/__init__.py`
- `mindforge/export/jsonld.py`
- `mindforge/export/rdf.py`
- `mindforge/export/bundle.py`

### Modified
- `mindforge/cli.py` — `mindforge export {jsonld,rdf,context-pack,bundle}`
- `pyproject.toml` — `rdflib` behind a new `[export]` extra (only needed for RDF)

---

## Testing

- `tests/test_export_jsonld.py` — schema-validate output against the JSON-LD context
- `tests/test_export_rdf.py` — roundtrip through rdflib, assert triple counts match
- `tests/test_export_bundle.py` — char-limit enforcement, TOC correctness

---

## Open questions

- **Licensing of exports:** MindForge is GPL-3.0. Exported knowledge derives from user content, not MindForge code — should be license-free from MindForge's perspective. Document explicitly.
- **Bundle ordering:** should it be by centrality, by confidence, by tag, or user-chosen? **Proposed:** centrality then confidence, configurable via `--order`.
- **JSON-LD context hosting:** ideally `mindforge.dev/ns` resolves. If that domain isn't owned, embed the context inline for v1.
