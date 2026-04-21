# Feature: Hybrid Retrieval

**Phase:** 3.4 (can ship in parallel with 3.1-3.3)
**Depends on:** —
**Unblocks:** better UI search, better MCP `search` + `get_context_pack`

---

## Motivation

Today, `query` is either keyword or semantic, picked by a flag. This underuses the graph entirely. Good retrieval blends all three signals.

Real retrieval quality comes from **hybrid**: keyword (exact matches), semantic (meaning), and graph-walk (neighborhood reinforcement). Default to hybrid; let advanced users tune.

---

## User-facing behavior

No change to the surface — `mindforge query "..."` and `/api/v1/search` just get quietly better.

Advanced:

```bash
mindforge query "vector databases" --mode hybrid   # default
mindforge query "vector databases" --mode keyword
mindforge query "vector databases" --mode semantic
mindforge query "vector databases" --weights 0.3,0.5,0.2  # k/s/g weights
```

---

## Design

### Retrieval stages

```
query
  │
  ├──► keyword scorer      → score_k per concept  (BM25-lite over name + definition + insights)
  ├──► semantic scorer     → score_s per concept  (cosine on embeddings, if available)
  └──► seed pool = top-N from keyword + semantic
         │
         └──► graph walker → score_g per concept  (walks 1-2 hops from seeds,
                                                  accumulates reinforcement weighted
                                                  by edge confidence & hop distance)

combined = w_k * score_k + w_s * score_s + w_g * score_g
return top-k by combined score
```

### Default weights

```python
weights = RetrievalWeights(
    keyword=0.4,
    semantic=0.4,
    graph=0.2,
)
```

If embeddings are unavailable: reweight to `(0.6, 0.0, 0.4)` automatically.

### Graph walk scoring

For each seed concept `s` with seed score `σ(s)`:
- Contribute `σ(s) * 1.0` to `score_g[s]`
- For each neighbor `n` at hop 1: `score_g[n] += σ(s) * edge_confidence * 0.5`
- For each neighbor at hop 2: `score_g[n] += σ(s) * edge_confidence * 0.2`

This rewards concepts that are **adjacent** to multiple strong keyword/semantic hits.

### BM25-lite

Full BM25 over a small corpus is overkill. Use TF-IDF over the set of (name, definition, key insights) strings. Existing `utils/text.py` has keyword extraction; reuse.

### Result shape

```python
@dataclass
class SearchResult:
    concept: Concept
    score_total: float
    score_breakdown: dict[str, float]   # {"keyword": 0.3, "semantic": 0.5, "graph": 0.1}
    matched_via: list[str]              # ["keyword", "graph"] etc., for explainability
```

UI uses `score_breakdown` to show why each result matched.

---

## Files touched

### New
- `mindforge/query/keyword_scorer.py`
- `mindforge/query/graph_walker.py`

### Modified
- `mindforge/query/engine.py` — `search()` becomes the hybrid orchestrator; old keyword/semantic modes become internal scorers
- `mindforge/cli.py` — `--mode` and `--weights` flags
- `mindforge/server/routes/search.py` — pass through mode + weights

---

## Testing

- `tests/test_hybrid_retrieval.py`:
  - Synthetic KB with 10 concepts and a tiny graph; assert that a graph-adjacent concept ranks higher in hybrid than in keyword-only.
  - Fallback-when-no-embeddings reweights correctly.
  - `score_breakdown` sums to `score_total` (within floating-point tolerance).

---

## Open questions

- **Weight tuning:** defaults are guesses. Ideal: learn weights from the eval harness's relevance judgments. **Proposed:** ship sensible defaults; add `mindforge eval --tune-retrieval` later that sweeps weights against the fixture corpus.
- **Graph walk depth:** 2 hops is a compromise. 3 hops can drown results in noise. **Proposed:** fixed at 2, config if demand emerges.
- **Re-ranking with LLM:** at top-k=5, an LLM re-rank of keyword+semantic+graph candidates could be excellent. **Proposed:** out of scope for v1; consider as `--mode llm-rerank` in a later pass.
