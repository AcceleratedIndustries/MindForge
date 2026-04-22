Human: Can you explain what a KV cache is and why it matters for LLM inference?

Assistant: KV Cache is a mechanism that stores the Key and Value matrices from the attention computation of previously processed tokens, avoiding redundant recomputation during autoregressive generation.

During generation, each new token requires attending to all previous tokens. Without caching, the model would recompute the Key and Value projections for every prior token at every step. That is O(n^2) work for a sequence of length n.

The KV cache stores these projections after the first pass, so subsequent tokens only compute Key and Value for themselves and reuse the cached values. This trades memory for speed — a critical trade-off in production LLM serving.

Optimizations like Multi-Query Attention reduce KV cache size by sharing Key and Value heads across attention heads.
