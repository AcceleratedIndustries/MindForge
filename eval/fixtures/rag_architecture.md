## Retrieval-Augmented Generation

RAG is a pattern that retrieves relevant documents from a knowledge base and provides them as context to an LLM before generating a response.

### Why RAG Matters

Foundation models have fixed training cutoffs. For a model to answer questions about new data, you either fine-tune it (expensive, slow) or retrieve context dynamically at query time. RAG picks the second path.

### Components

A typical RAG pipeline has four pieces:

- **Ingestion**: documents are chunked, embedded into vectors, and stored in a vector database.
- **Retrieval**: the query is embedded, nearest-neighbor search finds the top-k chunks.
- **Reranking**: a cross-encoder or smaller model scores the retrieved chunks more precisely.
- **Generation**: the LLM receives the top chunks as context and generates a grounded response.

### Vector Embeddings

Vector Embeddings are dense numerical representations of text in a high-dimensional space. Semantically similar strings map to nearby vectors. Embedding models are trained so cosine similarity reflects meaning rather than surface form.

### Hybrid Retrieval

Hybrid Retrieval combines keyword search (BM25) and semantic search (vector similarity), then reranks the union. This handles both exact-term matches and paraphrases.
