"""Embeddings index: optional semantic search over concepts using vector similarity.

Uses FAISS for fast nearest-neighbor search. Vectors are produced by a swappable
``Embedder`` (sentence-transformers by default; Ollama or OpenAI-compatible HTTP
providers are also supported via ``embedder=`` constructor arg).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from mindforge.distillation.concept import Concept


@runtime_checkable
class Embedder(Protocol):
    """Structural type for swappable embedding providers."""

    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


def _check_storage_deps() -> bool:
    """Storage layer requires faiss + numpy regardless of embedder choice."""
    try:
        import faiss  # noqa: F401
        import numpy  # noqa: F401

        return True
    except ImportError:
        return False


def _check_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


class EmbeddingIndex:
    """Semantic search index for MindForge concepts.

    Encodes concept definitions + explanations into dense vectors
    and supports nearest-neighbor queries. Vector generation is delegated
    to ``embedder`` when provided; otherwise the legacy sentence-transformers
    path is used.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedder: Embedder | None = None,
    ) -> None:
        self._model_name = model_name
        self._embedder = embedder
        self._model = None
        self._index = None
        self._slugs: list[str] = []
        self._dimension: int = 0
        has_storage = _check_storage_deps()
        has_encoder = embedder is not None or _check_sentence_transformers()
        self._available = has_storage and has_encoder

    @property
    def available(self) -> bool:
        return self._available

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)

    def _concept_text(self, concept: Concept) -> str:
        """Build the text to encode for a concept."""
        parts = [concept.name, concept.definition]
        if concept.explanation != concept.definition:
            parts.append(concept.explanation)
        if concept.insights:
            parts.extend(concept.insights[:3])
        return " ".join(parts)

    def _encode_batch(self, texts: list[str]):
        import numpy as np

        if self._embedder is not None:
            raw = self._embedder.embed_batch(texts)
            return np.array(raw, dtype=np.float32)
        self._ensure_model()
        if self._model is None:
            raise RuntimeError("Embedding model failed to load.")
        encoded = self._model.encode(texts, show_progress_bar=False)
        return np.array(encoded, dtype=np.float32)

    def _encode_one(self, text: str):
        import numpy as np

        if self._embedder is not None:
            return np.array([self._embedder.embed(text)], dtype=np.float32)
        self._ensure_model()
        if self._model is None:
            raise RuntimeError("Embedding model failed to load.")
        encoded = self._model.encode([text])
        return np.array(encoded, dtype=np.float32)

    def build(self, concepts: list[Concept]) -> None:
        """Build the index from a list of concepts."""
        if not self._available:
            return

        import faiss
        import numpy as np

        texts = [self._concept_text(c) for c in concepts]
        self._slugs = [c.slug for c in concepts]

        embeddings = self._encode_batch(texts)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        self._dimension = embeddings.shape[1]

        # Build FAISS index
        faiss_index = faiss.IndexFlatIP(self._dimension)
        faiss_index.add(embeddings)
        self._index = faiss_index

    def query(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Query the index with natural language. Returns (slug, score) pairs."""
        if not self._available or self._index is None:
            return []

        import numpy as np

        query_embedding = self._encode_one(text)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        # Search
        k = min(top_k, len(self._slugs))
        scores, indices = self._index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < len(self._slugs):
                results.append((self._slugs[idx], float(score)))
        return results

    def save(self, directory: Path) -> None:
        """Save the index to disk."""
        if not self._available or self._index is None:
            return

        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "concepts.faiss"))

        metadata = {
            "slugs": self._slugs,
            "model": self._model_name,
            "dimension": self._dimension,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(
        cls,
        directory: Path,
        embedder: Embedder | None = None,
    ) -> EmbeddingIndex:
        """Load an index from disk. Pass ``embedder`` to query without sentence-transformers."""
        index = cls(embedder=embedder)
        if not index._available:
            return index

        import faiss

        metadata_path = directory / "metadata.json"
        index_path = directory / "concepts.faiss"

        if metadata_path.exists() and index_path.exists():
            metadata = json.loads(metadata_path.read_text())
            index._slugs = metadata["slugs"]
            index._model_name = metadata["model"]
            index._dimension = metadata["dimension"]
            index._index = faiss.read_index(str(index_path))

        return index
