"""Embeddings index: optional semantic search over concepts using vector similarity.

Uses sentence-transformers for encoding and FAISS for fast nearest-neighbor search.
Falls back gracefully when dependencies are not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from mindforge.distillation.concept import Concept

if TYPE_CHECKING:
    import numpy as np


def _check_deps() -> bool:
    """Check if embedding dependencies are available."""
    try:
        import sentence_transformers  # noqa: F401
        import faiss  # noqa: F401
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


class EmbeddingIndex:
    """Semantic search index for MindForge concepts.

    Encodes concept definitions + explanations into dense vectors
    and supports nearest-neighbor queries.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._index = None
        self._slugs: list[str] = []
        self._dimension: int = 0
        self._available = _check_deps()

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

    def build(self, concepts: list[Concept]) -> None:
        """Build the index from a list of concepts."""
        if not self._available:
            return

        import faiss
        import numpy as np

        self._ensure_model()

        texts = [self._concept_text(c) for c in concepts]
        self._slugs = [c.slug for c in concepts]

        # Encode all concepts
        embeddings = self._model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        self._dimension = embeddings.shape[1]

        # Build FAISS index
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(embeddings)

    def query(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Query the index with natural language. Returns (slug, score) pairs."""
        if not self._available or self._index is None:
            return []

        import numpy as np

        self._ensure_model()

        # Encode query
        query_embedding = self._model.encode([text])
        query_embedding = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        # Search
        k = min(top_k, len(self._slugs))
        scores, indices = self._index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self._slugs):
                results.append((self._slugs[idx], float(score)))
        return results

    def save(self, directory: Path) -> None:
        """Save the index to disk."""
        if not self._available or self._index is None:
            return

        import faiss
        import numpy as np

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "concepts.faiss"))

        metadata = {
            "slugs": self._slugs,
            "model": self._model_name,
            "dimension": self._dimension,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, directory: Path) -> EmbeddingIndex:
        """Load an index from disk."""
        index = cls()
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
