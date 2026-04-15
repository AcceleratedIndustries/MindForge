"""Tests for the knowledge graph builder."""

from pathlib import Path

from mindforge.distillation.concept import Concept, ConceptStore, Relationship, RelationshipType
from mindforge.graph.builder import KnowledgeGraph


class TestKnowledgeGraph:
    def _make_store(self) -> ConceptStore:
        store = ConceptStore()
        c1 = Concept(
            name="Embeddings",
            definition="Dense vector representations.",
            explanation="Maps data to vectors.",
            relationships=[
                Relationship("embeddings", "search", RelationshipType.ENABLES),
            ],
        )
        c2 = Concept(
            name="Search",
            definition="Finding relevant items.",
            explanation="Uses vectors for matching.",
            relationships=[
                Relationship("search", "embeddings", RelationshipType.DEPENDS_ON),
            ],
        )
        c3 = Concept(
            name="RAG",
            definition="Retrieval augmented generation.",
            explanation="Combines search with LLMs.",
            relationships=[
                Relationship("rag", "search", RelationshipType.USES),
                Relationship("rag", "embeddings", RelationshipType.USES),
            ],
        )
        store.add(c1)
        store.add(c2)
        store.add(c3)
        return store

    def test_build_from_store(self):
        store = self._make_store()
        graph = KnowledgeGraph.from_store(store)
        data = graph.to_json()
        assert data["metadata"]["node_count"] == 3
        assert data["metadata"]["edge_count"] == 4

    def test_neighbors(self):
        store = self._make_store()
        graph = KnowledgeGraph.from_store(store)
        neighbors = graph.neighbors("rag")
        assert "search" in neighbors
        assert "embeddings" in neighbors

    def test_central_concepts(self):
        store = self._make_store()
        graph = KnowledgeGraph.from_store(store)
        top = graph.central_concepts(top_n=3)
        assert len(top) > 0
        # Embeddings and search should be most central (both have 2+ connections)
        slugs = [s for s, _ in top]
        assert "embeddings" in slugs or "search" in slugs

    def test_save_and_load(self, tmp_path):
        store = self._make_store()
        graph = KnowledgeGraph.from_store(store)

        path = tmp_path / "graph.json"
        graph.save(path)

        loaded = KnowledgeGraph.load(path)
        assert loaded.to_json()["metadata"]["node_count"] == 3
        assert loaded.to_json()["metadata"]["edge_count"] == 4

    def test_stats(self):
        store = self._make_store()
        graph = KnowledgeGraph.from_store(store)
        stats = graph.stats()
        assert stats["nodes"] == 3
        assert stats["edges"] == 4
        assert stats["clusters"] >= 1

    def test_find_clusters(self):
        store = self._make_store()
        graph = KnowledgeGraph.from_store(store)
        clusters = graph.find_clusters()
        # All 3 concepts are connected, so should be 1 cluster
        assert len(clusters) == 1
        assert len(clusters[0]) == 3
