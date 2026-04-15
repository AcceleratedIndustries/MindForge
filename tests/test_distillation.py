"""Tests for concept distillation, deduplication, and rendering."""

from pathlib import Path

from mindforge.distillation.concept import Concept, ConceptStore, Relationship, RelationshipType
from mindforge.distillation.deduplicator import deduplicate_concepts
from mindforge.distillation.distiller import distill_concept
from mindforge.distillation.renderer import render_concept, write_concept
from mindforge.ingestion.extractor import RawConcept


class TestConcept:
    def test_slug(self):
        c = Concept(name="Vector Embeddings", definition="test", explanation="test")
        assert c.slug == "vector-embeddings"

    def test_hash_deterministic(self):
        c = Concept(name="Test", definition="def", explanation="exp")
        assert c.hash == c.hash

    def test_to_dict_and_back(self):
        c = Concept(
            name="Test Concept",
            definition="A test concept.",
            explanation="Longer explanation.",
            tags=["test", "demo"],
            confidence=0.9,
        )
        data = c.to_dict()
        restored = Concept.from_dict(data)
        assert restored.name == c.name
        assert restored.definition == c.definition
        assert restored.tags == c.tags

    def test_merge(self):
        c1 = Concept(
            name="Test",
            definition="Short def.",
            explanation="Short.",
            insights=["insight 1"],
            tags=["a"],
        )
        c2 = Concept(
            name="Test",
            definition="A longer definition here.",
            explanation="A longer explanation.",
            insights=["insight 2"],
            tags=["b"],
        )
        merged = c1.merge_with(c2)
        assert merged.name == "Test"
        assert "insight 1" in merged.insights
        assert "insight 2" in merged.insights
        assert "a" in merged.tags
        assert "b" in merged.tags


class TestConceptStore:
    def test_add_and_get(self):
        store = ConceptStore()
        c = Concept(name="Test", definition="def", explanation="exp")
        store.add(c)
        assert store.get("test") is not None

    def test_add_duplicate_merges(self):
        store = ConceptStore()
        c1 = Concept(name="Test", definition="def1", explanation="exp1", tags=["a"])
        c2 = Concept(name="Test", definition="def2 longer", explanation="exp2", tags=["b"])
        store.add(c1)
        store.add(c2)
        result = store.get("test")
        assert "a" in result.tags
        assert "b" in result.tags

    def test_save_and_load(self, tmp_path):
        store = ConceptStore()
        store.add(Concept(name="Alpha", definition="First", explanation="First concept"))
        store.add(Concept(name="Beta", definition="Second", explanation="Second concept"))

        path = tmp_path / "concepts.json"
        store.save(path)

        loaded = ConceptStore.load(path)
        assert len(loaded.all()) == 2
        assert loaded.get("alpha") is not None


class TestDeduplicator:
    def test_exact_slug_dedup(self):
        raws = [
            RawConcept(name="Vector DB", raw_content="A database for vectors."),
            RawConcept(name="Vector DB", raw_content="Stores vector embeddings."),
        ]
        result = deduplicate_concepts(raws)
        assert len(result) == 1

    def test_similar_content_dedup(self):
        raws = [
            RawConcept(name="Embedding", raw_content="Dense numerical representations of text data in vector space"),
            RawConcept(name="Embeddings", raw_content="Dense numerical representations of text data in a vector space"),
        ]
        result = deduplicate_concepts(raws, similarity_threshold=0.6)
        assert len(result) == 1

    def test_different_concepts_kept(self):
        raws = [
            RawConcept(name="Vector Database", raw_content="Stores and queries high-dimensional vectors"),
            RawConcept(name="KV Cache", raw_content="Caches key-value pairs during transformer inference"),
        ]
        result = deduplicate_concepts(raws)
        assert len(result) == 2


class TestDistiller:
    def test_distill_basic(self):
        raw = RawConcept(
            name="Test Concept",
            raw_content="Test Concept is a fundamental building block. It provides key functionality for systems. This is important for performance.",
            source_files=["test.md"],
            confidence=0.8,
        )
        concept = distill_concept(raw)
        assert concept.name == "Test Concept"
        assert len(concept.definition) > 0
        assert concept.confidence == 0.8

    def test_removes_fluff(self):
        raw = RawConcept(
            name="Test",
            raw_content="Great question! Let me explain. Test is a method for verification. Hope this helps!",
        )
        concept = distill_concept(raw)
        assert "great question" not in concept.definition.lower()
        assert "hope this helps" not in concept.definition.lower()


class TestRenderer:
    def test_render_basic(self):
        c = Concept(
            name="Vector Embeddings",
            definition="Dense numerical representations.",
            explanation="Used in ML systems.",
            tags=["ml", "vectors"],
            confidence=0.85,
        )
        md = render_concept(c)
        assert "# Vector Embeddings" in md
        assert "Dense numerical representations." in md
        assert "confidence: 0.85" in md

    def test_render_with_links(self):
        c = Concept(
            name="Semantic Search",
            definition="Search by meaning.",
            explanation="Uses embeddings.",
            links=["Vector Embeddings", "FAISS"],
        )
        md = render_concept(c)
        assert "[[Vector Embeddings]]" in md
        assert "[[FAISS]]" in md

    def test_render_with_relationships(self):
        c = Concept(
            name="RAG",
            definition="Retrieval augmented generation.",
            explanation="Combines retrieval with generation.",
            relationships=[
                Relationship("rag", "semantic-search", RelationshipType.USES),
            ],
        )
        md = render_concept(c)
        assert "**uses**" in md

    def test_write_concept(self, tmp_path):
        c = Concept(
            name="Test Concept",
            definition="A test.",
            explanation="A test concept.",
        )
        path = write_concept(c, tmp_path)
        assert path.exists()
        assert path.name == "test-concept.md"
        content = path.read_text()
        assert "# Test Concept" in content
