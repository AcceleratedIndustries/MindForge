"""Tests for the linking engine and wiki-link insertion."""

from mindforge.distillation.concept import Concept, ConceptStore
from mindforge.linking.linker import detect_links, insert_wiki_links


class TestInsertWikiLinks:
    def test_basic_insertion(self):
        text = "Vector search relies on Embeddings for matching."
        result = insert_wiki_links(text, ["Embeddings"])
        assert "[[Embeddings]]" in result

    def test_no_double_link(self):
        text = "This uses [[Embeddings]] already."
        result = insert_wiki_links(text, ["Embeddings"])
        assert result.count("[[Embeddings]]") == 1

    def test_case_insensitive(self):
        text = "Using embeddings for search."
        result = insert_wiki_links(text, ["Embeddings"])
        assert "[[embeddings]]" in result

    def test_only_first_occurrence(self):
        text = "Embeddings are great. Embeddings are everywhere."
        result = insert_wiki_links(text, ["Embeddings"])
        assert result.count("[[") == 1


class TestDetectLinks:
    def test_name_mention_creates_link(self):
        store = ConceptStore()
        store.add(Concept(
            name="Vector Embeddings",
            definition="Dense numerical representations of data.",
            explanation="Used for similarity search.",
            tags=["ml", "vectors"],
        ))
        store.add(Concept(
            name="Semantic Search",
            definition="Search that uses Vector Embeddings to find similar items.",
            explanation="Relies on dense vectors for matching.",
            tags=["search", "vectors"],
        ))

        detect_links(store, confidence_threshold=0.2)

        search_concept = store.get("semantic-search")
        assert "Vector Embeddings" in search_concept.links

    def test_no_self_links(self):
        store = ConceptStore()
        store.add(Concept(
            name="Test",
            definition="Test is a test concept about testing.",
            explanation="Testing the test.",
            tags=["test"],
        ))

        detect_links(store)

        concept = store.get("test")
        assert "Test" not in concept.links
