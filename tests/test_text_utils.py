"""Tests for mindforge.utils.text."""

from mindforge.utils.text import (
    slugify,
    content_hash,
    normalize_whitespace,
    extract_sentences,
    extract_keywords,
    compute_text_similarity,
)


class TestSlugify:
    def test_basic(self):
        assert slugify("Vector Embeddings") == "vector-embeddings"

    def test_special_chars(self):
        assert slugify("KV Cache (GPU)") == "kv-cache-gpu"

    def test_multiple_spaces(self):
        assert slugify("  hello   world  ") == "hello-world"

    def test_hyphens(self):
        assert slugify("Retrieval-Augmented Generation") == "retrieval-augmented-generation"


class TestContentHash:
    def test_deterministic(self):
        h1 = content_hash("hello world")
        h2 = content_hash("hello world")
        assert h1 == h2

    def test_different_input(self):
        h1 = content_hash("hello")
        h2 = content_hash("world")
        assert h1 != h2

    def test_length(self):
        h = content_hash("test")
        assert len(h) == 12


class TestNormalizeWhitespace:
    def test_collapse(self):
        assert normalize_whitespace("hello   world") == "hello world"

    def test_newlines(self):
        assert normalize_whitespace("hello\n\nworld") == "hello world"

    def test_strip(self):
        assert normalize_whitespace("  hello  ") == "hello"


class TestExtractSentences:
    def test_basic(self):
        sentences = extract_sentences("Hello world. This is a test. Done!")
        assert len(sentences) == 3

    def test_single_sentence(self):
        sentences = extract_sentences("Hello world.")
        assert len(sentences) == 1

    def test_empty(self):
        sentences = extract_sentences("")
        assert len(sentences) == 0


class TestExtractKeywords:
    def test_returns_list(self):
        keywords = extract_keywords("Vector embeddings are dense numerical representations.")
        assert isinstance(keywords, list)

    def test_filters_stopwords(self):
        keywords = extract_keywords("The quick brown fox jumps over the lazy dog.")
        assert "the" not in keywords

    def test_top_n(self):
        text = "embeddings embeddings vectors vectors search search"
        keywords = extract_keywords(text, top_n=2)
        assert len(keywords) <= 2


class TestComputeTextSimilarity:
    def test_identical(self):
        sim = compute_text_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_different(self):
        sim = compute_text_similarity("hello world", "foo bar baz")
        assert sim == 0.0

    def test_partial(self):
        sim = compute_text_similarity("hello world test", "hello world other")
        assert 0 < sim < 1

    def test_empty(self):
        sim = compute_text_similarity("", "hello")
        assert sim == 0.0
