"""Tests for Ollama embedding provider."""

from unittest.mock import MagicMock, patch

from mindforge.embeddings.ollama_provider import OllamaEmbeddingProvider


def test_embed_single_text_calls_correct_endpoint():
    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434",
        model="nomic-embed-text",
    )
    fake_response = MagicMock()
    fake_response.read.return_value = b'{"embedding": [0.1, 0.2, 0.3]}'
    fake_response.__enter__ = lambda s: s
    fake_response.__exit__ = lambda *a: None
    with patch("urllib.request.urlopen", return_value=fake_response) as urlopen:
        vec = provider.embed("hello world")
    args, _ = urlopen.call_args
    assert "/api/embeddings" in args[0].full_url
    assert vec == [0.1, 0.2, 0.3]


def test_embed_batch_returns_list_of_vectors():
    provider = OllamaEmbeddingProvider(base_url="http://localhost:11434", model="nomic-embed-text")
    with patch.object(provider, "embed", side_effect=lambda t: [float(len(t))]):
        vecs = provider.embed_batch(["a", "bb", "ccc"])
    assert vecs == [[1.0], [2.0], [3.0]]


def test_unreachable_server_raises_clear_error():
    import urllib.error

    provider = OllamaEmbeddingProvider(
        base_url="http://nonexistent:11434", model="nomic-embed-text"
    )
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("unreachable")):
        try:
            provider.embed("x")
        except RuntimeError as e:
            assert "embedding server" in str(e).lower()
        else:
            raise AssertionError("expected RuntimeError")
