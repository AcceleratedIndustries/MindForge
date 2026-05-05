"""Tests for OpenAI-compatible embedding provider."""

import json
from unittest.mock import MagicMock, patch

from mindforge.embeddings.openai_compat_provider import OpenAICompatibleEmbeddingProvider


def test_embed_calls_v1_embeddings_endpoint():
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:8080/v1",
        model="text-embedding-3-small",
        api_key="test-key",
    )
    fake_response = MagicMock()
    fake_response.read.return_value = json.dumps(
        {"data": [{"embedding": [0.1, 0.2]}]}
    ).encode()
    fake_response.__enter__ = lambda s: s
    fake_response.__exit__ = lambda *a: None
    with patch("urllib.request.urlopen", return_value=fake_response) as urlopen:
        vec = provider.embed("hello")
    req = urlopen.call_args[0][0]
    assert "/v1/embeddings" in req.full_url
    assert req.headers.get("Authorization") == "Bearer test-key"
    assert vec == [0.1, 0.2]


def test_embed_batch_uses_single_request():
    """OpenAI-style /v1/embeddings accepts a list as input — no per-text loop."""
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:8080/v1", model="x", api_key=""
    )
    fake_response = MagicMock()
    fake_response.read.return_value = json.dumps(
        {"data": [{"embedding": [0.1]}, {"embedding": [0.2]}]}
    ).encode()
    fake_response.__enter__ = lambda s: s
    fake_response.__exit__ = lambda *a: None
    with patch("urllib.request.urlopen", return_value=fake_response) as urlopen:
        vecs = provider.embed_batch(["a", "b"])
    assert vecs == [[0.1], [0.2]]
    assert urlopen.call_count == 1
