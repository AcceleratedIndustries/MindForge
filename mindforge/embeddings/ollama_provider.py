"""Ollama embedding provider — uses /api/embeddings.

Conforms to the implicit embedder protocol shared with
``mindforge.embeddings.openai_compat_provider.OpenAICompatibleEmbeddingProvider``:

    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class OllamaEmbeddingProvider:
    base_url: str
    model: str
    timeout: int = 60

    def embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            raise RuntimeError(
                f"Ollama embedding server unreachable at {self.base_url}: {e}"
            ) from e
        return list(body.get("embedding", []))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
