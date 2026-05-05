"""OpenAI-compatible embedding provider — works with llama.cpp, vLLM, OpenAI proper.

Conforms to the implicit embedder protocol shared with
``mindforge.embeddings.ollama_provider.OllamaEmbeddingProvider``:

    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class OpenAICompatibleEmbeddingProvider:
    base_url: str  # e.g. "http://localhost:8080/v1" or "https://api.openai.com/v1"
    model: str
    api_key: str = ""
    timeout: int = 60

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            raise RuntimeError(
                f"OpenAI-compatible embedding server unreachable at {self.base_url}: {e}"
            ) from e
        return [list(item.get("embedding", [])) for item in body.get("data", [])]
