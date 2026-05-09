"""Deterministic content-derivative LLM client for pipeline tests.

Produces wire-format JSON responses derived from chunk text via a fixed
rule. No network. Reproducible byte-for-byte across runs. Selected via
`config.llm.provider == "mock"` and the make_llm_client factory.
"""

from __future__ import annotations

import json
import re

from mindforge.llm.client import LLMClient, LLMConfig, LLMResponse

_TITLECASE_PHRASE = re.compile(
    r"\b([A-Z][a-zA-Z0-9]*(?:-[A-Z][a-zA-Z0-9]*)*"
    r"(?:[ \t]+[A-Z][a-zA-Z0-9]*(?:-[A-Z][a-zA-Z0-9]*)*){0,3})\b"
)
_SENTENCE_BOUNDARY = re.compile(r"[.!?]\s+")
_MAX_CONCEPTS_PER_CALL = 3
_MIN_NAME_LEN = 3
_MAX_DEFINITION_CHARS = 300

# Real extraction prompts wrap the chunk in a TEXT: ... \n\nRespond with ...
# envelope. Slice that out so the mock acts on chunk content only — otherwise
# title-case words in the prompt boilerplate ("Extract", "TEXT") would shadow
# the actual concepts.
_PROMPT_TEXT_BLOCK = re.compile(
    r"TEXT:\s*\n(?P<body>.*?)\n\s*Respond with",
    re.DOTALL,
)


def _surrounding_sentence(text: str, position: int) -> str:
    """Return the sentence containing ``position`` in ``text``.

    Sentence boundaries are simple [.!?] followed by whitespace. Falls back to
    the whole text when no boundary is found.
    """
    start = 0
    end = len(text)
    for m in _SENTENCE_BOUNDARY.finditer(text):
        if m.end() <= position:
            start = m.end()
        elif m.start() >= position:
            end = m.start()
            break
    sentence = text[start:end].strip()
    return sentence or text.strip()


def _mock_concepts_from_text(text: str) -> list[dict[str, object]]:
    """Apply the deterministic content-derivative rule to ``text``."""
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for m in _TITLECASE_PHRASE.finditer(text):
        phrase = m.group(1).strip()
        if len(phrase) < _MIN_NAME_LEN or phrase in seen:
            continue
        seen.add(phrase)
        definition = _surrounding_sentence(text, m.start())[:_MAX_DEFINITION_CHARS]
        out.append(
            {
                "name": phrase,
                "definition": definition,
                "tags": ["mock"],
                "insights": [],
                "examples": [],
                "relationships": [],
            }
        )
        if len(out) >= _MAX_CONCEPTS_PER_CALL:
            break
    return out


class MockLLMClient(LLMClient):
    """Deterministic mock that returns canned JSON without network calls."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        super().__init__(config)
        self._available = True

    def generate(self, prompt: str, system: str = "", response_format: str = "") -> LLMResponse:
        m = _PROMPT_TEXT_BLOCK.search(prompt)
        text = m.group("body") if m else prompt
        concepts = _mock_concepts_from_text(text)
        body = json.dumps({"concepts": concepts})
        return LLMResponse(content=body, success=True)
