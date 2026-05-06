"""Indirect-prompt-injection mitigations for MCP tool responses.

Two interventions:

1. Content tagging — every tool wraps user-derived content in
   ``<mindforge_retrieved_content>...</mindforge_retrieved_content>`` so
   the calling agent's system prompt can treat the body as data, not as
   trusted instructions.

2. Hidden-Unicode stripping — remove zero-width / BOM / bidi-override /
   tag-block characters used for steganographic prompt injection.
"""

from __future__ import annotations

import re

OPEN_TAG = "<mindforge_retrieved_content>"
CLOSE_TAG = "</mindforge_retrieved_content>"

# Hidden-Unicode ranges used for steganographic injection. Explicit \u
# escapes keep the source reviewable — embedded invisible characters in
# the regex literal would defeat the purpose.
_HIDDEN_UNICODE = re.compile(
    "["
    "​-‏"  # ZWSP, ZWNJ, ZWJ, LRM, RLM
    "‪-‮"  # bidi overrides (LRE, RLE, PDF, LRO, RLO)
    "⁠-⁤"  # word joiner, invisible separators/times/plus
    "﻿"  # byte order mark (zero-width no-break space)
    "\U000e0000-\U000e007f"  # tag block
    "]"
)


def strip_hidden_unicode(text: str) -> str:
    """Remove hidden-Unicode characters from ``text``."""
    return _HIDDEN_UNICODE.sub("", text)


def wrap_retrieved_content(text: str) -> str:
    """Wrap retrieved content in agent-facing tags, after stripping hidden chars."""
    cleaned = strip_hidden_unicode(text)
    return f"{OPEN_TAG}\n{cleaned}\n{CLOSE_TAG}"
