"""Indirect-prompt-injection mitigations for MCP tool responses.

Two interventions:

1. Content tagging - every tool wraps user-derived content in
   ``<mindforge_retrieved_content>...</mindforge_retrieved_content>`` so
   the calling agent's system prompt can treat the body as data, not as
   trusted instructions.

2. Hidden-Unicode stripping - remove zero-width / BOM / bidi-override /
   tag-block characters used for steganographic prompt injection.
"""

from __future__ import annotations

import re

OPEN_TAG = "<mindforge_retrieved_content>"
CLOSE_TAG = "</mindforge_retrieved_content>"

# Hidden-Unicode ranges used for steganographic injection. Encoded with
# explicit escape sequences only - embedding the literal characters in the
# regex source defeats the purpose (and bandit B613:trojansource rightly
# flags any source file that contains them). Ranges:
#
#   U+200B-U+200F    ZWSP, ZWNJ, ZWJ, LRM, RLM
#   U+202A-U+202E    bidi overrides (LRE, RLE, PDF, LRO, RLO)
#   U+2060-U+2064    word joiner, invisible separator/times/plus/function
#   U+FEFF           BOM (zero-width no-break space)
#   U+E0000-U+E007F  tag block
_HIDDEN_UNICODE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\U000e0000-\U000e007f]")


def strip_hidden_unicode(text: str) -> str:
    """Remove hidden-Unicode characters from ``text``."""
    return _HIDDEN_UNICODE.sub("", text)


def wrap_retrieved_content(text: str) -> str:
    """Wrap retrieved content in agent-facing tags, after stripping hidden chars."""
    cleaned = strip_hidden_unicode(text)
    return f"{OPEN_TAG}\n{cleaned}\n{CLOSE_TAG}"
