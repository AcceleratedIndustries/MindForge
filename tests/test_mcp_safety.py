"""Tests for MCP safety helpers (content tagging + hidden-Unicode stripping)."""

from mindforge.mcp.safety import strip_hidden_unicode, wrap_retrieved_content


def test_wrap_returns_tagged_content() -> None:
    out = wrap_retrieved_content("hello world")
    assert out.startswith("<mindforge_retrieved_content>")
    assert out.endswith("</mindforge_retrieved_content>")
    assert "hello world" in out


def test_wrap_handles_empty_string() -> None:
    out = wrap_retrieved_content("")
    assert out == "<mindforge_retrieved_content>\n\n</mindforge_retrieved_content>"


def test_wrap_strips_hidden_unicode_inside_tags() -> None:
    out = wrap_retrieved_content("hel​lo")
    assert "hel​lo" not in out
    assert "hello" in out


def test_strip_zero_width_space() -> None:
    src = "hel​lo"  # ZERO WIDTH SPACE inside
    assert strip_hidden_unicode(src) == "hello"


def test_strip_zero_width_joiner() -> None:
    src = "a‍b"  # ZERO WIDTH JOINER
    assert strip_hidden_unicode(src) == "ab"


def test_strip_bom() -> None:
    src = "﻿hello"  # BYTE ORDER MARK
    assert strip_hidden_unicode(src) == "hello"


def test_strip_bidi_override() -> None:
    src = "abc‮def"  # RIGHT-TO-LEFT OVERRIDE
    assert strip_hidden_unicode(src) == "abcdef"


def test_strip_word_joiner() -> None:
    src = "abc⁠def"  # WORD JOINER
    assert strip_hidden_unicode(src) == "abcdef"


def test_strip_preserves_visible_unicode() -> None:
    src = "hello 世界 🌍"
    assert strip_hidden_unicode(src) == "hello 世界 🌍"


def test_strip_tag_block() -> None:
    src = "abc\U000e0061def"  # TAG LATIN SMALL LETTER A
    assert strip_hidden_unicode(src) == "abcdef"


def test_strip_full_tag_block_range() -> None:
    src = "x\U000e0000y\U000e007fz"  # both ends of the tag block
    assert strip_hidden_unicode(src) == "xyz"


def test_strip_idempotent() -> None:
    src = "hel​lo﻿world"
    once = strip_hidden_unicode(src)
    twice = strip_hidden_unicode(once)
    assert once == twice == "helloworld"
