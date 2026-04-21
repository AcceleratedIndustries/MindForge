"""Tests for the MCP client adapter seam."""

from __future__ import annotations

from mindforge.mcp.adapter import (
    ClientAdapter,
    DefaultAdapter,
    get_adapter,
    register_adapter,
)


def test_default_adapter_is_registered():
    adapter = get_adapter("default")
    assert isinstance(adapter, DefaultAdapter)


def test_unknown_adapter_name_falls_back_to_default():
    adapter = get_adapter("does-not-exist")
    assert isinstance(adapter, DefaultAdapter)


def test_adapter_respects_env(monkeypatch):
    monkeypatch.setenv("MINDFORGE_MCP_ADAPTER", "default")
    adapter = get_adapter()
    assert isinstance(adapter, DefaultAdapter)


def test_default_adapter_passes_description_through():
    adapter = DefaultAdapter()
    desc = "Long " * 100
    assert adapter.format_tool_description(desc) == desc


def test_default_adapter_passes_response_through():
    adapter = DefaultAdapter()
    payload = {"items": [1, 2, 3]}
    assert adapter.format_tool_response(payload) is payload


def test_adapter_is_extensible(monkeypatch):
    class TruncatingAdapter(ClientAdapter):
        name = "truncating"

        def format_tool_description(self, description: str) -> str:
            return description[:50]

    register_adapter("truncating", TruncatingAdapter)
    monkeypatch.setenv("MINDFORGE_MCP_ADAPTER", "truncating")
    adapter = get_adapter()
    assert adapter.format_tool_description("x" * 500) == "x" * 50
