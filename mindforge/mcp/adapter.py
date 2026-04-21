"""Client adapter seam for the MCP server.

Purpose: future MCP adaptations (client-specific quirks like tool-description
length limits, response shape tweaks, schema compatibility) plug in here
without touching the core server. This session ships ONE DefaultAdapter —
it is a behavior-preserving pass-through. The seam exists so future sessions
have somewhere to hang per-client logic.

Selection order:
    1. Explicit argument to get_adapter(name)
    2. Env var MINDFORGE_MCP_ADAPTER
    3. "default"
"""

from __future__ import annotations

import os
from typing import Any


class ClientAdapter:
    """Base class for MCP client adapters. Override to customize per client."""

    name: str = "base"

    def format_tool_description(self, description: str) -> str:
        """Return a description string safe for the target client."""
        return description

    def format_tool_response(self, payload: Any) -> Any:
        """Return a response payload safe for the target client."""
        return payload


class DefaultAdapter(ClientAdapter):
    """Pass-through adapter. Matches current behavior."""

    name = "default"


_REGISTRY: dict[str, type[ClientAdapter]] = {
    "default": DefaultAdapter,
}


def register_adapter(name: str, cls: type[ClientAdapter]) -> None:
    """Register a ClientAdapter subclass under a short name."""
    _REGISTRY[name] = cls


def get_adapter(name: str | None = None) -> ClientAdapter:
    """Resolve an adapter by name/env var, falling back to DefaultAdapter."""
    chosen = name or os.environ.get("MINDFORGE_MCP_ADAPTER") or "default"
    cls = _REGISTRY.get(chosen, DefaultAdapter)
    return cls()
