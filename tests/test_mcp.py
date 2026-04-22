"""Tests for the MindForge multi-KB MCP server.

The multi-KB server reads MINDFORGE_ROOT from the environment. These tests
isolate the filesystem by pointing MINDFORGE_ROOT at a tmp dir.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def mcp_root(monkeypatch, tmp_path):
    """Isolate MCP server state to a temp dir and reload the server module."""
    monkeypatch.setenv("MINDFORGE_ROOT", str(tmp_path))
    # Reload so module-level MINDFORGE_ROOT / KBS_DIR re-resolve.
    import mindforge.mcp.server as server
    importlib.reload(server)
    return server


def test_ensure_structure_creates_dirs_and_registry(mcp_root, tmp_path):
    mcp_root.ensure_structure()
    assert (tmp_path / "kbs").is_dir()
    assert (tmp_path / "trash").is_dir()
    assert (tmp_path / "registry.json").is_file()
    registry = json.loads((tmp_path / "registry.json").read_text())
    assert registry["version"] == "1.0"
    assert registry["kbs"] == {}


def test_kebab_case_slugifies_names(mcp_root):
    assert mcp_root.kebab_case("Neural Networks") == "neural-networks"
    assert mcp_root.kebab_case("MLOps Best Practices") == "mlops-best-practices"


def test_manager_can_create_list_and_select_kb(mcp_root, tmp_path):
    manager = mcp_root.MultiKBManager()
    result = manager.create_kb("Quantum Computing")
    assert result["success"] is True
    assert result["id"] == "quantum-computing"
    assert (tmp_path / "kbs" / "quantum-computing").is_dir()

    listed = manager.list_kbs()
    assert any(item["id"] == "quantum-computing" for item in listed)

    select_result = manager.select_kb("quantum-computing")
    assert select_result["success"] is True
    current = manager.get_current()
    assert current is not None
    assert current["id"] == "quantum-computing"


def test_create_server_returns_runnable(mcp_root):
    server = mcp_root.create_server()
    assert hasattr(server, "run")
    assert callable(server.run)


@pytest.mark.asyncio
async def test_list_tools_returns_all_tools(mcp_root):
    tools = await mcp_root.list_tools()
    tool_names = {t.name for t in tools}
    # Spot-check a few expected tools from each group.
    assert "kb_list" in tool_names
    assert "search" in tool_names
    assert "get_concept" in tool_names
