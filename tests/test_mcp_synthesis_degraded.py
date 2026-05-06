"""Synthesis tools degrade gracefully when the LLM is unreachable.

PR-C1 lays the framework: the runtime registry, the health check, and the
canonical ``synthesis_backend_unavailable`` reply. Tool-call routing
through this path is exercised by PR-C2's runtime tests.
"""

from __future__ import annotations

import json

from mindforge.config_file import ConfigFile, LLMConfigSection
from mindforge.mcp import server as mcp_server


def test_check_llm_health_returns_false_for_unreachable(monkeypatch) -> None:
    cfg = ConfigFile(llm=LLMConfigSection(base_url="http://nonexistent.example:12345"))

    class _DeadClient:
        def __init__(self, _cfg) -> None:
            self.config = _cfg

        @property
        def available(self) -> bool:
            return False

    monkeypatch.setattr(mcp_server, "LLMClient", _DeadClient)
    assert mcp_server._check_llm_health(cfg) is False


def test_check_llm_health_returns_true_when_reachable(monkeypatch) -> None:
    cfg = ConfigFile(llm=LLMConfigSection(base_url="http://localhost:11434"))

    class _LiveClient:
        def __init__(self, _cfg) -> None:
            self.config = _cfg

        @property
        def available(self) -> bool:
            return True

    monkeypatch.setattr(mcp_server, "LLMClient", _LiveClient)
    assert mcp_server._check_llm_health(cfg) is True


def test_configure_runtime_records_synthesis_enabled(monkeypatch) -> None:
    cfg = ConfigFile(llm=LLMConfigSection(base_url="http://nonexistent.example:12345"))
    monkeypatch.setattr(mcp_server, "_check_llm_health", lambda _cfg: False)
    enabled = mcp_server.configure_runtime(cfg)
    assert enabled is False
    assert mcp_server._runtime["synthesis_enabled"] is False
    assert mcp_server._runtime["config"] is cfg


def test_configure_runtime_warns_when_unreachable(monkeypatch, capsys) -> None:
    cfg = ConfigFile(llm=LLMConfigSection(base_url="http://nonexistent.example:12345"))
    monkeypatch.setattr(mcp_server, "_check_llm_health", lambda _cfg: False)
    mcp_server.configure_runtime(cfg)
    err = capsys.readouterr().err
    assert "unreachable" in err
    assert "Synthesis tools" in err


def test_synthesis_unavailable_response_shape() -> None:
    cfg = ConfigFile(llm=LLMConfigSection(base_url="http://localhost:11434"))
    out = mcp_server.synthesis_unavailable_response(cfg)
    assert len(out) == 1
    payload = json.loads(out[0].text)
    assert payload["error"] == "synthesis_backend_unavailable"
    assert "http://localhost:11434" in payload["message"]
    assert "get_concept" in payload["message"]
