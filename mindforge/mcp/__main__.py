"""Entry point for ``python -m mindforge.mcp``.

Runs the MindForge MCP server with multi-KB support. Reads MINDFORGE_ROOT
from the environment (defaults to ~/.mindforge). Accepts the same
``--llm-*`` flags as ``mindforge mcp`` so MCP hosts that prefer launching
via ``python -m`` get equivalent control.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from mindforge.config_file import load_config, merge_with_overrides
from mindforge.mcp.server import configure_runtime, main


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m mindforge.mcp")
    parser.add_argument("--llm-provider", choices=["ollama", "openai"], default=None)
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-summarize-model", default="")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--llm-timeout", type=int, default=None)
    return parser


def run() -> int:
    args = _build_parser().parse_args()
    cfg = load_config()
    cfg = merge_with_overrides(
        cfg,
        llm_provider=args.llm_provider,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_summarize_model=args.llm_summarize_model,
        llm_api_key=args.llm_api_key,
        llm_timeout=args.llm_timeout,
    )
    configure_runtime(cfg)
    asyncio.run(main())
    return 0


if __name__ == "__main__":
    sys.exit(run())
