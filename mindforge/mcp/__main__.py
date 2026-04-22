"""Entry point for `python -m mindforge.mcp.server`.

Runs the MindForge MCP server with multi-KB support. Reads MINDFORGE_ROOT
from the environment (defaults to ~/.mindforge).
"""

from __future__ import annotations

import asyncio
import sys

from mindforge.mcp.server import main


def run() -> int:
    asyncio.run(main())
    return 0


if __name__ == "__main__":
    sys.exit(run())
