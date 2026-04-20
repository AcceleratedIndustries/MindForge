"""Entry point for python -m mindforge.mcp.server

Runs the MindForge MCP server.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    """Start the MCP server."""
    from mindforge.mcp.server import create_server
    from mindforge.config import MindForgeConfig
    
    # Check for MINDFORGE_OUTPUT env var first
    output_dir = Path(os.environ.get("MINDFORGE_OUTPUT", "/home/will/knowledge-base"))
    
    config = MindForgeConfig(output_dir=output_dir)
    
    server = create_server(config)
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
