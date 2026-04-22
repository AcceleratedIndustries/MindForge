# Integrating MindForge with OpenAI Agents SDK

The OpenAI Agents SDK is a Python library for building custom agents. It ships an `MCPServerStdio` client for talking to MCP servers.

## Prerequisites

- Python 3.10+
- `pip install openai-agents` (or whatever package name the current Agents SDK uses)
- `pip install -e .` from the MindForge checkout

## Configuration

No config file — it's programmatic. Example:

```python
import asyncio
import os

from agents import Agent, Runner
from agents.mcp import MCPServerStdio


async def main():
    mindforge = MCPServerStdio(
        params={
            "command": "python",
            "args": ["-m", "mindforge.mcp.server"],
            "env": {"MINDFORGE_ROOT": os.path.expanduser("~/.mindforge")},
        }
    )
    async with mindforge:
        agent = Agent(
            name="KB Researcher",
            instructions="Use the mindforge tools to answer questions about my knowledge bases.",
            mcp_servers=[mindforge],
        )
        result = await Runner.run(agent, "What KBs do I have?")
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
```

If the Agents SDK API differs in your version (class name, init signature), consult the SDK's MCP integration docs — the MindForge command/args/env triple does not change.

## Verification

Running the snippet should print a list of KBs from your `~/.mindforge/kbs/` directory (or an empty list if you have none). If MindForge is not running, you'll see a stderr stack trace from the stdio client.

## Known limitations

- The SDK's MCP client caches tool lists; if you add tools to MindForge, recreate the `MCPServerStdio` instance.
- Env vars in the `env` dict must be expanded by the caller; `${HOME}` is NOT expanded automatically.
