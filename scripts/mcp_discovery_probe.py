#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import dotenv_values
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    for key, value in dotenv_values(ROOT / ".env").items():
        if value is not None:
            os.environ[key] = value


async def main() -> None:
    load_env()
    url = os.environ.get("HA_MCP_URL", "").strip()
    if not url:
        raise SystemExit("HA_MCP_URL missing")
    async with streamablehttp_client(url) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            tools = await session.list_tools()
            names = sorted(tool.name for tool in tools.tools)
            print(f"initialized=true server={init.serverInfo.name} version={init.serverInfo.version}")
            print(f"tool_count={len(names)}")
            print("tool_names=" + ",".join(names))


if __name__ == "__main__":
    asyncio.run(main())
