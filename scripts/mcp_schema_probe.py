#!/usr/bin/env python3
from __future__ import annotations
import asyncio, json, os
from pathlib import Path
from dotenv import dotenv_values
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    'ha_get_state','ha_search','ha_get_entity','ha_get_device','ha_get_history',
    'ha_list_floors_areas','ha_call_service','ha_bulk_control','ha_get_todo',
    'ha_set_todo_item','ha_config_get_calendar_events','ha_config_set_calendar_event',
    'ha_config_remove_calendar_event','ha_get_overview'
}

def load_env():
    for k,v in dotenv_values(ROOT/'.env').items():
        if v is not None: os.environ[k]=v

async def main():
    load_env(); url=os.environ['HA_MCP_URL']
    async with streamablehttp_client(url) as (r,w,_):
        async with ClientSession(r,w) as s:
            await s.initialize(); result=await s.list_tools()
            for tool in result.tools:
                if tool.name in NAMES:
                    print(json.dumps({'name':tool.name,'description':tool.description,'inputSchema':tool.inputSchema},ensure_ascii=False))

if __name__=='__main__': asyncio.run(main())
