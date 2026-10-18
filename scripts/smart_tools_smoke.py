#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import dotenv_values

from backend.realtime_provider import GeminiLiveProvider

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    for key, value in dotenv_values(ROOT / ".env").items():
        if value is not None:
            os.environ[key] = value


async def collect_reply(provider: GeminiLiveProvider, text: str, timeout: float = 30.0) -> str:
    await provider.send_text_turn(text)
    chunks: list[str] = []

    async def receive_one() -> None:
        async for event in provider.events():
            if event.type == "assistant_transcript_delta":
                chunks.append(str(event.data.get("text", "")))
            elif event.type == "response_completed":
                return

    await asyncio.wait_for(receive_one(), timeout=timeout)
    return "".join(chunks).strip()


async def main() -> None:
    requested_model = os.environ.get("SMART_SMOKE_MODEL", "").strip()
    requested_web = os.environ.get("SMART_SMOKE_WEB_SEARCH", "").strip()
    requested_ha = os.environ.get("SMART_SMOKE_HA", "").strip()
    load_env()
    if requested_model:
        os.environ["GEMINI_LIVE_MODEL"] = requested_model
    if requested_web:
        os.environ["GEMINI_GOOGLE_SEARCH"] = requested_web
    if requested_ha.lower() in {"0", "false", "off", "no"}:
        os.environ["HA_MCP_URL"] = ""
    provider = GeminiLiveProvider(
        instructions="Ты тестовая русскоязычная умная колонка. Выполняй просьбы точно и кратко."
    )
    provider.session_id = "smart-smoke"
    try:
        await asyncio.wait_for(provider.connect(), timeout=15)
        print(f"connected=true web_search={provider.enable_google_search} ha_tools={provider._ha_tool_count}")
        ha_reply = await collect_reply(
            provider,
            "Используй Home Assistant ha_search и только read-only. Найди до трёх сущностей, связанных с кухней, и коротко назови одну из них. Ничего не меняй.",
        )
        print("ha_reply_present=" + str(bool(ha_reply)))
        print("ha_reply_preview=" + ha_reply[:300].replace("\n", " "))
        print("ha_tool_calls=" + str(provider._ha_tool_calls))
        web_reply = await collect_reply(
            provider,
            "Обязательно используй инструмент web_search. Найди в интернете свежую публичную информацию о погоде в Москве и ответь по-русски одним коротким предложением. Не отвечай только из памяти модели.",
        )
        print("web_reply_present=" + str(bool(web_reply)))
        print("web_reply_preview=" + web_reply[:300].replace("\n", " "))
        print("web_tool_calls=" + str(provider._web_tool_calls))
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
