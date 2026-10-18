#!/usr/bin/env python3
"""Small Gemini Live model A/B probe using ADA's real provider and proxy."""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.realtime_provider import GeminiLiveProvider
MODELS = ["gemini-3.1-flash-live-preview", "gemini-3.8-live"]
TRIALS = 2
PROMPT = "Ответь по-русски одним коротким предложением: проверка голосового соединения прошла."


def load_env() -> None:
    for key, value in dotenv_values(ROOT / ".env").items():
        if value is not None:
            os.environ[key] = value


async def one_trial(model: str, trial: int) -> dict[str, object]:
    os.environ["GEMINI_LIVE_MODEL"] = model
    provider = GeminiLiveProvider(
        instructions=(
            "Ты тестовая русскоязычная голосовая система. Отвечай только по-русски, "
            "кратко и без инструментов, если они не обязательны."
        )
    )
    provider.session_id = f"ab-{model}-{trial}"
    connected_at = 0.0
    sent_at = 0.0
    first_audio_at = 0.0
    completed_at = 0.0
    audio_bytes = 0
    transcript = ""
    started = time.monotonic()
    try:
        await asyncio.wait_for(provider.connect(), timeout=15)
        connected_at = time.monotonic()
        await provider.send_text_turn(PROMPT)
        sent_at = time.monotonic()

        async def consume() -> None:
            nonlocal first_audio_at, completed_at, audio_bytes, transcript
            async for event in provider.events():
                if event.type == "audio":
                    if not first_audio_at:
                        first_audio_at = time.monotonic()
                    audio_bytes += len(event.data.get("pcm16", b""))
                elif event.type == "assistant_transcript_delta":
                    transcript += str(event.data.get("text", ""))
                elif event.type == "response_completed":
                    completed_at = time.monotonic()
                    return

        await asyncio.wait_for(consume(), timeout=20)
        return {
            "model": model,
            "trial": trial,
            "ok": bool(first_audio_at and completed_at and audio_bytes),
            "connect_ms": round((connected_at - started) * 1000, 1),
            "first_audio_ms": round((first_audio_at - sent_at) * 1000, 1) if first_audio_at else None,
            "response_ms": round((completed_at - sent_at) * 1000, 1) if completed_at else None,
            "audio_bytes": audio_bytes,
            "transcript": transcript.strip()[:200],
        }
    except Exception as exc:
        return {
            "model": model,
            "trial": trial,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    finally:
        try:
            await provider.close()
        except Exception:
            pass


async def main() -> None:
    load_env()
    results: list[dict[str, object]] = []
    for model in MODELS:
        for trial in range(1, TRIALS + 1):
            results.append(await one_trial(model, trial))
            await asyncio.sleep(0.5)

    summary: dict[str, object] = {}
    for model in MODELS:
        rows = [row for row in results if row["model"] == model]
        good = [row for row in rows if row.get("ok")]
        summary[model] = {
            "successes": len(good),
            "trials": len(rows),
            "median_connect_ms": round(statistics.median(float(row["connect_ms"]) for row in good), 1) if good else None,
            "median_first_audio_ms": round(statistics.median(float(row["first_audio_ms"]) for row in good), 1) if good else None,
            "median_response_ms": round(statistics.median(float(row["response_ms"]) for row in good), 1) if good else None,
        }
    print(json.dumps({"results": results, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
