#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import wave
from pathlib import Path

from dotenv import dotenv_values
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "audio-debug/20261017-155248-f8668d8163/gemini_input.wav"
MODEL = "gemini-3.8-live"
START_S = 22.5
END_S = 25.5


def load_env() -> None:
    for key, value in dotenv_values(ROOT / ".env").items():
        if value is not None:
            os.environ[key] = value


def load_segment() -> bytes:
    with wave.open(str(AUDIO), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        wav.setpos(int(START_S * wav.getframerate()))
        return wav.readframes(int((END_S - START_S) * wav.getframerate()))


async def run_case(name: str, input_transcription: dict) -> dict:
    proxy = os.environ.get("GEMINI_PROXY_URL", "").strip()
    http_options = types.HttpOptions(async_client_args={"proxy": proxy}) if proxy else None
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=http_options)
    config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": (
            "Пользователь говорит по-русски. Не переключайся на другие языки, если пользователь явно об этом не попросил."
        ),
        "input_audio_transcription": input_transcription,
        "output_audio_transcription": {},
        "speech_config": {
            "language_code": "ru-RU",
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}},
        },
        "realtime_input_config": {
            "activity_handling": types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            "automatic_activity_detection": {
                "disabled": False,
                "start_of_speech_sensitivity": types.StartSensitivity.START_SENSITIVITY_LOW,
                "end_of_speech_sensitivity": types.EndSensitivity.END_SENSITIVITY_HIGH,
                "prefix_padding_ms": 200,
                "silence_duration_ms": 350,
            },
        },
    }
    segment = load_segment()
    transcript = ""
    response_transcript = ""
    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            for offset in range(0, len(segment), 640):
                chunk = segment[offset:offset + 640]
                if chunk:
                    await session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                    )
            await session.send_realtime_input(audio_stream_end=True)

            async def receive() -> None:
                nonlocal transcript, response_transcript
                async for message in session.receive():
                    content = message.server_content
                    if content is None:
                        continue
                    if content.input_transcription and content.input_transcription.text:
                        transcript += content.input_transcription.text
                    if content.output_transcription and content.output_transcription.text:
                        response_transcript += content.output_transcription.text
                    if content.turn_complete:
                        return

            await asyncio.wait_for(receive(), timeout=15)
        return {
            "name": name,
            "ok": True,
            "input_transcript": transcript.strip(),
            "output_transcript": response_transcript.strip(),
        }
    except Exception as exc:
        return {
            "name": name,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    finally:
        await client.aio.aclose()


async def main() -> None:
    load_env()
    results = [
        await run_case("auto", {}),
        await run_case(
            "ru_hint",
            {
                "language_codes": ["ru"],
                "custom_vocabulary": ["Окей Набу", "Набу"],
            },
        ),
        await run_case(
            "ru_RU_hint",
            {
                "language_codes": ["ru-RU"],
                "custom_vocabulary": ["Окей Набу", "Набу"],
            },
        ),
        await run_case(
            "ru_hint_extended_vocab",
            {
                "language_codes": ["ru"],
                "custom_vocabulary": ["Окей Набу", "Набу", "Okay Nabu", "OK Nabu"],
            },
        ),
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
