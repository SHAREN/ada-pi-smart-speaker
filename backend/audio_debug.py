"""Temporary rich audio diagnostics for ADA Pi voice sessions.

This module is intentionally enabled during the current tuning phase. It stores
per-session microphone/provider/render tracks for 24 hours so full-duplex,
barge-in, wake buffering, and echo-cancellation behavior can be inspected.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INPUT_RATE = 16_000
OUTPUT_RATE = 24_000
RENDER_RATE = 48_000
RETENTION_HOURS = float(os.environ.get("ADA_AUDIO_DEBUG_RETENTION_HOURS", "24") or 24)
RETENTION_SECONDS = RETENTION_HOURS * 3600.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class _WavWriter:
    def __init__(self, path: Path, rate: int, channels: int = 1) -> None:
        self.path = path
        self.rate = rate
        self.channels = channels
        self.bytes_written = 0
        self._writer = wave.open(str(path), "wb")
        self._writer.setnchannels(channels)
        self._writer.setsampwidth(2)
        self._writer.setframerate(rate)

    def write(self, pcm16: bytes) -> None:
        if not pcm16:
            return
        self._writer.writeframesraw(pcm16)
        self.bytes_written += len(pcm16)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def info(self) -> dict[str, Any]:
        samples = self.bytes_written // max(1, 2 * self.channels)
        return {
            "filename": self.path.name,
            "bytes": self.bytes_written,
            "sample_rate": self.rate,
            "channels": self.channels,
            "duration_seconds": round(samples / self.rate, 3),
        }


class AudioDebugSession:
    def __init__(self, root: Path, browser_session_id: str) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.id = f"{stamp}-{browser_session_id}"
        self.root = root
        self.dir = root / "audio-debug" / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.origin_monotonic = time.monotonic()
        self.started_at = _utc_now()
        self.timeline: list[dict[str, Any]] = []
        self.raw_mic = _WavWriter(self.dir / "mic_raw.wav", INPUT_RATE)
        self.gemini_input = _WavWriter(self.dir / "gemini_input.wav", INPUT_RATE)
        self.assistant_raw = _WavWriter(self.dir / "assistant_raw.wav", OUTPUT_RATE)
        self.assistant_sent = _WavWriter(self.dir / "assistant_sent.wav", OUTPUT_RATE)
        self.render_path = self.dir / "speaker_render.wav"
        self.mix_path = self.dir / "mix_stereo.wav"
        self.metadata_path = self.dir / "timeline.json"
        self.render_process: asyncio.subprocess.Process | None = None
        self.closed = False

    async def start(self) -> None:
        cleanup_audio_debug(self.root)
        try:
            env = os.environ.copy()
            env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
            self.render_process = await asyncio.create_subprocess_exec(
                "pw-record",
                "--target=ada_aec_sink.monitor",
                "--rate=48000",
                "--channels=2",
                "--format=s16",
                str(self.render_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=env,
            )
        except Exception as exc:
            self.record_event("speaker_render_start_failed", {"error": str(exc)})
        self.record_event("capture_started", {})

    def record_event(self, event: str, data: dict[str, Any] | None = None) -> None:
        self.timeline.append({
            "event": event,
            "at": _utc_now(),
            "offset_ms": round((time.monotonic() - self.origin_monotonic) * 1000.0, 1),
            "data": data or {},
        })
        if len(self.timeline) <= 4 or event in {
            "wake", "stop", "idle", "semantic_end", "response_started",
            "response_completed", "response_interrupted", "session_closed",
            "fresh_session", "wake_cue_started", "wake_cue_finished",
            "end_cue_started", "end_cue_finished",
        }:
            self.write_metadata()

    def write_raw_mic(self, pcm16: bytes) -> None:
        self.raw_mic.write(pcm16)

    def write_gemini_input(self, pcm16: bytes) -> None:
        self.gemini_input.write(pcm16)

    def write_assistant_raw(self, pcm16: bytes) -> None:
        self.assistant_raw.write(pcm16)

    def write_assistant_sent(self, pcm16: bytes) -> None:
        self.assistant_sent.write(pcm16)

    def write_metadata(self) -> None:
        payload = {
            "id": self.id,
            "started_at": self.started_at,
            "retention_hours": RETENTION_HOURS,
            "timeline": self.timeline,
        }
        tmp = self.metadata_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.metadata_path)

    async def close(self, reason: str) -> None:
        if self.closed:
            return
        self.closed = True
        self.record_event("session_closed", {"reason": reason})
        if self.render_process is not None and self.render_process.returncode is None:
            self.render_process.terminate()
            try:
                await asyncio.wait_for(self.render_process.wait(), timeout=2)
            except asyncio.TimeoutError:
                self.render_process.kill()
                await self.render_process.wait()
        self.raw_mic.close()
        self.gemini_input.close()
        self.assistant_raw.close()
        self.assistant_sent.close()
        await asyncio.to_thread(self._build_mix)
        payload = {
            "id": self.id,
            "started_at": self.started_at,
            "finished_at": _utc_now(),
            "retention_hours": RETENTION_HOURS,
            "end_reason": reason,
            "files": {
                "mic_raw": self.raw_mic.info(),
                "gemini_input": self.gemini_input.info(),
                "assistant_raw": self.assistant_raw.info(),
                "assistant_sent": self.assistant_sent.info(),
                "speaker_render": _file_info(self.render_path),
                "mix_stereo": _file_info(self.mix_path),
            },
            "timeline": self.timeline,
        }
        tmp = self.metadata_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.metadata_path)
        cleanup_audio_debug(self.root)

    def _build_mix(self) -> None:
        if not self.raw_mic.path.exists() or not self.render_path.exists():
            return
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(self.raw_mic.path),
            "-i", str(self.render_path),
            "-filter_complex",
            "[0:a]aresample=48000,pan=mono|c0=c0[m];"
            "[1:a]pan=mono|c0=0.5*c0+0.5*c1[s];"
            "[m][s]amerge=inputs=2[a]",
            "-map", "[a]", "-c:a", "pcm_s16le", str(self.mix_path),
        ]
        subprocess.run(command, capture_output=True, check=False, timeout=30)


def _file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"filename": path.name, "exists": False, "bytes": 0}
    return {"filename": path.name, "exists": True, "bytes": path.stat().st_size}


def cleanup_audio_debug(root: Path) -> None:
    debug_root = root / "audio-debug"
    if not debug_root.exists():
        return
    cutoff = time.time() - RETENTION_SECONDS
    for item in debug_root.iterdir():
        if not item.is_dir():
            continue
        try:
            if item.stat().st_mtime < cutoff:
                shutil.rmtree(item)
        except OSError:
            continue
