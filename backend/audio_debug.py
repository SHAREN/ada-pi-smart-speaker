"""Temporary rich audio diagnostics for ADA Pi voice sessions.

This module is intentionally enabled during the current tuning phase. It stores
per-session microphone/provider/render tracks for 24 hours so full-duplex,
barge-in, wake buffering, and echo-cancellation behavior can be inspected.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import queue
import shutil
import subprocess
import threading
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


def _find_physical_p610_source() -> str | None:
    override = os.environ.get("ADA_AUDIO_DEBUG_PHYSICAL_SOURCE", "").strip()
    if override:
        return override
    env = os.environ.copy()
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
            env=env,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[1]
        if (
            name.startswith("alsa_input.usb-Plantronics_Plantronics_P610_")
            and name.endswith(".analog-stereo")
        ):
            return name
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    except OSError:
        pass
    return values


def _read_pressure(kind: str) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        text = Path(f"/proc/pressure/{kind}").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return result
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        result[parts[0]] = " ".join(parts[1:])
    return result


def _network_counters() -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    try:
        lines = Path("/proc/net/dev").read_text(encoding="utf-8", errors="replace").splitlines()[2:]
    except OSError:
        return result
    for line in lines:
        if ":" not in line:
            continue
        name, payload = line.split(":", 1)
        fields = payload.split()
        if len(fields) < 16:
            continue
        result[name.strip()] = {
            "rx_bytes": int(fields[0]), "rx_packets": int(fields[1]),
            "rx_errors": int(fields[2]), "rx_dropped": int(fields[3]),
            "tx_bytes": int(fields[8]), "tx_packets": int(fields[9]),
            "tx_errors": int(fields[10]), "tx_dropped": int(fields[11]),
        }
    return result


def _source_fingerprints(root: Path) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for relative in ("backend/main.py", "backend/realtime_provider.py", "backend/audio_debug.py", "frontend/app.js"):
        path = root / relative
        try:
            fingerprints[relative] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        except OSError:
            continue
    return fingerprints


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
        self._timeline_lock = threading.Lock()
        self.wake_preroll = _WavWriter(self.dir / "wake_preroll.wav", INPUT_RATE)
        self.raw_mic = _WavWriter(self.dir / "mic_raw.wav", INPUT_RATE)
        self.gemini_input = _WavWriter(self.dir / "gemini_input.wav", INPUT_RATE)
        self.assistant_raw = _WavWriter(self.dir / "assistant_raw.wav", OUTPUT_RATE)
        self.assistant_sent = _WavWriter(self.dir / "assistant_sent.wav", OUTPUT_RATE)
        self.render_path = self.dir / "speaker_render.wav"
        self.physical_mic_path = self.dir / "physical_p610_mic.wav"
        self.tv_reference_path = self.dir / "tv_reference.wav"
        self.mix_path = self.dir / "mix_stereo.wav"
        self.metadata_path = self.dir / "timeline.json"
        self.events_path = self.dir / "events.jsonl"
        self.render_stderr_path = self.dir / "speaker_render.stderr.log"
        self.physical_mic_stderr_path = self.dir / "physical_p610_mic.stderr.log"
        self.tv_reference_stderr_path = self.dir / "tv_reference.stderr.log"
        self.session_journal_path = self.dir / "session_journal.log"
        self.source_fingerprints = _source_fingerprints(root)
        self._events_file = self.events_path.open("a", encoding="utf-8", buffering=1)
        self._render_stderr_file = None
        self._physical_mic_stderr_file = None
        self._tv_reference_stderr_file = None
        self.render_process: asyncio.subprocess.Process | None = None
        self.physical_mic_process: asyncio.subprocess.Process | None = None
        self.tv_reference_process: asyncio.subprocess.Process | None = None
        # Never perform debug disk I/O on the realtime asyncio loop. A dedicated
        # FIFO worker owns all JSONL and in-process WAV writes; if storage stalls,
        # diagnostics can lag behind without delaying assistant audio delivery.
        self._io_queue: queue.SimpleQueue[tuple[Any, ...]] = queue.SimpleQueue()
        self._io_thread = threading.Thread(
            target=self._io_worker,
            name=f"ada-audio-debug-{browser_session_id}",
            daemon=True,
        )
        self._io_thread.start()
        self.closed = False

    async def start(self) -> None:
        await asyncio.to_thread(cleanup_audio_debug, self.root)
        self.record_system_snapshot("session_start")
        try:
            env = os.environ.copy()
            env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
            self._render_stderr_file = self.render_stderr_path.open("wb")
            self.render_process = await asyncio.create_subprocess_exec(
                "pw-record",
                "--target=ada_aec_sink.monitor",
                "--rate=48000",
                "--channels=2",
                "--format=s16",
                str(self.render_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=self._render_stderr_file,
                env=env,
            )
            self.record_event("speaker_render_capture_started", {"pid": self.render_process.pid})
        except Exception as exc:
            if self._render_stderr_file is not None:
                self._render_stderr_file.close()
                self._render_stderr_file = None
            self.record_event("speaker_render_start_failed", {"error": str(exc)})

        physical_source = await asyncio.to_thread(_find_physical_p610_source)
        if physical_source:
            try:
                env = os.environ.copy()
                env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
                self._physical_mic_stderr_file = self.physical_mic_stderr_path.open("wb")
                self.physical_mic_process = await asyncio.create_subprocess_exec(
                    "pw-record",
                    f"--target={physical_source}",
                    "--rate=48000",
                    "--channels=2",
                    "--format=s16",
                    str(self.physical_mic_path),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=self._physical_mic_stderr_file,
                    env=env,
                )
                self.record_event("physical_mic_capture_started", {"source": physical_source, "pid": self.physical_mic_process.pid})
            except Exception as exc:
                if self._physical_mic_stderr_file is not None:
                    self._physical_mic_stderr_file.close()
                    self._physical_mic_stderr_file = None
                self.record_event("physical_mic_start_failed", {"error": str(exc)})
        else:
            self.record_event("physical_mic_source_not_found", {})

        # Capture the exact TV/HDMI reference independently of whether active TV
        # suppression is enabled. This is diagnostic-only and never gates audio.
        try:
            env = os.environ.copy()
            env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
            tv_reference_source = os.environ.get("ADA_TV_REFERENCE_SOURCE", "@DEFAULT_MONITOR@")
            self._tv_reference_stderr_file = self.tv_reference_stderr_path.open("wb")
            self.tv_reference_process = await asyncio.create_subprocess_exec(
                "pw-record",
                f"--target={tv_reference_source}",
                "--rate=48000",
                "--channels=2",
                "--format=s16",
                str(self.tv_reference_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=self._tv_reference_stderr_file,
                env=env,
            )
            self.record_event(
                "tv_reference_capture_started",
                {"source": tv_reference_source, "pid": self.tv_reference_process.pid},
            )
        except Exception as exc:
            if self._tv_reference_stderr_file is not None:
                self._tv_reference_stderr_file.close()
                self._tv_reference_stderr_file = None
            self.record_event("tv_reference_capture_start_failed", {"error": str(exc)})
        self.record_event("capture_started", {})

    def _make_event_item(
        self,
        event: str,
        data: dict[str, Any] | None = None,
        *,
        at: str | None = None,
        offset_ms: float | None = None,
    ) -> dict[str, Any]:
        return {
            "event": event,
            "at": at or _utc_now(),
            "offset_ms": (
                round((time.monotonic() - self.origin_monotonic) * 1000.0, 1)
                if offset_ms is None else offset_ms
            ),
            "data": data or {},
        }

    def _write_event_item(self, item: dict[str, Any]) -> None:
        with self._timeline_lock:
            self.timeline.append(item)
        self._events_file.write(
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        # Flush from the background writer, never from the realtime asyncio loop.
        self._events_file.flush()

    def _collect_system_snapshot(self, reason: str, extra: dict[str, Any]) -> dict[str, Any]:
        memory = _read_key_values(Path("/proc/meminfo"))
        process = _read_key_values(Path("/proc/self/status"))
        try:
            disk = shutil.disk_usage(self.root)
            disk_info = {"total": disk.total, "used": disk.used, "free": disk.free}
        except OSError:
            disk_info = {}
        try:
            load = os.getloadavg()
        except OSError:
            load = (0.0, 0.0, 0.0)
        return {
            "reason": reason,
            "loadavg": [round(value, 3) for value in load],
            "memory": {key: memory.get(key) for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")},
            "process": {key: process.get(key) for key in ("VmRSS", "VmSize", "Threads")},
            "pressure": {kind: _read_pressure(kind) for kind in ("cpu", "memory", "io")},
            "network": _network_counters(),
            "disk": disk_info,
            "extra": extra,
        }

    def _io_worker(self) -> None:
        while True:
            work = self._io_queue.get()
            kind = work[0]
            if kind == "stop":
                return
            try:
                if kind == "event":
                    self._write_event_item(work[1])
                elif kind == "wav":
                    writer = getattr(self, work[1])
                    writer.write(work[2])
                elif kind == "system_snapshot":
                    _, at, offset_ms, reason, extra = work
                    item = self._make_event_item(
                        "system_snapshot",
                        self._collect_system_snapshot(reason, extra),
                        at=at,
                        offset_ms=offset_ms,
                    )
                    self._write_event_item(item)
                elif kind == "barrier":
                    work[1].set()
            except Exception:
                # Diagnostics must never take down or stall the voice path.
                if kind == "barrier":
                    work[1].set()

    def record_event(self, event: str, data: dict[str, Any] | None = None) -> None:
        self._io_queue.put(("event", self._make_event_item(event, data)))

    def record_system_snapshot(self, reason: str, extra: dict[str, Any] | None = None) -> None:
        self._io_queue.put((
            "system_snapshot",
            _utc_now(),
            round((time.monotonic() - self.origin_monotonic) * 1000.0, 1),
            reason,
            dict(extra or {}),
        ))

    def write_wake_preroll(self, pcm16: bytes) -> None:
        if pcm16:
            self._io_queue.put(("wav", "wake_preroll", pcm16))

    def write_raw_mic(self, pcm16: bytes) -> None:
        if pcm16:
            self._io_queue.put(("wav", "raw_mic", pcm16))

    def write_gemini_input(self, pcm16: bytes) -> None:
        if pcm16:
            self._io_queue.put(("wav", "gemini_input", pcm16))

    def write_assistant_raw(self, pcm16: bytes) -> None:
        if pcm16:
            self._io_queue.put(("wav", "assistant_raw", pcm16))

    def write_assistant_sent(self, pcm16: bytes) -> None:
        if pcm16:
            self._io_queue.put(("wav", "assistant_sent", pcm16))

    async def flush_pending(self) -> None:
        barrier = threading.Event()
        self._io_queue.put(("barrier", barrier))
        await asyncio.to_thread(barrier.wait)

    def _write_metadata_payload(self, payload: dict[str, Any]) -> None:
        tmp = self.metadata_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.metadata_path)

    def write_metadata(self) -> None:
        with self._timeline_lock:
            timeline = list(self.timeline)
        self._write_metadata_payload({
            "schema_version": 3,
            "id": self.id,
            "started_at": self.started_at,
            "retention_hours": RETENTION_HOURS,
            "source_fingerprints": self.source_fingerprints,
            "events_file": self.events_path.name,
            "timeline": timeline,
        })

    async def close(self, reason: str) -> None:
        if self.closed:
            return
        self.closed = True
        self.record_system_snapshot("session_close", {"end_reason": reason})

        async def stop_capture(process: asyncio.subprocess.Process | None) -> None:
            if process is None or process.returncode is not None:
                return
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

        await stop_capture(self.render_process)
        await stop_capture(self.physical_mic_process)
        await stop_capture(self.tv_reference_process)
        self.record_event("capture_processes_stopped", {
            "speaker_render_returncode": self.render_process.returncode if self.render_process is not None else None,
            "physical_mic_returncode": self.physical_mic_process.returncode if self.physical_mic_process is not None else None,
            "tv_reference_returncode": self.tv_reference_process.returncode if self.tv_reference_process is not None else None,
        })
        for attr in (
            "_render_stderr_file",
            "_physical_mic_stderr_file",
            "_tv_reference_stderr_file",
        ):
            handle = getattr(self, attr)
            if handle is not None:
                handle.close()
                setattr(self, attr, None)

        # Queue the final event, wait until every preceding event/WAV write is on
        # disk, then stop the worker. This draining is allowed during teardown;
        # it can no longer create audible pauses in an active response.
        self.record_event("session_closed", {"reason": reason})
        await self.flush_pending()
        self._io_queue.put(("stop",))
        await asyncio.to_thread(self._io_thread.join)

        self.wake_preroll.close()
        self.raw_mic.close()
        self.gemini_input.close()
        self.assistant_raw.close()
        self.assistant_sent.close()
        await asyncio.to_thread(self._build_mix)
        await asyncio.to_thread(self._capture_session_journal)
        with self._timeline_lock:
            timeline = sorted(self.timeline, key=lambda item: float(item.get("offset_ms", 0.0)))
        payload = {
            "schema_version": 3,
            "id": self.id,
            "started_at": self.started_at,
            "finished_at": _utc_now(),
            "retention_hours": RETENTION_HOURS,
            "end_reason": reason,
            "source_fingerprints": self.source_fingerprints,
            "events_file": self.events_path.name,
            "files": {
                "wake_preroll": self.wake_preroll.info(),
                "mic_raw": self.raw_mic.info(),
                "gemini_input": self.gemini_input.info(),
                "assistant_raw": self.assistant_raw.info(),
                "assistant_sent": self.assistant_sent.info(),
                "speaker_render": _file_info(self.render_path),
                "speaker_render_stderr": _file_info(self.render_stderr_path),
                "physical_p610_mic": _file_info(self.physical_mic_path),
                "physical_p610_mic_stderr": _file_info(self.physical_mic_stderr_path),
                "tv_reference": _file_info(self.tv_reference_path),
                "tv_reference_stderr": _file_info(self.tv_reference_stderr_path),
                "session_journal": _file_info(self.session_journal_path),
                "mix_stereo": _file_info(self.mix_path),
                "events": _file_info(self.events_path),
            },
            "timeline": timeline,
        }
        await asyncio.to_thread(self._write_metadata_payload, payload)
        self._events_file.close()
        await asyncio.to_thread(cleanup_audio_debug, self.root)

    def _capture_session_journal(self) -> None:
        sections = [
            ("ada-pi.service", ["journalctl", "--user", "-u", "ada-pi.service"]),
            ("pipecat-assist-watchdog", ["journalctl", "-u", "pipecat-assist-watchdog.service"]),
            ("pipewire", ["journalctl", "_COMM=pipewire"]),
            ("pipewire-pulse", ["journalctl", "_COMM=pipewire-pulse"]),
            ("wireplumber", ["journalctl", "_COMM=wireplumber"]),
        ]
        chunks: list[str] = []
        since_arg = datetime.fromisoformat(
            self.started_at.replace("Z", "+00:00")
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
        for label, command in sections:
            try:
                result = subprocess.run(
                    [*command, "--since", since_arg, "--no-pager", "-o", "short-iso", "-n", "1000"],
                    capture_output=True, text=True, check=False, timeout=5,
                )
                chunks.append(f"===== {label} =====\n{result.stdout}")
                if result.stderr:
                    chunks.append(f"----- {label} stderr -----\n{result.stderr}")
            except Exception as exc:
                chunks.append(f"===== {label} capture failed =====\n{exc}\n")
        self.session_journal_path.write_text("\n".join(chunks), encoding="utf-8")

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
