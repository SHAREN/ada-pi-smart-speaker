"""Render-aware guard for residual ADA/P610 self echo.

The hardware/PipeWire AEC remains the primary echo canceller.  This guard is a
second, conservative classifier used only while ADA is speaking.  It compares
the final post-AEC microphone stream with the exact dedicated ADA render
(`ada_aec_sink.monitor`).  Chunks that still strongly resemble ADA's own render
can be replaced with silence before Gemini's automatic activity detector sees
them, while unrelated near-end speech opens a short pass-through hold.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("voice.self_echo")

_SAMPLE_RATE = 16_000
_FEATURE_FRAME = 512
_FEATURE_WINDOW = np.hanning(_FEATURE_FRAME)
_BAND_EDGES = np.asarray([8, 12, 18, 25, 34, 45, 58, 73, 90, 110, 135, 170, 225])


def _pcm16_float(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16[: len(pcm16) & ~1], dtype="<i2").astype(np.float64) / 32768.0


def _rms(samples: np.ndarray) -> float:
    if not len(samples):
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)))


def _band_feature(samples: np.ndarray) -> np.ndarray:
    frame = samples[-_FEATURE_FRAME:] * _FEATURE_WINDOW
    power = np.abs(np.fft.rfft(frame)) ** 2
    return np.asarray(
        [math.log(float(np.mean(power[a:b])) + 1e-10) for a, b in zip(_BAND_EDGES[:-1], _BAND_EDGES[1:])],
        dtype=np.float64,
    )


def _feature_score(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] < 4:
        return 0.0
    aa = a - np.mean(a, axis=0, keepdims=True)
    bb = b - np.mean(b, axis=0, keepdims=True)
    aa /= np.maximum(np.std(aa, axis=0, keepdims=True), 0.25)
    bb /= np.maximum(np.std(bb, axis=0, keepdims=True), 0.25)
    aa = np.clip(aa, -3.0, 3.0)
    bb = np.clip(bb, -3.0, 3.0)
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.sum(aa * bb) / den) if den > 1e-12 else 0.0


@dataclass(slots=True)
class _Frame:
    at: float
    feature: np.ndarray
    rms: float


@dataclass(slots=True)
class SelfEchoDecision:
    suppress: bool = False
    ready: bool = False
    mic_rms: float = 0.0
    reference_rms: float = 0.0
    match_score: float | None = None
    match_delay_ms: float = 0.0
    near_end_hold: bool = False


class SelfEchoGate:
    def __init__(self) -> None:
        self.enabled = os.environ.get("ADA_SELF_ECHO_GATE", "true").strip().lower() in {
            "1", "true", "yes", "on"
        }
        self.reference_device = os.environ.get("ADA_SELF_ECHO_REFERENCE", "ada_aec_sink.monitor")
        self.reference_delay = max(0.0, float(os.environ.get("ADA_SELF_ECHO_DELAY_MS", "155")) / 1000.0)
        self.delay_search = max(0.02, float(os.environ.get("ADA_SELF_ECHO_DELAY_SEARCH_MS", "300")) / 1000.0)
        self.match_window = max(0.12, float(os.environ.get("ADA_SELF_ECHO_MATCH_WINDOW_MS", "240")) / 1000.0)
        self.match_threshold = float(os.environ.get("ADA_SELF_ECHO_MATCH_THRESHOLD", "0.50"))
        self.match_interval_seconds = max(
            0.02, float(os.environ.get("ADA_SELF_ECHO_MATCH_INTERVAL_MS", "40")) / 1000.0
        )
        self.reference_active_rms = float(os.environ.get("ADA_SELF_ECHO_REFERENCE_RMS", "0.0015"))
        self.near_end_min_rms = float(os.environ.get("ADA_SELF_ECHO_NEAR_END_RMS", "0.006"))
        self.near_end_hold_seconds = float(os.environ.get("ADA_SELF_ECHO_NEAR_END_HOLD", "0.8"))
        self.warmup_seconds = max(
            0.0, float(os.environ.get("ADA_SELF_ECHO_WARMUP_MS", "220")) / 1000.0
        )

        self._reference: deque[_Frame] = deque(maxlen=600)
        self._mic: deque[_Frame] = deque(maxlen=100)
        self._mic_tail = np.empty(0, dtype=np.float64)
        self._reader_task: asyncio.Task[None] | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._active = False
        self._active_started_at = 0.0
        self._near_end_until = 0.0
        self._last_decision = SelfEchoDecision()
        self._last_echo_only_at = 0.0
        self._last_match_at = 0.0
        self._last_match: tuple[bool, float | None, float, float] = (
            False, None, self.reference_delay, 0.0
        )
        self._suppressed_chunks = 0
        self._passed_chunks = 0
        self._reader_restarts = 0

    async def start(self) -> None:
        if not self.enabled or self._reader_task is not None:
            return
        self._reader_task = asyncio.create_task(self._reader_loop(), name="ada-self-echo-reference")

    async def stop(self) -> None:
        task = self._reader_task
        self._reader_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._stop_process()

    async def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=1.5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def _reader_loop(self) -> None:
        chunk_bytes = _SAMPLE_RATE * 20 // 1000 * 2
        tail = np.empty(0, dtype=np.float64)
        while self.enabled:
            try:
                self._process = await asyncio.create_subprocess_exec(
                    "parec", "--raw", f"--device={self.reference_device}",
                    f"--rate={_SAMPLE_RATE}", "--format=s16le", "--channels=1",
                    "--latency-msec=40", "--process-time-msec=20",
                    "--client-name=ADA-self-echo-reference",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                )
                assert self._process.stdout is not None
                while True:
                    data = await self._process.stdout.readexactly(chunk_bytes)
                    now = time.monotonic()
                    samples = _pcm16_float(data)
                    tail = np.concatenate((tail, samples))[-_FEATURE_FRAME:]
                    if len(tail) >= _FEATURE_FRAME:
                        self._reference.append(_Frame(now, _band_feature(tail), _rms(samples)))
                    cutoff = now - 5.0
                    while self._reference and self._reference[0].at < cutoff:
                        self._reference.popleft()
            except asyncio.IncompleteReadError:
                pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("self-echo reference capture failed: %s", exc)
            finally:
                await self._stop_process()
            self._reader_restarts += 1
            await asyncio.sleep(1.0)

    def _reset_mic_window(self) -> None:
        self._mic.clear()
        self._mic_tail = np.empty(0, dtype=np.float64)
        self._near_end_until = 0.0
        self._last_match_at = 0.0
        self._last_match = (False, None, self.reference_delay, 0.0)
        self._last_decision = SelfEchoDecision()

    def _append_mic(self, pcm16: bytes) -> tuple[float, float]:
        now = time.monotonic()
        samples = _pcm16_float(pcm16)
        mic_rms = _rms(samples)
        self._mic_tail = np.concatenate((self._mic_tail, samples))[-_FEATURE_FRAME:]
        if len(self._mic_tail) >= _FEATURE_FRAME:
            self._mic.append(_Frame(now, _band_feature(self._mic_tail), mic_rms))
        cutoff = now - 2.0
        while self._mic and self._mic[0].at < cutoff:
            self._mic.popleft()
        return now, mic_rms

    def _content_match(self) -> tuple[bool, float | None, float, float]:
        """Match mic features against precomputed render features.

        The old matcher rebuilt hundreds of FFT windows on every ~20 ms microphone
        packet. On Raspberry Pi 5 that took ~145 ms per call and starved the asyncio
        voice loop. Both streams already maintain one spectral feature per frame, so
        only compare those cached features here. Packet-arrival jitter is tolerated by
        searching a broad render-delay window around the current estimate.
        """
        now = time.monotonic()
        if self._last_match_at and now - self._last_match_at < self.match_interval_seconds:
            return self._last_match
        self._last_match_at = now

        frame_step = 0.020
        nwin = max(4, int(round(self.match_window / frame_step)))
        mic_frames = list(self._mic)
        reference_frames = list(self._reference)
        if len(mic_frames) < nwin or len(reference_frames) < nwin + 1:
            self._last_match = (False, None, self.reference_delay, 0.0)
            return self._last_match

        selected_mic = mic_frames[-nwin:]
        mic_features = np.stack([frame.feature for frame in selected_mic])
        mic_end = selected_mic[-1].at
        reference_times = np.asarray([frame.at for frame in reference_frames], dtype=np.float64)

        min_delay = max(0.0, self.reference_delay - self.delay_search)
        max_delay = self.reference_delay + self.delay_search
        search_step = frame_step
        best_score = -2.0
        best_delay = self.reference_delay
        best_rms = 0.0

        for delay in np.arange(min_delay, max_delay + search_step / 2.0, search_step):
            target_end = mic_end - float(delay)
            idx = int(np.argmin(np.abs(reference_times - target_end)))
            if idx + 1 < nwin:
                continue
            # A 20 ms feature cadence plus normal scheduling jitter should keep a
            # usable reference frame close to the searched target. Larger mismatch
            # means this delay candidate is not trustworthy.
            if abs(reference_frames[idx].at - target_end) > 0.055:
                continue
            selected_ref = reference_frames[idx - nwin + 1 : idx + 1]
            ref_features = np.stack([frame.feature for frame in selected_ref])
            score = _feature_score(mic_features, ref_features)
            if score > best_score:
                best_score = score
                best_delay = float(delay)
                best_rms = math.sqrt(
                    sum(frame.rms * frame.rms for frame in selected_ref) / len(selected_ref)
                )

        if best_score <= -1.5:
            self._last_match = (False, None, self.reference_delay, 0.0)
        else:
            self._last_match = (True, best_score, best_delay, best_rms)
        return self._last_match

    def classify(self, pcm16: bytes, *, assistant_active: bool) -> SelfEchoDecision:
        if not self.enabled:
            return SelfEchoDecision(mic_rms=_rms(_pcm16_float(pcm16)))

        if assistant_active != self._active:
            self._active = assistant_active
            self._active_started_at = time.monotonic() if assistant_active else 0.0
            self._reset_mic_window()

        if not assistant_active:
            decision = SelfEchoDecision(mic_rms=_rms(_pcm16_float(pcm16)))
            self._last_decision = decision
            return decision

        now, mic_rms = self._append_mic(pcm16)
        ready, score, delay, ref_rms = self._content_match()
        hold = now < self._near_end_until
        decision = SelfEchoDecision(
            ready=ready,
            mic_rms=mic_rms,
            reference_rms=ref_rms,
            match_score=score,
            match_delay_ms=delay * 1000.0,
            near_end_hold=hold,
        )

        recent_reference = [frame for frame in self._reference if frame.at >= now - 0.10]
        recent_ref_rms = (
            math.sqrt(sum(frame.rms * frame.rms for frame in recent_reference) / len(recent_reference))
            if recent_reference else 0.0
        )
        decision.reference_rms = max(decision.reference_rms, recent_ref_rms)

        if not ready or score is None or ref_rms < self.reference_active_rms:
            # The content matcher needs several microphone frames before it can
            # correlate. During this tiny startup interval, block only while the
            # exact ADA render is already active. This is an AEC warm-up window,
            # not a general mute: once the matcher is ready, real near-end speech
            # can immediately open the gate.
            if (
                self._active_started_at
                and now - self._active_started_at <= self.warmup_seconds
                and recent_ref_rms >= self.reference_active_rms
            ):
                decision.suppress = True
                self._last_echo_only_at = now
                self._suppressed_chunks += 1
            else:
                self._passed_chunks += 1
            self._last_decision = decision
            return decision

        # This gate has one job only: identify confirmed render echo.  It must not
        # decide that a low similarity frame is near-end speech, because transient
        # browser/WebSocket jitter can briefly lower the score and accidentally
        # authorize a barge-in.  Near-end speech is confirmed separately by the
        # manual-VAD controller in main.py after a sustained non-echo interval.
        decision.near_end_hold = False
        self._near_end_until = 0.0
        if score >= self.match_threshold:
            decision.suppress = True
            self._last_echo_only_at = now
            self._suppressed_chunks += 1
            if score >= max(0.70, self.match_threshold + 0.15):
                self.reference_delay = 0.98 * self.reference_delay + 0.02 * delay
        else:
            self._passed_chunks += 1

        self._last_decision = decision
        return decision

    def current_echo_only(self) -> bool:
        decision = self._last_decision
        return bool(decision.suppress and not decision.near_end_hold)

    def recent_echo_only(self, within_seconds: float = 1.5) -> bool:
        return bool(
            self._last_echo_only_at
            and time.monotonic() - self._last_echo_only_at <= max(0.0, within_seconds)
        )

    def snapshot(self) -> dict[str, object]:
        decision = self._last_decision
        return {
            "enabled": self.enabled,
            "ready": decision.ready,
            "reference_device": self.reference_device,
            "reference_delay_ms": round(self.reference_delay * 1000.0, 1),
            "match_threshold": self.match_threshold,
            "match_score": round(decision.match_score, 4) if decision.match_score is not None else None,
            "match_delay_ms": round(decision.match_delay_ms, 1),
            "reference_rms": round(decision.reference_rms, 6),
            "mic_rms": round(decision.mic_rms, 6),
            "echo_only": decision.suppress,
            "near_end_hold": decision.near_end_hold,
            "suppressed_chunks": self._suppressed_chunks,
            "passed_chunks": self._passed_chunks,
            "reference_frames": len(self._reference),
            "mic_frames": len(self._mic),
            "reader_restarts": self._reader_restarts,
        }
