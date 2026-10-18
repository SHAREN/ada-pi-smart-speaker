"""Reference-aware near-end speech guard for TV/HDMI playback.

The acoustic echo canceller removes as much correlated TV audio as it can.  This
module handles the remaining failure mode: speech from the TV can still look
like a near-end speaker to Gemini and interrupt an assistant response.

It never blocks the first user turn after the wake word.  Once the first ADA
response starts, the conversation bridge can arm this guard for the remainder
of that active voice session.  Frames whose energy is consistent with the
recent residual-TV floor learned during standby are then replaced with silence.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from array import array
from collections import deque
from dataclasses import dataclass
from statistics import median

logger = logging.getLogger("voice.tv_reference")


def pcm16_rms(pcm16: bytes) -> float:
    """Return normalized RMS (0..1) for little-endian signed 16-bit PCM."""
    if len(pcm16) < 2:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm16[: len(pcm16) & ~1])
    if os.sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0.0
    power = sum(float(value) * float(value) for value in samples) / len(samples)
    return math.sqrt(power) / 32768.0


@dataclass(slots=True)
class TVGateDecision:
    suppress: bool
    mic_rms: float
    ref_rms: float
    expected_echo_rms: float
    threshold: float
    learned_ratio: float
    ready: bool
    near_end_hold: bool


class TVReferenceGate:
    """Track HDMI render energy and reject probable TV-only barge-in frames."""

    def __init__(self) -> None:
        self.enabled = os.environ.get("ADA_TV_REFERENCE_GATE", "true").strip().lower() in {
            "1", "true", "yes", "on"
        }
        self.sample_rate = 16000
        self.chunk_ms = 20
        self.chunk_bytes = self.sample_rate * self.chunk_ms // 1000 * 2
        self.reference_delay = max(
            0.0, float(os.environ.get("ADA_TV_REFERENCE_DELAY_MS", "255")) / 1000.0
        )
        self.reference_active_rms = float(os.environ.get("ADA_TV_REFERENCE_ACTIVE_RMS", "0.0018"))
        self.absolute_near_end_rms = float(os.environ.get("ADA_TV_NEAR_END_MIN_RMS", "0.006"))
        self.echo_margin = float(os.environ.get("ADA_TV_ECHO_MARGIN", "1.55"))
        self.near_end_hold_seconds = float(os.environ.get("ADA_TV_NEAR_END_HOLD", "0.9"))
        self.min_training_samples = int(os.environ.get("ADA_TV_GATE_MIN_TRAINING", "25"))
        self._reference_history: deque[tuple[float, float]] = deque(maxlen=500)
        self._ratio_history: deque[float] = deque(maxlen=400)
        self._reader_task: asyncio.Task[None] | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._near_end_until = 0.0
        self._last_decision: TVGateDecision | None = None
        self._suppressed_chunks = 0
        self._passed_chunks = 0
        self._restart_count = 0

    async def start(self) -> None:
        if not self.enabled or self._reader_task is not None:
            return
        self._reader_task = asyncio.create_task(self._reader_loop(), name="ada-tv-reference")

    async def stop(self) -> None:
        task, self._reader_task = self._reader_task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._stop_process()

    async def _stop_process(self) -> None:
        process, self._process = self._process, None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=1.5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def _reader_loop(self) -> None:
        while self.enabled:
            try:
                self._process = await asyncio.create_subprocess_exec(
                    "parec",
                    "--raw",
                    "--device=@DEFAULT_MONITOR@",
                    f"--rate={self.sample_rate}",
                    "--format=s16le",
                    "--channels=1",
                    "--latency-msec=40",
                    "--process-time-msec=20",
                    "--client-name=ADA-TV-reference",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                assert self._process.stdout is not None
                while True:
                    data = await self._process.stdout.readexactly(self.chunk_bytes)
                    now = time.monotonic()
                    self._reference_history.append((now, pcm16_rms(data)))
                    cutoff = now - 3.0
                    while self._reference_history and self._reference_history[0][0] < cutoff:
                        self._reference_history.popleft()
            except asyncio.IncompleteReadError:
                pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("TV reference capture failed: %s", exc)
            finally:
                await self._stop_process()
            self._restart_count += 1
            await asyncio.sleep(1.0)

    def _delayed_reference_rms(self, now: float) -> float:
        if not self._reference_history:
            return 0.0
        target = now - self.reference_delay
        # History is at ~20 ms granularity. Search backwards so the common case
        # finds the target in a handful of entries rather than copying the deque.
        best_rms = 0.0
        best_delta = float("inf")
        for stamp, level in reversed(self._reference_history):
            delta = abs(stamp - target)
            if delta < best_delta:
                best_delta = delta
                best_rms = level
            elif stamp < target:
                break
        # Do not use stale reference data after playback has stopped/restarted.
        return best_rms if best_delta <= 0.12 else 0.0

    def _learned_ratio(self) -> float:
        if not self._ratio_history:
            return 0.0
        values = sorted(self._ratio_history)
        # A high-ish robust quantile learns ordinary TV residual peaks while
        # ignoring the rare much-louder near-end utterance.
        index = min(len(values) - 1, int(round((len(values) - 1) * 0.78)))
        return float(values[index])

    def observe(self, pcm16: bytes, *, learn: bool) -> TVGateDecision:
        """Observe one microphone frame and optionally learn the TV residual floor."""
        now = time.monotonic()
        mic_rms = pcm16_rms(pcm16)
        ref_rms = self._delayed_reference_rms(now)
        ratio = mic_rms / max(ref_rms, 1e-6)
        if learn and ref_rms >= self.reference_active_rms and 0.0 < ratio < 20.0:
            # During standby most HDMI-correlated input is the TV itself. A user
            # speaking without the wake word becomes an upper outlier and is
            # intentionally prevented from dominating the robust quantile.
            if len(self._ratio_history) < self.min_training_samples:
                self._ratio_history.append(ratio)
            else:
                center = median(self._ratio_history)
                if ratio <= max(center * 3.0, 0.08):
                    self._ratio_history.append(ratio)

        learned_ratio = self._learned_ratio()
        ready = len(self._ratio_history) >= self.min_training_samples
        expected = ref_rms * learned_ratio if ready else 0.0
        threshold = max(self.absolute_near_end_rms, expected * self.echo_margin)
        hold = now < self._near_end_until
        decision = TVGateDecision(
            suppress=False,
            mic_rms=mic_rms,
            ref_rms=ref_rms,
            expected_echo_rms=expected,
            threshold=threshold,
            learned_ratio=learned_ratio,
            ready=ready,
            near_end_hold=hold,
        )
        self._last_decision = decision
        return decision

    def classify_active(self, pcm16: bytes) -> TVGateDecision:
        """Classify a frame after the conversation gate has been armed."""
        decision = self.observe(pcm16, learn=False)
        now = time.monotonic()
        if not self.enabled or not decision.ready or decision.ref_rms < self.reference_active_rms:
            self._passed_chunks += 1
            return decision

        if decision.near_end_hold:
            self._passed_chunks += 1
            return decision

        # Require a clear rise above the learned TV residual envelope. Once a
        # near-end rise is seen, hold the gate open so quiet syllables are not
        # chopped out of the same utterance.
        if decision.mic_rms > decision.threshold:
            self._near_end_until = now + self.near_end_hold_seconds
            decision.near_end_hold = True
            self._passed_chunks += 1
            return decision

        decision.suppress = True
        self._suppressed_chunks += 1
        return decision

    def recent_tv_only(self) -> bool:
        decision = self._last_decision
        return bool(decision and decision.suppress and not decision.near_end_hold)

    def snapshot(self) -> dict[str, object]:
        decision = self._last_decision
        return {
            "enabled": self.enabled,
            "ready": len(self._ratio_history) >= self.min_training_samples,
            "training_samples": len(self._ratio_history),
            "learned_ratio": round(self._learned_ratio(), 4),
            "reference_delay_ms": round(self.reference_delay * 1000.0, 1),
            "reference_rms": round(decision.ref_rms, 6) if decision else 0.0,
            "mic_rms": round(decision.mic_rms, 6) if decision else 0.0,
            "threshold": round(decision.threshold, 6) if decision else 0.0,
            "tv_only": bool(decision.suppress) if decision else False,
            "near_end_hold": bool(decision.near_end_hold) if decision else False,
            "suppressed_chunks": self._suppressed_chunks,
            "passed_chunks": self._passed_chunks,
            "reference_restarts": self._restart_count,
        }
