"""Content-matched TV/HDMI guard for full-duplex voice sessions.

The TV is a difficult acoustic echo source: its own DSP, room reflections and a
~300 ms acoustic path mean ordinary AEC may leave intelligible dialogue behind.
Rather than deciding from residual loudness, this guard observes two independent
signals in parallel:

* the physical P610 microphone (before the software AEC cascade), and
* the exact digital monitor of the normal system/HDMI sink.

Every 20 ms each stream is reduced to coarse log-band energies.  Over a rolling
window we compare the *temporal pattern* of those bands while searching a small
delay range.  Per-band normalization removes most fixed room/microphone colour.
A high score means "the microphone is hearing the same content that HDMI played".
A near-end speaker adds unrelated content and drives the score down.

The first user turn after the wake word is never gated.  Once armed by the first
assistant response, callers may replace TV-matched microphone chunks with
silence.  A short downstream delay buffer lets a detected near-end syllable open
pre-roll instead of clipping the start of the utterance.
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

import numpy as np

logger = logging.getLogger("voice.tv_reference")


P610_SOURCE_DEFAULT = (
    "alsa_input.usb-Plantronics_Plantronics_P610_"
    "EE1EBC80EE86964E9094E76D3A28EF37-00.analog-stereo"
)
_BAND_EDGES = np.asarray([8, 12, 18, 25, 34, 45, 58, 73, 90, 110, 135, 170, 225])
_FEATURE_FRAME = 512
_FEATURE_WINDOW = np.hanning(_FEATURE_FRAME)


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


def _pcm16_float(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16[: len(pcm16) & ~1], dtype="<i2").astype(np.float64) / 32768.0


def _band_feature(samples: np.ndarray) -> np.ndarray:
    """Return 12 coarse log-power bands for one 32 ms frame at 16 kHz."""
    if len(samples) < _FEATURE_FRAME:
        raise ValueError("feature frame is too short")
    frame = samples[-_FEATURE_FRAME:] * _FEATURE_WINDOW
    power = np.abs(np.fft.rfft(frame)) ** 2
    return np.asarray(
        [
            math.log(float(np.mean(power[a:b])) + 1e-10)
            for a, b in zip(_BAND_EDGES[:-1], _BAND_EDGES[1:])
        ],
        dtype=np.float64,
    )


def feature_match_score(a: np.ndarray, b: np.ndarray) -> float:
    """Channel-insensitive temporal similarity for two N x 12 feature windows."""
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] < 4:
        return 0.0
    aa = a - np.mean(a, axis=0, keepdims=True)
    bb = b - np.mean(b, axis=0, keepdims=True)
    aa /= np.maximum(np.std(aa, axis=0, keepdims=True), 0.25)
    bb /= np.maximum(np.std(bb, axis=0, keepdims=True), 0.25)
    aa = np.clip(aa, -3.0, 3.0)
    bb = np.clip(bb, -3.0, 3.0)
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 1e-12:
        return 0.0
    return float(np.sum(aa * bb) / den)


@dataclass(slots=True)
class _FeatureFrame:
    at: float
    feature: np.ndarray
    rms: float


@dataclass(slots=True)
class _ContentMatch:
    ready: bool = False
    score: float | None = None
    delay_seconds: float = 0.0
    reference_rms: float = 0.0
    physical_rms: float = 0.0


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
    match_score: float | None = None
    match_threshold: float = 0.0
    match_delay_ms: float = 0.0
    physical_rms: float = 0.0


class TVReferenceGate:
    """Reject HDMI-derived speech while preserving unrelated near-end speech."""

    def __init__(self) -> None:
        self.enabled = os.environ.get("ADA_TV_REFERENCE_GATE", "true").strip().lower() in {
            "1", "true", "yes", "on"
        }
        self.sample_rate = 16000
        self.chunk_ms = 20
        self.chunk_bytes = self.sample_rate * self.chunk_ms // 1000 * 2
        self.reference_device = os.environ.get("ADA_TV_REFERENCE_SOURCE", "@DEFAULT_MONITOR@")
        self.physical_device = os.environ.get("ADA_TV_PHYSICAL_SOURCE", P610_SOURCE_DEFAULT)

        # The real-video capture on this installation measured ~283 ms.  The
        # matcher searches around this value and slowly updates it only from
        # strong TV-only matches, so TV processing latency may drift safely.
        self.reference_delay = max(
            0.0, float(os.environ.get("ADA_TV_REFERENCE_DELAY_MS", "283")) / 1000.0
        )
        self.delay_search_seconds = max(
            0.02, float(os.environ.get("ADA_TV_MATCH_DELAY_SEARCH_MS", "120")) / 1000.0
        )
        self.match_window_seconds = max(
            0.30, float(os.environ.get("ADA_TV_MATCH_WINDOW_MS", "500")) / 1000.0
        )
        self.match_interval_seconds = max(
            0.05, float(os.environ.get("ADA_TV_MATCH_INTERVAL_MS", "100")) / 1000.0
        )
        self.match_floor = float(os.environ.get("ADA_TV_MATCH_FLOOR", "0.24"))
        self.match_ceiling = float(os.environ.get("ADA_TV_MATCH_CEILING", "0.38"))
        self.match_scale = float(os.environ.get("ADA_TV_MATCH_SCALE", "0.70"))
        self.reference_active_rms = float(os.environ.get("ADA_TV_REFERENCE_ACTIVE_RMS", "0.003"))
        self.absolute_near_end_rms = float(os.environ.get("ADA_TV_NEAR_END_MIN_RMS", "0.004"))
        self.near_end_hold_seconds = float(os.environ.get("ADA_TV_NEAR_END_HOLD", "1.0"))
        self.min_training_samples = int(os.environ.get("ADA_TV_GATE_MIN_TRAINING", "20"))
        self.decision_delay_seconds = float(os.environ.get("ADA_TV_GATE_DECISION_DELAY", "0.40"))
        self.decision_delay_bytes = int(self.sample_rate * 2 * self.decision_delay_seconds)

        self._reference_features: deque[_FeatureFrame] = deque(maxlen=500)
        self._physical_features: deque[_FeatureFrame] = deque(maxlen=500)
        self._match_history: deque[float] = deque(maxlen=500)
        self._reader_tasks: dict[str, asyncio.Task[None]] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._near_end_until = 0.0
        self._last_decision: TVGateDecision | None = None
        self._last_match = _ContentMatch()
        self._last_match_at = 0.0
        self._suppressed_chunks = 0
        self._passed_chunks = 0
        self._restart_count = 0

    async def start(self) -> None:
        if not self.enabled or self._reader_tasks:
            return
        self._reader_tasks = {
            "reference": asyncio.create_task(
                self._reader_loop("reference", self.reference_device, "ADA-TV-reference"),
                name="ada-tv-reference",
            ),
            "physical": asyncio.create_task(
                self._reader_loop("physical", self.physical_device, "ADA-TV-physical"),
                name="ada-tv-physical",
            ),
        }

    async def stop(self) -> None:
        tasks = list(self._reader_tasks.values())
        self._reader_tasks.clear()
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        for kind in list(self._processes):
            await self._stop_process(kind)

    async def _stop_process(self, kind: str) -> None:
        process = self._processes.pop(kind, None)
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=1.5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def _reader_loop(self, kind: str, device: str, client_name: str) -> None:
        history = self._reference_features if kind == "reference" else self._physical_features
        tail = np.empty(0, dtype=np.float64)
        while self.enabled:
            try:
                process = await asyncio.create_subprocess_exec(
                    "parec",
                    "--raw",
                    f"--device={device}",
                    f"--rate={self.sample_rate}",
                    "--format=s16le",
                    "--channels=1",
                    "--latency-msec=40",
                    "--process-time-msec=20",
                    f"--client-name={client_name}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                self._processes[kind] = process
                assert process.stdout is not None
                while True:
                    data = await process.stdout.readexactly(self.chunk_bytes)
                    now = time.monotonic()
                    samples = _pcm16_float(data)
                    tail = np.concatenate((tail, samples))[-_FEATURE_FRAME:]
                    if len(tail) >= _FEATURE_FRAME:
                        history.append(
                            _FeatureFrame(
                                at=now,
                                feature=_band_feature(tail),
                                rms=pcm16_rms(data),
                            )
                        )
                    cutoff = now - 8.0
                    while history and history[0].at < cutoff:
                        history.popleft()
            except asyncio.IncompleteReadError:
                pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("%s capture failed: %s", client_name, exc)
            finally:
                await self._stop_process(kind)
            self._restart_count += 1
            await asyncio.sleep(1.0)

    def _match_threshold(self) -> float:
        if len(self._match_history) < self.min_training_samples:
            return self.match_floor
        values = np.asarray(self._match_history, dtype=np.float64)
        q10 = float(np.quantile(values, 0.10))
        return max(self.match_floor, min(self.match_ceiling, q10 * self.match_scale))

    def _content_match(self) -> _ContentMatch:
        now = time.monotonic()
        if now - self._last_match_at < self.match_interval_seconds:
            return self._last_match
        self._last_match_at = now

        nwin = max(15, int(round(self.match_window_seconds / (self.chunk_ms / 1000.0))))
        physical = list(self._physical_features)
        reference = list(self._reference_features)
        if len(physical) < nwin or len(reference) < nwin + 3:
            self._last_match = _ContentMatch()
            return self._last_match

        pframes = physical[-nwin:]
        p = np.stack([item.feature for item in pframes])
        p_end = pframes[-1].at
        p_rms = math.sqrt(sum(item.rms * item.rms for item in pframes) / len(pframes))
        ref_times = np.asarray([item.at for item in reference], dtype=np.float64)

        step = self.chunk_ms / 1000.0
        search_steps = max(1, int(round(self.delay_search_seconds / step)))
        best_score = -2.0
        best_delay = self.reference_delay
        best_ref_rms = 0.0
        for offset in range(-search_steps, search_steps + 1):
            delay = max(0.0, self.reference_delay + offset * step)
            target_end = p_end - delay
            idx = int(np.argmin(np.abs(ref_times - target_end)))
            if idx + 1 < nwin:
                continue
            if abs(reference[idx].at - target_end) > 0.045:
                continue
            rframes = reference[idx - nwin + 1 : idx + 1]
            r = np.stack([item.feature for item in rframes])
            score = feature_match_score(p, r)
            if score > best_score:
                best_score = score
                best_delay = delay
                best_ref_rms = math.sqrt(
                    sum(item.rms * item.rms for item in rframes) / len(rframes)
                )

        if best_score <= -1.5:
            self._last_match = _ContentMatch()
        else:
            self._last_match = _ContentMatch(
                ready=True,
                score=best_score,
                delay_seconds=best_delay,
                reference_rms=best_ref_rms,
                physical_rms=p_rms,
            )
        return self._last_match

    def observe(self, pcm16: bytes, *, learn: bool) -> TVGateDecision:
        """Observe final post-AEC mic audio and optionally learn real TV match scores."""
        now = time.monotonic()
        match = self._content_match()
        mic_rms = pcm16_rms(pcm16)

        if (
            learn
            and match.ready
            and match.score is not None
            and match.reference_rms >= self.reference_active_rms
            and match.score >= self.match_floor
        ):
            self._match_history.append(match.score)
            # Only a strong content match is trusted to retune the acoustic delay.
            if match.score >= 0.45:
                self.reference_delay = 0.97 * self.reference_delay + 0.03 * match.delay_seconds

        ready = bool(
            match.ready
            and len(self._match_history) >= self.min_training_samples
            and len(self._reader_tasks) >= 2
        )
        threshold = self._match_threshold()
        hold = now < self._near_end_until
        decision = TVGateDecision(
            suppress=False,
            mic_rms=mic_rms,
            ref_rms=match.reference_rms,
            expected_echo_rms=0.0,
            threshold=self.absolute_near_end_rms,
            learned_ratio=0.0,
            ready=ready,
            near_end_hold=hold,
            match_score=match.score,
            match_threshold=threshold,
            match_delay_ms=match.delay_seconds * 1000.0,
            physical_rms=match.physical_rms,
        )
        self._last_decision = decision
        return decision

    def classify_active(self, pcm16: bytes) -> TVGateDecision:
        """Classify post-wake audio using physical-mic/HDMI content matching."""
        decision = self.observe(pcm16, learn=False)
        now = time.monotonic()
        if (
            not self.enabled
            or not decision.ready
            or decision.ref_rms < self.reference_active_rms
            or decision.match_score is None
        ):
            self._passed_chunks += 1
            return decision

        if decision.near_end_hold:
            self._passed_chunks += 1
            return decision

        # Near-end requires two independent facts: the physical microphone no
        # longer resembles HDMI content, and the final post-AEC stream contains
        # meaningful energy. A loud actor can satisfy the second condition, but
        # keeps a high HDMI match and therefore remains suppressed.
        if (
            decision.match_score < decision.match_threshold
            and decision.mic_rms >= self.absolute_near_end_rms
        ):
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
            "ready": bool(decision.ready) if decision else False,
            "training_samples": len(self._match_history),
            "reference_delay_ms": round(self.reference_delay * 1000.0, 1),
            "reference_rms": round(decision.ref_rms, 6) if decision else 0.0,
            "physical_rms": round(decision.physical_rms, 6) if decision else 0.0,
            "mic_rms": round(decision.mic_rms, 6) if decision else 0.0,
            "match_score": round(decision.match_score, 4) if decision and decision.match_score is not None else None,
            "match_threshold": round(decision.match_threshold, 4) if decision else round(self._match_threshold(), 4),
            "near_end_min_rms": self.absolute_near_end_rms,
            "decision_delay_ms": round(self.decision_delay_seconds * 1000.0, 1),
            "tv_only": bool(decision.suppress) if decision else False,
            "near_end_hold": bool(decision.near_end_hold) if decision else False,
            "suppressed_chunks": self._suppressed_chunks,
            "passed_chunks": self._passed_chunks,
            "reference_feature_frames": len(self._reference_features),
            "physical_feature_frames": len(self._physical_features),
            "reader_restarts": self._restart_count,
        }
