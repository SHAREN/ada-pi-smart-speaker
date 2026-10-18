"""Local wake/stop/session gate for ADA Pi's always-open P610 microphone.

The browser continues capturing the microphone at all times so microWakeWord can
run locally. PCM is forwarded to Gemini only while the gate is active. This
preserves ADA's native full-duplex barge-in inside an active conversation while
providing a privacy-preserving standby mode.
"""
from __future__ import annotations

import math
import os
import time
from array import array
from collections import deque
from pathlib import Path

from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures


PCM_RATE = 16_000
PCM_WIDTH = 2
PCM_CHANNELS = 1
PCM_BYTES_PER_SECOND = PCM_RATE * PCM_WIDTH * PCM_CHANNELS


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


class WakeSessionGate:
    """Small deterministic state machine around local wake/stop detection."""

    def __init__(self, root: Path) -> None:
        wake_dir = root / "wakewords"
        self._features = MicroWakeWordFeatures()
        self._wake_model = MicroWakeWord.from_config(wake_dir / "okay_nabu.json")
        self._stop_model = MicroWakeWord.from_config(wake_dir / "stop.json")

        # Production values copied from proxy39.
        self.wake_threshold = _env_float("ADA_WAKE_THRESHOLD", 0.85, 0.0, 1.0)
        self.stop_threshold = _env_float("ADA_STOP_THRESHOLD", 0.50, 0.0, 1.0)
        self.refractory_seconds = _env_float("ADA_WAKE_REFRACTORY_SECONDS", 2.0, 0.0, 30.0)
        self.stop_guard_seconds = _env_float("ADA_STOP_GUARD_SECONDS", 0.25, 0.0, 5.0)
        self.end_cue_wake_guard_seconds = _env_float("ADA_END_CUE_WAKE_GUARD_SECONDS", 1.0, 0.0, 5.0)
        self.pre_roll_seconds = _env_float("ADA_AUDIO_PREROLL_SECONDS", 2.0, 0.0, 5.0)
        # Wake is an arming command, never user content. Do not forward any
        # pre-wake overlap to Gemini after the activation cue.
        self.command_overlap_seconds = 0.0
        self.activation_buffer_max_seconds = _env_float("ADA_ACTIVATION_BUFFER_MAX_SECONDS", 30.0, 2.0, 60.0)
        self.idle_timeout_seconds = _env_float("ADA_ACTIVE_IDLE_TIMEOUT", 30.0, 5.0, 300.0)

        self._wake_model.probability_cutoff = self.wake_threshold
        self._stop_model.probability_cutoff = self.stop_threshold
        self._wake_model.debug_probabilities = False
        self._stop_model.debug_probabilities = False

        self.state = "standby"  # standby | active | closing
        self.last_end_reason = ""
        self._last_wake_monotonic = 0.0
        self._last_activity_monotonic = 0.0
        self._stop_guard_until = 0.0
        self._wake_guard_until = 0.0
        self._wake_detected_monotonic = 0.0

        self._pre_roll: deque[bytes] = deque()
        self._pre_roll_duration = 0.0
        self._pending: deque[bytes] = deque()
        self._pending_duration = 0.0
        self.activation_buffer_overflow = False

        self.wake_count = 0
        self.stop_count = 0
        self.idle_count = 0
        self.semantic_end_count = 0

    @property
    def active(self) -> bool:
        return self.state == "active"

    @property
    def closing(self) -> bool:
        return self.state == "closing"

    @staticmethod
    def frame_duration(pcm16: bytes) -> float:
        return len(pcm16) / PCM_BYTES_PER_SECOND

    @staticmethod
    def normalized_rms(pcm16: bytes) -> float:
        if not pcm16:
            return 0.0
        samples = array("h")
        samples.frombytes(pcm16)
        if not samples:
            return 0.0
        mean_square = sum(sample * sample for sample in samples) / len(samples)
        return math.sqrt(mean_square) / 32768.0

    def _reset_models(self) -> None:
        self._features.reset()
        self._wake_model.reset()
        self._stop_model.reset()

    def remember_pre_roll(self, pcm16: bytes) -> None:
        if self.state != "standby" or not pcm16:
            return
        self._pre_roll.append(pcm16)
        self._pre_roll_duration += self.frame_duration(pcm16)
        while self._pre_roll and self._pre_roll_duration > self.pre_roll_seconds:
            dropped = self._pre_roll.popleft()
            self._pre_roll_duration -= self.frame_duration(dropped)

    def begin_activation_buffer(self) -> None:
        """Start a clean post-wake command buffer.

        The wake phrase itself is only an arming command.  Never seed Gemini's
        input with standby/pre-wake audio, otherwise the tail of "Okay Nabu" or
        the activation cue can be interpreted as a user turn and make ADA speak
        before the user has asked anything.
        """
        self._pending.clear()
        self._pending_duration = 0.0
        self._pre_roll.clear()
        self._pre_roll_duration = 0.0
        self.activation_buffer_overflow = False
        self._wake_detected_monotonic = time.monotonic()

    def buffer_activation_frame(self, pcm16: bytes) -> bool:
        """Keep earliest command audio if provider/cue is temporarily unavailable."""
        duration = self.frame_duration(pcm16)
        if self._pending_duration + duration > self.activation_buffer_max_seconds:
            self.activation_buffer_overflow = True
            return False
        self._pending.append(pcm16)
        self._pending_duration += duration
        return True

    def activation_buffer_snapshot(self) -> list[bytes]:
        return list(self._pending)

    def drain_activation_buffer(self) -> list[bytes]:
        frames = list(self._pending)
        self._pending.clear()
        self._pending_duration = 0.0
        self.activation_buffer_overflow = False
        return frames

    def clear_activation_buffer(self) -> None:
        self._pending.clear()
        self._pending_duration = 0.0
        self._pre_roll.clear()
        self._pre_roll_duration = 0.0
        self.activation_buffer_overflow = False

    def activate(self) -> None:
        now = time.monotonic()
        self.state = "active"
        self.last_end_reason = ""
        self._last_activity_monotonic = now
        self._stop_guard_until = now + self.stop_guard_seconds
        self.wake_count += 1

    def begin_close(self, reason: str) -> None:
        if self.state == "standby":
            return
        self.state = "closing"
        self.last_end_reason = reason
        self.clear_activation_buffer()
        if reason == "stop":
            self.stop_count += 1
        elif reason == "idle":
            self.idle_count += 1
        elif reason == "semantic":
            self.semantic_end_count += 1

    def finish_standby(self) -> None:
        self.state = "standby"
        self._last_activity_monotonic = 0.0
        self._wake_guard_until = time.monotonic() + self.end_cue_wake_guard_seconds
        self.clear_activation_buffer()
        self._reset_models()

    def reset_after_wake_cue(self) -> None:
        """Do not let the activation cue or its acoustic tail trigger Stop."""
        self._features.reset()
        self._wake_model.reset()
        self._stop_model.reset()
        self._stop_guard_until = time.monotonic() + self.stop_guard_seconds

    def note_activity(self) -> None:
        if self.state == "active":
            self._last_activity_monotonic = time.monotonic()

    def idle_elapsed(self) -> float:
        if self.state != "active" or not self._last_activity_monotonic:
            return 0.0
        return max(0.0, time.monotonic() - self._last_activity_monotonic)

    def idle_expired(self) -> bool:
        return self.state == "active" and self.idle_elapsed() >= self.idle_timeout_seconds

    def process_audio(self, pcm16: bytes, *, allow_stop: bool = True) -> str | None:
        """Return ``wake`` or ``stop`` on a local detection, otherwise ``None``.

        ``allow_stop`` must come from an independent near-end/local-speech VAD.
        The Stop model is intentionally not allowed to close a session from the
        assistant's own speaker audio or residual echo alone.
        """
        if self.state == "closing":
            return None
        if self.state == "standby" and time.monotonic() < self._wake_guard_until:
            return None

        for features in self._features.process_streaming(pcm16):
            if self.state == "active":
                if time.monotonic() < self._stop_guard_until:
                    self._stop_model.reset()
                    continue
                if not allow_stop:
                    # Never accumulate assistant playback / residual echo inside
                    # the Stop detector. A fresh local-speech VAD window must be
                    # present before the detector can make a closing decision.
                    self._stop_model.reset()
                    continue
                if self._stop_model.process_streaming(features):
                    self.begin_close("stop")
                    return "stop"
                continue

            if self._wake_model.process_streaming(features):
                now = time.monotonic()
                if now - self._last_wake_monotonic < self.refractory_seconds:
                    continue
                self._last_wake_monotonic = now
                self.activate()
                return "wake"
        return None

    def status(self) -> dict[str, object]:
        return {
            "state": self.state,
            "active": self.active,
            "closing": self.closing,
            "wake_word": "Okay Nabu",
            "stop_word": "Stop",
            "wake_threshold": self.wake_threshold,
            "stop_threshold": self.stop_threshold,
            "refractory_seconds": self.refractory_seconds,
            "stop_guard_seconds": self.stop_guard_seconds,
            "end_cue_wake_guard_seconds": self.end_cue_wake_guard_seconds,
            "audio_preroll_seconds": self.pre_roll_seconds,
            "command_overlap_seconds": self.command_overlap_seconds,
            "activation_buffer_max_seconds": self.activation_buffer_max_seconds,
            "active_idle_timeout_seconds": self.idle_timeout_seconds,
            "idle_elapsed_seconds": round(self.idle_elapsed(), 3),
            "buffered_input_frames": len(self._pending),
            "buffered_input_bytes": sum(len(frame) for frame in self._pending),
            "buffered_input_seconds": round(self._pending_duration, 3),
            "activation_buffer_overflow": self.activation_buffer_overflow,
            "wake_count": self.wake_count,
            "stop_count": self.stop_count,
            "idle_count": self.idle_count,
            "semantic_end_count": self.semantic_end_count,
            "last_end_reason": self.last_end_reason,
        }
