#!/usr/bin/env python3
"""Generate tiny deterministic local success cues for ADA household controls."""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOUNDS = ROOT / "sounds"
RATE = 24_000
AMPLITUDE = 0.20


def _tone(freq: float, seconds: float, *, phase: float = 0.0) -> list[float]:
    count = max(1, int(RATE * seconds))
    fade = max(1, int(RATE * 0.012))
    out: list[float] = []
    for index in range(count):
        envelope = 1.0
        if index < fade:
            envelope = index / fade
        elif index >= count - fade:
            envelope = max(0.0, (count - 1 - index) / fade)
        # Fundamental plus a quiet octave makes the cue audible on the P610
        # without sounding like a harsh pure beep.
        value = math.sin(2 * math.pi * freq * index / RATE + phase)
        value += 0.22 * math.sin(2 * math.pi * (freq * 2) * index / RATE + phase)
        out.append(value * AMPLITUDE * envelope / 1.22)
    return out


def _silence(seconds: float) -> list[float]:
    return [0.0] * int(RATE * seconds)


def _write(path: Path, samples: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = bytearray()
    for sample in samples:
        clipped = max(-1.0, min(1.0, sample))
        pcm.extend(struct.pack("<h", round(clipped * 32767)))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(bytes(pcm))


def main() -> None:
    # Rising two-note chime = ON / enabled.
    on = _tone(660.0, 0.085) + _silence(0.018) + _tone(880.0, 0.11)
    # Descending two-note chime = OFF / disabled.
    off = _tone(660.0, 0.085) + _silence(0.018) + _tone(440.0, 0.12)
    _write(SOUNDS / "control_on.wav", on)
    _write(SOUNDS / "control_off.wav", off)
    print("generated", SOUNDS / "control_on.wav", SOUNDS / "control_off.wav")


if __name__ == "__main__":
    main()
