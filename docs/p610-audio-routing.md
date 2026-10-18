# Plantronics P610 voice-speaker audio routing

## Current production intent (2026-10-17)

The Plantronics P610 is the dedicated audio endpoint for the ADA / smart-speaker voice loop. It must **not** be the general desktop/system speaker.

Required routing:

- General Raspberry Pi / desktop / VLC / browser / media audio -> **whatever normal system output the user selected** (for example HDMI or ordinary speakers).
- ADA smart-speaker microphone -> **Plantronics P610 microphone** through `ada_aec_source`.
- ADA smart-speaker replies, wake/stop/session sounds and other ADA-owned playback -> **Plantronics P610 speaker** through `ada_aec_sink`.
- `ada_aec_source` and `ada_aec_sink` are a paired PipeWire/PulseAudio WebRTC echo-cancel path. Chromium-side echo cancellation stays disabled to avoid double AEC.

In other words, changing the system output for a film or music must not move the smart-speaker voice path away from the P610.

## How this is implemented

`ada-p610-aec.service` starts `scripts/p610-aec.sh`.

The script:

1. Finds the physical P610 input and output.
2. Leaves the user's general system playback selection untouched.
3. Creates `ada_aec_source` + `ada_aec_sink` with `module-echo-cancel`, using the physical P610 as both AEC masters.
4. Sets the physical P610 speaker to 80% and the virtual AEC sink to 100%.
5. Makes `ada_aec_source` the default capture source, but deliberately does **not** make `ada_aec_sink` the global default playback sink.

`ada-pi.service` pins the whole ADA process tree to the dedicated voice path with:

```ini
Environment=PULSE_SINK=ada_aec_sink
Environment=PULSE_SOURCE=ada_aec_source
```

This makes the ADA Chromium process and ADA-owned PulseAudio playback use the P610/AEC path regardless of which normal system output the user chooses.

The voice watchdog also checks the actual `echo-cancel-playback` sink. Desktop audio controls can move existing streams when the user switches the normal output, so the watchdog moves only that AEC render stream back to the physical P610 if necessary. It never changes the user's system default sink.

## Important failure mode

A previous version of `p610-aec.sh` executed:

```bash
pactl set-default-sink ada_aec_sink
```

That made the P610/AEC path the global system output, so ordinary media and desktop audio could come out of the smart-speaker speaker. Conversely, changing the normal desktop output can move the already-running AEC render stream unless the route is guarded.

Therefore the invariant is:

> **System default = user's normal speakers/output; ADA process = `ada_aec_sink` / `ada_aec_source`; AEC physical master = P610.**

## Voice-loop conclusions from the P610 investigation

- The P610 is the USB microphone/speaker front end; the assistant logic, wake word, session control, Gemini/Home Assistant integration and recovery logic live in software on the Raspberry Pi.
- The microphone is kept available locally for wake/stop handling; the normal wake phrase is **Okay Nabu**, with local **Stop** handling.
- The voice loop is designed for full-duplex interaction and barge-in, so echo cancellation and correct speaker reference routing are essential.
- Do not enable Chromium/browser AEC on top of the system AEC path. `ada_aec_source` already uses `ada_aec_sink` as the render reference.
- Earlier debugging showed that very low P610 hardware volume could make replies effectively inaudible even when playback was active. The production script therefore keeps the physical P610 output at 80%.
- Gemini Live start-of-speech sensitivity is intentionally set to `START_SENSITIVITY_HIGH`. In the October 18 failure, local speech detection saw the user and microphone audio reached Gemini, but Gemini produced no `user_transcript` or response because the post-AEC speech level was substantially quieter than in the previous working capture. High sensitivity restored reliable turns; the P610 AEC path is relied on to prevent speaker echo from causing false starts.
- If wake/end cues are audible but spoken replies are absent, inspect the latest audio debug capture. `assistant_raw.wav` and `assistant_sent.wav` remaining at zero bytes means the provider never generated speech; distinguish that from a playback-routing problem before changing sinks.
- If ADA hears itself, first verify that `ada_aec_source` and `ada_aec_sink` exist, that ADA is pinned to them, and that the echo-cancel playback master still terminates on the physical P610 rather than HDMI.

## Verification

Expected high-level state:

```text
Default system sink: user's currently selected normal output
ADA playback sink:   ada_aec_sink
ADA capture source:  ada_aec_source
AEC sink master:     physical Plantronics P610 speaker
AEC source master:   physical Plantronics P610 microphone
```

Useful checks:

```bash
wpctl status
pactl list short sinks
pactl list short sources
pactl list sink-inputs
systemctl --user status ada-p610-aec.service ada-pi.service
```

When checking `pactl list sink-inputs`, the echo-cancel playback stream must ultimately be attached to the physical P610 sink, while unrelated media such as VLC should remain on whichever normal system output the user selected.

## TV / system-audio reference cancellation

ADA now uses the system playback signal as an additional far-end reference so dialogue from the TV is less likely to be interpreted as the user. The implementation deliberately does **not** change the user's normal playback device.

The live capture chain is:

1. `ada-tv-aec.service` starts `scripts/tv-aec-native.sh` after PipeWire/WirePlumber expose the P610. A native PipeWire WebRTC echo-cancel module runs in `monitor.mode`, observing the monitor of the current default system sink (normally HDMI/TV) while capturing the physical P610 microphone. It publishes `ada_tv_aec_source`.
2. `ada-p610-aec.service` waits briefly for `ada_tv_aec_source` and then uses it as the capture master for the existing P610 self-echo AEC. If the TV layer is unavailable, it safely falls back to the physical P610 microphone.
3. The final `ada_aec_source` therefore has two cascaded acoustic references: system/TV playback first, ADA's own P610 playback second.
4. ADA playback remains pinned to `ada_aec_sink`/P610. Ordinary system audio remains on the user-selected sink and is never redirected to P610.

The TV layer uses WebRTC `extended_filter` and `delay_agnostic` processing. Tests with explicit fixed HDMI delays, mono conversion, and a virtual render sink did not consistently beat monitor mode, so the production path leaves delay estimation adaptive.

### Reference-aware anti-barge-in gate

AEC alone does not perfectly remove distant TV speech. The important observed failure mode was not false wake-up in standby; it was residual TV dialogue being transcribed as new user speech during an ADA response, causing Gemini to interrupt itself.

`backend/tv_reference_gate.py` adds a conservative second defense:

- `parec @DEFAULT_MONITOR@` samples the exact system render reference at 16 kHz without changing playback routing.
- While ADA is in standby, the gate learns a robust ratio between the delayed HDMI reference and the final post-AEC microphone residual. The current calibrated reference delay is 255 ms.
- The first user turn after `Okay Nabu` is never gated. When ADA starts the first reply, the reference gate becomes armed for the remainder of that active voice session, covering both ADA playback and the pauses between turns where TV dialogue previously created fake user messages.
- A frame consistent with the learned TV residual is replaced with equal-duration silence before being sent to Gemini. This preserves realtime timing while preventing TV dialogue from opening Gemini's speech detector.
- A clear near-end rise opens the gate and holds it open for 0.9 s so a user's quieter syllables are not chopped. The minimum near-end threshold is intentionally conservative (`0.006` normalized RMS) and can be overridden with environment variables.
- Provider/local `speech_started` and local Stop authorization also ignore frames currently classified as probable TV-only input.

On the retained real TV-only capture, the calibrated gate would reject about 92% of 43 ms TV-only frames at the conservative `0.006` threshold. A separate real user-speech capture contains large peaks well above the threshold; the 0.9 s hold preserves the rest of the utterance after the first near-end rise.

### Calibration findings

- The HDMI-to-P610 acoustic delay is large. In one continuous controlled run it stayed very stable around 254.5–255.4 ms across consecutive two-second windows.
- A frozen room FIR did not generalize well to later samples even though the delay stayed stable. The P610 itself appears to adapt its microphone gain/noise processing over time, so a one-time fixed impulse response is not a reliable production canceller.
- A custom streaming FDAF improved recorded TV leakage by roughly 3–4 dB in conservative 16 kHz simulations; the best observed native WebRTC live run was better, so the custom filter was not put into production.
- P610 exposes two capture channels but they were measured as bit-identical (`corr = 1.0`), so there is no independent microphone pair available for beamforming.

### Recovery and verification

The voice watchdog now treats `ada_tv_aec_source` as a separate health layer. If it disappears for repeated checks, it restarts `ada-tv-aec.service` first and then rebuilds `ada-p610-aec.service` so the cascade binds to the restored source. The P610 script no longer trusts a cached numeric PulseAudio module ID because PipeWire can recycle those IDs after restart.

Useful checks:

```bash
systemctl --user status ada-tv-aec.service ada-p610-aec.service ada-pi.service
wpctl status
curl -s http://127.0.0.1:18765/api/voice/status
```

The expected invariant remains: **system audio -> user-selected output (normally HDMI); TV reference -> TV-AEC only; ADA playback -> P610; final ADA microphone -> TV-AEC -> P610-self AEC -> Gemini.**
