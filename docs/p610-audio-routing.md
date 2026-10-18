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

### Real-video content-matched anti-barge-in gate

AEC alone does not remove enough distant TV dialogue. A 29.9-second capture of the actual video playing through HDMI measured only about **2.16 dB** attenuation in the TV-AEC stage and **2.78 dB** across the complete two-stage AEC path. That is not enough to prevent intelligible actor dialogue from being detected as a user turn, so the production gate no longer decides from residual loudness and no longer needs pink-noise calibration.

`backend/tv_reference_gate.py` adds a content-based second defense:

- Two independent `parec` readers run at 16 kHz: one samples the exact HDMI/default-sink monitor and the other samples the **physical P610 microphone before software AEC**.
- Every 20 ms both streams are reduced to 12 coarse log-power frequency bands. Over a rolling 500 ms window, the gate compares the temporal pattern of those bands. Per-band normalization largely removes static speaker, room and microphone coloration.
- The matcher searches a small delay range around the current HDMI-to-P610 delay. Standby periods containing real TV playback continuously teach both the useful match-score distribution and the effective realtime delay. The delay seen by live `parec` streams can differ from an offline WAV correlation because the capture streams have their own buffering; this is why it is learned from the real running video instead of hard-coded from a pink-noise measurement.
- A **high content-match score** means the physical microphone is hearing the same content that HDMI just played, even if an actor suddenly becomes loud. That remains TV-only and is suppressed.
- A **low content-match score plus meaningful energy on the final post-AEC mic** is treated as independent near-end speech. The gate then stays open for 1.0 s so quieter syllables in the same utterance are not chopped.
- The first user turn after `Okay Nabu` is never gated or delayed. When ADA begins its first response, the content gate becomes armed for the remainder of that active voice session, including the pauses between turns where TV dialogue previously created fake user messages.
- Once armed, mic delivery to Gemini has a roughly 400 ms decision/pre-roll buffer. If an independent near-end syllable is detected, the queued original audio is released instead of losing the start of the phrase. TV-matched queued audio is replaced with equal-duration silence, preserving the realtime clock.
- Provider/local `speech_started` and local Stop authorization also respect the content-match verdict, so loud TV dialogue cannot authorize a false interruption merely by crossing an amplitude threshold.

On the retained real-video capture, the 500 ms physical-P610/HDMI content score had a TV-only median of about `0.583`, 10th percentile `0.461`, and minimum `0.327`. In an offline double-talk check made from that **same real TV recording** plus a retained real user-voice recording, strong near-end syllables drove the score far lower (for example around `0.14` at equal nominal level and around `0.07` at +6 dB). Production does not use those fixed numbers directly: it learns a conservative threshold from the current real TV content while in standby.

### Calibration findings

- On the current real-video capture, offline HDMI-to-P610 correlation was about 283 ms. Live capture timing can differ, so the content matcher searches and slowly adapts the effective delay from strong TV-only matches.
- A frozen room FIR did not generalize well even when delay was stable; P610/TV processing changes the effective transfer function enough that one static impulse response is not reliable.
- Speaker verification was evaluated as an alternative but rejected for gating: quiet physical/post-AEC TV fragments could overlap the user's speaker-embedding similarity. Content identity against the exact HDMI reference provided a safer discriminator for this installation.
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
