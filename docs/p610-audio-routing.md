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

## Planned TV / system-audio reference cancellation

The next useful extension is to cancel TV/system playback from the P610 microphone as well. This is standard far-end/reference acoustic echo cancellation: instead of treating the TV as unknown background noise, feed the exact digital system playback signal into an AEC stage as the render/reference signal and subtract the correlated acoustic copy that arrives at the microphone.

For this installation, the preferred design is a second AEC stage after the existing P610 self-echo stage:

1. Stage 1 keeps removing ADA's own P610 playback from the P610 microphone.
2. Stage 2 uses the monitor/reference signal of the user's normal system sink (for example HDMI feeding the TV) and removes that playback from the already-cleaned microphone.
3. The final source is what wake-word detection and Gemini receive.
4. System playback routing must remain user-controlled; the cancellation path must observe it rather than forcing the system default to another sink.

PipeWire's native echo-cancel module supports a monitor mode that can take the current sink monitor as the reference instead of requiring applications to render through a special virtual sink. WebRTC AEC3 performs adaptive delay estimation between render and capture, which is important because HDMI/TV processing adds latency and that latency can change.

Important engineering constraints:

- HDMI/TV playback and the USB P610 can have different latency and clock behavior, so delay/drift must be measured rather than hard-coded.
- The existing P610 playback reference and the TV reference are different acoustic paths. Prefer cascaded stages or a true multi-reference/multi-channel design rather than simply summing the two references before a single one-channel adaptive filter.
- Validate with recordings containing TV-only, user-only, and simultaneous TV+user speech. The goal is strong TV attenuation without damaging near-end speech or barge-in.
- Keep the current working routing invariant unchanged: ordinary system audio follows the user's selected output; ADA playback remains pinned to P610.
