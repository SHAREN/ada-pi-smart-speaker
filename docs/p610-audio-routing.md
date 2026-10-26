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

The voice watchdog also checks the actual `echo-cancel-playback` sink. Desktop audio controls and PipeWire stream-restore can move this live stream when the default output changes or after reboot. The watchdog therefore reads machine-readable `pactl -f json` state and treats the numeric/current sink as authoritative; `target.object=P610` alone is **not** accepted as proof, because a retained target can coexist with an actual HDMI route. If the current sink differs, only the ADA AEC render stream is moved back to the physical P610. The user's system default sink is never changed.

`p610-aec.sh` enforces the same invariant at creation time: after `module-echo-cancel` is loaded it waits for the module's playback stream to appear and explicitly pins that stream to the physical P610. `ada-pi.service` now `Requires=` and is `PartOf=` `ada-p610-aec.service`, so ADA does not start without the dedicated AEC endpoint and is restarted with the AEC layer instead of continuing on a stale/default output route.

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
- Wake semantics are strict: **Okay Nabu -> play the activation cue -> start listening**. The wake phrase, standby pre-roll and audio captured while the cue is playing are never forwarded to Gemini as user content, so wake alone must not make ADA answer first.
- **Stop** is a dedicated local microWakeWord detector on the post-AEC microphone path. The model still starts evaluating from its `0.60` cutoff, but an actual session close now requires the stronger calibrated confirmation threshold `0.82` plus the local speech/self-echo guards. This was added after two retained false-Stop captures where ADA's own/background audio crossed the old threshold. The latest raw Stop probability is exposed as `last_stop_probability` for live tuning.
- Normal conversation end (`Stop`, idle or semantic close) no longer creates a fresh Gemini session. The persistent Live session/resumption context is kept, so a later **Okay Nabu** continues the same conversation instead of forgetting it. For semantic close, Gemini is the intent classifier: if the current user utterance means they want to leave/end the conversation, `end_conversation` must be the model's first and only action for that turn; it must not speak «хорошо/заканчиваю/до встречи» before or instead of the tool. Backend accepts that tool call without a fixed-phrase allowlist, immediately clears queued speech, and plays only the local end cue — no spoken farewell. Gemini Live can emit the function call before `input_transcription` for the same voice turn, so backend also accepts an end tool during a bounded `ADA_SEMANTIC_END_ACTIVITY_GRACE_SECONDS` window (default 4 s) after a fresh local `activity_end`; the activity sequence is consumed once. A tool call with neither current/recent transcript nor a fresh just-finished voice turn still fails closed. The post-end wake guard is only `0.35 s`, and the duplicate-wake refractory window is `1.0 s` instead of 2 seconds; the audible end cue remains the main acoustic protection.
- Successful immediate Home Assistant household controls executed through `ha_call_service` or `ha_bulk_control` are returned to Gemini with `FunctionResponseScheduling.SILENT`. The action result stays in model context, but a successful light/switch/fan/cover/media/vacuum command does not trigger a spoken acknowledgement. A successful SILENT control still emits the internal `silent_turn_completed` activity marker, but local VAD `activity_start`/`activity_end` segments are **never** counted as semantic user turns and can no longer suppress later assistant speech. Gemini Live is authoritative for response ordering: every valid `response_started` is forwarded, while a genuinely newer user interruption is handled by Gemini's existing `response_interrupted` event, which clears current playback. This avoids the former failure mode where pauses inside one utterance accumulated phantom `pending_user_turns` and made normal replies such as «молодец» or a weather question disappear. The `pending_user_turns` status field remains only as a compatibility diagnostic and stays at zero. For successful on/off controls, HA returns an internal `_voice_feedback` marker based on the validated action actually sent; backend strips that marker from the model response and plays a local P610 cue instead: `sounds/control_on.wav` is a short rising chime and `sounds/control_off.wav` is a short falling chime. Simple `ha_bulk_control` on/off operations are dispatched with per-operation `timeout_seconds=0` (the MCP's documented no-wait dispatch mode) and `parallel=true`, then ADA performs its own bounded read-after-write confirmation against exact entity IDs with `ha_get_state` (default timeout `ADA_POWER_CONFIRM_TIMEOUT_SECONDS=2.0`, poll interval `ADA_POWER_CONFIRM_POLL_MS=150`). The success cue is emitted only after every confirmable target reports the requested `on`/`off` state. The confirmed tool result includes a compact `voice_state_snapshot` (`entity_id`, actual state, desired state, confirmation flag) that remains in the Gemini Live turn context as the current device-state cheat sheet. If the requested state is not observed before the deadline, the result becomes `success=false`, `unconfirmed=true`, no success cue is emitted, and Gemini receives the latest actual HA state so it cannot treat dispatch alone as completion. Blocked or failed dispatches do not play a success cue and remain speakable so ADA can report that the action did not happen. Fast-dispatched on/off results are also remembered for 20 seconds without overwriting Home Assistant's raw state. If a subsequent `ha_get_state` still reports the opposite value, the result includes `voice_recent_controls` with the desired state, reported state and age; Gemini must describe this as confirmation lag rather than confidently asserting the old HA value as physical truth. The mutation guard is intentionally narrow: Gemini owns natural-language intent and synonyms, while the guard rejects missing current-user context, raw/admin/security calls, non-allowlisted services, malformed/missing targets, explicitly deferred requests sent to an immediate-control tool, and one critical linguistic failure mode: factual state reports/corrections such as «свет выключен», «лампы горят» or first-person past reports such as «я всё выключил» cannot authorize an on/off mutation unless that same utterance also contains an explicit power request such as «выключи». This is not a general imperative allowlist; natural phrasing such as «сделай на кухне темно» remains Gemini's responsibility. It does not re-interpret calendar wording and does not rewrite Gemini's chosen on/off direction. In `ha_bulk_control` operations mode redundant top-level `action`/`selector` fields are still normalized away before the HA call so harmless model formatting noise does not force a retry. When asked what the user just said, ADA must report the transcript it actually received as «Я распознала это как …» rather than substituting the tool action it performed.
- Local weather is anchored to Home Assistant entity `weather.yandex_pogoda` by default (`ADA_WEATHER_ENTITY` can override it). On every Gemini Live connection ADA force-refreshes and compacts that entity into the system context as a weather cheat sheet: current condition/temperature/feels-like/wind/gust/bearing, the next 12 hourly forecast points, and up to 6 day/night forecast segments; large icon arrays and provider attribution are excluded. The snapshot also contains a deterministic Russian `voice_summary`, e.g. `Сейчас +11°. Сегодня от −1° до +11°. Осадки не ожидаются.` For a generic «погода/какая погода» request Gemini should speak only that short summary (1–3 sentences), without naming Yandex/Home Assistant or volunteering wind/other details unless explicitly asked. The snapshot carries `captured_at` plus the cache freshness window (`ADA_WEATHER_CACHE_TTL_SECONDS`, default 300 s). While fresh, Gemini answers directly from context; when stale or when the requested horizon exceeds the embedded forecast, it calls local `get_local_weather` rather than web search and receives the same `voice_summary`. The parser accepts direct and nested/bulk `ha_get_state` response shapes so a wrapper change cannot silently produce a cached snapshot full of nulls. Provider status exposes entity, cache state, capture time, TTL, current condition/temperature, hourly/day-night forecast counts, and the current `weather_voice_summary`.
- A compact transcript memory is also persisted under ignored runtime `data/voice-conversation-memory.json` (up to 40 recent user/ADA turns and about 16k characters). New Live sessions receive that history as prior conversational context, so a process restart or unavoidable clean reconnect does not erase the recent conversation. Raw microphone audio is not stored in this memory file.
- Ordinary speech during ADA TTS is **heard but does not automatically interrupt**. From `response_started` until the browser AudioWorklet reports the physical PCM queue `drained`, `assistant_playback_locked` prevents overlap speech from opening a live Gemini activity, but the microphone remains active and the filtered near-end audio is captured into a bounded overlap buffer (`ADA_OVERLAP_BUFFER_SECONDS`, default `20 s`). If the user finishes speaking before ADA finishes, that buffered utterance is replayed to Gemini as a complete user turn immediately after `drained`; if the user is still speaking at `drained`, the buffered prefix is replayed and the same Gemini activity remains open so the rest of the utterance continues live. Thus casual overlap does not cut ADA off, but Gemini still receives what the user said. A browser near-end VAD hit during playback also opens a short local keyword-only window (`ADA_LOCAL_INTERRUPT_WINDOW_SECONDS`, default `2.5 s`) for the dedicated Stop detector; echo-classified frames are fed to that detector as silence rather than repeatedly resetting its streaming state. If local Stop is detected while ADA is speaking, backend clears playback and sends an interrupt pulse to Gemini without closing the conversation; if Stop is detected while ADA is already quiet, it retains the old meaning of ending the active voice session. After `drained`, ordinary listening reopens only after `ADA_PLAYBACK_REOPEN_DELAY_MS` (default `500 ms`). The bundled `stop.tflite` is English-trained, so additional Russian interrupt words such as «стой», «подожди», «хватит», «тише» require dedicated local interrupt models rather than enabling general barge-in. A `response_completed` arriving after the PCM queue is already empty must emit `drained` immediately from the worklet's `flush` handler`; otherwise the early empty-buffer return can strand the speaking lock. Backend also has a fail-safe `ADA_PLAYBACK_DRAIN_TIMEOUT_SECONDS` (default `4 s`) that releases a post-response playback lock if `drained` telemetry is ever lost. Gemini Live reconnects are response-boundary aware: a `live_reconnecting` event immediately aborts any unfinished assistant response/playback lock, clears stale browser audio, resets stale manual activity and marks playback idle; once the replacement Live session is connected, any pending overlap utterance is replayed with `provider_recovered` instead of remaining trapped behind the dead response. As a second fail-safe, `ADA_RESPONSE_STALL_TIMEOUT_SECONDS` (default `12 s`) clears an active response if no assistant audio chunk arrives for that interval even though the socket still appears connected. Runtime status exposes `provider_reconnect_recovery_count` and `response_stall_timeout_count`.
- Proactive internal turns (habit alerts, visual checks, other `send_text_turn` / `send_habit_alert` calls) are serialized behind the same physical playback boundary by `LiveSessionManager`. They wait for playback idle instead of sending a new Gemini user turn while ADA is still speaking, so internal automation cannot create a second form of self-interruption.
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

## TV / system-audio reference cancellation — historical / matcher disabled

**Current production state (2026-10-24): the TV content matcher is fully disabled.** `ADA_TV_REFERENCE_ENABLED=false`, `ADA_TV_REFERENCE_OBSERVE=false`, `ADA_TV_ACTIVITY_GUARD=false`, and `ADA_TV_WAKE_GUARD=false`; the `ADA-TV-reference` / `ADA-TV-physical` matcher readers do not run and TV matching has no authority over wake, VAD, Stop, or user-audio forwarding. The material below is retained as investigation history and must not be treated as the active conversation policy.

The separate PipeWire/PulseAudio preprocessing route may still contain the previously built system-audio AEC layer; that is audio routing/AEC, not `backend/tv_reference_gate.py`. Do not re-enable the content matcher merely to address ADA self-interruption: the production solution is the TTS speaking lock described above.

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

### Voice-only Chromium window isolation

The always-on voice browser still needs a real Chromium page for microphone/WebRTC audio, but it must not appear as a normal desktop window or steal focus. Native Wayland/labwc did not reliably honor Chromium's `--start-minimized`, `--window-size=1,1`, or off-screen position, and the tiny app window could later become focused again.

`start.sh` therefore forces only the voice-only browser onto XWayland with `--ozone-platform=x11` and gives it a stable `WM_CLASS` via `--class=ada-pi-voice`. After launch an event-driven guard finds the top-level `ada-pi-voice` window, applies `hidden`, `skip_taskbar`, `skip_pager`, and `below`, then watches `_NET_WM_STATE` with `xprop -spy`. If Chromium ever removes the hidden state while trying to activate itself, the guard immediately restores it. The guard is tied to the browser lifetime and is cleaned up by the normal `start.sh` shutdown path.

The local `~/.config/labwc/rc.xml` also keeps matching fallback rules, but they are no longer relied on for correctness. This was validated by manually removing `_NET_WM_STATE_HIDDEN` from the live ADA window; the guard restored `HIDDEN` automatically while the browser websocket and Gemini Live session stayed connected.
