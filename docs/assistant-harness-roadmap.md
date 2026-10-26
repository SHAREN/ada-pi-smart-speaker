# ADA/Nabu — Product-grade Voice Assistant Harness roadmap

Status: active implementation roadmap
Umbrella issue: https://github.com/SHAREN/ada-pi-smart-speaker/issues/1
Scope: ADA/Nabu on Raspberry Pi 5 + Gemini Live + Home Assistant MCP + Plantronics P610

## Goal

Turn ADA/Nabu from a capable LLM voice stack into a predictable, practical household assistant with commercial-speaker UX:

> **Коротко понял → надёжно сделал → проверил → минимально подтвердил → закончил.**

The harness is a product-behavior layer, not a single giant prompt. It must separate lifecycle, response policy, smart-home action rules, technical tool instructions, persona, context and observability.

## Non-negotiable baseline — preserve, do not regress

The current implementation already has useful behavior that must be treated as baseline and integrated into the harness rather than replaced blindly:

- wake-word flow without spoken “я здесь / я слушаю” filler;
- P610 audio routing and AEC/self-echo protection;
- speaking/playback lock to prevent ordinary speech and self-voice from interrupting TTS;
- selective local interrupt path for `stop` while preserving normal speaking-lock behavior;
- silent on/off success cues for ordinary smart-home actions;
- Home Assistant action verification/read-after-write direction via `voice_state_snapshot`;
- compact weather `voice_summary` without provider attribution;
- semantic `end_conversation` path that can end silently;
- conversation memory/follow-up support already present in the backend;
- audio-debug traces, watchdogs and regression tests.

Any implementation from research must pass compatibility checks against these behaviors.

## Research intake rule

Deep-research findings are not copied into production because they sound good. Every reusable idea must go through this decision ledger:

| Field | Required decision |
|---|---|
| Source | Exact repo/doc/file/guide |
| What it solves | Concrete ADA problem |
| Type | prompt / runtime policy / engine / dataset / UX pattern / observability |
| Reusable as-is? | yes / adapt / no |
| Russian support | native / partial / irrelevant / missing |
| Runtime impact | latency, CPU/RAM, network, model-token cost |
| Interaction risk | stop, overlap, tool calls, HA state, session lifecycle |
| Privacy impact | audio/transcript retention, cloud dependency |
| Acceptance test | exact regression scenario |
| Decision | ADOPT / EXPERIMENT / REJECT |
| Status | planned / in progress / deployed / removed |

### Sources explicitly in scope for review

- Home Assistant Voice Recipes
- MCP Assist
- OHF-Voice intents
- OHF-Voice default-agent
- Home-LLM datasets/models/examples
- OpenVoiceOS
- Home Assistant Assist / LLM integration / Wyoming ecosystem
- Rhasspy
- Almond / Genie
- Pipecat voice agents
- LiveKit Agents
- Gemini Live prompting and lifecycle docs
- OpenAI Realtime voice-agent patterns where architecture is portable
- Alexa conversational-design guidance
- Rasa CALM
- Voiceflow conversation-design patterns

The research must distinguish **reusable instructions/policies** from engines, training data and marketing claims.

## Target harness architecture

```text
assistant_harness/
  core_behavior.md
  lifecycle.yaml
  response_policy.yaml
  smart_home_policy.yaml
  context_policy.yaml
  error_recovery.yaml
  persona.md
  capabilities/
    home_assistant.yaml
    weather.yaml
    timers.yaml
    media.yaml
  prompts/
    system.txt
    technical_tools.txt
  tests/
    scenarios/
    golden_responses/
```

Runtime prompt/config should be assembled from modules. Persona is the lowest-priority layer and must never override lifecycle, safety, tool discipline, silence policy or correctness.

## Target behavior layers

### 1. Core behavior

- Do the task first; do not narrate internal reasoning or tool use.
- Prefer the cheapest reliable path.
- Ask only for information that actually blocks execution.
- Do not repeat the user's request unless it resolves ambiguity.
- Do not append “что-нибудь ещё?”, “если хотите…” or similar chatter by default.
- Never invent successful action state.
- Spoken Russian should be brief, natural and free of UI/Markdown/technical jargon.

### 2. Conversation lifecycle

Explicit state machine instead of prompt-only assumptions:

```text
STANDBY
 -> WAKE_DETECTED
 -> LISTENING
 -> THINKING / TOOL_EXECUTION
 -> SPEAKING
 -> optional FOLLOWUP_WINDOW
 -> STANDBY
```

Exceptional transitions:

```text
LOCAL_INTERRUPT -> stop current playback -> LISTENING
STOP_WHEN_IDLE   -> STANDBY
SEMANTIC_END     -> STANDBY silently
TIMEOUT          -> STANDBY
FATAL_ERROR      -> short error -> STANDBY
```

Important ADA-specific rule: **ordinary user speech during assistant TTS must not automatically barge in**. Only dedicated local interrupt commands should break playback unless we later prove a safer selective-overlap design.

### 3. Response modes

Adopt a small set of explicit conversation modes, inspired by systems such as MCP Assist but adapted to ADA:

- `none` — one-shot task, then close;
- `smart` — default, follow-up only when context makes it useful;
- `always` — conversational mode until explicit end.

Default: `smart`.

### 4. Response classes

Runtime should classify outputs rather than making the model improvise delivery style every turn:

```text
SILENT_SUCCESS
CUE_SUCCESS
SHORT_CONFIRMATION
SHORT_ANSWER
CLARIFY
ERROR
LONG_FORM
```

Examples:

- “Выключи свет” -> action + readback + cue, no spoken filler.
- “Поставь 22 градуса” -> action + readback + “22 градуса.” when spoken confirmation is useful.
- “Какая температура дома?” -> one compact factual answer.
- ambiguous target -> one narrow question.
- failed readback -> short error, never success.

### 5. Smart-home action contract

For basic home control, deterministic/skills-first execution should be preferred when confidence is sufficient:

- on/off;
- brightness/temperature/position;
- open/close;
- state queries;
- timer/alarm if capability exists;
- scenes/scripts;
- simple media control;
- cancel/end.

LLM fallback is for free-form language, compound requests, explanations, ambiguous resolution and unsupported intents.

Every side-effecting capability should expose metadata conceptually equivalent to:

```yaml
risk: low | medium | high
confirmation: never | contextual | always
reversible: true | false
readback_required: true | false
response_class: CUE_SUCCESS
room_context_allowed: true
```

For state-changing HA actions:

```text
action accepted -> read actual state -> expected state matches -> success
```

Never map transport success alone to spoken success.

### 6. Context

Separate ephemeral conversational context from live device state.

Conversation context may retain:

- previous intent/domain;
- resolved target;
- important slots;
- last action/result;
- conversation mode.

Live Home Assistant state must be refreshed before a result depends on it. Do not answer from stale memory when a physical switch may have changed the device.

### 7. Selective interruption / overlap

Current product requirement:

- microphone/VAD may remain physically active during TTS;
- ordinary speech does **not** interrupt playback;
- local interrupt vocabulary (`стоп`, later `стой`, `подожди`, `хватит`, etc.) should interrupt current TTS without ending the whole session unless policy says otherwise;
- when assistant is idle, `stop` may end/close the active session;
- self-voice/TV/echo must not trigger semantic turns;
- local interrupt must remain independent of cloud/model availability.

Russian keyword spotting is a separate implementation item because the bundled Stop microWakeWord model is currently English-trained; do not reduce thresholds blindly and reintroduce false positives.

### 8. Error recovery

Errors must be product-facing, not infrastructure-facing:

- timeout -> short retryable message;
- unavailable entity -> identify the device/status simply;
- action accepted but state mismatch -> report failure;
- provider reconnect -> restore only safe/necessary conversation state;
- duplicate tool/model event -> do not duplicate side effects;
- long-running operation -> use a non-verbal cue before spoken “подождите”.

### 9. Persona

Nabu should feel calm, fast and natural, not overly friendly or chatbot-like.

Persona must not introduce filler such as:

- “Конечно”
- “Разумеется”
- “С удовольствием”
- “Я здесь”
- “Я вас слушаю”
- “Чем ещё могу помочь?”
- “Если хотите, я могу…”

These phrases are not globally banned when semantically useful, but they must never be default scaffolding.

## Implementation phases

### Phase 0 — research normalization

- collect verified findings from the deep-research report;
- fill the decision ledger;
- mark each finding ADOPT / EXPERIMENT / REJECT;
- explicitly record things we will **not** copy.

### Phase 1 — policy extraction and modular prompt

- extract existing behavior from `backend/main.py`, provider/tool instructions and docs;
- separate behavior / lifecycle / technical tools / persona;
- define response classes;
- add anti-filler and short-answer regression cases;
- preserve current wake, weather, cue and semantic-end behavior.

### Phase 2 — lifecycle controller

- formalize session state machine;
- formalize `none/smart/always` follow-up behavior;
- normalize response-completed vs physical audio-drained semantics;
- make reconnect/resume behavior explicit;
- define interrupt/end/timeout reasons in structured traces.

### Phase 3 — selective interruption

- validate current local Stop path end-to-end with live audio;
- implement Russian interrupt KWS or another local low-latency Russian command recognizer;
- add `стоп / стой / подожди / хватит` policy;
- preserve ordinary-speech lock during assistant TTS;
- quantify false interrupt rate before widening vocabulary.

### Phase 4 — deterministic smart-home fast path

- classify common HA commands before general reasoning;
- add room/satellite context where trustworthy;
- unify read-after-write contract;
- add risk/confirmation metadata;
- add idempotency/duplicate-action guard.

### Phase 5 — context and repair

- make compact follow-up context explicit;
- add ASR repair rules for incomplete/nearby phrases without over-guessing;
- resolve pronouns/domain follow-ups (“а на кухне?”, “а там?”) using bounded context;
- preserve live-state freshness.

### Phase 6 — proactive behavior

Only after reactive quality is strong:

- non-verbal notification cue first;
- ask/offer before speaking long unsolicited content;
- rate limits / quiet hours / relevance threshold;
- no repeated nagging;
- user can dismiss/snooze/disable categories.

### Phase 7 — observability and quality gates

Emit structured turn traces with at least:

```json
{
  "session_id": "...",
  "turn_id": "...",
  "transcript": "...",
  "intent_source": "deterministic|llm|fallback",
  "intent": "...",
  "targets": [],
  "tools": [],
  "action_result": "...",
  "readback": "...",
  "response_class": "...",
  "spoken_text": "...",
  "latency_ms": {},
  "end_reason": "completed|interrupt|stop|timeout|error"
}
```

Raw audio/transcript retention must remain diagnostic/configurable rather than mandatory.

## Required scenario matrix

At minimum, keep automated or reproducible tests for:

1. wake -> cue only, no spoken greeting;
2. simple light on/off -> verified action + cue;
3. requested setpoint -> concise confirmation;
4. state query -> compact factual response;
5. weather now -> compact `voice_summary`;
6. weather detailed -> expanded only when requested;
7. room context resolves target without useless question;
8. true ambiguity -> one narrow clarification;
9. action transport succeeds but readback mismatches -> error;
10. entity unavailable -> short useful error;
11. duplicate model/tool event -> no duplicate side effect;
12. normal user speech during TTS -> does not interrupt;
13. local `стоп` during TTS -> playback stops quickly, session remains usable;
14. local stop while idle -> session closes according to policy;
15. semantic “закончи разговор” -> silent end;
16. self-voice/echo -> no false user turn;
17. TV/background content -> no false semantic turn;
18. follow-up “а на кухне?” -> prior domain retained;
19. partial/uncertain ASR -> repair or narrow clarification, not confident hallucination;
20. reconnect during active session -> safe recovery without duplicate action;
21. tool timeout -> no false success;
22. long-running task -> non-verbal progress cue before chatter;
23. anti-filler corpus -> no default “конечно/я слушаю/что-нибудь ещё”;
24. proactive notification -> cue/permission pattern, no unsolicited monologue;
25. privacy mode -> diagnostics can be disabled without breaking core behavior.

## Metrics

Track p50 and p95, not individual anecdotes:

- wake -> listening ready;
- end-of-user-speech -> first tool call;
- end-of-user-speech -> first audio;
- end-of-user-speech -> action completion;
- interrupt keyword -> playback stopped;
- action verification latency;
- provider reconnect latency.

Quality counters:

- false interrupt rate;
- missed local interrupt rate;
- false wake rate;
- false success confirmations;
- duplicate side effects;
- clarification rate on actually unambiguous commands;
- chatter/filler per 100 turns;
- semantic-end success rate;
- tool failure converted to correct user-facing error;
- ASR ambiguity repaired vs hallucinated.

## Definition of done for the umbrella issue

- [ ] Harness is modular rather than one monolithic prompt.
- [ ] Lifecycle is explicit and covered by tests.
- [ ] Response classes are explicit.
- [ ] Common HA actions have a deterministic fast path where appropriate.
- [ ] Action success is tied to state verification when required.
- [ ] Simple home commands default to cue/silence instead of chatter.
- [ ] Selective local interruption works in Russian without enabling generic barge-in.
- [ ] Semantic end is reliable and quiet.
- [ ] Follow-up modes are explicit and default to `smart`.
- [ ] Short weather/status answers are preserved.
- [ ] No infrastructure/tool details leak into normal voice UX.
- [ ] Structured traces and latency/quality metrics exist.
- [ ] Deep-research findings have explicit ADOPT / EXPERIMENT / REJECT decisions.
- [ ] Regression matrix is green before each rollout.

## Implementation discipline

For each adopted research idea:

1. add it to the decision ledger;
2. define the observable behavior change;
3. add/extend a regression test first where practical;
4. implement the smallest compatible change;
5. run the focused suite, then full suite;
6. restart ADA only after tests pass;
7. verify one real voice scenario when the feature depends on audio/provider behavior;
8. update this roadmap and issue #1 with result, metrics and any rejected alternative.

This file is the durable implementation map. Issue #1 is the public tracking thread; code-level work can be split into child issues/PRs as phases become concrete.
