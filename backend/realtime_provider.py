"""Gemini Live provider for the browser audio bridge."""

from __future__ import annotations

import abc
import asyncio
import logging
import os
import re
import time
import unicodedata
from datetime import datetime
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from .assistant_harness import (
    DEFAULT_ADA_INSTRUCTIONS,
    ResponseScenario,
    build_assistant_instructions,
    response_class_for_scenario,
    response_policy,
)
from .ha_mcp_voice import VOICE_TOOL_NAMES, call_voice_tool, get_voice_tool_declarations
from .speaker_clock import CLOCK_TOOL_DECLARATIONS, CLOCK_TOOL_NAMES, speaker_clock
from .web_tools import WEB_TOOL_DECLARATIONS, web_read, web_search

logger = logging.getLogger("voice.provider")

EXPRESSION_NAMES = (
    "neutral",
    "sassy",
    "amused",
    "skeptical",
    "annoyed",
    "mad",
    "concerned",
    "surprised",
    "mischievous",
    "serious",
    "alert",
)

_END_COMMANDS = {
    "закончи", "заканчивай", "закончить", "заверши", "завершить",
    "закрой", "закрыть", "прекрати", "прекратить", "перестань", "перестать",
    "хватит", "отключись", "отключиться",
}
_END_CONTEXT_STEMS = ("разговор", "диалог", "сесси", "слуш")
SEMANTIC_END_ACTIVITY_GRACE_SECONDS = max(
    1.0, float(os.environ.get("ADA_SEMANTIC_END_ACTIVITY_GRACE_SECONDS", "4.0"))
)


def _normalize_utterance(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return " ".join(text.split())


def _explicit_end_request(text: str) -> bool:
    """Legacy phrase-level semantic helper retained for diagnostics/tests."""
    normalized = _normalize_utterance(text)
    tokens = normalized.split()
    if not tokens:
        return False
    if normalized in {"стоп", "все стоп", "всё стоп"}:
        return True
    if "не" in tokens and any(stem in normalized for stem in ("заканч", "заверш", "закры", "прекрат", "перестан")):
        return False
    command_indexes = [index for index, token in enumerate(tokens) if token in _END_COMMANDS]
    if command_indexes and any(stem in normalized for stem in _END_CONTEXT_STEMS):
        if not all("не" in tokens[max(0, index - 2):index] for index in command_indexes):
            return True
    request_lead = any(
        phrase in normalized
        for phrase in ("хочу чтобы ты", "прошу тебя", "командую", "давай", "можешь")
    )
    end_verb = any(stem in normalized for stem in ("законч", "заверш", "закры", "прекрат", "отключ"))
    return bool(request_lead and end_verb and any(stem in normalized for stem in _END_CONTEXT_STEMS))


def _compact_weather_payload(payload: dict[str, Any], entity_id: str) -> dict[str, Any]:
    """Shrink HA weather state to the fields useful in voice context."""
    record: Any = payload.get("data", payload)
    if isinstance(record, dict) and isinstance(record.get("states"), dict):
        candidate = record["states"].get(entity_id)
        if isinstance(candidate, dict):
            record = candidate
    if isinstance(record, dict) and isinstance(record.get("data"), dict):
        record = record["data"]
    if not isinstance(record, dict):
        return {"success": False, "entity_id": entity_id, "error": "invalid weather payload"}
    if record.get("error"):
        return {"success": False, "entity_id": entity_id, "error": record.get("error")}
    attributes = record.get("attributes") if isinstance(record.get("attributes"), dict) else {}
    hourly_raw = attributes.get("forecast_hourly") if isinstance(attributes.get("forecast_hourly"), list) else []
    daily_raw = attributes.get("forecast_twice_daily") if isinstance(attributes.get("forecast_twice_daily"), list) else []

    hourly: list[dict[str, Any]] = []
    for item in hourly_raw[:12]:
        if not isinstance(item, dict):
            continue
        hourly.append({
            key: item.get(key)
            for key in (
                "datetime", "condition", "native_temperature", "native_apparent_temperature",
                "native_wind_speed", "native_wind_gust_speed", "wind_bearing",
            )
            if item.get(key) is not None
        })

    twice_daily: list[dict[str, Any]] = []
    for item in daily_raw[:6]:
        if not isinstance(item, dict):
            continue
        twice_daily.append({
            key: item.get(key)
            for key in (
                "datetime", "is_daytime", "condition", "native_temperature", "native_templow",
                "native_apparent_temperature", "native_wind_speed", "native_wind_gust_speed", "wind_bearing",
            )
            if item.get(key) is not None
        })

    return {
        "success": True,
        "entity_id": entity_id,
        "condition": record.get("state"),
        "temperature": attributes.get("temperature"),
        "apparent_temperature": attributes.get("apparent_temperature"),
        "temperature_unit": attributes.get("temperature_unit") or "°C",
        "wind_speed": attributes.get("wind_speed"),
        "wind_gust_speed": attributes.get("wind_gust_speed"),
        "wind_speed_unit": attributes.get("wind_speed_unit") or "km/h",
        "wind_bearing": attributes.get("wind_bearing"),
        "hourly": hourly,
        "twice_daily": twice_daily,
    }


def _signed_temperature(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    rounded = int(round(number))
    if rounded > 0:
        return f"+{rounded}°"
    if rounded < 0:
        return f"−{abs(rounded)}°"
    return "0°"


def _weather_voice_summary(snapshot: dict[str, Any]) -> str:
    """Build the short default spoken weather answer used by the voice assistant."""
    if not snapshot.get("success"):
        return ""
    current = _signed_temperature(snapshot.get("temperature"))
    captured_raw = str(snapshot.get("captured_at") or "")
    try:
        today = datetime.fromisoformat(captured_raw).date()
    except ValueError:
        today = datetime.now().astimezone().date()

    temperatures: list[float] = []
    today_conditions: list[str] = []
    for item in list(snapshot.get("twice_daily") or []):
        if not isinstance(item, dict):
            continue
        stamp = str(item.get("datetime") or "")
        try:
            if datetime.fromisoformat(stamp).date() != today:
                continue
        except ValueError:
            continue
        for key in ("native_templow", "native_temperature"):
            try:
                temperatures.append(float(item.get(key)))
            except (TypeError, ValueError):
                pass
        if item.get("condition"):
            today_conditions.append(str(item.get("condition")).lower())

    hourly_temperatures: list[float] = []
    for item in list(snapshot.get("hourly") or []):
        if not isinstance(item, dict):
            continue
        stamp = str(item.get("datetime") or "")
        try:
            if datetime.fromisoformat(stamp).date() != today:
                continue
        except ValueError:
            continue
        try:
            hourly_temperatures.append(float(item.get("native_temperature")))
        except (TypeError, ValueError):
            pass
        if item.get("condition"):
            today_conditions.append(str(item.get("condition")).lower())
    if not temperatures:
        temperatures.extend(hourly_temperatures)

    if current:
        try:
            temperatures.append(float(snapshot.get("temperature")))
        except (TypeError, ValueError):
            pass

    parts: list[str] = []
    if current:
        parts.append(f"Сейчас {current}.")
    if temperatures:
        low = _signed_temperature(min(temperatures))
        high = _signed_temperature(max(temperatures))
        if low and high:
            parts.append(f"Сегодня от {low} до {high}.")

    conditions = " ".join(today_conditions)
    if any(token in conditions for token in ("snow", "snowy", "sleet")):
        parts.append("Ожидается снег.")
    elif any(token in conditions for token in ("rain", "rainy", "pouring", "lightning", "hail")):
        parts.append("Ожидается дождь.")
    else:
        parts.append("Осадки не ожидаются.")
    return " ".join(parts)


def _weather_context_text(snapshot: dict[str, Any], *, ttl_seconds: float = 300.0) -> str:
    if not snapshot.get("success"):
        return ""
    captured = str(snapshot.get("captured_at") or "")
    unit = str(snapshot.get("temperature_unit") or "°C")
    current = (
        f"condition={snapshot.get('condition')}; temperature={snapshot.get('temperature')}{unit}; "
        f"feels_like={snapshot.get('apparent_temperature')}{unit}; "
        f"wind={snapshot.get('wind_speed')} {snapshot.get('wind_speed_unit')}; "
        f"gust={snapshot.get('wind_gust_speed')} {snapshot.get('wind_speed_unit')}; "
        f"bearing={snapshot.get('wind_bearing')}"
    )
    hourly_bits: list[str] = []
    for item in list(snapshot.get("hourly") or [])[:12]:
        if not isinstance(item, dict):
            continue
        stamp = str(item.get("datetime") or "")
        try:
            dt = datetime.fromisoformat(stamp)
            label = dt.strftime("%d.%m %H:%M")
        except ValueError:
            label = stamp
        hourly_bits.append(
            f"{label} temp={item.get('native_temperature')}°C feels={item.get('native_apparent_temperature')}°C "
            f"condition={item.get('condition')} wind={item.get('native_wind_speed')}"
        )

    daily_bits: list[str] = []
    for item in list(snapshot.get("twice_daily") or [])[:6]:
        if not isinstance(item, dict):
            continue
        stamp = str(item.get("datetime") or "")
        try:
            dt = datetime.fromisoformat(stamp)
            label = dt.strftime("%d.%m") + (" day" if item.get("is_daytime") else " night")
        except ValueError:
            label = stamp
        daily_bits.append(
            f"{label} temp={item.get('native_temperature')}°C low={item.get('native_templow')}°C "
            f"feels={item.get('native_apparent_temperature')}°C condition={item.get('condition')} "
            f"wind={item.get('native_wind_speed')}"
        )

    hourly_text = "; ".join(hourly_bits)
    daily_text = "; ".join(daily_bits)
    freshness_minutes = max(1, int(ttl_seconds // 60))
    voice_summary = str(snapshot.get("voice_summary") or _weather_voice_summary(snapshot))
    return (
        "Local weather cheat sheet for the user's default home location. "
        f"captured_at={captured}; fresh_for_about={freshness_minutes}min; current: {current}. "
        f"Default spoken weather summary: {voice_summary} "
        f"Hourly forecast: {hourly_text}. "
        f"Day/night forecast: {daily_text}. "
        "For a generic local-weather request such as 'погода' or 'какая погода', answer with the Default spoken weather summary only: "
        "keep it to one to three short sentences and do not add provider names, source attribution, Home Assistant details, wind, humidity, "
        "or other extra facts unless the user explicitly asks for them. Never say that the weather was provided by Yandex or another provider unless asked about the source. "
        "Use this weather context directly while it is fresh. If captured_at is older than the stated freshness window, or the user asks beyond the included forecast horizon, "
        "call get_local_weather before answering. Never use web_search for ordinary local weather while Home Assistant weather is available."
    )


@dataclass(slots=True)
class ProviderEvent:
    type: str
    data: dict[str, Any]


class RealtimeProvider(abc.ABC):
    @abc.abstractmethod
    async def connect(self, resumption_handle: str | None = None) -> None: ...

    @abc.abstractmethod
    async def send_audio(self, pcm16: bytes) -> None: ...

    @abc.abstractmethod
    async def send_activity_start(self) -> None: ...

    @abc.abstractmethod
    async def send_activity_end(self) -> None: ...

    @abc.abstractmethod
    async def send_video(self, jpeg: bytes) -> None: ...

    @abc.abstractmethod
    async def send_text_turn(self, text: str) -> None: ...

    @abc.abstractmethod
    def events(self) -> AsyncIterator[ProviderEvent]: ...

    @abc.abstractmethod
    async def close(self) -> None: ...


class GeminiLiveProvider(RealtimeProvider):
    """Gemini 3.1 Flash Live over Google's asynchronous Live API SDK."""

    def __init__(self, instructions: str | None = None, office_state_getter: Any = None,
                 habit_state_getter: Any = None) -> None:
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
        self.voice = os.environ.get("GEMINI_LIVE_VOICE", "Kore")
        self.language_code = os.environ.get("GEMINI_LIVE_LANGUAGE", "ru-RU").strip() or "ru-RU"
        self.input_language_codes = [
            code.strip()
            for code in os.environ.get("GEMINI_INPUT_LANGUAGE_CODES", "ru-RU").split(",")
            if code.strip()
        ] or ["ru-RU"]
        self.input_custom_vocabulary = [
            phrase.strip()
            for phrase in os.environ.get(
                "GEMINI_INPUT_CUSTOM_VOCABULARY", "Окей Набу,Набу,Okay Nabu,OK Nabu"
            ).split(",")
            if phrase.strip()
        ]
        for phrase in ("Бавлы", "Бавлах"):
            if phrase not in self.input_custom_vocabulary:
                self.input_custom_vocabulary.append(phrase)
        self.video_resolution = os.environ.get("GEMINI_VIDEO_RESOLUTION", "high").lower()
        self.voice_only = os.environ.get("ADA_UI_MODE", "full").strip().lower() == "voice-only"
        base_instructions = instructions or os.environ.get("GEMINI_LIVE_INSTRUCTIONS") or DEFAULT_ADA_INSTRUCTIONS
        self.instructions = build_assistant_instructions(
            base_instructions,
            voice_only=self.voice_only,
        )
        self._client: Any = None
        self._session_context: Any = None
        self._session: Any = None
        self._closed = False
        self._send_lock = asyncio.Lock()
        self.session_id = "-"
        self.resumption_handle: str | None = None
        self.go_away_time_left: str | None = None
        self.office_state_getter = office_state_getter
        self.habit_state_getter = habit_state_getter
        # Native Gemini Google Search grounding is optional because some API projects
        # reject Live setup with a separate grounding quota.  ADA's local web tools
        # are the default internet path and do not consume that quota.
        self.enable_google_search = os.environ.get("GEMINI_GOOGLE_SEARCH", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.enable_web_tools = os.environ.get("ADA_WEB_TOOLS", "true").strip().lower() in {"1", "true", "yes", "on"}
        self._last_user_transcript = ""
        self._last_user_transcript_at = 0.0
        self._voice_activity_sequence = 0
        self._last_activity_start_at = 0.0
        self._last_activity_end_at = 0.0
        self._last_activity_end_sequence = 0
        self._last_semantic_end_activity_sequence = 0
        self._ha_tool_count = 0
        self._web_tool_calls = 0
        self._ha_tool_calls = 0
        self.weather_entity = os.environ.get("ADA_WEATHER_ENTITY", "weather.yandex_pogoda").strip() or "weather.yandex_pogoda"
        self.weather_cache_ttl_seconds = max(
            60.0, float(os.environ.get("ADA_WEATHER_CACHE_TTL_SECONDS", "300"))
        )
        self._weather_cache: dict[str, Any] | None = None
        self._weather_cache_at = 0.0

    async def _get_local_weather(self, *, force: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        if (
            not force
            and self._weather_cache is not None
            and now - self._weather_cache_at < self.weather_cache_ttl_seconds
        ):
            return dict(self._weather_cache)
        if not os.environ.get("HA_MCP_URL", "").strip():
            return {"success": False, "entity_id": self.weather_entity, "error": "Home Assistant MCP is not configured"}
        try:
            raw = await asyncio.wait_for(
                call_voice_tool(
                    "ha_get_state",
                    {"entity_id": self.weather_entity, "fields": ["state", "attributes"]},
                    transcript="какая погода",
                ),
                timeout=4.0,
            )
            snapshot = _compact_weather_payload(raw, self.weather_entity)
            snapshot["captured_at"] = datetime.now().astimezone().isoformat(timespec="minutes")
            if snapshot.get("success"):
                snapshot["voice_summary"] = _weather_voice_summary(snapshot)
                self._weather_cache = dict(snapshot)
                self._weather_cache_at = now
            return snapshot
        except Exception as exc:
            logger.warning("session=%s local weather refresh failed: %s", self.session_id, exc)
            if self._weather_cache is not None:
                stale = dict(self._weather_cache)
                stale["stale"] = True
                stale["refresh_error"] = f"{type(exc).__name__}: {exc}"
                return stale
            return {"success": False, "entity_id": self.weather_entity, "error": f"weather refresh failed: {type(exc).__name__}: {exc}"}

    def _effective_user_transcript(self, current: str, *, ttl_seconds: float = 6.0) -> str:
        text = str(current or "").strip()
        if text:
            return text
        if self._last_user_transcript and (time.monotonic() - self._last_user_transcript_at) <= ttl_seconds:
            return self._last_user_transcript
        return ""

    def _has_fresh_voice_turn_for_semantic_end(self) -> bool:
        """Allow semantic close when Gemini's tool call races ahead of ASR text."""
        now = time.monotonic()
        if self._last_activity_end_sequence <= self._last_semantic_end_activity_sequence:
            return False
        if self._last_activity_end_sequence != self._voice_activity_sequence:
            return False
        if self._last_activity_start_at <= 0.0 or self._last_activity_end_at < self._last_activity_start_at:
            return False
        age = now - self._last_activity_end_at
        return 0.0 <= age <= SEMANTIC_END_ACTIVITY_GRACE_SECONDS

    async def connect(self, resumption_handle: str | None = None) -> None:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        proxy_url = os.environ.get("GEMINI_PROXY_URL", "").strip()
        if proxy_url:
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    async_client_args={"proxy": proxy_url},
                ),
            )
            logger.info("session=%s Gemini Live proxy enabled", self.session_id)
        else:
            self._client = genai.Client(api_key=self.api_key)

        system_instruction = self.instructions
        if os.environ.get("HA_MCP_URL", "").strip():
            snapshot = await self._get_local_weather(force=True)
            weather_context = _weather_context_text(snapshot, ttl_seconds=self.weather_cache_ttl_seconds)
            if weather_context:
                system_instruction += "\n\n" + weather_context
                logger.info(
                    "session=%s weather context primed entity=%s captured_at=%s",
                    self.session_id,
                    self.weather_entity,
                    snapshot.get("captured_at"),
                )

        config = {
            "response_modalities": ["AUDIO"],
            "media_resolution": (
                types.MediaResolution.MEDIA_RESOLUTION_HIGH
                if self.video_resolution == "high"
                else types.MediaResolution.MEDIA_RESOLUTION_LOW
            ),
            "system_instruction": system_instruction,
            "input_audio_transcription": {
                "language_codes": self.input_language_codes,
                "custom_vocabulary": self.input_custom_vocabulary,
            },
            "output_audio_transcription": {},
            "speech_config": {
                "language_code": self.language_code,
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": self.voice},
                }
            },
            "realtime_input_config": {
                # Server-side VAD previously re-opened an already-finished user
                # utterance while the assistant was speaking, causing Gemini to
                # interrupt itself.  ADA already has a local VAD plus a render-aware
                # self-echo gate, so make those local decisions authoritative.
                "activity_handling": types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
                "automatic_activity_detection": {
                    "disabled": True,
                },
            },
            # Native audio consumes context quickly. Sliding-window compression
            # prevents an extended voice conversation from exhausting the
            # session context while retaining recent conversational history.
            "context_window_compression": types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
            "session_resumption": types.SessionResumptionConfig(
                handle=resumption_handle,
            ),
            "tools": [{
                "function_declarations": [{
                    "name": "set_facial_expression",
                    "description": (
                        "Changes Ada's visible facial expression without interrupting speech. "
                        "Select the expression matching the tone of Ada's current response."
                    ),
                    "behavior": types.Behavior.NON_BLOCKING,
                    "parameters_json_schema": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "enum": list(EXPRESSION_NAMES),
                                "description": "The facial expression Ada should display.",
                            }
                        },
                        "required": ["expression"],
                        "additionalProperties": False,
                    },
                }, {
                    "name": "get_office_state",
                    "description": (
                        "Returns current read-only Home Assistant office-light state, "
                        "whether the user is home, recent local office occupancy, and any "
                        "active five-minute or latched habit condition. Use this when asked "
                        "about office lights, occupancy, or whether lights were left on."
                    ),
                    "behavior": types.Behavior.NON_BLOCKING,
                    "parameters_json_schema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                }, {
                    "name": "report_habit_observation",
                    "description": (
                        "Reports the structured result of a water or junk-food camera "
                        "observation requested by the ADA backend. Call this only when a "
                        "backend prompt supplies a challenge_id, after reviewing the full "
                        "observation window. Mere containers or food presence are not consumption."
                    ),
                    "behavior": types.Behavior.NON_BLOCKING,
                    "parameters_json_schema": {
                        "type": "object",
                        "properties": {
                            "challenge_id": {"type": "string"},
                            "habit_key": {"type": "string", "enum": ["not_drinking_enough_water", "junk_food"]},
                            "observed": {"type": "boolean"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "reason": {"type": "string"},
                            "item_identified": {"type": "string", "description": "For junk-food checks, the specific visible food or drink; empty when none is identifiable."},
                            "consumption_visible": {"type": "boolean", "description": "For junk-food checks, true only when actual eating or drinking is visibly confirmed."},
                            "classified_unhealthy": {"type": "boolean", "description": "For junk-food checks, true only when the identified item clearly belongs to the configured unhealthy categories."},
                        },
                        "required": ["challenge_id", "habit_key", "observed", "confidence", "reason"],
                        "additionalProperties": False,
                    },
                }, {
                    "name": "get_habit_status",
                    "description": (
                        "Returns every habit Ada tracks, including habits with no occurrences, "
                        "their lifecycle status, rolling seven-day occurrences and days, and "
                        "current monitor state and progress. Use this whenever the user asks "
                        "what habits are tracked or how their habits are progressing."
                    ),
                    "behavior": types.Behavior.NON_BLOCKING,
                    "parameters_json_schema": {"type": "object", "properties": {}, "additionalProperties": False},
                }, {
                    "name": "get_local_weather",
                    "description": (
                        "Returns the user's local current weather, compact forecast, and a short voice_summary from the configured "
                        "Home Assistant weather entity. For a generic request such as 'погода' or 'какая погода', speak only voice_summary "
                        "without provider/source attribution or extra details. Use this tool when the embedded weather context may be stale "
                        "or the user asks beyond its shown horizon. Do not use web_search for ordinary local weather while this tool is available."
                    ),
                    "behavior": types.Behavior.NON_BLOCKING,
                    "parameters_json_schema": {"type": "object", "properties": {}, "additionalProperties": False},
                }, {
                    "name": "end_conversation",
                    "description": (
                        "Call when the user's current utterance semantically expresses a desire to end, leave, "
                        "close, or stop the current voice conversation/listening session, including natural or "
                        "indirect wording. Do not require a fixed phrase. Never use this to stop media, timers, "
                        "devices, to discuss a previous conversation ending, or merely because a normal request "
                        "is complete. Do not call it for nonsense or genuinely ambiguous audio."
                    ),
                    "behavior": types.Behavior.NON_BLOCKING,
                    "parameters_json_schema": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Short semantic reason the user wants to end this voice session."
                            }
                        },
                        "additionalProperties": False,
                    },
                }, {
                    "name": "thinking_signal",
                    "description": (
                        "Call once before work expected to leave the user waiting over about one second because "
                        "it needs search, multiple tools, or nontrivial analysis. Never call it for a simple reply."
                    ),
                    "behavior": types.Behavior.NON_BLOCKING,
                    "parameters_json_schema": {"type": "object", "properties": {}, "additionalProperties": False},
                }]
            }],
        }

        if self.voice_only:
            declarations = config["tools"][0]["function_declarations"]
            config["tools"][0]["function_declarations"] = [
                item for item in declarations if item.get("name") != "set_facial_expression"
            ]

        config["tools"][0]["function_declarations"].extend(CLOCK_TOOL_DECLARATIONS)

        if self.enable_web_tools:
            config["tools"][0]["function_declarations"].extend(WEB_TOOL_DECLARATIONS)

        # Extend the local ADA functions with a deliberately restricted Home
        # Assistant MCP catalog.  Discovery failure must not take down ordinary
        # voice/chat or internet search.
        if os.environ.get("HA_MCP_URL", "").strip():
            try:
                ha_declarations = await asyncio.wait_for(
                    get_voice_tool_declarations(), timeout=6.0
                )
                config["tools"][0]["function_declarations"].extend(ha_declarations)
                self._ha_tool_count = len(ha_declarations)
            except Exception as exc:
                self._ha_tool_count = 0
                logger.warning("session=%s HA MCP catalog unavailable: %s", self.session_id, exc)

        # Google Search is handled natively by Gemini Live and can coexist with
        # custom function declarations in the same session.
        if self.enable_google_search:
            config["tools"].insert(0, {"google_search": {}})

        self._session_context = self._client.aio.live.connect(
            model=self.model,
            config=config,
        )
        self._session = await self._session_context.__aenter__()
        logger.info(
            "session=%s Gemini Live session connected (model=%s, voice=%s, language=%s, video=%s, internet_tools=%s, google_grounding=%s, ha_tools=%d)",
            self.session_id, self.model,
            self.voice,
            self.language_code,
            self.video_resolution,
            self.enable_web_tools,
            self.enable_google_search,
            self._ha_tool_count,
        )

    async def send_audio(self, pcm16: bytes) -> None:
        if self._session is None:
            raise RuntimeError("provider is not connected")
        async with self._send_lock:
            await self._session.send_realtime_input(
                audio=types.Blob(data=pcm16, mime_type="audio/pcm;rate=16000")
            )

    async def send_activity_start(self) -> None:
        if self._session is None:
            raise RuntimeError("provider is not connected")
        async with self._send_lock:
            await self._session.send_realtime_input(activity_start=types.ActivityStart())
        self._voice_activity_sequence += 1
        self._last_activity_start_at = time.monotonic()

    async def send_activity_end(self) -> None:
        if self._session is None:
            raise RuntimeError("provider is not connected")
        async with self._send_lock:
            await self._session.send_realtime_input(activity_end=types.ActivityEnd())
        self._last_activity_end_at = time.monotonic()
        self._last_activity_end_sequence = self._voice_activity_sequence

    async def interrupt_response(self) -> None:
        """Interrupt current model speech without creating a semantic user turn."""
        if self._session is None:
            raise RuntimeError("provider is not connected")
        async with self._send_lock:
            await self._session.send_realtime_input(activity_start=types.ActivityStart())
            await self._session.send_realtime_input(activity_end=types.ActivityEnd())

    async def send_video(self, jpeg: bytes) -> None:
        if self._session is None:
            raise RuntimeError("provider is not connected")
        async with self._send_lock:
            await self._session.send_realtime_input(
                video=types.Blob(data=jpeg, mime_type="image/jpeg")
            )

    async def send_text_turn(self, text: str) -> None:
        if self._session is None:
            raise RuntimeError("provider is not connected")
        async with self._send_lock:
            await self._session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=text)],
                ),
                turn_complete=True,
            )

    async def send_habit_alert(self, jpeg: bytes, alert: str) -> None:
        if self._session is None:
            raise RuntimeError("provider is not connected")
        async with self._send_lock:
            await self._session.send_client_content(
                turns=types.Content(role="user", parts=[
                    types.Part.from_text(text=alert),
                    types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                ]),
                turn_complete=True,
            )

    async def events(self) -> AsyncIterator[ProviderEvent]:
        if self._session is None:
            raise RuntimeError("provider is not connected")

        response_active = False
        input_transcript = ""
        response_started_at = 0.0
        response_audio_chunks = 0
        response_audio_bytes = 0
        response_last_audio_at = 0.0
        response_last_audio_duration = 0.0
        response_max_audio_gap = 0.0
        response_max_jitter_deficit = 0.0

        while not self._closed:
            async for message in self._session.receive():
                update = message.session_resumption_update
                if update and update.resumable and update.new_handle:
                    self.resumption_handle = update.new_handle
                if message.go_away:
                    self.go_away_time_left = str(message.go_away.time_left or "")
                    yield ProviderEvent("go_away", {"time_left": self.go_away_time_left})
                    return

                # Consume input transcription before function calls from the same
                # server message.  This gives the mutation guard the freshest
                # possible wording and fails closed when Gemini has not emitted
                # transcription yet.
                content = message.server_content
                if content is not None:
                    transcription = content.input_transcription
                    if transcription and transcription.text:
                        input_transcript += transcription.text
                        current_text = input_transcript.strip()
                        if current_text:
                            self._last_user_transcript = current_text
                            self._last_user_transcript_at = time.monotonic()

                tool_call = message.tool_call
                if tool_call and tool_call.function_calls:
                    function_responses = []
                    silent_control_completed = False
                    control_feedback_kind = ""
                    for call in tool_call.function_calls:
                        response_scheduling = types.FunctionResponseScheduling.WHEN_IDLE
                        logger.info(
                            "session=%s function_call received id=%s name=%s args=%r",
                            self.session_id, call.id, call.name, call.args,
                        )
                        requested = (call.args or {}).get("expression")
                        if call.name == "set_facial_expression" and requested in EXPRESSION_NAMES:
                            yield ProviderEvent("expression", {"name": requested})
                            result = {"output": f"Ada is now {requested}"}
                        elif call.name == "get_office_state" and self.office_state_getter is not None:
                            result = {"output": self.office_state_getter()}
                        elif call.name == "report_habit_observation":
                            args = dict(call.args or {})
                            yield ProviderEvent("habit_observation", args)
                            result = {"output": "Observation delivered to the habit monitor"}
                        elif call.name == "get_habit_status" and self.habit_state_getter is not None:
                            result = {"output": self.habit_state_getter()}
                        elif call.name == "get_local_weather":
                            self._ha_tool_calls += 1
                            result = await self._get_local_weather()
                        elif call.name == "end_conversation":
                            reason = " ".join(str((call.args or {}).get("reason", "")).split())[:240]
                            current_utterance = self._effective_user_transcript(input_transcript)
                            fresh_voice_turn = self._has_fresh_voice_turn_for_semantic_end()
                            # Gemini is the semantic classifier. Live API tool calls can arrive
                            # before input_transcription for the same just-finished voice turn,
                            # so a fresh local activity boundary is also valid evidence that this
                            # tool call belongs to a real current utterance. This avoids the race
                            # where ADA says she will end the conversation but the backend rejects
                            # the tool because ASR text has not arrived yet.
                            if current_utterance.strip() or fresh_voice_turn:
                                if fresh_voice_turn:
                                    self._last_semantic_end_activity_sequence = self._last_activity_end_sequence
                                transcript_for_log = current_utterance[:300] if current_utterance.strip() else "<ASR pending for fresh voice turn>"
                                logger.info(
                                    "session=%s accepting semantic end tool transcript=%r fresh_voice_turn=%s model_reason=%r",
                                    self.session_id,
                                    transcript_for_log,
                                    fresh_voice_turn,
                                    reason,
                                )
                                yield ProviderEvent("semantic_end_requested", {"reason": reason})
                                result = {
                                    "success": True,
                                    "instruction": "Close silently. Do not generate any spoken farewell or follow-up.",
                                }
                                response_class = response_class_for_scenario(
                                    ResponseScenario.SEMANTIC_END_SUCCESS
                                )
                                response_scheduling = (
                                    types.FunctionResponseScheduling.SILENT
                                    if not response_policy(response_class).speak
                                    else types.FunctionResponseScheduling.WHEN_IDLE
                                )
                            else:
                                logger.warning(
                                    "session=%s semantic end tool ignored without a current user transcript or fresh voice turn reason=%r",
                                    self.session_id,
                                    reason,
                                )
                                result = {
                                    "success": False,
                                    "blocked": True,
                                    "error": "No current user utterance or fresh voice turn is available to justify ending the voice session.",
                                }
                        elif call.name == "thinking_signal":
                            yield ProviderEvent("thinking_signal", {})
                            result = {"success": True}
                        elif call.name in CLOCK_TOOL_NAMES:
                            result = await speaker_clock.call_tool(
                                str(call.name),
                                dict(call.args or {}),
                            )
                        elif call.name == "web_search":
                            self._web_tool_calls += 1
                            args = dict(call.args or {})
                            result = await web_search(
                                str(args.get("query") or ""),
                                int(args.get("max_results") or 5),
                                str(args.get("kind") or "text"),
                            )
                        elif call.name == "web_read":
                            self._web_tool_calls += 1
                            result = await web_read(str((call.args or {}).get("url") or ""))
                        elif call.name in VOICE_TOOL_NAMES:
                            self._ha_tool_calls += 1
                            result = await call_voice_tool(
                                str(call.name),
                                dict(call.args or {}),
                                transcript=self._effective_user_transcript(input_transcript),
                            )
                            # Immediate household controls should feel like a switch,
                            # not a conversation. On successful device mutation, keep
                            # the tool result in context but do not trigger spoken ack.
                            if (
                                call.name in {"ha_call_service", "ha_bulk_control"}
                                and isinstance(result, dict)
                                and result.get("success", True) is not False
                                and not result.get("blocked")
                                and not result.get("error")
                            ):
                                feedback_kind = str(result.pop("_voice_feedback", "") or "")
                                if feedback_kind in {"on", "off"}:
                                    response_class = response_class_for_scenario(
                                        ResponseScenario.SIMPLE_HOME_ACTION_SUCCESS
                                    )
                                    delivery_policy = response_policy(response_class)
                                    if delivery_policy.cue:
                                        control_feedback_kind = feedback_kind
                                    silent_control_completed = not delivery_policy.speak
                                    response_scheduling = (
                                        types.FunctionResponseScheduling.SILENT
                                        if not delivery_policy.speak
                                        else types.FunctionResponseScheduling.WHEN_IDLE
                                    )
                                else:
                                    # Preserve the current baseline for other successful
                                    # mutations until their spoken response classes are
                                    # wired in a separately reviewed slice.
                                    silent_control_completed = True
                                    response_scheduling = types.FunctionResponseScheduling.SILENT
                        else:
                            result = {"error": "Unsupported or unavailable function"}
                        function_responses.append(types.FunctionResponse(
                            id=call.id,
                            name=call.name or "set_facial_expression",
                            response=result,
                            # The expression tool often arrives before audio.
                            # WHEN_IDLE lets Gemini continue the spoken reply;
                            # SILENT would add the result to context without
                            # triggering generation, leaving Ada mute.
                            scheduling=response_scheduling,
                        ))
                        logger.info(
                            "session=%s function_call result id=%s name=%s result=%r",
                            self.session_id, call.id, call.name, result,
                        )
                    await self._session.send_tool_response(
                        function_responses=function_responses
                    )
                    logger.info(
                        "session=%s function_call responses sent count=%d",
                        self.session_id, len(function_responses),
                    )
                    if control_feedback_kind in {"on", "off"}:
                        yield ProviderEvent("control_feedback", {"kind": control_feedback_kind})
                    if silent_control_completed:
                        yield ProviderEvent("silent_turn_completed", {})

                if content is None:
                    continue

                if content.interrupted:
                    elapsed = time.monotonic() - response_started_at if response_started_at else 0.0
                    logger.warning(
                        "session=%s assistant interrupted response_active=%s age_ms=%d audio_chunks=%d "
                        "audio_bytes=%d pending_input_transcript=%r",
                        self.session_id, response_active, elapsed * 1000, response_audio_chunks,
                        response_audio_bytes, input_transcript.strip(),
                    )
                    response_active = False
                    yield ProviderEvent("response_interrupted", {})
                    # Gemini 3.1 can include several content parts in one event.
                    # Any audio/transcript accompanying an interruption belongs
                    # to the cancelled response and must not restart playback.
                    continue

                output_transcription = content.output_transcription
                if output_transcription and output_transcription.text:
                    if not response_active:
                        response_active = True
                        response_started_at = time.monotonic()
                        response_audio_chunks = 0
                        response_audio_bytes = 0
                        response_last_audio_at = 0.0
                        response_last_audio_duration = 0.0
                        response_max_audio_gap = 0.0
                        response_max_jitter_deficit = 0.0
                        yield ProviderEvent("response_started", {})
                    yield ProviderEvent(
                        "assistant_transcript_delta",
                        {"text": output_transcription.text},
                    )

                model_turn = content.model_turn
                if model_turn:
                    for part in model_turn.parts or []:
                        inline_data = part.inline_data
                        if inline_data and inline_data.data:
                            if not response_active:
                                response_active = True
                                response_started_at = time.monotonic()
                                response_audio_chunks = 0
                                response_audio_bytes = 0
                                response_last_audio_at = 0.0
                                response_last_audio_duration = 0.0
                                response_max_audio_gap = 0.0
                                response_max_jitter_deficit = 0.0
                                yield ProviderEvent("response_started", {})
                            if len(inline_data.data) <= 2:
                                logger.warning(
                                    "session=%s ignoring zero-duration assistant PCM sentinel bytes=%d",
                                    self.session_id,
                                    len(inline_data.data),
                                )
                                continue
                            now = time.monotonic()
                            chunk_gap = (now - response_last_audio_at) if response_last_audio_at else 0.0
                            jitter_deficit = max(0.0, chunk_gap - response_last_audio_duration) if response_last_audio_at else 0.0
                            if response_last_audio_at:
                                response_max_audio_gap = max(response_max_audio_gap, chunk_gap)
                                response_max_jitter_deficit = max(response_max_jitter_deficit, jitter_deficit)
                            response_last_audio_at = now
                            response_audio_chunks += 1
                            response_audio_bytes += len(inline_data.data)
                            chunk_audio_duration = len(inline_data.data) / (24000.0 * 2.0)
                            response_last_audio_duration = chunk_audio_duration
                            yield ProviderEvent("audio", {
                                "pcm16": inline_data.data,
                                "sequence": response_audio_chunks,
                                "provider_gap_ms": round(chunk_gap * 1000.0, 1),
                                "provider_jitter_deficit_ms": round(jitter_deficit * 1000.0, 1),
                                "chunk_audio_ms": round(chunk_audio_duration * 1000.0, 1),
                                "provider_elapsed_ms": round((now - response_started_at) * 1000.0, 1) if response_started_at else 0.0,
                            })

                if content.turn_complete:
                    transcript = input_transcript.strip()
                    if transcript:
                        yield ProviderEvent("user_transcript", {"text": transcript})
                    input_transcript = ""
                    if response_active:
                        elapsed = time.monotonic() - response_started_at if response_started_at else 0.0
                        audio_seconds = response_audio_bytes / (24000.0 * 2.0) if response_audio_bytes else 0.0
                        delivery_ratio = (elapsed / audio_seconds) if audio_seconds > 0 else 0.0
                        log = logger.warning if (response_max_jitter_deficit >= 0.25 or delivery_ratio >= 1.8) else logger.info
                        log(
                            "session=%s assistant response completed duration_ms=%d audio_chunks=%d audio_bytes=%d audio_ms=%d max_chunk_gap_ms=%d max_jitter_deficit_ms=%d delivery_ratio=%.2f",
                            self.session_id, elapsed * 1000, response_audio_chunks, response_audio_bytes,
                            audio_seconds * 1000, response_max_audio_gap * 1000,
                            response_max_jitter_deficit * 1000, delivery_ratio,
                        )
                        yield ProviderEvent("response_completed", {
                            "duration_ms": round(elapsed * 1000.0, 1),
                            "audio_chunks": response_audio_chunks,
                            "audio_bytes": response_audio_bytes,
                            "audio_ms": round(audio_seconds * 1000.0, 1),
                            "max_chunk_gap_ms": round(response_max_audio_gap * 1000.0, 1),
                            "max_jitter_deficit_ms": round(response_max_jitter_deficit * 1000.0, 1),
                            "delivery_ratio": round(delivery_ratio, 3),
                        })
                    response_active = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._session_context is not None:
            try:
                await asyncio.wait_for(
                    self._session_context.__aexit__(None, None, None), timeout=3
                )
            except (TimeoutError, Exception):
                logger.debug("Gemini session close did not complete cleanly", exc_info=True)
        if self._client is not None:
            try:
                await self._client.aio.aclose()
            except Exception:
                logger.debug("Gemini client close did not complete cleanly", exc_info=True)


def create_provider(instructions: str | None = None, office_state_getter: Any = None,
                    habit_state_getter: Any = None) -> RealtimeProvider:
    return GeminiLiveProvider(
        instructions=instructions,
        office_state_getter=office_state_getter,
        habit_state_getter=habit_state_getter,
    )
