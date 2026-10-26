"""Voice-safe Home Assistant MCP bridge for ADA Pi.

The underlying HA-MCP server exposes a large administrative tool catalog.  A
voice assistant must never inherit that catalog wholesale, so this module
publishes a small allowlist and applies a second deterministic safety gate to
all state-changing calls.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger("voice.ha_mcp")

# Only these MCP tools are ever visible to Gemini.
VOICE_TOOL_NAMES = {
    "ha_search",
    "ha_get_state",
    "ha_get_entity",
    "ha_get_device",
    "ha_get_history",
    "ha_list_floors_areas",
    "ha_get_todo",
    "ha_set_todo_item",
    "ha_config_get_calendar_events",
    "ha_config_set_calendar_event",
    "ha_config_remove_calendar_event",
    "ha_bulk_control",
    "ha_call_service",
}

MUTATING_TOOLS = {
    "ha_set_todo_item",
    "ha_config_set_calendar_event",
    "ha_config_remove_calendar_event",
    "ha_bulk_control",
    "ha_call_service",
}

# Low-risk household domains that the voice bridge may mutate. Security/admin
# domains such as lock/alarm_control_panel/homeassistant/hassio are deliberately
# absent; this is the core deterministic safety boundary.
VOICE_CONTROL_DOMAINS = {
    "light", "switch", "fan", "cover", "climate", "media_player", "vacuum",
    "humidifier", "water_heater", "input_boolean", "input_number", "number",
    "select", "input_select", "button", "input_button", "scene", "script",
    "automation",
}

SERVICE_ALLOWLIST: dict[str, set[str]] = {
    "light": {"turn_on", "turn_off", "toggle"},
    "switch": {"turn_on", "turn_off", "toggle"},
    "fan": {"turn_on", "turn_off", "toggle", "set_percentage", "set_preset_mode", "oscillate"},
    "cover": {"open_cover", "close_cover", "stop_cover", "set_cover_position", "open_cover_tilt", "close_cover_tilt", "set_cover_tilt_position"},
    "climate": {"turn_on", "turn_off", "set_temperature", "set_hvac_mode", "set_fan_mode", "set_preset_mode", "set_humidity"},
    "media_player": {"turn_on", "turn_off", "toggle", "volume_set", "volume_up", "volume_down", "volume_mute", "media_play", "media_pause", "media_play_pause", "media_stop", "media_next_track", "media_previous_track", "select_source", "select_sound_mode", "shuffle_set", "repeat_set"},
    "vacuum": {"start", "stop", "pause", "return_to_base", "locate", "clean_spot"},
    "humidifier": {"turn_on", "turn_off", "toggle", "set_humidity", "set_mode"},
    "water_heater": {"turn_on", "turn_off", "set_temperature", "set_operation_mode", "set_away_mode"},
    "input_boolean": {"turn_on", "turn_off", "toggle"},
    "input_number": {"set_value", "increment", "decrement"},
    "number": {"set_value"},
    "select": {"select_option", "select_next", "select_previous", "select_first", "select_last"},
    "input_select": {"select_option", "select_next", "select_previous", "select_first", "select_last"},
    "button": {"press"},
    "input_button": {"press"},
    "scene": {"turn_on"},
    "script": {"turn_on", "turn_off", "toggle"},
    "automation": {"trigger", "turn_on", "turn_off", "toggle"},
}

# Transcript-language checks in the safety gate are intentionally narrow. They
# cover only cases where executing a control would be clearly unsafe/incorrect:
# a deferred request sent to an immediate tool, or a factual state report/correction
# that Gemini accidentally converted into an on/off action.
_DEFERRED_RE = re.compile(r"(?:^|\s)(?:через|когда|после\s+того\s+как|как\s+только)(?:\s|$)", re.I)
_STATE_PREDICATE_RE = re.compile(
    r"(?:^|\s)(?:включ[её]н\w*|выключ[её]н\w*|отключ[её]н\w*|горит|горят|не\s+горит|не\s+горят|"
    r"открыт\w*|закрыт\w*)(?:\s|$)",
    re.I,
)
_FIRST_PERSON_PAST_CONTROL_RE = re.compile(
    r"(?:^|\s)я(?:\s+\S+){0,6}\s+(?:включил\w*|выключил\w*|отключил\w*|открыл\w*|закрыл\w*|"
    r"запустил\w*|остановил\w*|убрал\w*|поставил\w*)(?:\s|$)",
    re.I,
)
_EXPLICIT_POWER_REQUEST_RE = re.compile(
    r"(?:^|\s)(?:включи|включай|выключи|выключай|отключи|отключай|зажги|зажигай|погаси|гаси)(?:\s|$)",
    re.I,
)

# Canonical Home Assistant area names with spoken/translated forms.  The
# resolver always returns the real HA area name, so Gemini cannot accidentally
# translate «Кухня» to an area_filter that does not exist.
_AREA_ALIASES: dict[str, tuple[str, ...]] = {
    "Кухня": ("кухн", "kitchen"),
    "Туалет": ("туалет", "toilet", "wc"),
    "Ванная": ("ванн", "bathroom", "bath room"),
    "Зал": ("зал", "гостин", "living room", "livingroom", "hall"),
    "Коридор": ("коридор", "corridor", "hallway"),
    "Прихожая": ("прихож", "entryway", "entry hall", "foyer"),
}

_TOOL_DESCRIPTIONS = {
    "ha_search": "Resolve Home Assistant entities by the user's spoken Russian name/room. Never translate Russian room names to English. For a generic room command such as 'свет на кухне', use the curated targets returned by this tool directly; they already exclude unavailable/service indicators and may prefer an HA aggregate when one exists.",
    "ha_get_state": "Read the current state and useful attributes of one or more exact Home Assistant entity IDs.",
    "ha_get_entity": "Read registry metadata for exact Home Assistant entities, including area and friendly name.",
    "ha_get_device": "Read Home Assistant device information, optionally by entity or area.",
    "ha_get_history": "Read recent Home Assistant state history or long-term statistics for exact entities.",
    "ha_list_floors_areas": "List Home Assistant floors and areas/rooms for location reasoning.",
    "ha_bulk_control": "Immediately control one or more exact low-risk household entities or an exact area/floor for the user's current explicit request. Do not ask for confirmation when the target and action are clear.",
    "ha_call_service": "Immediately call an allowed low-risk household Home Assistant service for an exact entity when bulk_control cannot express the requested parameter. Do not ask for confirmation when the target and action are clear.",
    "ha_get_todo": "Read Home Assistant todo/shopping lists and their items.",
    "ha_set_todo_item": "Add or update an item in a Home Assistant todo/shopping list when explicitly requested.",
    "ha_config_get_calendar_events": "Read events from a Home Assistant calendar for a requested time range.",
    "ha_config_set_calendar_event": "Create or update a Home Assistant calendar event when explicitly requested.",
    "ha_config_remove_calendar_event": "Delete a Home Assistant calendar event only when the user explicitly asks to remove/cancel it.",
}

_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_AT = 0.0
_CACHE_LOCK: asyncio.Lock | None = None
CACHE_TTL = 300.0
RESULT_LIMIT = 6000

# Short-lived optimistic record of simple voice controls. Some integrations
# report the old entity state for several seconds after a successfully dispatched
# service call. We never overwrite HA's raw state; instead state reads are
# annotated so Gemini can describe the confirmation lag rather than confidently
# reporting the stale value as physical truth.
_RECENT_POWER_DISPATCH: dict[str, dict[str, Any]] = {}
RECENT_POWER_DISPATCH_TTL = 20.0
_VOICE_STATE_CACHE: dict[str, dict[str, Any]] = {}
POWER_CONFIRM_TIMEOUT_SECONDS = max(0.5, float(os.environ.get("ADA_POWER_CONFIRM_TIMEOUT_SECONDS", "2.0")))
POWER_CONFIRM_POLL_SECONDS = max(0.05, float(os.environ.get("ADA_POWER_CONFIRM_POLL_MS", "150")) / 1000.0)


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    arguments: dict[str, Any]
    reason: str = ""


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return " ".join(text.split())


def _canonical_area(*values: str) -> str | None:
    haystack = " ".join(_normalize(value) for value in values if value)
    if not haystack:
        return None
    for canonical, aliases in _AREA_ALIASES.items():
        canonical_norm = _normalize(canonical)
        if canonical_norm and canonical_norm in haystack:
            return canonical
        if any(alias in haystack for alias in aliases):
            return canonical
    return None


def _infer_domain_from_transcript(text: str) -> str | None:
    norm = _normalize(text)
    if any(word in norm for word in ("свет", "ламп", "подсвет", "лента", "торшер")):
        return "light"
    if any(word in norm for word in ("пылесос", "уборк")):
        return "vacuum"
    if any(word in norm for word in ("штор", "жалюз", "роллет")):
        return "cover"
    if any(word in norm for word in ("вентилят", "вытяжк")):
        return "fan"
    if any(word in norm for word in ("кондиционер", "климат", "температур", "котел", "отоплен")):
        return "climate"
    if any(word in norm for word in ("телевизор", "телек", "тв ", " тв", "медиаплеер")):
        return "media_player"
    if any(word in norm for word in ("розетк", "гирлянд")):
        return "switch"
    return None


def _rewrite_search_arguments(arguments: dict[str, Any], transcript: str) -> dict[str, Any]:
    args = dict(arguments or {})
    current_area = _canonical_area(str(args.get("area_filter") or ""), str(args.get("query") or ""), transcript)
    inferred_domain = _infer_domain_from_transcript(transcript)
    if current_area:
        args["area_filter"] = current_area
        # For a clear area-scoped household command, registry listing is more
        # deterministic than fuzzy text search (and avoids kitchen->English drift).
        if _has_action_intent(transcript) and (args.get("domain_filter") or inferred_domain):
            args.pop("query", None)
    if inferred_domain and _has_action_intent(transcript):
        args.setdefault("domain_filter", inferred_domain)
    if args.get("domain_filter") in VOICE_CONTROL_DOMAINS:
        args["limit"] = max(20, int(args.get("limit") or 10))
        args["result_fields"] = ["entity_id", "friendly_name", "domain", "state", "area", "aliases"]
        args["fields"] = ["entities", "entity_total_matches", "area_names", "message", "warnings", "errors", "partial"]
    return args


def _curate_search_result(payload: dict[str, Any], transcript: str) -> dict[str, Any]:
    entities = payload.get("entities")
    if not isinstance(entities, list):
        return payload
    norm = _normalize(transcript)
    action = _has_action_intent(transcript)
    domain = str(payload.get("domain_filter") or "")
    area = _canonical_area(str(payload.get("area_filter") or ""), transcript)
    if action:
        entities = [
            item for item in entities
            if isinstance(item, dict)
            and str(item.get("state") or "").lower() != "unavailable"
        ]
        known_state = [
            item for item in entities
            if str(item.get("state") or "").lower() != "unknown"
        ]
        if known_state:
            entities = known_state
    if action and domain == "light":
        # Remove obvious device/status LEDs from household lighting intent.
        entities = [
            item for item in entities
            if "led ring" not in _normalize(item.get("friendly_name") or "")
            and "assist satellite" not in _normalize(item.get("friendly_name") or "")
        ]
        if "лампоч" in norm or "ламп " in f"{norm} ":
            preferred = [item for item in entities if "ламп" in _normalize(item.get("friendly_name") or item.get("entity_id") or "")]
            if preferred:
                entities = preferred
        elif any(token in norm for token in ("лента", "подсвет")):
            preferred = [item for item in entities if any(token in _normalize(item.get("friendly_name") or item.get("entity_id") or "") for token in ("led", "strip", "лент", "подсвет"))]
            if preferred:
                entities = preferred
        else:
            # Prefer an HA aggregate/group if its aliases explicitly describe the
            # generic room light; otherwise keep all controllable lights in area.
            aggregate = []
            for item in entities:
                aliases = [_normalize(alias) for alias in (item.get("aliases") or [])]
                if any("свет в " in alias or "основной свет" in alias for alias in aliases):
                    aggregate.append(item)
            if aggregate:
                entities = aggregate
    payload = dict(payload)
    payload["entities"] = entities
    payload["count"] = len(entities)
    payload["voice_resolution"] = {
        "area": area,
        "domain": domain or _infer_domain_from_transcript(transcript),
        "generic_area_control": bool(action and area and domain),
        "instruction": (
            "Use these curated Home Assistant targets directly. Do not translate the area name or ask for confirmation when the current user command is clear."
        ),
    }
    return payload


def _curate_light_switch_fallback(payload: dict[str, Any], transcript: str) -> dict[str, Any]:
    entities = payload.get("entities")
    if not isinstance(entities, list):
        return payload
    kept: list[dict[str, Any]] = []
    for item in entities:
        if not isinstance(item, dict):
            continue
        if str(item.get("state") or "").lower() == "unavailable":
            continue
        text = _normalize(" ".join([
            str(item.get("friendly_name") or ""),
            str(item.get("entity_id") or ""),
            " ".join(str(alias) for alias in (item.get("aliases") or [])),
        ]))
        if any(term in text for term in ("свет", "ламп", "освещ", "light")):
            kept.append(item)
    result = dict(payload)
    result["entities"] = kept
    result["count"] = len(kept)
    result["voice_resolution"] = {
        "area": _canonical_area(str(payload.get("area_filter") or ""), transcript),
        "domain": "light",
        "semantic_fallback_domain": "switch",
        "instruction": (
            "These switch entities semantically represent room lighting because no controllable light-domain entities were found. "
            "Use them directly for the clear user lighting command without asking for confirmation."
        ),
    }
    return result


def _has_action_intent(text: str) -> bool:
    """Permissive resolver hint only; never used to authorize or block mutations."""
    norm = _normalize(text)
    if not norm:
        return False
    return bool(re.search(
        r"(?:^|\s)(?:включ\w*|выключ\w*|отключ\w*|откр\w*|закр\w*|запуст\w*|"
        r"останов\w*|сдел\w*|постав\w*|установ\w*|зада\w*|измен\w*|поменя\w*|"
        r"добав\w*|убер\w*|удал\w*|отмен\w*|переключ\w*|продолж\w*|пропылесос\w*|"
        r"громче|тише|ярче|темнее)(?:\s|$)",
        norm,
    )) or any(phrase in norm for phrase in ("в список", "в покупки", "в календар", "напомни", "напоминание"))


def _entity_domains(entity_id: Any) -> set[str]:
    values: list[str] = []
    if isinstance(entity_id, str):
        values = [part.strip() for part in entity_id.split(",") if part.strip()]
    elif isinstance(entity_id, list):
        values = [str(part).strip() for part in entity_id if str(part).strip()]
    return {value.split(".", 1)[0] for value in values if "." in value}


def _looks_like_factual_state_report(text: str) -> bool:
    """Detect a narrow class of reports/corrections that must not mutate devices.

    This intentionally is not a general imperative classifier. Natural commands
    remain Gemini's job. The backstop only catches state predicates ("свет
    выключен", "лампы горят") and first-person past reports ("я всё выключил")
    when there is no explicit on/off request in the same utterance.
    """
    norm = _normalize(text)
    if not norm or _EXPLICIT_POWER_REQUEST_RE.search(norm):
        return False
    return bool(_STATE_PREDICATE_RE.search(norm) or _FIRST_PERSON_PAST_CONTROL_RE.search(norm))


def guard_mutation(name: str, arguments: dict[str, Any], transcript: str) -> GuardResult:
    """Apply only hard safety/shape constraints to state-changing HA calls.

    Gemini owns natural-language intent. This gate deliberately does not try to
    classify arbitrary Russian commands or maintain a synonym allowlist. Its job
    is to prevent dangerous/administrative calls, malformed targets, obviously
    deferred requests, and the narrow high-cost failure where a factual device
    state report/correction is accidentally converted into an on/off mutation.
    """
    args = dict(arguments or {})
    if name not in MUTATING_TOOLS:
        return GuardResult(True, args)
    if not transcript.strip():
        return GuardResult(False, args, "No current user transcript is available for this state-changing request")

    if name in {"ha_bulk_control", "ha_call_service"} and _DEFERRED_RE.search(_normalize(transcript)):
        return GuardResult(False, args, "Deferred/scheduled actions are not executed by the immediate voice-control tool")

    if (
        name in {"ha_bulk_control", "ha_call_service"}
        and _control_feedback_kind(name, args) in {"on", "off"}
        and _looks_like_factual_state_report(transcript)
    ):
        return GuardResult(
            False,
            args,
            "The current utterance is a factual device-state report/correction, not an explicit on/off request",
        )

    if name == "ha_call_service":
        if args.get("ws_command"):
            return GuardResult(False, args, "Raw Home Assistant WebSocket commands are blocked for voice")
        domain = str(args.get("domain") or "").strip()
        service = str(args.get("service") or "").strip()
        if domain not in VOICE_CONTROL_DOMAINS:
            return GuardResult(False, args, f"Home Assistant domain '{domain}' is not voice-enabled")
        allowed_services = SERVICE_ALLOWLIST.get(domain, set())
        if service not in allowed_services:
            return GuardResult(False, args, f"Service '{domain}.{service}' is not voice-enabled")
        domains = _entity_domains(args.get("entity_id"))
        if not domains:
            return GuardResult(False, args, "Voice service calls require an explicit entity_id target")
        if not domains.issubset(VOICE_CONTROL_DOMAINS):
            return GuardResult(False, args, "One or more target entity domains are not voice-enabled")
    elif name == "ha_bulk_control":
        operations = args.get("operations")
        selector = args.get("selector")
        if isinstance(operations, list):
            # In operations mode Home Assistant rejects selector-style top-level
            # fields such as action. Gemini occasionally includes both; the
            # per-operation actions are authoritative, so normalize the payload
            # before it reaches HA instead of forcing a second model retry.
            args.pop("action", None)
            args.pop("selector", None)
            for operation in operations:
                if not isinstance(operation, dict):
                    return GuardResult(False, args, "Invalid bulk-control operation")
                domains = _entity_domains(operation.get("entity_id"))
                if not domains or not domains.issubset(VOICE_CONTROL_DOMAINS):
                    return GuardResult(False, args, "Bulk-control target is not voice-enabled")
                action = str(operation.get("action") or "").strip()
                if not action:
                    return GuardResult(False, args, "Bulk-control operation requires an action")
        elif isinstance(selector, dict):
            domain = str(selector.get("domain") or "")
            if domain not in VOICE_CONTROL_DOMAINS:
                return GuardResult(False, args, f"Selector domain '{domain}' is not voice-enabled")
            action = str(args.get("action") or "").strip()
            if not action:
                return GuardResult(False, args, "Selector bulk control requires an action")
        else:
            return GuardResult(False, args, "Bulk control requires explicit operations or selector")

    return GuardResult(True, args)


def _is_null_schema(schema: Any) -> bool:
    return isinstance(schema, dict) and schema.get("type") == "null"


def sanitize_schema(schema: Any) -> dict[str, Any]:
    """Return a compact Gemini-compatible subset of JSON Schema."""
    if not isinstance(schema, dict):
        return {"type": "string"}
    out: dict[str, Any] = {}
    typ = schema.get("type")
    if isinstance(typ, str) and typ in {"object", "array", "string", "integer", "number", "boolean"}:
        out["type"] = typ
    elif isinstance(schema.get("properties"), dict) or "additionalProperties" in schema:
        out["type"] = "object"
    elif isinstance(schema.get("items"), dict):
        out["type"] = "array"
    if isinstance(schema.get("description"), str):
        out["description"] = schema["description"][:500]
    if isinstance(schema.get("enum"), list):
        enum = [v for v in schema["enum"] if isinstance(v, (str, int, float, bool))]
        if enum:
            out["enum"] = enum
    if "const" in schema and isinstance(schema.get("const"), (str, int, float, bool)):
        out.setdefault("enum", [schema["const"]])
    props = schema.get("properties")
    if isinstance(props, dict):
        out["properties"] = {str(k): sanitize_schema(v) for k, v in props.items() if isinstance(k, str)}
    required = schema.get("required")
    if isinstance(required, list):
        valid = set(out.get("properties", {}))
        req = [x for x in required if isinstance(x, str) and (not valid or x in valid)]
        if req:
            out["required"] = req
    if isinstance(schema.get("items"), dict):
        out["items"] = sanitize_schema(schema["items"])
    alternatives = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(alternatives, list):
        non_null = [sanitize_schema(x) for x in alternatives if isinstance(x, dict) and not _is_null_schema(x)]
        if len(non_null) == 1:
            # Flatten the common Optional[T] pattern; Gemini does not need a null branch.
            merged = non_null[0]
            if "description" in out and "description" not in merged:
                merged["description"] = out["description"]
            out = merged
        elif non_null:
            out["anyOf"] = non_null
    if not out:
        out = {"type": "string"}
    return out


async def _discover() -> dict[str, dict[str, Any]]:
    url = os.environ.get("HA_MCP_URL", "").strip()
    if not url:
        return {}
    async with streamablehttp_client(url) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
    found: dict[str, dict[str, Any]] = {}
    for tool in result.tools:
        if tool.name not in VOICE_TOOL_NAMES:
            continue
        found[tool.name] = {
            "name": tool.name,
            "description": _TOOL_DESCRIPTIONS.get(tool.name, (tool.description or "")[:500]),
            "parameters_json_schema": sanitize_schema(tool.inputSchema or {"type": "object"}),
        }
    return found


async def get_voice_tool_declarations(*, force: bool = False) -> list[dict[str, Any]]:
    global _CACHE_AT, _CACHE_LOCK
    if _CACHE_LOCK is None:
        _CACHE_LOCK = asyncio.Lock()
    now = time.monotonic()
    if not force and _CACHE and now - _CACHE_AT < CACHE_TTL:
        return list(_CACHE.values())
    async with _CACHE_LOCK:
        now = time.monotonic()
        if not force and _CACHE and now - _CACHE_AT < CACHE_TTL:
            return list(_CACHE.values())
        discovered = await _discover()
        _CACHE.clear()
        _CACHE.update(discovered)
        _CACHE_AT = time.monotonic()
        logger.info("HA MCP voice catalog ready tools=%d", len(_CACHE))
        return list(_CACHE.values())


def _content_to_python(result: Any) -> Any:
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    pieces: list[str] = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            pieces.append(str(text))
    text = "\n".join(pieces).strip()
    if not text:
        return {"success": not bool(getattr(result, "isError", False))}
    try:
        return json.loads(text)
    except Exception:
        return text


def _bounded_result(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        payload: Any = value
    else:
        payload = {"result": value}
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) <= RESULT_LIMIT:
        return payload
    return {
        "success": True,
        "truncated": True,
        "result_preview": text[: RESULT_LIMIT - 200],
    }


_WEATHER_ATTRIBUTE_KEYS = {
    "temperature",
    "apparent_temperature",
    "temperature_unit",
    "wind_bearing",
    "wind_gust_speed",
    "wind_speed",
    "wind_speed_unit",
    "forecast_hourly",
    "forecast_twice_daily",
    "friendly_name",
}


def _compact_weather_record(record: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in record.items()
        if key in {"entity_id", "state", "last_changed", "last_reported", "last_updated", "context"}
    }
    attributes = record.get("attributes")
    if isinstance(attributes, dict):
        compact_attributes = {
            key: value
            for key, value in attributes.items()
            if key in _WEATHER_ATTRIBUTE_KEYS
        }
        if isinstance(compact_attributes.get("forecast_hourly"), list):
            compact_attributes["forecast_hourly"] = compact_attributes["forecast_hourly"][:12]
        if isinstance(compact_attributes.get("forecast_twice_daily"), list):
            compact_attributes["forecast_twice_daily"] = compact_attributes["forecast_twice_daily"][:6]
        result["attributes"] = compact_attributes
    return result


def _compact_weather_state_value(value: Any, entity_id: Any) -> Any:
    """Compact weather ha_get_state payload before the generic result-size cap."""
    ids: list[str] = []
    if isinstance(entity_id, str):
        ids = [entity_id]
    elif isinstance(entity_id, list):
        ids = [str(item) for item in entity_id]
    weather_ids = {item for item in ids if item.startswith("weather.")}
    if not weather_ids or not isinstance(value, dict):
        return value

    payload = dict(value)
    data = payload.get("data")
    if isinstance(data, dict):
        compact_data = dict(data)
        states = compact_data.get("states")
        if isinstance(states, dict):
            compact_states = dict(states)
            for current_id in weather_ids:
                record = compact_states.get(current_id)
                if isinstance(record, dict):
                    compact_states[current_id] = _compact_weather_record(record)
            compact_data["states"] = compact_states
        elif "state" in compact_data and isinstance(compact_data.get("attributes"), dict):
            compact_data = _compact_weather_record(compact_data)
        payload["data"] = compact_data
        return payload

    if "state" in payload and isinstance(payload.get("attributes"), dict):
        return _compact_weather_record(payload)
    return payload


async def _mcp_call(url: str, name: str, arguments: dict[str, Any]) -> tuple[Any, Any]:
    async with streamablehttp_client(url) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            raw = await session.call_tool(name, arguments)
    return raw, _content_to_python(raw)


def _control_feedback_kind(name: str, arguments: dict[str, Any]) -> str | None:
    """Return a local success-cue kind from already-guarded HA arguments."""
    if name == "ha_call_service":
        service = str(arguments.get("service") or "").strip().lower()
        if service == "turn_on":
            return "on"
        if service == "turn_off":
            return "off"
        return None
    if name == "ha_bulk_control":
        actions: list[str] = []
        direct = str(arguments.get("action") or "").strip().lower()
        if direct:
            actions.append(direct)
        operations = arguments.get("operations")
        if isinstance(operations, list):
            for operation in operations:
                if isinstance(operation, dict):
                    action = str(operation.get("action") or "").strip().lower()
                    if action:
                        actions.append(action)
        normalized = {
            "on" if action in {"on", "turn_on"} else
            "off" if action in {"off", "turn_off"} else
            action
            for action in actions
        }
        if normalized == {"on"}:
            return "on"
        if normalized == {"off"}:
            return "off"
    return None


def _prepare_immediate_voice_control(name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Make simple voice on/off controls return after dispatch, not state convergence."""
    args = dict(arguments or {})
    if name != "ha_bulk_control":
        return args, False

    operations = args.get("operations")
    if isinstance(operations, list) and operations:
        copied: list[dict[str, Any]] = []
        simple_power = True
        for operation in operations:
            if not isinstance(operation, dict):
                simple_power = False
                break
            item = dict(operation)
            action = str(item.get("action") or "").strip().lower()
            if action not in {"on", "off"}:
                simple_power = False
            copied.append(item)
        if simple_power:
            for item in copied:
                # ha_bulk_control documents 0 as "dispatch without waiting for
                # confirmation". The actual state is verified asynchronously below.
                item["timeout_seconds"] = 0
            args["operations"] = copied
            args["parallel"] = True
            return args, True

    selector = args.get("selector")
    action = str(args.get("action") or "").strip().lower()
    if isinstance(selector, dict) and action in {"on", "off"}:
        args["timeout_seconds"] = 0
        return args, True
    return args, False


def _remember_recent_power_dispatch(payload: dict[str, Any]) -> None:
    """Remember accepted on/off targets briefly without replacing HA state."""
    results = payload.get("results")
    if not isinstance(results, list):
        return
    now = time.monotonic()
    for item in results:
        if not isinstance(item, dict) or item.get("success") is False:
            continue
        entity_id = str(item.get("entity_id") or "").strip()
        action = str(item.get("action") or "").strip().lower()
        command_sent = bool(item.get("command_sent")) or bool(item.get("dispatched"))
        if entity_id and action in {"on", "off"} and command_sent:
            _RECENT_POWER_DISPATCH[entity_id] = {
                "desired_state": action,
                "at": now,
            }


def _state_records(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("states"), dict):
        return {
            str(entity_id): record
            for entity_id, record in data["states"].items()
            if isinstance(record, dict)
        }
    if isinstance(data, dict) and data.get("entity_id"):
        return {str(data.get("entity_id")): data}
    return {}


def _cache_state_records(records: dict[str, dict[str, Any]]) -> None:
    now = time.monotonic()
    for entity_id, record in records.items():
        state = str(record.get("state") or "").strip()
        if not entity_id or not state:
            continue
        _VOICE_STATE_CACHE[entity_id] = {
            "state": state,
            "at": now,
            "last_changed": record.get("last_changed"),
            "last_updated": record.get("last_updated"),
        }


def _requested_power_targets(
    name: str,
    arguments: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, str]:
    """Return exact entity -> desired on/off state for commands actually dispatched."""
    targets: dict[str, str] = {}
    results = payload.get("results")
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict) or item.get("success") is False:
                continue
            entity_id = str(item.get("entity_id") or "").strip()
            action = str(item.get("action") or "").strip().lower()
            sent = bool(item.get("command_sent")) or bool(item.get("dispatched"))
            if entity_id and action in {"on", "off"} and sent:
                targets[entity_id] = action
    if targets:
        return targets

    if name == "ha_call_service":
        service = str(arguments.get("service") or "").strip().lower()
        desired = "on" if service == "turn_on" else "off" if service == "turn_off" else ""
        if desired:
            entity_id = arguments.get("entity_id")
            values = [entity_id] if isinstance(entity_id, str) else entity_id if isinstance(entity_id, list) else []
            for value in values:
                target = str(value or "").strip()
                if target:
                    targets[target] = desired
    return targets


async def _confirm_power_targets(
    url: str,
    targets: dict[str, str],
) -> tuple[bool, dict[str, dict[str, Any]], float]:
    """Read HA until every target reports the requested state or timeout expires."""
    if not targets:
        return False, {}, 0.0
    started = time.monotonic()
    deadline = started + POWER_CONFIRM_TIMEOUT_SECONDS
    latest: dict[str, dict[str, Any]] = {}
    entity_ids = list(targets)
    while True:
        try:
            raw, value = await _mcp_call(url, "ha_get_state", {"entity_id": entity_ids})
            if not bool(getattr(raw, "isError", False)):
                state_payload = _bounded_result(value)
                if isinstance(state_payload, dict):
                    latest = _state_records(state_payload)
                    _cache_state_records(latest)
                    if latest and all(
                        str(latest.get(entity_id, {}).get("state") or "") == desired
                        for entity_id, desired in targets.items()
                    ):
                        for entity_id in targets:
                            _RECENT_POWER_DISPATCH.pop(entity_id, None)
                        return True, latest, time.monotonic() - started
        except Exception:
            logger.exception("HA voice confirmation read failed targets=%r", targets)
        now = time.monotonic()
        if now >= deadline:
            return False, latest, now - started
        await asyncio.sleep(min(POWER_CONFIRM_POLL_SECONDS, max(0.0, deadline - now)))


def _voice_state_snapshot(
    targets: dict[str, str],
    records: dict[str, dict[str, Any]],
    *,
    confirmed: bool,
    elapsed_seconds: float,
) -> dict[str, Any]:
    states = {
        entity_id: {
            "state": str(records.get(entity_id, {}).get("state") or "unknown"),
            "desired_state": desired,
            "confirmed": str(records.get(entity_id, {}).get("state") or "") == desired,
        }
        for entity_id, desired in targets.items()
    }
    return {
        "confirmed": confirmed,
        "confirmation_ms": int(elapsed_seconds * 1000),
        "states": states,
        "instruction": (
            "These are the actual Home Assistant states read after the voice command. "
            "Treat confirmed entries as the current device-state cheat sheet for this turn."
        ),
    }


def _annotate_recent_power_state(payload: dict[str, Any]) -> dict[str, Any]:
    """Annotate a state read when HA still disagrees with a recent voice command."""
    now = time.monotonic()
    for entity_id, recent in list(_RECENT_POWER_DISPATCH.items()):
        if now - float(recent.get("at") or 0.0) > RECENT_POWER_DISPATCH_TTL:
            _RECENT_POWER_DISPATCH.pop(entity_id, None)

    data = payload.get("data")
    records: dict[str, dict[str, Any]] = {}
    if isinstance(data, dict) and isinstance(data.get("states"), dict):
        records = {
            str(entity_id): record
            for entity_id, record in data["states"].items()
            if isinstance(record, dict)
        }
    elif isinstance(data, dict) and data.get("entity_id"):
        records = {str(data.get("entity_id")): data}

    pending: list[dict[str, Any]] = []
    for entity_id, record in records.items():
        recent = _RECENT_POWER_DISPATCH.get(entity_id)
        if not recent:
            continue
        desired = str(recent.get("desired_state") or "")
        reported = str(record.get("state") or "")
        age = max(0.0, now - float(recent.get("at") or now))
        if reported == desired:
            _RECENT_POWER_DISPATCH.pop(entity_id, None)
            continue
        pending.append(
            {
                "entity_id": entity_id,
                "desired_state": desired,
                "reported_state": reported,
                "age_seconds": round(age, 1),
                "confirmation_pending": True,
            }
        )

    if pending:
        payload["voice_recent_controls"] = {
            "pending": pending,
            "instruction": (
                "Home Assistant still reports an older/conflicting state for a very recent voice control. "
                "Do not assert the reported state as certain physical truth. Mention that the recent command "
                "was sent but state confirmation is still pending; answer other confirmed entities normally."
            ),
        }
    return payload


async def call_voice_tool(name: str, arguments: dict[str, Any], *, transcript: str = "") -> dict[str, Any]:
    if name not in VOICE_TOOL_NAMES:
        return {"success": False, "error": "Tool is not available to the voice assistant"}
    resolved_arguments = _rewrite_search_arguments(arguments, transcript) if name == "ha_search" else dict(arguments or {})
    guard = guard_mutation(name, resolved_arguments, transcript)
    if not guard.allowed:
        logger.warning("HA MCP voice guard denied tool=%s reason=%s transcript=%r", name, guard.reason, transcript[:300])
        return {"success": False, "blocked": True, "error": guard.reason}

    url = os.environ.get("HA_MCP_URL", "").strip()
    if not url:
        return {"success": False, "error": "Home Assistant MCP is not configured"}
    started = time.monotonic()
    dispatch_arguments, fast_confirmation = _prepare_immediate_voice_control(name, guard.arguments)
    try:
        raw, value = await _mcp_call(url, name, dispatch_arguments)
        if name == "ha_get_state":
            value = _compact_weather_state_value(value, dispatch_arguments.get("entity_id"))
        payload = _bounded_result(value)
        if name == "ha_get_state" and isinstance(payload, dict) and not bool(getattr(raw, "isError", False)):
            payload = _annotate_recent_power_state(payload)
        if name == "ha_search" and isinstance(payload, dict) and not bool(getattr(raw, "isError", False)):
            payload = _curate_search_result(payload, transcript)
            wants_room_light = (
                guard.arguments.get("domain_filter") == "light"
                and bool(guard.arguments.get("area_filter"))
                and _has_action_intent(transcript)
            )
            if wants_room_light and not payload.get("entities"):
                fallback_args = dict(guard.arguments)
                fallback_args["domain_filter"] = "switch"
                fallback_raw, fallback_value = await _mcp_call(url, "ha_search", fallback_args)
                if not bool(getattr(fallback_raw, "isError", False)):
                    fallback_payload = _curate_light_switch_fallback(
                        _bounded_result(fallback_value), transcript
                    )
                    if fallback_payload.get("entities"):
                        payload = fallback_payload
                        logger.info(
                            "HA MCP voice resolver used semantic light->switch fallback area=%s targets=%d",
                            fallback_args.get("area_filter"), len(fallback_payload.get("entities") or []),
                        )
        if bool(getattr(raw, "isError", False)):
            payload = {"success": False, "error": payload}
        elif (
            isinstance(payload, dict)
            and payload.get("success", True) is not False
            and not payload.get("blocked")
            and not payload.get("error")
        ):
            feedback_kind = _control_feedback_kind(name, dispatch_arguments)
            if feedback_kind in {"on", "off"}:
                if fast_confirmation:
                    _remember_recent_power_dispatch(payload)
                targets = _requested_power_targets(name, dispatch_arguments, payload)
                if targets:
                    confirmed, records, confirm_elapsed = await _confirm_power_targets(url, targets)
                    payload["voice_state_snapshot"] = _voice_state_snapshot(
                        targets,
                        records,
                        confirmed=confirmed,
                        elapsed_seconds=confirm_elapsed,
                    )
                    if confirmed:
                        payload["_voice_feedback"] = feedback_kind
                        payload["voice_confirmation"] = "confirmed"
                    else:
                        payload["success"] = False
                        payload["dispatched"] = True
                        payload["unconfirmed"] = True
                        payload["voice_confirmation"] = "unconfirmed"
                        payload["error"] = (
                            "The control command was dispatched, but Home Assistant did not confirm the requested "
                            f"{feedback_kind} state within {POWER_CONFIRM_TIMEOUT_SECONDS:.1f}s. Do not claim it succeeded."
                        )
                else:
                    payload["success"] = False
                    payload["unconfirmed"] = True
                    payload["voice_confirmation"] = "unconfirmed"
                    payload["error"] = "The control result did not expose any exact entity whose state could be confirmed."
        logger.info("HA MCP voice tool=%s ok=%s duration_ms=%d fast_confirmation=%s", name, not bool(getattr(raw, "isError", False)), (time.monotonic()-started)*1000, fast_confirmation)
        return payload
    except Exception as exc:
        logger.exception("HA MCP voice tool failed name=%s", name)
        return {"success": False, "error": f"Home Assistant MCP call failed: {type(exc).__name__}: {exc}"}
