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

# Domains appropriate for a household voice assistant.  Security/admin domains
# such as lock/alarm_control_panel/homeassistant/hassio are deliberately absent.
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

_ACTION_WORDS = {
    "включи", "включите", "включить", "выключи", "выключите", "выключить",
    "отключи", "отключите", "отключить", "открой", "откройте", "открыть",
    "закрой", "закройте", "закрыть", "запусти", "запустите", "запустить",
    "останови", "остановите", "остановить", "поставь", "поставьте", "поставить",
    "установи", "установите", "установить", "задай", "задайте", "задать",
    "сделай", "сделайте", "сделать", "измени", "измените", "изменить",
    "поменяй", "поменяйте", "поменять", "добавь", "добавьте", "добавить",
    "убери", "уберите", "убрать", "удали", "удалите", "удалить", "отмени",
    "отмените", "отменить", "отметь", "отметьте", "отметить", "заверши",
    "завершите", "завершить", "переключи", "переключите", "переключить",
    "громче", "тише", "ярче", "темнее", "пауза", "продолжи", "продолжите",
    "продолжить", "пропылесось", "пропылесосьте", "пропылесосить",
}
_ON_WORDS = {"включи", "включите", "включить", "запусти", "запустите", "запустить", "открой", "откройте", "открыть"}
_OFF_WORDS = {"выключи", "выключите", "выключить", "отключи", "отключите", "отключить", "останови", "остановите", "остановить", "закрой", "закройте", "закрыть"}
_DEFERRED_RE = re.compile(r"(?:^|\s)(?:через|когда|после\s+того\s+как|как\s+только)(?:\s|$)", re.I)

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


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    arguments: dict[str, Any]
    reason: str = ""


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return " ".join(text.split())


def _tokens(text: str) -> list[str]:
    return _normalize(text).split()


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
    norm = _normalize(text)
    tokens = norm.split()
    if any(token in _ACTION_WORDS for token in tokens):
        return True
    # Natural short forms for list/calendar operations.
    return any(phrase in norm for phrase in ("в список", "в покупки", "в календар", "напомни", "напоминание"))


def _requested_power(text: str) -> str | None:
    requested: str | None = None
    for token in _tokens(text):
        if token in _ON_WORDS:
            requested = "on"
        elif token in _OFF_WORDS:
            requested = "off"
    return requested


def _entity_domains(entity_id: Any) -> set[str]:
    values: list[str] = []
    if isinstance(entity_id, str):
        values = [part.strip() for part in entity_id.split(",") if part.strip()]
    elif isinstance(entity_id, list):
        values = [str(part).strip() for part in entity_id if str(part).strip()]
    return {value.split(".", 1)[0] for value in values if "." in value}


def guard_mutation(name: str, arguments: dict[str, Any], transcript: str) -> GuardResult:
    args = dict(arguments or {})
    if name not in MUTATING_TOOLS:
        return GuardResult(True, args)
    if not transcript.strip():
        return GuardResult(False, args, "No current user transcript is available for this state-changing request")
    if not _has_action_intent(transcript):
        return GuardResult(False, args, "The current user utterance does not contain an explicit action request")

    if name in {"ha_bulk_control", "ha_call_service"} and _DEFERRED_RE.search(_normalize(transcript)):
        return GuardResult(False, args, "Deferred/scheduled actions are not executed by the immediate voice-control tool")

    requested_power = _requested_power(transcript)

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
        # Correct a model-side on/off inversion only when the user's wording is explicit.
        if requested_power == "on" and service == "turn_off" and "turn_on" in allowed_services:
            args["service"] = "turn_on"
        elif requested_power == "off" and service == "turn_on" and "turn_off" in allowed_services:
            args["service"] = "turn_off"

    elif name == "ha_bulk_control":
        operations = args.get("operations")
        selector = args.get("selector")
        if isinstance(operations, list):
            for operation in operations:
                if not isinstance(operation, dict):
                    return GuardResult(False, args, "Invalid bulk-control operation")
                domains = _entity_domains(operation.get("entity_id"))
                if not domains or not domains.issubset(VOICE_CONTROL_DOMAINS):
                    return GuardResult(False, args, "Bulk-control target is not voice-enabled")
                action = str(operation.get("action") or "")
                if requested_power == "on" and action == "off":
                    operation["action"] = "on"
                elif requested_power == "off" and action == "on":
                    operation["action"] = "off"
        elif isinstance(selector, dict):
            domain = str(selector.get("domain") or "")
            if domain not in VOICE_CONTROL_DOMAINS:
                return GuardResult(False, args, f"Selector domain '{domain}' is not voice-enabled")
            action = str(args.get("action") or "")
            if requested_power == "on" and action == "off":
                args["action"] = "on"
            elif requested_power == "off" and action == "on":
                args["action"] = "off"
        else:
            return GuardResult(False, args, "Bulk control requires explicit operations or selector")

    elif name == "ha_config_remove_calendar_event":
        norm = _normalize(transcript)
        if not any(word in norm.split() for word in {"удали", "удалите", "удалить", "отмени", "отмените", "отменить", "убери", "уберите", "убрать"}):
            return GuardResult(False, args, "Deleting a calendar event requires an explicit delete/cancel request")

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


async def _mcp_call(url: str, name: str, arguments: dict[str, Any]) -> tuple[Any, Any]:
    async with streamablehttp_client(url) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            raw = await session.call_tool(name, arguments)
    return raw, _content_to_python(raw)


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
    try:
        raw, value = await _mcp_call(url, name, guard.arguments)
        payload = _bounded_result(value)
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
        logger.info("HA MCP voice tool=%s ok=%s duration_ms=%d", name, not bool(getattr(raw, "isError", False)), (time.monotonic()-started)*1000)
        return payload
    except Exception as exc:
        logger.exception("HA MCP voice tool failed name=%s", name)
        return {"success": False, "error": f"Home Assistant MCP call failed: {type(exc).__name__}: {exc}"}
