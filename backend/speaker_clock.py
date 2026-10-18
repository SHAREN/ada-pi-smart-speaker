"""Persistent local clock, timers and alarms for ADA's smart-speaker role."""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_PATH = ROOT / "data" / "speaker_clock.json"
MAX_TIMER_SECONDS = 48 * 60 * 60
MAX_ALARM_DAYS = 31

CLOCK_TOOL_NAMES = {
    "get_local_time",
    "set_timer",
    "set_alarm",
    "list_timers",
    "cancel_timer",
}

CLOCK_TOOL_DECLARATIONS = [
    {
        "name": "get_local_time",
        "description": "Return the smart speaker's authoritative local date, time, weekday and timezone. Use this instead of guessing the current time or date.",
        "parameters_json_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "set_timer",
        "description": "Create a local countdown timer that will ring on the speaker even after the Gemini conversation returns to standby. Use for requests like 'таймер на 10 минут'.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "hours": {"type": "integer", "description": "Whole hours, normally 0."},
                "minutes": {"type": "integer", "description": "Whole minutes, normally 0."},
                "seconds": {"type": "integer", "description": "Whole seconds, normally 0."},
                "label": {"type": "string", "description": "Short optional purpose, e.g. 'макароны'."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "set_alarm",
        "description": "Set a local clock alarm. day_offset=0 means today if that time is still ahead, otherwise tomorrow; use 1 for explicitly tomorrow.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "hour": {"type": "integer", "description": "Local hour 0-23."},
                "minute": {"type": "integer", "description": "Local minute 0-59."},
                "day_offset": {"type": "integer", "description": "Days from today, 0-31."},
                "label": {"type": "string", "description": "Short optional alarm label."},
            },
            "required": ["hour", "minute"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_timers",
        "description": "List active local countdown timers and alarms with remaining time and local due time.",
        "parameters_json_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "cancel_timer",
        "description": "Cancel one local timer/alarm by its id or spoken label. Set all=true only when the user explicitly asks to cancel every timer and alarm.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Timer id or label/name to cancel."},
                "all": {"type": "boolean", "description": "Cancel all active timers and alarms."},
            },
            "additionalProperties": False,
        },
    },
]


class SpeakerClockManager:
    def __init__(self, state_path: Path = DEFAULT_STATE_PATH) -> None:
        self.state_path = state_path
        self.timezone_name = os.environ.get("ADA_TIMEZONE", "Asia/Yekaterinburg").strip() or "Asia/Yekaterinburg"
        self.timezone = ZoneInfo(self.timezone_name)
        self._entries: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._on_fire: Callable[[dict[str, Any]], Awaitable[None]] | None = None
        self._started = False
        self.last_fired: dict[str, Any] | None = None

    def _local_iso(self, timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp, self.timezone).isoformat(timespec="seconds")

    def local_time(self) -> dict[str, Any]:
        now = datetime.now(self.timezone)
        return {
            "success": True,
            "timezone": self.timezone_name,
            "iso": now.isoformat(timespec="seconds"),
            "date": now.date().isoformat(),
            "time": now.strftime("%H:%M:%S"),
            "weekday": now.strftime("%A"),
            "utc_offset": now.strftime("%z"),
        }

    def _load(self) -> None:
        self._entries.clear()
        if not self.state_path.is_file():
            return
        try:
            raw = json.loads(self.state_path.read_text())
        except Exception:
            return
        entries = raw.get("entries", []) if isinstance(raw, dict) else []
        for item in entries:
            if not isinstance(item, dict):
                continue
            timer_id = str(item.get("id") or "").strip()
            try:
                due_at = float(item.get("due_at"))
            except (TypeError, ValueError):
                continue
            if not timer_id or due_at <= 0:
                continue
            self._entries[timer_id] = {
                "id": timer_id,
                "kind": "alarm" if item.get("kind") == "alarm" else "timer",
                "label": str(item.get("label") or "").strip()[:160],
                "created_at": float(item.get("created_at") or time.time()),
                "due_at": due_at,
            }

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "timezone": self.timezone_name, "entries": list(self._entries.values())}
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        temp.replace(self.state_path)

    async def start(self, on_fire: Callable[[dict[str, Any]], Awaitable[None]] | None = None) -> None:
        self._on_fire = on_fire
        if self._started:
            return
        self._started = True
        self._load()
        now = time.time()
        changed = False
        for timer_id, item in list(self._entries.items()):
            overdue = now - float(item["due_at"])
            if overdue > 300:
                self._entries.pop(timer_id, None)
                changed = True
                continue
            self._schedule(timer_id)
        if changed:
            self._save()

    async def stop(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._started = False

    def _schedule(self, timer_id: str) -> None:
        old = self._tasks.pop(timer_id, None)
        if old is not None:
            old.cancel()
        self._tasks[timer_id] = asyncio.create_task(self._wait_and_fire(timer_id), name=f"ada-clock-{timer_id}")

    async def _wait_and_fire(self, timer_id: str) -> None:
        try:
            item = self._entries.get(timer_id)
            if item is None:
                return
            delay = max(0.0, float(item["due_at"]) - time.time())
            await asyncio.sleep(delay)
            item = self._entries.pop(timer_id, None)
            self._tasks.pop(timer_id, None)
            if item is None:
                return
            self._save()
            fired = dict(item)
            fired["due_local"] = self._local_iso(float(item["due_at"]))
            self.last_fired = fired
            if self._on_fire is not None:
                await self._on_fire(fired)
        except asyncio.CancelledError:
            raise

    def _public_item(self, item: dict[str, Any]) -> dict[str, Any]:
        due_at = float(item["due_at"])
        return {
            "id": item["id"],
            "kind": item["kind"],
            "label": item.get("label", ""),
            "remaining_seconds": max(0, round(due_at - time.time())),
            "due_local": self._local_iso(due_at),
        }

    async def set_timer(self, *, hours: int = 0, minutes: int = 0, seconds: int = 0, label: str = "") -> dict[str, Any]:
        total = int(hours or 0) * 3600 + int(minutes or 0) * 60 + int(seconds or 0)
        if total < 1 or total > MAX_TIMER_SECONDS:
            return {"success": False, "error": f"Timer duration must be between 1 and {MAX_TIMER_SECONDS} seconds"}
        timer_id = "timer-" + uuid.uuid4().hex[:8]
        now = time.time()
        item = {
            "id": timer_id,
            "kind": "timer",
            "label": " ".join(str(label or "").split())[:160],
            "created_at": now,
            "due_at": now + total,
        }
        self._entries[timer_id] = item
        self._save()
        self._schedule(timer_id)
        return {"success": True, "timer": self._public_item(item)}

    async def set_alarm(self, *, hour: int, minute: int, day_offset: int = 0, label: str = "") -> dict[str, Any]:
        hour = int(hour)
        minute = int(minute)
        day_offset = int(day_offset or 0)
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            return {"success": False, "error": "Alarm hour/minute is out of range"}
        if not 0 <= day_offset <= MAX_ALARM_DAYS:
            return {"success": False, "error": f"day_offset must be between 0 and {MAX_ALARM_DAYS}"}
        now = datetime.now(self.timezone)
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0) + timedelta(days=day_offset)
        if day_offset == 0 and target <= now:
            target += timedelta(days=1)
        timer_id = "alarm-" + uuid.uuid4().hex[:8]
        item = {
            "id": timer_id,
            "kind": "alarm",
            "label": " ".join(str(label or "").split())[:160],
            "created_at": time.time(),
            "due_at": target.timestamp(),
        }
        self._entries[timer_id] = item
        self._save()
        self._schedule(timer_id)
        return {"success": True, "alarm": self._public_item(item)}

    def list_active(self) -> dict[str, Any]:
        items = [self._public_item(item) for item in sorted(self._entries.values(), key=lambda row: float(row["due_at"]))]
        return {"success": True, "count": len(items), "items": items, "last_fired": self.last_fired}

    async def cancel(self, *, query: str = "", all_items: bool = False) -> dict[str, Any]:
        if all_items:
            ids = list(self._entries)
        else:
            needle = " ".join(str(query or "").split()).casefold()
            if not needle:
                return {"success": False, "error": "Specify a timer/alarm id or label"}
            if needle in self._entries:
                ids = [needle]
            else:
                exact = [timer_id for timer_id, item in self._entries.items() if str(item.get("label") or "").casefold() == needle]
                partial = [timer_id for timer_id, item in self._entries.items() if needle in str(item.get("label") or "").casefold()]
                ids = exact or partial
                if len(ids) > 1:
                    return {"success": False, "ambiguous": True, "matches": [self._public_item(self._entries[timer_id]) for timer_id in ids]}
        if not ids:
            return {"success": False, "error": "No matching active timer or alarm"}
        cancelled: list[dict[str, Any]] = []
        for timer_id in ids:
            item = self._entries.pop(timer_id, None)
            task = self._tasks.pop(timer_id, None)
            if task is not None:
                task.cancel()
            if item is not None:
                cancelled.append(self._public_item(item))
        self._save()
        return {"success": True, "cancelled": cancelled}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        args = dict(arguments or {})
        if name == "get_local_time":
            return self.local_time()
        if name == "set_timer":
            return await self.set_timer(
                hours=int(args.get("hours") or 0),
                minutes=int(args.get("minutes") or 0),
                seconds=int(args.get("seconds") or 0),
                label=str(args.get("label") or ""),
            )
        if name == "set_alarm":
            return await self.set_alarm(
                hour=int(args.get("hour")),
                minute=int(args.get("minute")),
                day_offset=int(args.get("day_offset") or 0),
                label=str(args.get("label") or ""),
            )
        if name == "list_timers":
            return self.list_active()
        if name == "cancel_timer":
            return await self.cancel(query=str(args.get("query") or ""), all_items=bool(args.get("all", False)))
        return {"success": False, "error": "Unknown local clock tool"}

    def snapshot(self) -> dict[str, Any]:
        return {
            "timezone": self.timezone_name,
            "started": self._started,
            "active_count": len(self._entries),
            "items": [self._public_item(item) for item in sorted(self._entries.values(), key=lambda row: float(row["due_at"]))],
            "last_fired": self.last_fired,
        }


speaker_clock = SpeakerClockManager()
