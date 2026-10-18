"""Persistent Gemini Live connection with documented session resumption."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from typing import Any

from .realtime_provider import ProviderEvent, create_provider

logger = logging.getLogger("voice.live_manager")


class LiveSessionManager:
    def __init__(
        self, prompt_getter: Callable[[], str],
        office_state_getter: Callable[[], dict[str, Any]] | None = None,
        habit_state_getter: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.prompt_getter = prompt_getter
        self.office_state_getter = office_state_getter
        self.habit_state_getter = habit_state_getter
        self.provider: Any = None
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._connected = asyncio.Event()
        self._subscribers: set[asyncio.Queue[ProviderEvent]] = set()
        self._generation = 0
        self.reconnect_count = 0
        self.fresh_session_count = 0
        self.status = "stopped"
        self.last_error = ""

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run())
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._connected.wait(), timeout=8)

    async def _run(self) -> None:
        handle: str | None = None
        generation = self._generation
        delay = 1.0
        while not self._stopping:
            provider = create_provider(
                self.prompt_getter(), office_state_getter=self.office_state_getter,
                habit_state_getter=self.habit_state_getter,
            )
            provider.session_id = "persistent"
            self.provider = provider
            self.status = "connecting"
            try:
                await provider.connect(resumption_handle=handle)
                self.status = "connected"
                self.last_error = ""
                self._connected.set()
                delay = 1.0
                async for event in provider.events():
                    if provider.resumption_handle:
                        handle = provider.resumption_handle
                    if event.type == "go_away":
                        logger.info("Gemini Live GoAway received time_left=%s; resuming", event.data.get("time_left"))
                        break
                    await self._broadcast(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stopping:
                    break
                self.last_error = str(exc)
                logger.warning("Gemini Live connection lost; reconnecting: %s", exc)
                await self._broadcast(ProviderEvent("live_reconnecting", {"message": str(exc)}))
            finally:
                if provider.resumption_handle:
                    handle = provider.resumption_handle
                self._connected.clear()
                with suppress(Exception):
                    await provider.close()
                if self.provider is provider:
                    self.provider = None
            if self._stopping:
                break
            if generation != self._generation:
                # An explicit fresh-session request (or prompt change) must not
                # resume the previous conversational context.
                generation = self._generation
                handle = None
            self.reconnect_count += 1
            self.status = "reconnecting"
            await asyncio.sleep(delay)
            delay = min(15.0, delay * 2)
        self.status = "stopped"

    async def _broadcast(self, event: ProviderEvent) -> None:
        for queue in tuple(self._subscribers):
            if queue.full():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(event)

    async def events(self) -> AsyncIterator[ProviderEvent]:
        queue: asyncio.Queue[ProviderEvent] = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

    async def _ready_provider(self) -> Any:
        await asyncio.wait_for(self._connected.wait(), timeout=10)
        if self.provider is None:
            raise RuntimeError("Gemini Live is reconnecting")
        return self.provider

    async def wait_connected(self) -> None:
        await self._ready_provider()

    async def send_audio(self, pcm16: bytes) -> None:
        await (await self._ready_provider()).send_audio(pcm16)

    async def send_video(self, jpeg: bytes) -> None:
        await (await self._ready_provider()).send_video(jpeg)

    async def send_text_turn(self, message: str) -> None:
        await (await self._ready_provider()).send_text_turn(message)

    async def send_habit_alert(self, jpeg: bytes, alert: str) -> None:
        await (await self._ready_provider()).send_habit_alert(jpeg, alert)

    async def reload_prompt(self) -> None:
        self._generation += 1
        provider = self.provider
        if provider is not None:
            await provider.close()

    async def fresh_session(self, *, timeout: float = 12.0) -> None:
        """Drop conversational resumption state and establish a clean warm session."""
        self._generation += 1
        self.fresh_session_count += 1
        self._connected.clear()
        provider = self.provider
        if provider is not None:
            await provider.close()
        await asyncio.wait_for(self._connected.wait(), timeout=timeout)

    def snapshot(self) -> dict[str, Any]:
        provider = self.provider
        return {
            "status": self.status,
            "last_error": self.last_error,
            "generation": self._generation,
            "reconnect_count": self.reconnect_count,
            "fresh_session_count": self.fresh_session_count,
            "connected": self._connected.is_set(),
            "has_provider": provider is not None,
            "web_search_enabled": bool(
                getattr(provider, "enable_web_tools", False) or getattr(provider, "enable_google_search", False)
            ) if provider else False,
            "google_grounding_enabled": bool(getattr(provider, "enable_google_search", False)) if provider else False,
            "ha_mcp_tool_count": int(getattr(provider, "_ha_tool_count", 0) or 0) if provider else 0,
            "web_tool_calls": int(getattr(provider, "_web_tool_calls", 0) or 0) if provider else 0,
            "ha_tool_calls": int(getattr(provider, "_ha_tool_calls", 0) or 0) if provider else 0,
        }

    async def stop(self) -> None:
        self._stopping = True
        if self.provider is not None:
            with suppress(Exception):
                await self.provider.close()
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._connected.clear()
