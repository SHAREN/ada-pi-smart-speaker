import asyncio
import unittest
from unittest.mock import patch

from backend.live_manager import (
    LiveSessionManager,
    _contended_resumption_error,
    _expired_resumption_error,
)


class _FakeProvider:
    def __init__(self, mode: str, seen_handles: list[str | None]) -> None:
        self.mode = mode
        self.seen_handles = seen_handles
        self.session_id = "-"
        self.resumption_handle: str | None = None
        self.closed = False

    async def connect(self, resumption_handle: str | None = None) -> None:
        self.seen_handles.append(resumption_handle)
        if self.mode == "expired":
            self.resumption_handle = resumption_handle
            raise RuntimeError("1008 None. BidiGenerateContent session not found or expired")
        if self.mode == "contended":
            self.resumption_handle = resumption_handle
            raise RuntimeError(
                "1008 None. BidiGenerateContent session is already connected to an existing client"
            )

    async def events(self):
        if self.mode == "disconnect_with_handle":
            self.resumption_handle = "stale-handle"
            raise RuntimeError("temporary network disconnect")
        while not self.closed:
            await asyncio.sleep(0.01)
        if False:
            yield None

    async def close(self) -> None:
        self.closed = True


class LiveManagerResumptionTests(unittest.IsolatedAsyncioTestCase):
    def test_expired_error_classifier(self) -> None:
        self.assertTrue(_expired_resumption_error(
            RuntimeError("1008 None. BidiGenerateContent session not found or expired")
        ))
        self.assertFalse(_expired_resumption_error(RuntimeError("temporary network disconnect")))

    def test_contended_error_classifier(self) -> None:
        self.assertTrue(_contended_resumption_error(RuntimeError(
            "1008 None. BidiGenerateContent session is already connected to an existing client"
        )))
        self.assertFalse(_contended_resumption_error(RuntimeError("temporary network disconnect")))

    async def test_expired_handle_is_dropped_before_next_reconnect(self) -> None:
        seen_handles: list[str | None] = []
        providers = [
            _FakeProvider("disconnect_with_handle", seen_handles),
            _FakeProvider("expired", seen_handles),
            _FakeProvider("stable", seen_handles),
        ]
        manager = LiveSessionManager(lambda: "test")

        with patch("backend.live_manager.create_provider", side_effect=providers):
            await manager.start()
            for _ in range(500):
                if len(seen_handles) >= 3 and manager.status == "connected":
                    break
                await asyncio.sleep(0.01)
            self.assertGreaterEqual(len(seen_handles), 3)
            self.assertEqual(seen_handles[:3], [None, "stale-handle", None])
            await manager.stop()

    async def test_proactive_text_turn_waits_for_physical_playback_idle(self) -> None:
        class TurnProvider:
            def __init__(self) -> None:
                self.messages: list[str] = []

            async def send_text_turn(self, message: str) -> None:
                self.messages.append(message)

        manager = LiveSessionManager(lambda: "test")
        provider = TurnProvider()
        manager.provider = provider
        manager._connected.set()
        manager.set_playback_busy(True)

        task = asyncio.create_task(manager.send_text_turn("habit alert"))
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertEqual(provider.messages, [])
        self.assertTrue(manager.snapshot()["playback_busy"])
        self.assertEqual(manager.snapshot()["playback_deferred_turn_count"], 1)

        manager.set_playback_busy(False)
        await asyncio.wait_for(task, timeout=1.0)
        self.assertEqual(provider.messages, ["habit alert"])
        self.assertFalse(manager.snapshot()["playback_busy"])

    async def test_contended_handle_is_dropped_before_next_reconnect(self) -> None:
        seen_handles: list[str | None] = []
        providers = [
            _FakeProvider("disconnect_with_handle", seen_handles),
            _FakeProvider("contended", seen_handles),
            _FakeProvider("stable", seen_handles),
        ]
        manager = LiveSessionManager(lambda: "test")

        with patch("backend.live_manager.create_provider", side_effect=providers):
            await manager.start()
            for _ in range(500):
                if len(seen_handles) >= 3 and manager.status == "connected":
                    break
                await asyncio.sleep(0.01)
            self.assertGreaterEqual(len(seen_handles), 3)
            self.assertEqual(seen_handles[:3], [None, "stale-handle", None])
            await manager.stop()


if __name__ == "__main__":
    unittest.main()
