import asyncio
import tempfile
import unittest
from pathlib import Path

from backend.speaker_clock import SpeakerClockManager


class SpeakerClockTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = SpeakerClockManager(Path(self.temp.name) / "speaker_clock.json")
        self.fired = []

        async def on_fire(item):
            self.fired.append(item)

        await self.manager.start(on_fire)

    async def asyncTearDown(self):
        await self.manager.stop()
        self.temp.cleanup()

    async def test_local_time_uses_configured_timezone(self):
        result = self.manager.local_time()
        self.assertTrue(result["success"])
        self.assertEqual(result["timezone"], "Asia/Yekaterinburg")
        self.assertIn("+05:00", result["iso"])

    async def test_timer_persists_and_can_be_cancelled(self):
        result = await self.manager.set_timer(seconds=120, label="чай")
        self.assertTrue(result["success"])
        timer = result["timer"]
        self.assertEqual(timer["kind"], "timer")
        self.assertEqual(timer["label"], "чай")
        self.assertTrue(self.manager.state_path.is_file())
        listed = self.manager.list_active()
        self.assertEqual(listed["count"], 1)
        cancelled = await self.manager.cancel(query="чай")
        self.assertTrue(cancelled["success"])
        self.assertEqual(self.manager.list_active()["count"], 0)

    async def test_timer_fires_callback(self):
        result = await self.manager.set_timer(seconds=1, label="тест")
        self.assertTrue(result["success"])
        await asyncio.sleep(1.15)
        self.assertEqual(len(self.fired), 1)
        self.assertEqual(self.fired[0]["label"], "тест")
        self.assertEqual(self.manager.list_active()["count"], 0)

    async def test_alarm_validation(self):
        result = await self.manager.set_alarm(hour=25, minute=0)
        self.assertFalse(result["success"])
        result = await self.manager.set_alarm(hour=8, minute=99)
        self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main()
