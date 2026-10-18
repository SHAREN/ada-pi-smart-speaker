import os
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.wake_gate import WakeSessionGate


ROOT = Path(__file__).resolve().parents[1]


class WakeSessionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = WakeSessionGate(ROOT)

    @patch.dict(os.environ, {}, clear=True)
    def test_voice_gate_defaults_are_safe_for_p610(self) -> None:
        gate = WakeSessionGate(ROOT)
        self.assertEqual(gate.state, "standby")
        self.assertEqual(gate.wake_threshold, 0.85)
        self.assertEqual(gate.stop_threshold, 0.65)
        self.assertEqual(gate.refractory_seconds, 2.0)
        self.assertEqual(gate.stop_guard_seconds, 0.5)
        self.assertEqual(gate.pre_roll_seconds, 2.0)
        self.assertEqual(gate.command_overlap_seconds, 0.4)
        self.assertEqual(gate.idle_timeout_seconds, 30.0)

    def test_activation_buffer_uses_only_trailing_overlap(self) -> None:
        # 320 bytes at 16 kHz mono s16 == 10 ms.
        frame = b"\x00\x00" * 160
        for _ in range(100):
            self.gate.remember_pre_roll(frame)
        self.gate.activate()
        self.gate.begin_activation_buffer()
        pending = self.gate.activation_buffer_snapshot()
        duration = sum(self.gate.frame_duration(item) for item in pending)
        self.assertGreaterEqual(duration, 0.4)
        self.assertLess(duration, 0.42)
        self.assertLess(duration, self.gate.pre_roll_seconds)

    def test_end_lifecycle_is_closing_then_standby(self) -> None:
        self.gate.activate()
        self.assertTrue(self.gate.active)
        self.gate.begin_close("stop")
        self.assertTrue(self.gate.closing)
        self.assertFalse(self.gate.active)
        self.assertEqual(self.gate.stop_count, 1)
        self.assertEqual(self.gate.last_end_reason, "stop")
        self.gate.finish_standby()
        self.assertEqual(self.gate.state, "standby")
        self.assertFalse(self.gate.active)

    def test_semantic_and_idle_counters_are_distinct(self) -> None:
        self.gate.activate()
        self.gate.begin_close("semantic")
        self.assertEqual(self.gate.semantic_end_count, 1)
        self.gate.finish_standby()
        self.gate.activate()
        self.gate.begin_close("idle")
        self.assertEqual(self.gate.idle_count, 1)
        self.assertEqual(self.gate.stop_count, 0)

    def test_activation_buffer_keeps_earliest_audio_at_cap(self) -> None:
        self.gate.activation_buffer_max_seconds = 0.02
        frame = b"\x01\x00" * 160  # 10 ms
        self.assertTrue(self.gate.buffer_activation_frame(frame))
        self.assertTrue(self.gate.buffer_activation_frame(frame))
        self.assertFalse(self.gate.buffer_activation_frame(frame))
        self.assertTrue(self.gate.activation_buffer_overflow)
        self.assertEqual(len(self.gate.activation_buffer_snapshot()), 2)


if __name__ == "__main__":
    unittest.main()
