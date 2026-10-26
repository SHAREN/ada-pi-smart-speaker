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
        self.assertEqual(gate.stop_threshold, 0.60)
        self.assertEqual(gate.stop_confirm_threshold, 0.82)
        self.assertEqual(gate.refractory_seconds, 1.0)
        self.assertEqual(gate.stop_guard_seconds, 0.25)
        self.assertEqual(gate.end_cue_wake_guard_seconds, 0.35)
        self.assertEqual(gate.pre_roll_seconds, 2.0)
        self.assertEqual(gate.command_overlap_seconds, 0.0)
        self.assertEqual(gate.first_turn_timeout_seconds, 6.0)
        self.assertEqual(gate.idle_timeout_seconds, 30.0)

    def test_activation_buffer_discards_all_pre_wake_audio(self) -> None:
        # 320 bytes at 16 kHz mono s16 == 10 ms.
        frame = b"\x00\x00" * 160
        for _ in range(100):
            self.gate.remember_pre_roll(frame)
        self.gate.activate()
        self.gate.begin_activation_buffer()
        self.assertEqual(self.gate.activation_buffer_snapshot(), [])
        self.assertEqual(self.gate.status()["buffered_input_seconds"], 0.0)

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

    def test_first_turn_timeout_closes_false_wake_window_but_real_speech_disarms_it(self) -> None:
        with patch("backend.wake_gate.time.monotonic", return_value=100.0):
            self.gate.activate()
        self.assertTrue(self.gate.status()["first_turn_pending"])
        with patch("backend.wake_gate.time.monotonic", return_value=105.9):
            self.assertFalse(self.gate.first_turn_expired())
        with patch("backend.wake_gate.time.monotonic", return_value=106.1):
            self.assertTrue(self.gate.first_turn_expired())

        self.gate.confirm_first_turn()
        with patch("backend.wake_gate.time.monotonic", return_value=120.0):
            self.assertFalse(self.gate.first_turn_expired())
        self.assertFalse(self.gate.status()["first_turn_pending"])

    def test_independent_guard_can_reject_wake_without_raising_threshold(self) -> None:
        class Features:
            def process_streaming(self, pcm16):
                return [object()]

            def reset(self):
                return None

        class WakeModel:
            def __init__(self):
                self.reset_count = 0

            def process_streaming_prob(self, features):
                return 0.91

            def reset(self):
                self.reset_count += 1

        self.gate._features = Features()
        wake_model = WakeModel()
        self.gate._wake_model = wake_model
        event = self.gate.process_audio(b"\x00\x00" * 320, allow_wake=False)
        self.assertEqual(event, "wake_rejected")
        self.assertEqual(self.gate.state, "standby")
        self.assertEqual(self.gate.wake_count, 0)
        self.assertEqual(self.gate.wake_rejected_count, 1)
        self.assertAlmostEqual(self.gate.last_rejected_wake_probability, 0.91)
        self.assertEqual(wake_model.reset_count, 1)


if __name__ == "__main__":
    unittest.main()
