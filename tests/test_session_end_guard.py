import unittest

from backend.realtime_provider import _explicit_end_request
from backend.wake_gate import WakeSessionGate


class SemanticEndGuardTests(unittest.TestCase):
    def test_explicit_end_commands_are_allowed(self):
        self.assertTrue(_explicit_end_request("заверши разговор"))
        self.assertTrue(_explicit_end_request("можешь закончить этот диалог"))
        self.assertTrue(_explicit_end_request("перестань меня слушать"))
        self.assertTrue(_explicit_end_request("закрой сессию"))
        self.assertTrue(_explicit_end_request("стоп"))
        self.assertTrue(_explicit_end_request("всё, стоп"))
        self.assertTrue(_explicit_end_request("я хочу, чтобы ты завершил разговор"))
        self.assertTrue(_explicit_end_request("командую завершить разговор сессию"))

    def test_bug_report_about_previous_end_is_not_an_end_request(self):
        self.assertFalse(_explicit_end_request("Почему ты завершил разговор?"))
        self.assertFalse(_explicit_end_request("Ты сама закончила разговор"))

    def test_negated_end_command_is_blocked(self):
        self.assertFalse(_explicit_end_request("не заканчивай разговор"))
        self.assertFalse(_explicit_end_request("не надо завершить сессию"))

    def test_device_stop_is_not_conversation_end(self):
        self.assertFalse(_explicit_end_request("останови музыку"))
        self.assertFalse(_explicit_end_request("выключи пылесос"))


class _FakeFeatures:
    def process_streaming(self, _pcm16):
        return [object()]


class _FakeStopModel:
    def __init__(self):
        self.reset_count = 0
        self.process_count = 0

    def reset(self):
        self.reset_count += 1

    def process_streaming_prob(self, _features):
        self.process_count += 1
        return 0.9


class LocalStopVadGateTests(unittest.TestCase):
    @staticmethod
    def _gate():
        gate = WakeSessionGate.__new__(WakeSessionGate)
        gate.state = "active"
        gate._wake_guard_until = 0.0
        gate._stop_guard_until = 0.0
        gate._features = _FakeFeatures()
        gate._stop_model = _FakeStopModel()
        gate.stop_threshold = 0.60
        gate.last_stop_probability = 0.0
        gate.closed_reason = None

        def begin_close(reason):
            gate.closed_reason = reason
            gate.state = "closing"

        gate.begin_close = begin_close
        return gate

    def test_stop_model_cannot_close_without_independent_local_vad(self):
        gate = self._gate()
        result = gate.process_audio(b"\x00\x00" * 160, allow_stop=False)
        self.assertIsNone(result)
        self.assertIsNone(gate.closed_reason)
        self.assertEqual(gate._stop_model.process_count, 0)
        self.assertGreaterEqual(gate._stop_model.reset_count, 1)

    def test_stop_model_can_close_when_local_vad_confirms_speech(self):
        gate = self._gate()
        result = gate.process_audio(b"\x00\x00" * 160, allow_stop=True)
        self.assertEqual(result, "stop")
        self.assertEqual(gate.closed_reason, "stop")
        self.assertEqual(gate._stop_model.process_count, 1)
        self.assertEqual(gate.last_stop_probability, 0.9)

    def test_stop_probability_below_calibrated_threshold_does_not_close(self):
        gate = self._gate()
        gate._stop_model.process_streaming_prob = lambda _features: 0.57
        result = gate.process_audio(b"\x00\x00" * 160, allow_stop=True)
        self.assertIsNone(result)
        self.assertIsNone(gate.closed_reason)
        self.assertEqual(gate.last_stop_probability, 0.57)


if __name__ == "__main__":
    unittest.main()
