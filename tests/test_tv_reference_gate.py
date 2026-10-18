import time
import unittest
from array import array

from backend.tv_reference_gate import TVReferenceGate, pcm16_rms


def pcm(level: float, samples: int = 320) -> bytes:
    value = max(-32767, min(32767, int(round(level * 32768))))
    return array("h", [value] * samples).tobytes()


class TVReferenceGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = TVReferenceGate()
        self.gate.enabled = True
        self.gate.reference_delay = 0.0
        self.gate.reference_active_rms = 0.001
        self.gate.absolute_near_end_rms = 0.004
        self.gate.echo_margin = 1.5
        self.gate.near_end_hold_seconds = 1.0
        self.gate.min_training_samples = 3

    def push_reference(self, level: float) -> None:
        self.gate._reference_history.append((time.monotonic(), level))

    def test_pcm16_rms_is_normalized(self) -> None:
        self.assertAlmostEqual(pcm16_rms(pcm(0.25)), 0.25, places=3)

    def test_tv_only_frame_is_suppressed_after_standby_learning(self) -> None:
        for _ in range(3):
            self.push_reference(0.05)
            self.gate.observe(pcm(0.01), learn=True)
        self.push_reference(0.05)
        decision = self.gate.classify_active(pcm(0.01))
        self.assertTrue(decision.ready)
        self.assertTrue(decision.suppress)
        self.assertAlmostEqual(decision.learned_ratio, 0.2, places=2)

    def test_near_end_voice_opens_gate_and_holds_quiet_syllable(self) -> None:
        for _ in range(3):
            self.push_reference(0.05)
            self.gate.observe(pcm(0.01), learn=True)
        self.push_reference(0.05)
        loud = self.gate.classify_active(pcm(0.03))
        self.assertFalse(loud.suppress)
        self.assertTrue(loud.near_end_hold)
        self.push_reference(0.05)
        quiet = self.gate.classify_active(pcm(0.006))
        self.assertFalse(quiet.suppress)
        self.assertTrue(quiet.near_end_hold)

    def test_no_reference_never_suppresses(self) -> None:
        for _ in range(3):
            self.push_reference(0.05)
            self.gate.observe(pcm(0.01), learn=True)
        self.push_reference(0.0)
        decision = self.gate.classify_active(pcm(0.002))
        self.assertFalse(decision.suppress)


if __name__ == "__main__":
    unittest.main()
