import unittest
from unittest.mock import patch

import numpy as np

from backend.self_echo_gate import SelfEchoDecision, SelfEchoGate, _Frame, _feature_score


class SelfEchoGateTests(unittest.TestCase):
    def test_identical_feature_sequences_match_strongly(self):
        base = np.asarray([
            [np.sin((i + 1) * (j + 1) * 0.19) + i * 0.07 for j in range(12)]
            for i in range(8)
        ], dtype=np.float64)
        self.assertGreater(_feature_score(base, base.copy()), 0.99)

    def test_content_match_finds_expected_render_delay(self):
        gate = SelfEchoGate()
        gate.reference_delay = 0.16
        gate.delay_search = 0.08
        gate.match_window = 0.24
        gate.match_interval_seconds = 0.0
        rng = np.random.default_rng(42)

        reference = []
        for i in range(80):
            feature = rng.normal(size=12).astype(np.float64)
            frame = _Frame(at=i * 0.02, feature=feature, rms=0.08)
            gate._reference.append(frame)
            reference.append(frame)

        # Last 12 microphone feature frames are an exact delayed copy of the
        # render stream. Their timestamps are shifted by 160 ms, matching the
        # expected acoustic/render latency while avoiding any repeated FFT work.
        for frame in reference[45:57]:
            gate._mic.append(_Frame(
                at=frame.at + 0.16,
                feature=frame.feature.copy(),
                rms=0.02,
            ))

        ready, score, delay, ref_rms = gate._content_match()

        self.assertTrue(ready)
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.95)
        self.assertAlmostEqual(delay, 0.16, delta=0.021)
        self.assertGreater(ref_rms, 0.01)

    def test_recent_echo_timestamp_survives_last_decision_reset(self):
        gate = SelfEchoGate()
        gate._last_echo_only_at = 10.0
        gate._last_decision = SelfEchoDecision()
        with patch("backend.self_echo_gate.time.monotonic", return_value=10.7):
            self.assertTrue(gate.recent_echo_only(1.0))
            self.assertFalse(gate.current_echo_only())


if __name__ == "__main__":
    unittest.main()
