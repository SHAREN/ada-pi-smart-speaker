import unittest
from array import array

import numpy as np

from backend.tv_reference_gate import (
    TVReferenceGate,
    _ContentMatch,
    feature_match_score,
    pcm16_rms,
)


def pcm(level: float, samples: int = 320) -> bytes:
    value = max(-32767, min(32767, int(round(level * 32768))))
    return array("h", [value] * samples).tobytes()


class TVReferenceGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = TVReferenceGate()
        self.gate.enabled = True
        self.gate.reference_active_rms = 0.001
        self.gate.absolute_near_end_rms = 0.004
        self.gate.near_end_hold_seconds = 1.0
        self.gate.min_training_samples = 3
        self.gate.match_floor = 0.24
        self.gate.match_ceiling = 0.38
        self.gate.match_scale = 0.70
        self.gate._match_history.extend([0.52, 0.58, 0.64])
        # Unit tests bypass the async parec readers but retain the same readiness
        # invariant as production: both physical-P610 and HDMI-reference streams.
        self.gate._reader_tasks = {"reference": object(), "physical": object()}

    def use_match(
        self,
        score: float,
        *,
        reference_rms: float = 0.10,
        physical_rms: float = 0.02,
        delay_seconds: float = 0.283,
    ) -> None:
        match = _ContentMatch(
            ready=True,
            score=score,
            delay_seconds=delay_seconds,
            reference_rms=reference_rms,
            physical_rms=physical_rms,
        )
        self.gate._content_match = lambda: match

    def test_pcm16_rms_is_normalized(self) -> None:
        self.assertAlmostEqual(pcm16_rms(pcm(0.25)), 0.25, places=3)

    def test_feature_match_ignores_static_per_band_room_colour(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(size=(25, 12))
        scale = np.linspace(0.45, 2.4, 12)
        offset = np.linspace(-3.0, 2.0, 12)
        coloured = source * scale + offset
        self.assertGreater(feature_match_score(source, coloured), 0.99)

    def test_tv_content_match_is_suppressed_even_when_loud(self) -> None:
        self.use_match(0.58, physical_rms=0.08)
        decision = self.gate.classify_active(pcm(0.035))
        self.assertTrue(decision.ready)
        self.assertGreater(decision.match_score, decision.match_threshold)
        self.assertTrue(decision.suppress)

    def test_unmatched_near_end_voice_opens_gate_and_holds_quiet_syllable(self) -> None:
        self.use_match(0.15)
        loud = self.gate.classify_active(pcm(0.02))
        self.assertFalse(loud.suppress)
        self.assertTrue(loud.near_end_hold)

        # Once a strong near-end syllable opens the gate, a quieter following
        # syllable remains open even if the TV match becomes high again.
        self.use_match(0.62)
        quiet = self.gate.classify_active(pcm(0.001))
        self.assertFalse(quiet.suppress)
        self.assertTrue(quiet.near_end_hold)

    def test_low_match_without_post_aec_speech_does_not_open_gate(self) -> None:
        self.use_match(0.12)
        decision = self.gate.classify_active(pcm(0.001))
        self.assertTrue(decision.suppress)
        self.assertFalse(decision.near_end_hold)

    def test_no_hdmi_reference_fails_open(self) -> None:
        self.use_match(0.60, reference_rms=0.0)
        decision = self.gate.classify_active(pcm(0.002))
        self.assertFalse(decision.suppress)

    def test_real_content_threshold_is_bounded(self) -> None:
        self.gate._match_history.clear()
        self.gate._match_history.extend([0.45, 0.50, 0.55, 0.60, 0.65])
        threshold = self.gate._match_threshold()
        self.assertGreaterEqual(threshold, self.gate.match_floor)
        self.assertLessEqual(threshold, self.gate.match_ceiling)
        self.assertGreater(threshold, 0.30)


if __name__ == "__main__":
    unittest.main()
