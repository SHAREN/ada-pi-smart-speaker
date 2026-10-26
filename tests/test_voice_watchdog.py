import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ada_voice_watchdog", ROOT / "scripts" / "voice-watchdog.py"
)
assert SPEC is not None and SPEC.loader is not None
watchdog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watchdog)


class VoiceWatchdogRouteTests(unittest.TestCase):
    def test_actual_sink_id_wins_over_stale_target_object(self) -> None:
        sinks = '[{"index":62,"name":"alsa_output.usb-Plantronics_Plantronics_P610_ABC-00.analog-stereo"},{"index":85,"name":"alsa_output.platform-hdmi.hdmi-stereo"}]'
        sink_inputs = '[{"index":103,"sink":85,"properties":{"target.object":"alsa_output.usb-Plantronics_Plantronics_P610_ABC-00.analog-stereo","node.name":"echo-cancel-playback"}}]'
        calls: list[list[str]] = []

        def fake_run(
            command: list[str], timeout: float = 8.0, *, stdout_only_on_success: bool = False
        ):
            calls.append(command)
            if command == ["pactl", "-f", "json", "list", "sinks"]:
                return True, sinks
            if command == ["pactl", "-f", "json", "list", "sink-inputs"]:
                return True, sink_inputs
            if command == ["pactl", "move-sink-input", "103", "62"]:
                return True, ""
            raise AssertionError(f"unexpected command: {command}")

        with patch.object(watchdog, "run", side_effect=fake_run):
            result = watchdog.ensure_aec_playback_on_p610()

        self.assertTrue(result["ok"])
        self.assertTrue(result["repaired"])
        self.assertEqual(result["previous_sink_id"], "85")
        self.assertEqual(result["p610_sink_id"], "62")
        self.assertIn(["pactl", "move-sink-input", "103", "62"], calls)

    def test_matching_actual_sink_needs_no_repair(self) -> None:
        sinks = '[{"index":62,"name":"alsa_output.usb-Plantronics_Plantronics_P610_ABC-00.analog-stereo"}]'
        sink_inputs = '[{"index":103,"sink":62,"properties":{"target.object":"alsa_output.usb-Plantronics_Plantronics_P610_ABC-00.analog-stereo","node.name":"echo-cancel-playback"}}]'

        def fake_run(
            command: list[str], timeout: float = 8.0, *, stdout_only_on_success: bool = False
        ):
            if command == ["pactl", "-f", "json", "list", "sinks"]:
                return True, sinks
            if command == ["pactl", "-f", "json", "list", "sink-inputs"]:
                return True, sink_inputs
            raise AssertionError(f"unexpected command: {command}")

        with patch.object(watchdog, "run", side_effect=fake_run):
            result = watchdog.ensure_aec_playback_on_p610()

        self.assertTrue(result["ok"])
        self.assertFalse(result["repaired"])
        self.assertEqual(result["route"], "sink-id-json")


if __name__ == "__main__":
    unittest.main()
