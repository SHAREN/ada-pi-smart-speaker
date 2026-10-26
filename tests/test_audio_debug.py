import json
import tempfile
import unittest
from pathlib import Path

from backend.audio_debug import AudioDebugSession


class AudioDebugSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_events_are_crash_safe_and_final_metadata_describes_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = AudioDebugSession(root, "browser123")
            session.record_event("assistant_chunk_received", {
                "sequence": 1,
                "bytes": 4800,
                "provider_gap_ms": 0.0,
            })

            # Realtime callers only enqueue diagnostics; the background writer owns
            # filesystem I/O. A barrier makes the asynchronous trace deterministic in tests.
            await session.flush_pending()
            rows = [json.loads(line) for line in session.events_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[-1]["event"], "assistant_chunk_received")
            self.assertEqual(rows[-1]["data"]["sequence"], 1)

            session.record_system_snapshot("playback_underrun", {"underruns": 1})
            session._capture_session_journal = lambda: session.session_journal_path.write_text("test journal\n", encoding="utf-8")
            await session.close("test")

            metadata = json.loads(session.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], 3)
            self.assertEqual(metadata["events_file"], "events.jsonl")
            self.assertIn("source_fingerprints", metadata)
            self.assertIn("events", metadata["files"])
            self.assertIn("speaker_render_stderr", metadata["files"])
            self.assertIn("physical_p610_mic_stderr", metadata["files"])
            self.assertIn("tv_reference", metadata["files"])
            self.assertIn("tv_reference_stderr", metadata["files"])
            self.assertIn("session_journal", metadata["files"])
            self.assertTrue(metadata["files"]["session_journal"]["exists"])
            snapshots = [item for item in metadata["timeline"] if item["event"] == "system_snapshot"]
            self.assertTrue(snapshots)
            self.assertEqual(snapshots[-1]["data"]["extra"].get("end_reason"), "test")


if __name__ == "__main__":
    unittest.main()
