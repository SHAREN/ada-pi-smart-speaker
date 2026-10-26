import tempfile
import unittest
from pathlib import Path

from backend.conversation_memory import ConversationMemory


class ConversationMemoryTests(unittest.TestCase):
    def test_persists_and_reloads_recent_turns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.json"
            memory = ConversationMemory(path, max_turns=10, max_chars=1000)
            memory.append("user", "Напомни, о чём мы говорили")
            memory.append("assistant", "Мы обсуждали кофемашину.")

            reloaded = ConversationMemory(path, max_turns=10, max_chars=1000)
            prompt = reloaded.prompt_context()
            self.assertIn("User: Напомни, о чём мы говорили", prompt)
            self.assertIn("ADA: Мы обсуждали кофемашину.", prompt)
            self.assertEqual(reloaded.snapshot()["turn_count"], 2)

    def test_trims_oldest_turns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.json"
            memory = ConversationMemory(path, max_turns=2, max_chars=1000)
            memory.append("user", "one")
            memory.append("assistant", "two")
            memory.append("user", "three")
            prompt = memory.prompt_context()
            self.assertNotIn("User: one", prompt)
            self.assertIn("ADA: two", prompt)
            self.assertIn("User: three", prompt)

    def test_duplicate_adjacent_turn_is_not_written_twice(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.json"
            memory = ConversationMemory(path)
            memory.append("user", "  одна   фраза ")
            memory.append("user", "одна фраза")
            self.assertEqual(memory.snapshot()["turn_count"], 1)

    def test_single_letter_user_asr_noise_is_not_persisted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.json"
            memory = ConversationMemory(path)
            memory.append("user", "A")
            memory.append("user", "О")
            memory.append("user", "Да")
            self.assertEqual(memory.snapshot()["turn_count"], 1)
            self.assertIn("User: Да", memory.prompt_context())


if __name__ == "__main__":
    unittest.main()
