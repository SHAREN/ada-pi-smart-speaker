import hashlib
import unittest

from backend.assistant_harness import (
    DEFAULT_ADA_INSTRUCTIONS,
    LIVE_POLICY_INSTRUCTIONS,
    VOICE_ONLY_INSTRUCTIONS,
    build_assistant_instructions,
)


class AssistantHarnessTests(unittest.TestCase):
    def test_default_prompt_is_byte_identical_to_pre_refactor_contract(self) -> None:
        prompt = build_assistant_instructions(DEFAULT_ADA_INSTRUCTIONS, voice_only=False)
        self.assertEqual(len(prompt), 9408)
        self.assertEqual(
            hashlib.sha256(prompt.encode()).hexdigest(),
            "bc70a22c5714aa0fd47e19d42510df722ff3e92354d0cb1a7cebda7cb920ba62",
        )

    def test_voice_only_prompt_is_byte_identical_to_pre_refactor_contract(self) -> None:
        prompt = build_assistant_instructions(DEFAULT_ADA_INSTRUCTIONS, voice_only=True)
        self.assertEqual(len(prompt), 9580)
        self.assertEqual(
            hashlib.sha256(prompt.encode()).hexdigest(),
            "d27bb1f7588fb57d80cc421f36815e42fb68c6837683a8b61b2f233a37dd6580",
        )

    def test_custom_base_instructions_are_preserved_and_policy_is_appended(self) -> None:
        prompt = build_assistant_instructions("Custom Ada base", voice_only=False)
        self.assertTrue(prompt.startswith("Custom Ada base\n\n"))
        self.assertTrue(prompt.endswith(LIVE_POLICY_INSTRUCTIONS))
        self.assertNotIn(VOICE_ONLY_INSTRUCTIONS, prompt)

    def test_voice_only_modifier_is_appended_only_in_voice_only_mode(self) -> None:
        full_prompt = build_assistant_instructions("base", voice_only=False)
        voice_only_prompt = build_assistant_instructions("base", voice_only=True)
        self.assertNotIn(VOICE_ONLY_INSTRUCTIONS, full_prompt)
        self.assertTrue(voice_only_prompt.endswith(VOICE_ONLY_INSTRUCTIONS))
        self.assertEqual(voice_only_prompt[:-len(VOICE_ONLY_INSTRUCTIONS)], full_prompt)


if __name__ == "__main__":
    unittest.main()
