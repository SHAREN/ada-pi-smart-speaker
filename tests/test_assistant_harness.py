import hashlib
import unittest

from backend.assistant_harness import (
    DEFAULT_ADA_INSTRUCTIONS,
    LIVE_POLICY_INSTRUCTIONS,
    RESPONSE_POLICIES,
    SCENARIO_RESPONSE_CLASSES,
    VOICE_ONLY_INSTRUCTIONS,
    ResponseClass,
    ResponseScenario,
    build_assistant_instructions,
    response_class_for_scenario,
    response_policy,
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

    def test_response_class_catalog_matches_roadmap(self) -> None:
        self.assertEqual(set(RESPONSE_POLICIES), set(ResponseClass))
        self.assertEqual(set(SCENARIO_RESPONSE_CLASSES), set(ResponseScenario))
        self.assertEqual(
            {item.value for item in ResponseClass},
            {
                "SILENT_SUCCESS",
                "CUE_SUCCESS",
                "SHORT_CONFIRMATION",
                "SHORT_ANSWER",
                "CLARIFY",
                "ERROR",
                "LONG_FORM",
            },
        )

    def test_nonverbal_success_classes_never_speak(self) -> None:
        silent = response_policy(ResponseClass.SILENT_SUCCESS)
        cue = response_policy(ResponseClass.CUE_SUCCESS)
        self.assertFalse(silent.speak)
        self.assertFalse(silent.cue)
        self.assertEqual(silent.max_sentences, 0)
        self.assertFalse(cue.speak)
        self.assertTrue(cue.cue)
        self.assertEqual(cue.max_sentences, 0)

    def test_spoken_classes_are_bounded_except_long_form(self) -> None:
        self.assertEqual(response_policy(ResponseClass.SHORT_CONFIRMATION).max_sentences, 1)
        self.assertEqual(response_policy(ResponseClass.SHORT_ANSWER).max_sentences, 2)
        self.assertEqual(response_policy(ResponseClass.CLARIFY).max_sentences, 1)
        self.assertEqual(response_policy(ResponseClass.ERROR).max_sentences, 1)
        self.assertIsNone(response_policy(ResponseClass.LONG_FORM).max_sentences)
        self.assertTrue(response_policy(ResponseClass.CLARIFY).question_allowed)
        self.assertFalse(response_policy(ResponseClass.SHORT_ANSWER).question_allowed)

    def test_roadmap_scenarios_have_deterministic_default_classes(self) -> None:
        expected = {
            ResponseScenario.SEMANTIC_END_SUCCESS: ResponseClass.SILENT_SUCCESS,
            ResponseScenario.SIMPLE_HOME_ACTION_SUCCESS: ResponseClass.CUE_SUCCESS,
            ResponseScenario.SETPOINT_SUCCESS: ResponseClass.SHORT_CONFIRMATION,
            ResponseScenario.STATE_QUERY: ResponseClass.SHORT_ANSWER,
            ResponseScenario.AMBIGUOUS_TARGET: ResponseClass.CLARIFY,
            ResponseScenario.FAILED_ACTION: ResponseClass.ERROR,
            ResponseScenario.DETAILED_REQUEST: ResponseClass.LONG_FORM,
        }
        self.assertEqual(SCENARIO_RESPONSE_CLASSES, expected)
        for scenario, response_class in expected.items():
            self.assertEqual(response_class_for_scenario(scenario), response_class)


if __name__ == "__main__":
    unittest.main()
