import unittest

from backend.ha_mcp_voice import (
    _curate_light_switch_fallback,
    _curate_search_result,
    _rewrite_search_arguments,
    guard_mutation,
    sanitize_schema,
)


class HomeAssistantVoiceGuardTests(unittest.TestCase):
    def test_read_only_tool_needs_no_control_intent(self):
        result = guard_mutation("ha_get_state", {"entity_id": "light.kitchen"}, "")
        self.assertTrue(result.allowed)

    def test_mutation_without_transcript_fails_closed(self):
        result = guard_mutation(
            "ha_bulk_control",
            {"operations": [{"entity_id": "light.kitchen", "action": "off"}]},
            "",
        )
        self.assertFalse(result.allowed)

    def test_admin_service_is_blocked(self):
        result = guard_mutation(
            "ha_call_service",
            {"domain": "homeassistant", "service": "restart", "entity_id": "homeassistant.home"},
            "перезапусти home assistant",
        )
        self.assertFalse(result.allowed)

    def test_raw_websocket_is_blocked(self):
        result = guard_mutation(
            "ha_call_service",
            {"ws_command": "config/entity_registry/list"},
            "сделай это",
        )
        self.assertFalse(result.allowed)

    def test_explicit_light_control_is_allowed(self):
        result = guard_mutation(
            "ha_bulk_control",
            {"operations": [{"entity_id": "light.kitchen", "action": "off"}]},
            "выключи свет на кухне",
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.arguments["operations"][0]["action"], "off")

    def test_power_direction_is_corrected_from_current_utterance(self):
        result = guard_mutation(
            "ha_call_service",
            {"domain": "light", "service": "turn_on", "entity_id": "light.kitchen"},
            "выключи свет на кухне",
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.arguments["service"], "turn_off")

    def test_deferred_immediate_control_is_blocked(self):
        result = guard_mutation(
            "ha_bulk_control",
            {"operations": [{"entity_id": "light.kitchen", "action": "off"}]},
            "через десять минут выключи свет на кухне",
        )
        self.assertFalse(result.allowed)

    def test_lock_domain_is_not_voice_enabled(self):
        result = guard_mutation(
            "ha_call_service",
            {"domain": "lock", "service": "unlock", "entity_id": "lock.front_door"},
            "открой входную дверь",
        )
        self.assertFalse(result.allowed)

    def test_optional_json_schema_is_flattened(self):
        schema = {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
            "description": "Optional name",
        }
        self.assertEqual(sanitize_schema(schema)["type"], "string")

    def test_kitchen_area_translation_is_rewritten_to_real_ha_area(self):
        args = _rewrite_search_arguments(
            {"area_filter": "kitchen", "domain_filter": "light"},
            "выключи свет на кухне",
        )
        self.assertEqual(args["area_filter"], "Кухня")
        self.assertEqual(args["domain_filter"], "light")
        self.assertNotIn("query", args)

    def test_kitchen_query_is_area_scoped_for_generic_light_command(self):
        args = _rewrite_search_arguments(
            {"query": "kitchen"},
            "включи свет на кухне",
        )
        self.assertEqual(args["area_filter"], "Кухня")
        self.assertEqual(args["domain_filter"], "light")
        self.assertNotIn("query", args)

    def test_generic_kitchen_light_keeps_bulb_and_strip_but_drops_led_ring(self):
        payload = {
            "domain_filter": "light",
            "area_filter": "Кухня",
            "entities": [
                {"entity_id": "light.home_assistant_voice_096153_led_ring", "friendly_name": "Home Assistant Voice 096153 LED Ring", "state": "unavailable", "aliases": []},
                {"entity_id": "light.lampochka_kukhnia", "friendly_name": "Лампочка кухня", "state": "off", "aliases": []},
                {"entity_id": "light.led_kitchen_kitchen_led_strip", "friendly_name": "LED kitchen Kitchen LED Strip", "state": "off", "aliases": []},
            ],
        }
        result = _curate_search_result(payload, "включи свет на кухне")
        self.assertEqual(
            [item["entity_id"] for item in result["entities"]],
            ["light.lampochka_kukhnia", "light.led_kitchen_kitchen_led_strip"],
        )

    def test_specific_kitchen_bulb_or_strip_is_resolved(self):
        payload = {
            "domain_filter": "light",
            "area_filter": "Кухня",
            "entities": [
                {"entity_id": "light.lampochka_kukhnia", "friendly_name": "Лампочка кухня", "state": "off", "aliases": []},
                {"entity_id": "light.led_kitchen_kitchen_led_strip", "friendly_name": "LED kitchen Kitchen LED Strip", "state": "off", "aliases": []},
            ],
        }
        bulb = _curate_search_result(payload, "включи лампочку на кухне")
        strip = _curate_search_result(payload, "включи подсветку на кухне")
        self.assertEqual([item["entity_id"] for item in bulb["entities"]], ["light.lampochka_kukhnia"])
        self.assertEqual([item["entity_id"] for item in strip["entities"]], ["light.led_kitchen_kitchen_led_strip"])

    def test_hall_generic_light_prefers_ha_aggregate_alias(self):
        payload = {
            "domain_filter": "light",
            "area_filter": "Зал",
            "entities": [
                {"entity_id": "light.torsher", "friendly_name": "Торшер", "state": "off", "aliases": []},
                {"entity_id": "light.torshery", "friendly_name": "Торшеры", "state": "off", "aliases": ["свет в зале", "основной свет в зале"]},
            ],
        }
        result = _curate_search_result(payload, "включи свет в зале")
        self.assertEqual([item["entity_id"] for item in result["entities"]], ["light.torshery"])

    def test_corridor_light_can_fallback_to_semantic_switch(self):
        payload = {
            "domain_filter": "switch",
            "area_filter": "Коридор",
            "entities": [
                {"entity_id": "switch.vykliuchatel_koridor", "friendly_name": "Свет в коридоре", "state": "unknown", "aliases": ["коридорный свет", "освещение в коридоре"]},
                {"entity_id": "switch.router_wifi", "friendly_name": "Router WiFi", "state": "on", "aliases": []},
            ],
        }
        result = _curate_light_switch_fallback(payload, "включи свет в коридоре")
        self.assertEqual([item["entity_id"] for item in result["entities"]], ["switch.vykliuchatel_koridor"])
        self.assertEqual(result["voice_resolution"]["domain"], "light")


if __name__ == "__main__":
    unittest.main()
