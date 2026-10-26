import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import backend.ha_mcp_voice as ha_voice
from backend.ha_mcp_voice import (
    _RECENT_POWER_DISPATCH,
    _annotate_recent_power_state,
    _compact_weather_state_value,
    _control_feedback_kind,
    _prepare_immediate_voice_control,
    _remember_recent_power_dispatch,
    _curate_light_switch_fallback,
    _curate_search_result,
    _rewrite_search_arguments,
    guard_mutation,
    sanitize_schema,
)


class HomeAssistantVoiceGuardTests(unittest.TestCase):
    def setUp(self):
        _RECENT_POWER_DISPATCH.clear()

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

    def test_factual_state_correction_cannot_become_power_command(self):
        result = guard_mutation(
            "ha_bulk_control",
            {"operations": [{"entity_id": "switch.hall", "action": "off"}]},
            "нет, в коридоре свет выключен",
        )
        self.assertFalse(result.allowed)

    def test_first_person_past_report_cannot_become_power_command(self):
        result = guard_mutation(
            "ha_bulk_control",
            {"operations": [{"entity_id": "light.kitchen", "action": "off"}]},
            "нет, я всё в телевизоре выключил",
        )
        self.assertFalse(result.allowed)

    def test_correction_prefix_with_explicit_command_is_still_allowed(self):
        result = guard_mutation(
            "ha_bulk_control",
            {"operations": [{"entity_id": "light.kitchen", "action": "off"}]},
            "нет, выключи свет на кухне",
        )
        self.assertTrue(result.allowed)

    def test_guard_does_not_second_guess_model_semantics(self):
        result = guard_mutation(
            "ha_call_service",
            {"domain": "light", "service": "turn_on", "entity_id": "light.kitchen"},
            "выключи свет на кухне",
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.arguments["service"], "turn_on")
        self.assertEqual(_control_feedback_kind("ha_call_service", result.arguments), "on")

    def test_natural_language_without_allowlisted_verb_is_allowed(self):
        result = guard_mutation(
            "ha_bulk_control",
            {"operations": [{"entity_id": "light.kitchen", "action": "off"}]},
            "сделай на кухне темно",
        )
        self.assertTrue(result.allowed)

    def test_calendar_delete_does_not_use_phrase_allowlist(self):
        result = guard_mutation(
            "ha_config_remove_calendar_event",
            {"calendar_entity_id": "calendar.home", "event_id": "abc"},
            "это событие мне больше не нужно",
        )
        self.assertTrue(result.allowed)

    def test_natural_continuous_imperative_is_allowed(self):
        result = guard_mutation(
            "ha_bulk_control",
            {"operations": [{"entity_id": "light.kitchen", "action": "off"}]},
            "теперь обратно выключай",
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.arguments["operations"][0]["action"], "off")

    def test_operations_mode_drops_redundant_top_level_action(self):
        result = guard_mutation(
            "ha_bulk_control",
            {
                "operations": [{"entity_id": "light.kitchen", "action": "on"}],
                "action": "on",
            },
            "включай свет на кухне",
        )
        self.assertTrue(result.allowed)
        self.assertNotIn("action", result.arguments)
        self.assertEqual(result.arguments["operations"][0]["action"], "on")

    def test_simple_bulk_power_uses_zero_wait_and_background_confirmation(self):
        args, fast = _prepare_immediate_voice_control(
            "ha_bulk_control",
            {
                "operations": [
                    {"entity_id": "light.kitchen", "action": "on"},
                    {"entity_id": "switch.hall", "action": "on"},
                ]
            },
        )
        self.assertTrue(fast)
        self.assertTrue(args["parallel"])
        self.assertEqual([item["timeout_seconds"] for item in args["operations"]], [0, 0])

    def test_non_power_bulk_keeps_normal_confirmation(self):
        args, fast = _prepare_immediate_voice_control(
            "ha_bulk_control",
            {"operations": [{"entity_id": "vacuum.robot", "action": "start"}]},
        )
        self.assertFalse(fast)
        self.assertNotIn("timeout_seconds", args["operations"][0])

    def test_recent_unconfirmed_power_command_annotates_conflicting_state(self):
        _remember_recent_power_dispatch(
            {
                "results": [
                    {
                        "entity_id": "switch.hall",
                        "action": "off",
                        "command_sent": True,
                        "status": "dispatched_unconfirmed",
                    }
                ]
            }
        )
        payload = _annotate_recent_power_state(
            {
                "data": {
                    "success": True,
                    "states": {"switch.hall": {"entity_id": "switch.hall", "state": "on"}},
                }
            }
        )
        pending = payload["voice_recent_controls"]["pending"]
        self.assertEqual(pending[0]["entity_id"], "switch.hall")
        self.assertEqual(pending[0]["desired_state"], "off")
        self.assertEqual(pending[0]["reported_state"], "on")

    def test_recent_power_command_is_cleared_when_ha_confirms_state(self):
        _remember_recent_power_dispatch(
            {"results": [{"entity_id": "switch.hall", "action": "off", "command_sent": True}]}
        )
        payload = _annotate_recent_power_state(
            {"data": {"success": True, "states": {"switch.hall": {"entity_id": "switch.hall", "state": "off"}}}}
        )
        self.assertNotIn("voice_recent_controls", payload)
        self.assertNotIn("switch.hall", _RECENT_POWER_DISPATCH)

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

    def test_weather_state_is_compacted_before_generic_result_limit(self):
        raw = {
            "data": {
                "state": "sunny",
                "attributes": {
                    "temperature": 10,
                    "apparent_temperature": 8,
                    "temperature_unit": "°C",
                    "wind_speed": 1.8,
                    "wind_speed_unit": "km/h",
                    "forecast_hourly_icons": ["x" * 500 for _ in range(30)],
                    "forecast_hourly": [
                        {"datetime": f"2026-10-25T{hour % 24:02d}:00:00+05:00", "native_temperature": hour}
                        for hour in range(30)
                    ],
                    "forecast_twice_daily": [
                        {"datetime": f"2026-10-{25 + day:02d}T12:00:00+05:00", "native_temperature": day}
                        for day in range(8)
                    ],
                    "entity_picture": "x" * 5000,
                },
            }
        }
        compact = _compact_weather_state_value(raw, "weather.yandex_pogoda")
        attrs = compact["data"]["attributes"]
        self.assertEqual(len(attrs["forecast_hourly"]), 12)
        self.assertEqual(len(attrs["forecast_twice_daily"]), 6)
        self.assertNotIn("forecast_hourly_icons", attrs)
        self.assertNotIn("entity_picture", attrs)
        self.assertNotIn("truncated", ha_voice._bounded_result(compact))

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


class HomeAssistantVoiceConfirmationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _RECENT_POWER_DISPATCH.clear()
        ha_voice._VOICE_STATE_CACHE.clear()

    async def test_power_control_emits_success_only_after_state_confirmation(self):
        async def fake_call(_url, name, arguments):
            if name == "ha_bulk_control":
                return SimpleNamespace(isError=False), {
                    "results": [
                        {
                            "entity_id": "switch.hall",
                            "action": "off",
                            "command_sent": True,
                            "status": "dispatched_unconfirmed",
                        }
                    ],
                    "successful_commands": 1,
                    "failed_commands": 0,
                }
            if name == "ha_get_state":
                self.assertEqual(arguments["entity_id"], ["switch.hall"])
                return SimpleNamespace(isError=False), {
                    "data": {
                        "success": True,
                        "states": {
                            "switch.hall": {
                                "entity_id": "switch.hall",
                                "state": "off",
                                "last_changed": "2026-10-25T15:20:00+05:00",
                            }
                        },
                    }
                }
            raise AssertionError(name)

        with patch.dict(os.environ, {"HA_MCP_URL": "http://ha-mcp.test"}), patch.object(
            ha_voice, "_mcp_call", side_effect=fake_call
        ):
            result = await ha_voice.call_voice_tool(
                "ha_bulk_control",
                {"operations": [{"entity_id": "switch.hall", "action": "off"}]},
                transcript="выключи свет в коридоре",
            )

        self.assertEqual(result["voice_confirmation"], "confirmed")
        self.assertEqual(result["_voice_feedback"], "off")
        self.assertTrue(result["voice_state_snapshot"]["confirmed"])
        self.assertEqual(result["voice_state_snapshot"]["states"]["switch.hall"]["state"], "off")
        self.assertEqual(ha_voice._VOICE_STATE_CACHE["switch.hall"]["state"], "off")

    async def test_unconfirmed_power_control_is_not_reported_as_success(self):
        async def fake_call(_url, name, _arguments):
            if name == "ha_bulk_control":
                return SimpleNamespace(isError=False), {
                    "results": [
                        {
                            "entity_id": "switch.hall",
                            "action": "off",
                            "command_sent": True,
                            "status": "dispatched_unconfirmed",
                        }
                    ],
                    "successful_commands": 1,
                    "failed_commands": 0,
                }
            if name == "ha_get_state":
                return SimpleNamespace(isError=False), {
                    "data": {
                        "success": True,
                        "states": {"switch.hall": {"entity_id": "switch.hall", "state": "on"}},
                    }
                }
            raise AssertionError(name)

        with patch.dict(os.environ, {"HA_MCP_URL": "http://ha-mcp.test"}), patch.object(
            ha_voice, "POWER_CONFIRM_TIMEOUT_SECONDS", 0.01
        ), patch.object(ha_voice, "POWER_CONFIRM_POLL_SECONDS", 0.001), patch.object(
            ha_voice, "_mcp_call", side_effect=fake_call
        ):
            result = await ha_voice.call_voice_tool(
                "ha_bulk_control",
                {"operations": [{"entity_id": "switch.hall", "action": "off"}]},
                transcript="выключи свет в коридоре",
            )

        self.assertFalse(result["success"])
        self.assertTrue(result["unconfirmed"])
        self.assertEqual(result["voice_confirmation"], "unconfirmed")
        self.assertNotIn("_voice_feedback", result)
        self.assertFalse(result["voice_state_snapshot"]["confirmed"])
        self.assertEqual(result["voice_state_snapshot"]["states"]["switch.hall"]["state"], "on")


if __name__ == "__main__":
    unittest.main()
