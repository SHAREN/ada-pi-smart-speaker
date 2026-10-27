import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from google.genai import types

from backend.assistant_harness import RESPONSE_POLICIES, ResponseClass, ResponsePolicy
from backend.realtime_provider import (
    GeminiLiveProvider,
    _compact_weather_payload,
    _weather_context_text,
    _weather_voice_summary,
)


class InterruptedSession:
    def __init__(self, provider: GeminiLiveProvider) -> None:
        self.provider = provider

    async def receive(self):
        yield types.LiveServerMessage(
            server_content=types.LiveServerContent(
                interrupted=True,
                model_turn=types.Content(
                    parts=[types.Part(inline_data=types.Blob(data=b"stale", mime_type="audio/pcm;rate=24000"))]
                ),
                output_transcription=types.Transcription(text="stale transcript"),
            )
        )
        self.provider._closed = True


class ExpressionToolSession:
    def __init__(self, provider: GeminiLiveProvider) -> None:
        self.provider = provider
        self.responses = []

    async def receive(self):
        yield types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(function_calls=[
                types.FunctionCall(
                    id="expression-call-1",
                    name="set_facial_expression",
                    args={"expression": "sassy"},
                )
            ])
        )
        self.provider._closed = True

    async def send_tool_response(self, *, function_responses):
        self.responses.extend(function_responses)


class OfficeStateToolSession(ExpressionToolSession):
    async def receive(self):
        yield types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(function_calls=[
                types.FunctionCall(id="office-state-1", name="get_office_state", args={})
            ])
        )
        self.provider._closed = True


class HabitObservationToolSession(ExpressionToolSession):
    async def receive(self):
        yield types.LiveServerMessage(tool_call=types.LiveServerToolCall(function_calls=[
            types.FunctionCall(id="habit-1",name="report_habit_observation",args={"challenge_id":"water-1","habit_key":"not_drinking_enough_water","observed":True,"confidence":.9,"reason":"visible drink"})
        ]))
        self.provider._closed=True


class HabitStatusToolSession(ExpressionToolSession):
    async def receive(self):
        yield types.LiveServerMessage(tool_call=types.LiveServerToolCall(function_calls=[
            types.FunctionCall(id="habit-status-1",name="get_habit_status",args={})
        ]))
        self.provider._closed=True


class EndConversationToolSession(ExpressionToolSession):
    async def receive(self):
        yield types.LiveServerMessage(
            server_content=types.LiveServerContent(
                input_transcription=types.Transcription(text="заверши разговор"),
            ),
            tool_call=types.LiveServerToolCall(function_calls=[
                types.FunctionCall(id="end-1", name="end_conversation", args={"reason": "user said goodbye"})
            ]),
        )
        self.provider._closed = True


class NaturalEndConversationToolSession(ExpressionToolSession):
    async def receive(self):
        yield types.LiveServerMessage(
            server_content=types.LiveServerContent(
                input_transcription=types.Transcription(text="ладно, мне уже пора"),
            ),
            tool_call=types.LiveServerToolCall(function_calls=[
                types.FunctionCall(id="end-natural-1", name="end_conversation", args={"reason": "user wants to leave the conversation"})
            ]),
        )
        self.provider._closed = True


class NoTranscriptEndConversationToolSession(ExpressionToolSession):
    async def receive(self):
        yield types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(function_calls=[
                types.FunctionCall(id="end-empty-1", name="end_conversation", args={"reason": "model guessed an end"})
            ]),
        )
        self.provider._closed = True


class PreTranscriptEndConversationToolSession(ExpressionToolSession):
    async def receive(self):
        yield types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(function_calls=[
                types.FunctionCall(
                    id="end-pre-asr-1",
                    name="end_conversation",
                    args={"reason": "User requested to end the conversation."},
                )
            ]),
        )
        self.provider._closed = True


class HomeAssistantControlToolSession(ExpressionToolSession):
    async def receive(self):
        yield types.LiveServerMessage(
            server_content=types.LiveServerContent(
                input_transcription=types.Transcription(text="включи свет на кухне"),
            ),
            tool_call=types.LiveServerToolCall(function_calls=[
                types.FunctionCall(
                    id="ha-control-1",
                    name="ha_call_service",
                    args={"domain": "light", "service": "turn_on", "entity_id": "light.kitchen"},
                )
            ]),
        )
        self.provider._closed = True


class WeatherToolSession(ExpressionToolSession):
    async def receive(self):
        yield types.LiveServerMessage(tool_call=types.LiveServerToolCall(function_calls=[
            types.FunctionCall(id="weather-1", name="get_local_weather", args={})
        ]))
        self.provider._closed = True


class ThinkingSignalToolSession(ExpressionToolSession):
    async def receive(self):
        yield types.LiveServerMessage(tool_call=types.LiveServerToolCall(function_calls=[
            types.FunctionCall(id="think-1", name="thinking_signal", args={})
        ]))
        self.provider._closed = True


class TinyAudioSentinelSession:
    def __init__(self, provider: GeminiLiveProvider) -> None:
        self.provider = provider

    async def receive(self):
        yield types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(parts=[
                    types.Part(inline_data=types.Blob(data=b"\x00\x00", mime_type="audio/pcm;rate=24000"))
                ])
            )
        )
        self.provider._closed = True


class ProviderEventTests(unittest.IsolatedAsyncioTestCase):
    def test_live_config_guards_long_full_duplex_sessions(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "backend/realtime_provider.py").read_text()
        self.assertIn('"disabled": True', source)
        self.assertIn("types.ActivityStart()", source)
        self.assertIn("types.ActivityEnd()", source)
        self.assertIn("ActivityHandling.START_OF_ACTIVITY_INTERRUPTS", source)
        self.assertIn('"language_code": self.language_code', source)
        self.assertIn('"language_codes": self.input_language_codes', source)
        self.assertIn('"custom_vocabulary": self.input_custom_vocabulary', source)
        provider = GeminiLiveProvider()
        self.assertEqual(provider.language_code, "ru-RU")
        self.assertEqual(provider.input_language_codes, ["ru-RU"])
        self.assertIn("Окей Набу", provider.input_custom_vocabulary)
        self.assertIn("Okay Nabu", provider.input_custom_vocabulary)
        self.assertIn("Бавлы", provider.input_custom_vocabulary)
        self.assertIn("Бавлах", provider.input_custom_vocabulary)
        self.assertIn("ContextWindowCompressionConfig", source)
        self.assertIn("SlidingWindow", source)
        self.assertIn("SessionResumptionConfig", source)
        self.assertIn("session_resumption_update", source)
        self.assertIn("message.go_away", source)

    def test_voice_prompt_prefers_silence_over_presence_chatter(self) -> None:
        provider = GeminiLiveProvider()
        prompt = provider.instructions.casefold()
        self.assertIn("listening and silence are valid default states", prompt)
        self.assertIn("never fill silence or a pause with presence/listening chatter", prompt)
        self.assertIn("do not offer further help or invite another question", prompt)
        self.assertIn("partial/incomplete phrase", prompt)
        self.assertIn("at most a terse 'да'", prompt)
        self.assertIn("do not narrate your availability", prompt)

    def test_recent_transcript_survives_turn_complete_for_tool_guard(self) -> None:
        provider = GeminiLiveProvider()
        provider._last_user_transcript = "включи свет на кухне"
        provider._last_user_transcript_at = time.monotonic()
        self.assertEqual(provider._effective_user_transcript(""), "включи свет на кухне")
        provider._last_user_transcript_at = time.monotonic() - 7.0
        self.assertEqual(provider._effective_user_transcript(""), "")

    async def test_interruption_discards_coalesced_stale_audio(self) -> None:
        provider = GeminiLiveProvider()
        provider._session = InterruptedSession(provider)
        events = [event async for event in provider.events()]
        self.assertEqual([event.type for event in events], ["response_interrupted"])

    async def test_zero_duration_pcm_sentinel_starts_response_without_audio_event(self) -> None:
        provider = GeminiLiveProvider()
        provider._session = TinyAudioSentinelSession(provider)
        events = [event async for event in provider.events()]
        self.assertEqual([event.type for event in events], ["response_started"])

    async def test_expression_tool_is_forwarded_and_spoken_reply_can_continue(self) -> None:
        provider = GeminiLiveProvider()
        session = ExpressionToolSession(provider)
        provider._session = session

        events = [event async for event in provider.events()]

        self.assertEqual([(event.type, event.data) for event in events], [
            ("expression", {"name": "sassy"})
        ])
        self.assertEqual(len(session.responses), 1)
        response = session.responses[0]
        self.assertEqual(response.id, "expression-call-1")
        self.assertEqual(response.scheduling, types.FunctionResponseScheduling.WHEN_IDLE)

    async def test_office_state_tool_returns_monitor_snapshot(self) -> None:
        state = {"status": "monitoring", "person_state": "not_home", "lights_on": ["light.left_office_light"]}
        provider = GeminiLiveProvider(office_state_getter=lambda: state)
        session = OfficeStateToolSession(provider)
        provider._session = session

        events = [event async for event in provider.events()]

        self.assertEqual(events, [])
        self.assertEqual(session.responses[0].response, {"output": state})
        self.assertEqual(session.responses[0].name, "get_office_state")

    async def test_habit_observation_tool_emits_structured_event(self) -> None:
        provider=GeminiLiveProvider(); session=HabitObservationToolSession(provider); provider._session=session
        events=[event async for event in provider.events()]
        self.assertEqual(events[0].type,"habit_observation")
        self.assertEqual(events[0].data["challenge_id"],"water-1")
        self.assertTrue(events[0].data["observed"])
        self.assertEqual(session.responses[0].name,"report_habit_observation")

    async def test_habit_status_tool_returns_complete_snapshot(self) -> None:
        snapshot={"window_days":7,"habits":[{"habit_key":"posture","lifecycle_status":"possible"}]}
        provider=GeminiLiveProvider(habit_state_getter=lambda:snapshot)
        session=HabitStatusToolSession(provider); provider._session=session
        events=[event async for event in provider.events()]
        self.assertEqual(events,[])
        self.assertEqual(session.responses[0].name,"get_habit_status")
        self.assertEqual(session.responses[0].response,{"output":snapshot})

    def test_weather_payload_is_compact_and_context_ready(self) -> None:
        raw = {
            "data": {
                "state": "sunny",
                "attributes": {
                    "temperature": 7,
                    "apparent_temperature": 5,
                    "temperature_unit": "°C",
                    "wind_speed": 3.6,
                    "wind_speed_unit": "km/h",
                    "forecast_hourly_icons": ["huge-icon-list-must-not-survive"],
                    "forecast_hourly": [
                        {
                            "datetime": "2026-10-25T13:00:00+05:00",
                            "condition": "sunny",
                            "native_temperature": 9,
                            "native_apparent_temperature": 6,
                            "native_wind_speed": 0.6,
                        }
                    ],
                    "forecast_twice_daily": [
                        {
                            "datetime": "2026-10-25T12:00:00+05:00",
                            "is_daytime": True,
                            "condition": "partlycloudy",
                            "native_temperature": 11,
                            "native_templow": -2,
                        }
                    ],
                    "attribution": "Data provided by Yandex Pogoda",
                },
            }
        }
        snapshot = _compact_weather_payload(raw, "weather.yandex_pogoda")
        snapshot["captured_at"] = "2026-10-25T13:10+05:00"
        snapshot["voice_summary"] = _weather_voice_summary(snapshot)
        self.assertTrue(snapshot["success"])
        self.assertEqual(snapshot["temperature"], 7)
        self.assertNotIn("forecast_hourly_icons", snapshot)
        self.assertEqual(
            snapshot["voice_summary"],
            "Сейчас +7°. Сегодня от −2° до +11°. Осадки не ожидаются.",
        )
        context = _weather_context_text(snapshot, ttl_seconds=300)
        self.assertIn("Default spoken weather summary", context)
        self.assertIn(snapshot["voice_summary"], context)
        self.assertIn("25.10 13:00 temp=9°C", context)
        self.assertIn("25.10 day temp=11°C low=-2°C", context)
        self.assertIn("fresh_for_about=5min", context)
        self.assertIn("get_local_weather", context)
        self.assertNotIn("Data provided by Yandex Pogoda", context)

    def test_weather_payload_accepts_bulk_state_wrapper(self) -> None:
        raw = {
            "data": {
                "success": True,
                "states": {
                    "weather.yandex_pogoda": {
                        "entity_id": "weather.yandex_pogoda",
                        "state": "cloudy",
                        "attributes": {"temperature": 3, "temperature_unit": "°C"},
                    }
                },
            }
        }
        snapshot = _compact_weather_payload(raw, "weather.yandex_pogoda")
        self.assertTrue(snapshot["success"])
        self.assertEqual(snapshot["condition"], "cloudy")
        self.assertEqual(snapshot["temperature"], 3)

    async def test_local_weather_tool_uses_cached_snapshot(self) -> None:
        provider = GeminiLiveProvider()
        provider._weather_cache = {
            "success": True,
            "entity_id": "weather.yandex_pogoda",
            "condition": "sunny",
            "temperature": 7,
            "temperature_unit": "°C",
            "hourly": [],
            "twice_daily": [],
            "captured_at": "2026-10-25T13:10+05:00",
        }
        provider._weather_cache_at = time.monotonic()
        session = WeatherToolSession(provider)
        provider._session = session
        events = [event async for event in provider.events()]
        self.assertEqual(events, [])
        self.assertEqual(session.responses[0].name, "get_local_weather")
        self.assertEqual(session.responses[0].response["temperature"], 7)
        self.assertEqual(session.responses[0].scheduling, types.FunctionResponseScheduling.WHEN_IDLE)

    async def test_end_conversation_tool_emits_semantic_end_event_silently(self) -> None:
        provider = GeminiLiveProvider()
        session = EndConversationToolSession(provider)
        provider._session = session
        events = [event async for event in provider.events()]
        self.assertEqual([(event.type, event.data) for event in events], [
            ("semantic_end_requested", {"reason": "user said goodbye"})
        ])
        self.assertEqual(session.responses[0].name, "end_conversation")
        self.assertTrue(session.responses[0].response["success"])
        self.assertEqual(session.responses[0].scheduling, types.FunctionResponseScheduling.SILENT)
        self.assertIn("Do not generate any spoken farewell", session.responses[0].response["instruction"])

    async def test_natural_end_wording_is_accepted_when_model_calls_tool(self) -> None:
        provider = GeminiLiveProvider()
        session = NaturalEndConversationToolSession(provider)
        provider._session = session
        events = [event async for event in provider.events()]
        self.assertEqual([(event.type, event.data) for event in events], [
            ("semantic_end_requested", {"reason": "user wants to leave the conversation"})
        ])
        self.assertEqual(session.responses[0].scheduling, types.FunctionResponseScheduling.SILENT)

    async def test_end_conversation_tool_without_current_transcript_is_blocked(self) -> None:
        provider = GeminiLiveProvider()
        session = NoTranscriptEndConversationToolSession(provider)
        provider._session = session
        events = [event async for event in provider.events()]
        self.assertEqual(events, [])
        self.assertFalse(session.responses[0].response["success"])
        self.assertTrue(session.responses[0].response["blocked"])
        self.assertEqual(session.responses[0].scheduling, types.FunctionResponseScheduling.WHEN_IDLE)

    async def test_end_conversation_accepts_fresh_voice_turn_before_asr_transcript(self) -> None:
        provider = GeminiLiveProvider()
        provider._voice_activity_sequence = 4
        provider._last_activity_start_at = time.monotonic() - 1.0
        provider._last_activity_end_at = time.monotonic() - 0.2
        provider._last_activity_end_sequence = 4
        session = PreTranscriptEndConversationToolSession(provider)
        provider._session = session
        events = [event async for event in provider.events()]
        self.assertEqual([(event.type, event.data) for event in events], [
            ("semantic_end_requested", {"reason": "User requested to end the conversation."})
        ])
        self.assertTrue(session.responses[0].response["success"])
        self.assertEqual(session.responses[0].scheduling, types.FunctionResponseScheduling.SILENT)
        self.assertEqual(provider._last_semantic_end_activity_sequence, 4)

    async def test_end_conversation_does_not_trust_stale_voice_activity_without_transcript(self) -> None:
        provider = GeminiLiveProvider()
        provider._voice_activity_sequence = 2
        provider._last_activity_start_at = time.monotonic() - 8.0
        provider._last_activity_end_at = time.monotonic() - 7.0
        provider._last_activity_end_sequence = 2
        session = PreTranscriptEndConversationToolSession(provider)
        provider._session = session
        events = [event async for event in provider.events()]
        self.assertEqual(events, [])
        self.assertFalse(session.responses[0].response["success"])
        self.assertTrue(session.responses[0].response["blocked"])

    async def test_semantic_end_scheduling_follows_response_policy(self) -> None:
        provider = GeminiLiveProvider()
        session = EndConversationToolSession(provider)
        provider._session = session
        spoken_policy = ResponsePolicy(
            speak=True,
            cue=False,
            max_sentences=1,
            question_allowed=False,
        )
        with patch.dict(
            RESPONSE_POLICIES,
            {ResponseClass.SILENT_SUCCESS: spoken_policy},
        ):
            events = [event async for event in provider.events()]
        self.assertEqual([(event.type, event.data) for event in events], [
            ("semantic_end_requested", {"reason": "user said goodbye"})
        ])
        self.assertEqual(
            session.responses[0].scheduling,
            types.FunctionResponseScheduling.WHEN_IDLE,
        )

    async def test_successful_home_assistant_control_result_is_silent(self) -> None:
        provider = GeminiLiveProvider()
        session = HomeAssistantControlToolSession(provider)
        provider._session = session
        with patch(
            "backend.realtime_provider.call_voice_tool",
            new=AsyncMock(return_value={"success": True, "_voice_feedback": "on"}),
        ):
            events = [event async for event in provider.events()]
        self.assertEqual([(event.type, event.data) for event in events], [
            ("control_feedback", {"kind": "on"}),
            ("silent_turn_completed", {}),
        ])
        self.assertEqual(session.responses[0].name, "ha_call_service")
        self.assertEqual(session.responses[0].scheduling, types.FunctionResponseScheduling.SILENT)
        self.assertNotIn("_voice_feedback", session.responses[0].response)

    async def test_home_control_delivery_follows_response_policy(self) -> None:
        provider = GeminiLiveProvider()
        session = HomeAssistantControlToolSession(provider)
        provider._session = session
        spoken_policy = ResponsePolicy(
            speak=True,
            cue=False,
            max_sentences=1,
            question_allowed=False,
        )
        with (
            patch.dict(
                RESPONSE_POLICIES,
                {ResponseClass.CUE_SUCCESS: spoken_policy},
            ),
            patch(
                "backend.realtime_provider.call_voice_tool",
                new=AsyncMock(return_value={"success": True, "_voice_feedback": "on"}),
            ),
        ):
            events = [event async for event in provider.events()]
        self.assertEqual(events, [])
        self.assertEqual(
            session.responses[0].scheduling,
            types.FunctionResponseScheduling.WHEN_IDLE,
        )

    async def test_failed_home_assistant_control_does_not_emit_success_cue(self) -> None:
        provider = GeminiLiveProvider()
        session = HomeAssistantControlToolSession(provider)
        provider._session = session
        with patch(
            "backend.realtime_provider.call_voice_tool",
            new=AsyncMock(return_value={"success": False, "error": "device unavailable"}),
        ):
            events = [event async for event in provider.events()]
        self.assertEqual(events, [])
        self.assertEqual(session.responses[0].scheduling, types.FunctionResponseScheduling.WHEN_IDLE)

    async def test_thinking_signal_tool_emits_local_event(self) -> None:
        provider = GeminiLiveProvider()
        session = ThinkingSignalToolSession(provider)
        provider._session = session
        events = [event async for event in provider.events()]
        self.assertEqual([(event.type, event.data) for event in events], [
            ("thinking_signal", {})
        ])
        self.assertEqual(session.responses[0].name, "thinking_signal")
        self.assertTrue(session.responses[0].response["success"])

    async def test_video_frame_uses_live_video_input(self) -> None:
        class VideoSession:
            def __init__(self) -> None:
                self.video = None

            async def send_realtime_input(self, *, video):
                self.video = video

        provider = GeminiLiveProvider()
        session = VideoSession()
        provider._session = session

        await provider.send_video(b"jpeg-frame")

        self.assertEqual(session.video.data, b"jpeg-frame")
        self.assertEqual(session.video.mime_type, "image/jpeg")

    async def test_habit_alert_sends_text_and_current_image_as_a_turn(self) -> None:
        class AlertSession:
            async def send_client_content(self, **kwargs):
                self.payload = kwargs

        provider = GeminiLiveProvider()
        session = AlertSession()
        provider._session = session
        await provider.send_habit_alert(b"current-frame", "first possible habit")
        self.assertTrue(session.payload["turn_complete"])
        parts = session.payload["turns"].parts
        self.assertEqual(parts[0].text, "first possible habit")
        self.assertEqual(parts[1].inline_data.data, b"current-frame")
        self.assertEqual(parts[1].inline_data.mime_type, "image/jpeg")

    async def test_text_turn_can_trigger_a_boot_greeting(self) -> None:
        class TextSession:
            async def send_client_content(self, **kwargs):
                self.payload = kwargs

        provider = GeminiLiveProvider()
        session = TextSession()
        provider._session = session
        await provider.send_text_turn("Give a short greeting")
        self.assertTrue(session.payload["turn_complete"])
        self.assertEqual(session.payload["turns"].parts[0].text, "Give a short greeting")


if __name__ == "__main__":
    unittest.main()
