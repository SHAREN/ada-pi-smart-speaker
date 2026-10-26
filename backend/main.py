"""FastAPI browser-to-realtime-provider audio bridge."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import signal
import time
import uuid
from contextlib import suppress
from collections import deque
from pathlib import Path

from PIL import Image

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .realtime_provider import DEFAULT_ADA_INSTRUCTIONS
from .pironman import EXPRESSION_COLORS, PironmanClient, PironmanError
from .camera import CameraHub
from .hailo_vision import HAILO_ROTATE_180, VisionBackendManager, letterbox
from .posture import PoseService, PostureMonitor, PostureStore
from .posture_verifier import GeminiClutterVerifier, GeminiPostureVerifier
from .visual_habits import VisualHabitService
from .live_manager import LiveSessionManager
from .home_assistant import HomeAssistantClient
from .office_lights import OfficeLightMonitor
from .wake_gate import WakeSessionGate
from .audio_debug import AudioDebugSession, cleanup_audio_debug
from .speaker_clock import speaker_clock
from .tv_reference_gate import TVReferenceGate
from .self_echo_gate import SelfEchoGate
from .conversation_memory import ConversationMemory

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = Path(os.environ.get("ADA_LOG_DIR", ROOT / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_LEVEL = getattr(logging, os.environ.get("ADA_LOG_LEVEL", "INFO").upper(), logging.INFO)
LOG_FORMAT = logging.Formatter(
    "%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)
if not root_logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(LOG_FORMAT)
    root_logger.addHandler(console_handler)
if not any(getattr(handler, "baseFilename", None) for handler in root_logger.handlers):
    file_handler = RotatingFileHandler(
        LOG_DIR / "ada-pi.log",
        maxBytes=int(os.environ.get("ADA_LOG_MAX_BYTES", 10 * 1024 * 1024)),
        backupCount=int(os.environ.get("ADA_LOG_BACKUPS", 5)),
    )
    file_handler.setFormatter(LOG_FORMAT)
    root_logger.addHandler(file_handler)
logger = logging.getLogger("voice.backend")

FRONTEND = ROOT / "frontend"
app = FastAPI(title="Pi Full-Duplex Voice Prototype")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
pironman = PironmanClient()
camera = CameraHub(fps=5)
vision_backend = VisionBackendManager()
pose_estimator = vision_backend.pose
object_detector = vision_backend.detection
posture_store = PostureStore(Path(os.environ.get("ADA_HABIT_DB", ROOT / "data" / "habits.db")))
posture_monitor = PostureMonitor(posture_store)
posture_verifier = GeminiPostureVerifier()


async def notify_office_light_habit(alert: dict[str, object]) -> None:
    """Turn a confirmed HA occurrence into an immediate spoken Live turn."""
    lights = [str(entity).removeprefix("light.").replace("_", " ") for entity in alert.get("lights_on", [])]
    count = int(alert.get("rolling_occurrences", 1))
    message = (
        "ADA office-light habit alert: Home Assistant has confirmed that the user "
        f"is away while these office lights remain on: {', '.join(lights)}. "
        f"This is confirmed occurrence {count} in the current rolling week. Speak now. "
        "Give one concise, composed, mildly sarcastic observation, then tell the user "
        "which lights were left on and suggest turning them off."
    )
    await live_manager.send_text_turn(message)


def habit_tool_snapshot() -> dict[str, object]:
    """Return all tracked habits and live state without exposing settings secrets."""
    profiles = {str(item["habit_key"]): item for item in posture_store.habit_profiles()}
    monitors = visual_habits.snapshot()
    habits: list[dict[str, object]] = []
    for key in ["posture", *monitors.keys(), "office_lights_left_on"]:
        profile = profiles.get(key)
        item: dict[str, object] = {
            "habit_key": key,
            "lifecycle_status": profile["status"] if profile else "not_observed",
            "rolling_occurrences": int(profile["rolling_occurrences"]) if profile else 0,
            "rolling_days": int(profile["rolling_days"]) if profile else 0,
        }
        if key in monitors:
            monitor = monitors[key]
            item.update(enabled=bool(monitor["settings"]["enabled"]),
                        monitor_state=monitor["state"], progress=monitor["progress"])
        elif key == "posture":
            status = posture_monitor.status()
            item.update(enabled=True, monitor_state=status["state"], calibrated=status["calibrated"])
        else:
            office = office_light_monitor.snapshot()
            item.update(enabled=True, monitor_state=office.get("status", "unavailable"))
        habits.append(item)
    return {"window_days": 7, "established_requirements": {"occurrences": 10, "days": 3}, "habits": habits}


conversation_memory = ConversationMemory(ROOT / "data" / "voice-conversation-memory.json")


def ada_system_prompt() -> str:
    base = posture_store.system_prompt(DEFAULT_ADA_INSTRUCTIONS)
    memory = conversation_memory.prompt_context()
    if not memory:
        return base
    return f"{base}\n\n{memory}"


live_manager = LiveSessionManager(
    ada_system_prompt,
    office_state_getter=lambda: office_light_monitor.snapshot(),
    habit_state_getter=habit_tool_snapshot,
)
voice_gate = WakeSessionGate(ROOT)
tv_reference_gate = TVReferenceGate()
self_echo_gate = SelfEchoGate()
VOICE_RUNTIME_STATE: dict[str, object] = {
    "browser_connected": False,
    "session_id": None,
    "assistant_response_active": False,
    "manual_activity_active": False,
    "barge_candidate_active": False,
    "tv_gate_armed": False,
    "cue": None,
    "last_event": None,
    "last_event_at": None,
    "interruption_count": 0,
    "raw_mic_chunks": 0,
    "raw_mic_bytes": 0,
    "forwarded_gemini_chunks": 0,
    "forwarded_gemini_bytes": 0,
}
WAKE_CUE = ROOT / "sounds" / "wake_word_triggered.wav"
END_CUE = ROOT / "sounds" / "session_end.wav"
THINKING_CUE = ROOT / "sounds" / "thinking.wav"
TIMER_CUE = ROOT / "sounds" / "timer_alarm.wav"
CONTROL_ON_CUE = ROOT / "sounds" / "control_on.wav"
CONTROL_OFF_CUE = ROOT / "sounds" / "control_off.wav"
pose_service = PoseService(camera, pose_estimator, posture_monitor, posture_verifier, live_manager)
home_assistant = HomeAssistantClient()
office_light_monitor = OfficeLightMonitor(
    home_assistant, posture_store, pose_service.office_occupied,
    notifier=notify_office_light_habit,
)
visual_habits = VisualHabitService(camera, pose_service, object_detector, posture_store, live_manager, GeminiClutterVerifier())
oled_guard_task: asyncio.Task[None] | None = None


async def sync_expression_lighting(expression: str) -> None:
    """Best-effort case lighting sync that never disrupts the voice session."""
    try:
        await pironman.set_expression_lighting(expression)
    except PironmanError as exc:
        logger.warning("could not sync Pironman lighting for %s: %s", expression, exc)


async def keep_oled_awake() -> None:
    """Enforce Ada's always-on display and maximum-cooling policies."""
    while True:
        try:
            fans_changed = await pironman.ensure_fans_max()
            if fans_changed:
                logger.info("Pironman fans set to always-on maximum speed")
        except PironmanError as exc:
            logger.warning("could not enforce Pironman maximum fan policy: %s", exc)
        try:
            oled_changed = await pironman.ensure_oled_on()
            if oled_changed:
                logger.info("Pironman OLED enabled with sleep disabled")
        except PironmanError as exc:
            logger.warning("could not enforce Pironman OLED policy: %s", exc)
        await asyncio.sleep(60)


@app.on_event("startup")
async def start_oled_guard() -> None:
    global oled_guard_task
    oled_guard_task = asyncio.create_task(keep_oled_awake())
    cleanup_audio_debug(ROOT)
    await tv_reference_gate.start()
    await self_echo_gate.start()
    await live_manager.start()
    await speaker_clock.start(on_clock_fire)
    await pose_service.start()
    await visual_habits.start()
    await office_light_monitor.start()


@app.on_event("shutdown")
async def stop_oled_guard() -> None:
    global oled_guard_task
    if oled_guard_task is not None:
        oled_guard_task.cancel()
        with suppress(asyncio.CancelledError):
            await oled_guard_task
        oled_guard_task = None
    await office_light_monitor.stop()
    await visual_habits.stop()
    await pose_service.stop()
    await speaker_clock.stop()
    vision_backend.close()
    await live_manager.stop()
    await self_echo_gate.stop()
    await tv_reference_gate.stop()
    await camera.stop()


def _note_voice_event(name: str) -> None:
    VOICE_RUNTIME_STATE["last_event"] = name
    VOICE_RUNTIME_STATE["last_event_at"] = time.time()


async def _play_voice_cue(path: Path, name: str) -> bool:
    """Play local feedback through the exact AEC render-reference sink."""
    if not path.is_file():
        logger.warning("voice cue missing: %s", path)
        return False
    VOICE_RUNTIME_STATE["cue"] = name
    _note_voice_event(f"cue:{name}:start")
    try:
        process = await asyncio.create_subprocess_exec(
            "pw-play", "--target=ada_aec_sink", str(path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        return_code = await process.wait()
        if return_code != 0:
            logger.warning("voice cue %s exited with code %s", name, return_code)
        return return_code == 0
    except Exception:
        logger.exception("voice cue %s failed", name)
        return False
    finally:
        VOICE_RUNTIME_STATE["cue"] = None
        _note_voice_event(f"cue:{name}:finish")


async def on_clock_fire(item: dict[str, object]) -> None:
    """Ring locally even when the conversational gate is in standby."""
    label = str(item.get("label") or "")
    kind = str(item.get("kind") or "timer")
    logger.info("local %s fired id=%s label=%r", kind, item.get("id"), label)
    _note_voice_event(f"clock:{kind}:fired")
    # Three short chimes are intentional: audible in a kitchen, but finite so a
    # missed local Stop cannot create an endless alarm loop.
    for _ in range(3):
        await _play_voice_cue(TIMER_CUE, f"{kind}_alarm")
        await asyncio.sleep(0.25)


@app.get("/api/voice/status")
async def voice_status() -> dict[str, object]:
    return {
        "gate": voice_gate.status(),
        "provider": live_manager.snapshot(),
        "runtime": dict(VOICE_RUNTIME_STATE),
        "tv_reference_gate": tv_reference_gate.snapshot(),
        "self_echo_gate": self_echo_gate.snapshot(),
        "conversation_memory": conversation_memory.snapshot(),
        "clock": speaker_clock.snapshot(),
    }


@app.middleware("http")
async def disable_frontend_cache(request, call_next):
    """Keep long-running kiosk Chromium sessions on the current UI code."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/vision/status")
async def vision_status() -> dict[str, object]:
    """Expose the active accelerator, models, fallback, and last error."""
    status = await asyncio.to_thread(vision_backend.status, True)
    return {**status, "camera": {
        "frame_bytes": len(camera.latest_frame) if camera.latest_frame else 0,
        "frame_age_seconds": round(time.monotonic() - camera.latest_frame_at, 3) if camera.latest_frame_at else None,
        "generation": camera.generation,
    }}


@app.get("/api/vision/frame.jpg")
async def vision_frame() -> Response:
    """Return the exact latest JPEG used by local vision for diagnostics."""
    if camera.latest_frame is None:
        raise HTTPException(status_code=503, detail="No camera frame is available")
    return Response(camera.latest_frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/vision/input.jpg")
async def vision_input() -> Response:
    """Return the exact letterboxed RGB image submitted to both Hailo models."""
    if camera.latest_frame is None:
        raise HTTPException(status_code=503, detail="No camera frame is available")
    rgb, _, _ = await asyncio.to_thread(letterbox, camera.latest_frame, 640, HAILO_ROTATE_180)
    with io.BytesIO() as encoded:
        Image.fromarray(rgb).save(encoded, format="JPEG", quality=90)
        return Response(encoded.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/pironman")
async def pironman_snapshot() -> dict[str, object]:
    """Return hardware telemetry and user-facing display settings."""
    try:
        return await pironman.snapshot()
    except PironmanError as exc:
        return {
            "online": False,
            "dashboard_url": f"{pironman.base_url}/small",
            "error": str(exc),
            "data": {},
            "config": {},
        }


@app.patch("/api/pironman/controls")
async def update_pironman_controls(request: Request) -> dict[str, object]:
    """Update the allow-listed, reversible case controls."""
    try:
        controls = await request.json()
        return await pironman.update_controls(controls)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PironmanError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/pironman/expression")
async def update_pironman_expression(request: Request) -> dict[str, str]:
    """Apply case lighting for a locally selected Ada expression."""
    try:
        payload = await request.json()
        expression = payload.get("expression") if isinstance(payload, dict) else None
        if expression not in EXPRESSION_COLORS:
            raise ValueError("Unsupported expression")
        await pironman.set_expression_lighting(expression)
        return {"expression": expression, "status": "applied"}
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PironmanError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/shutdown", status_code=202)
async def shutdown(request: Request, background_tasks: BackgroundTasks) -> dict[str, str]:
    """Allow only the locally displayed kiosk to stop this backend process."""
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1"}:
        raise HTTPException(status_code=403, detail="Shutdown is available only from this device")
    background_tasks.add_task(os.kill, os.getpid(), signal.SIGTERM)
    return {"status": "shutting_down"}


@app.get("/api/habits/posture")
async def posture_status() -> dict[str, object]:
    return {
        **posture_monitor.status(),
        "habit": posture_store.habit_profile("posture"),
        "events": posture_store.events(10),
    }


@app.get("/api/habits")
async def habit_status() -> dict[str, object]:
    """Return the complete habit catalog for the tracker screen."""
    return {
        "habits": posture_store.habit_profiles(),
        "established_requirements": {"occurrences": 10, "days": 3, "window_days": 7},
        "office_lights": office_light_monitor.snapshot(),
        "monitors": visual_habits.snapshot(),
        "notifications": posture_store.notifications(pending_visual_only=True),
    }


@app.get("/api/habits/office-lights")
async def office_light_status() -> dict[str, object]:
    return office_light_monitor.snapshot()


@app.get("/api/habits/monitors")
async def visual_monitor_status() -> dict[str, object]:
    return {"monitors": visual_habits.snapshot(), "notifications": posture_store.notifications(pending_visual_only=True)}


@app.patch("/api/habits/monitors/{habit_key}/settings")
async def update_visual_monitor_settings(habit_key: str, request: Request) -> dict[str, object]:
    try:
        payload = await request.json()
        if not isinstance(payload, dict): raise ValueError("Settings must be an object")
        visual_habits.update_settings(habit_key, payload)
        return visual_habits.snapshot()[habit_key]
    except KeyError as exc: raise HTTPException(status_code=404, detail="Unknown monitor") from exc
    except (ValueError, json.JSONDecodeError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/habits/monitors/{habit_key}/trigger", status_code=202)
async def trigger_visual_monitor(habit_key: str) -> dict[str, object]:
    try:
        return await visual_habits.trigger_check(habit_key)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/habits/desk-clutter/calibration", status_code=202)
async def calibrate_clean_desk() -> dict[str, object]:
    return visual_habits.begin_calibration()


@app.post("/api/habits/notifications/{notification_id}/acknowledge")
async def acknowledge_habit_notification(notification_id: int) -> dict[str, object]:
    if not posture_store.acknowledge_notification(notification_id): raise HTTPException(status_code=404, detail="Notification not found")
    return {"id": notification_id, "visual_acknowledged": True}


@app.patch("/api/habits/office-lights/settings")
async def update_office_light_settings(request: Request) -> dict[str, object]:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Settings must be an object")
        settings = office_light_monitor.update_settings(payload)
        return {**office_light_monitor.snapshot(), "settings": settings}
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/home-assistant/entities")
async def home_assistant_entities() -> dict[str, object]:
    try:
        return {"status": "connected", "entities": await home_assistant.entities()}
    except Exception as exc:
        logger.warning("could not load Home Assistant entities: %s", exc)
        return {"status": "unavailable", "error": str(exc), "entities": []}


@app.post("/api/home-assistant/entities/{entity_id}/power")
async def set_home_assistant_power(entity_id: str, request: Request) -> dict[str, object]:
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1"}:
        raise HTTPException(status_code=403, detail="Device control is available only from this device")
    try:
        payload = await request.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("on"), bool):
            raise ValueError("on must be true or false")
        return await home_assistant.set_power(entity_id, payload["on"])
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("Home Assistant control failed entity=%s: %s", entity_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.patch("/api/habits/posture/settings")
async def update_posture_settings(request: Request) -> dict[str, object]:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Settings must be an object")
        posture_store.update_settings(payload)
        return posture_monitor.status()
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/habits/posture/calibration/start", status_code=202)
async def start_posture_calibration() -> dict[str, object]:
    posture_monitor.start_calibration(kind="good")
    return posture_monitor.status()


@app.post("/api/habits/posture/calibration/start/{kind}", status_code=202)
async def start_typed_posture_calibration(kind: str) -> dict[str, object]:
    try:
        posture_monitor.start_calibration(kind=kind)
        return posture_monitor.status()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/habits/posture/calibration")
async def posture_calibration_status() -> dict[str, object]:
    return posture_monitor.status()


@app.get("/api/habits/posture/events")
async def posture_events(limit: int = 30) -> dict[str, object]:
    return {"events": posture_store.events(max(1, min(100, limit)))}


@app.post("/api/habits/posture/events/{event_id}/correction")
async def correct_posture_event(event_id: int, request: Request) -> dict[str, object]:
    try:
        payload = await request.json()
        correction = payload.get("correction") if isinstance(payload, dict) else None
        if not isinstance(correction, str):
            raise ValueError("correction is required")
        if not posture_store.correct_event(event_id, correction):
            raise HTTPException(status_code=404, detail="Posture event not found")
        return {"id": event_id, "correction": correction}
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/habits/reset")
async def reset_habit_data(request: Request) -> dict[str, object]:
    """Clear local habit history while preserving posture calibration."""
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1"}:
        raise HTTPException(status_code=403, detail="Habit reset is available only from this device")
    posture_monitor.clear_habit_history()
    visual_habits.clear_history()
    return {"status": "cleared", **posture_monitor.status(), "events": []}


@app.get("/api/settings/ada")
async def ada_settings() -> dict[str, object]:
    return {
        "system_prompt": posture_store.system_prompt(DEFAULT_ADA_INSTRUCTIONS),
        "live_status": live_manager.status,
        "live_error": live_manager.last_error,
    }


@app.patch("/api/settings/ada")
async def update_ada_settings(request: Request) -> dict[str, object]:
    try:
        payload = await request.json()
        prompt = payload.get("system_prompt") if isinstance(payload, dict) else None
        if not isinstance(prompt, str):
            raise ValueError("system_prompt is required")
        saved = posture_store.update_system_prompt(prompt)
        await live_manager.reload_prompt()
        return {"system_prompt": saved, "live_status": "reconnecting"}
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def send_json(socket: WebSocket, event_type: str, **data: object) -> None:
    await socket.send_text(json.dumps({"type": event_type, **data}))


@app.websocket("/ws/pose")
async def pose_socket(browser: WebSocket) -> None:
    """Stream the shared background pose results and camera frames."""
    await browser.accept()
    try:
        status = vision_backend.status()
        await send_json(browser, "pose_status", message=f"Pose tracking active · {status['backend']}", **status)
        async for result, frame in pose_service.results():
            await send_json(browser, "pose", **result)
            await browser.send_bytes(frame)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("pose stream failed")
        with suppress(Exception):
            await send_json(browser, "pose_error", message=str(exc))
            await browser.close(code=1011)


@app.websocket("/ws/habits")
async def habits_socket(browser: WebSocket) -> None:
    """Publish posture state, calibration progress, and new episode IDs."""
    await browser.accept()
    previous_habits: dict[str, dict[str, object]] | None = None
    try:
        while True:
            habits = posture_store.habit_profiles()
            current_habits = {str(item["habit_key"]): item for item in habits}
            habit_signal = None
            if previous_habits is not None:
                for key, habit in current_habits.items():
                    previous = previous_habits.get(key)
                    if previous is None:
                        habit_signal = {"kind": "first_added", **habit}
                        break
                    if habit["status"] != previous["status"]:
                        habit_signal = {"kind": "status_changed", **habit}
                        break
                    if habit["rolling_occurrences"] > previous["rolling_occurrences"]:
                        habit_signal = {"kind": "occurrence", **habit}
                        break
            await send_json(
                browser,
                "posture",
                **posture_monitor.status(),
                habit=posture_store.habit_profile("posture"),
                habits=habits,
                monitors=visual_habits.snapshot(),
                notifications=posture_store.notifications(pending_visual_only=True),
                habit_signal=habit_signal,
            )
            previous_habits = current_habits
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/detect")
async def detection_socket(browser: WebSocket) -> None:
    """Stream the background monitor's shared EfficientDet results."""
    # object_detector.infer is centralized in VisualHabitService so this page
    # cannot create a competing camera inference loop.
    await browser.accept()
    try:
        status = vision_backend.status()
        await send_json(browser, "detection_status", message=f"Object detection active · {status['backend']}", **status)
        async for result, frame in visual_habits.detections():
            await send_json(browser, "detection", **result)
            await browser.send_bytes(frame)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("detection stream failed")
        with suppress(Exception):
            await send_json(browser, "detection_error", message=str(exc))
            await browser.close(code=1011)


@app.websocket("/ws")
async def voice_socket(browser: WebSocket) -> None:
    await browser.accept()
    session_id = uuid.uuid4().hex[:10]
    logger.info("session=%s browser connected (%s)", session_id, browser.client)
    provider = live_manager
    closed = asyncio.Event()
    speech_active = asyncio.Event()
    activation_cue_done = asyncio.Event()
    activation_cue_done.set()
    finalize_lock = asyncio.Lock()
    video_mode = os.environ.get("ADA_VIDEO_MODE", "activity").lower()
    last_video_sent = 0.0
    forwarded_video_count = 0
    video_send_lock = asyncio.Lock()
    speech_clear_task: asyncio.Task[None] | None = None
    close_task: asyncio.Task[None] | None = None
    semantic_close_watchdog_task: asyncio.Task[None] | None = None
    semantic_close_timeout = max(
        1.0, float(os.environ.get("ADA_SEMANTIC_CLOSE_TIMEOUT_SECONDS", "4.0"))
    )
    local_speech_detected = False
    local_speech_stop_grace_until = 0.0
    assistant_response_active = False
    # Backend-authoritative speaking lock. It stays set from response_started
    # until the browser AudioWorklet reports that the actual PCM queue drained.
    # response_completed only means Gemini finished generating, not that the
    # speaker finished playing.
    assistant_playback_locked = False
    playback_drain_wait_started = 0.0
    playback_drain_timeout_seconds = max(
        2.0, float(os.environ.get("ADA_PLAYBACK_DRAIN_TIMEOUT_SECONDS", "4.0"))
    )
    last_assistant_audio_at = 0.0
    response_stall_timeout_seconds = max(
        6.0, float(os.environ.get("ADA_RESPONSE_STALL_TIMEOUT_SECONDS", "12.0"))
    )
    assistant_echo_guard_until = 0.0
    assistant_memory_parts: list[str] = []
    semantic_close_pending = False
    tv_gate_armed = False
    manual_activity_active = False
    # Kept in status for compatibility/diagnostics only. Local VAD boundaries are
    # not Gemini turn boundaries, so this value must never drive response routing.
    pending_user_turns = 0
    audio_debug_session: AudioDebugSession | None = None

    VOICE_RUNTIME_STATE.update(
        browser_connected=True,
        session_id=session_id,
        assistant_response_active=False,
        assistant_playback_locked=False,
        playback_drain_timeout_seconds=playback_drain_timeout_seconds,
        playback_drain_timeout_count=0,
        response_stall_timeout_seconds=response_stall_timeout_seconds,
        response_stall_timeout_count=0,
        provider_reconnect_recovery_count=0,
        manual_activity_active=False,
        pending_user_turns=0,
        barge_candidate_active=False,
        overlap_speech_active=False,
        overlap_turn_pending=False,
        tv_gate_armed=False,
    )
    live_manager.set_playback_busy(False)
    _note_voice_event("browser_connected")

    async def send_session_state() -> None:
        with suppress(WebSocketDisconnect, RuntimeError):
            await send_json(
                browser,
                "session_state",
                **voice_gate.status(),
                provider=live_manager.snapshot(),
            )

    def remember_conversation_turn(role: str, text: str) -> None:
        try:
            conversation_memory.append(role, text)
        except Exception:
            logger.exception("failed to persist %s conversation turn", role)

    async def play_wake_cue() -> None:
        activation_cue_done.clear()
        if audio_debug_session is not None:
            audio_debug_session.record_event("wake_cue_started", {})
        try:
            await _play_voice_cue(WAKE_CUE, "wake")
        finally:
            if audio_debug_session is not None:
                audio_debug_session.record_event("wake_cue_finished", {})
            # The wake phrase and everything heard while the activation cue is
            # playing are control audio, not a Gemini user turn. Start listening
            # from a clean boundary only after the cue has finished.
            voice_gate.clear_activation_buffer()
            voice_gate.reset_after_wake_cue()
            activation_cue_done.set()
            await send_session_state()

    async def finalize_session(reason: str, *, clear_playback: bool) -> None:
        nonlocal assistant_response_active, assistant_playback_locked, playback_drain_wait_started, assistant_echo_guard_until, semantic_close_pending, tv_gate_armed, manual_activity_active, pending_user_turns, audio_debug_session, local_speech_detected, local_speech_stop_grace_until, semantic_close_watchdog_task
        async with finalize_lock:
            if voice_gate.state == "standby":
                return
            if voice_gate.state == "active":
                voice_gate.begin_close(reason)
            _note_voice_event(f"session_close:{reason}")
            if audio_debug_session is not None:
                audio_debug_session.record_event(
                    "semantic_end" if reason == "semantic" else reason, {}
                )
            assistant_response_active = False
            assistant_playback_locked = False
            playback_drain_wait_started = 0.0
            assistant_echo_guard_until = 0.0
            VOICE_RUNTIME_STATE["assistant_response_active"] = False
            VOICE_RUNTIME_STATE["assistant_playback_locked"] = False
            live_manager.set_playback_busy(False)
            VOICE_RUNTIME_STATE["barge_candidate_active"] = False
            if manual_activity_active:
                with suppress(Exception):
                    await provider.send_activity_end()
                manual_activity_active = False
                VOICE_RUNTIME_STATE["manual_activity_active"] = False
            pending_user_turns = 0
            VOICE_RUNTIME_STATE["pending_user_turns"] = 0
            if clear_playback:
                with suppress(Exception):
                    await send_json(browser, "clear_audio")
            # A semantic end is finalized only after the farewell response has
            # completed. Give the browser's small jitter queue time to drain.
            if reason == "semantic" and not clear_playback:
                await asyncio.sleep(0.35)
            if reason != "first_turn_timeout":
                if audio_debug_session is not None:
                    audio_debug_session.record_event("end_cue_started", {})
                await _play_voice_cue(END_CUE, "end")
                if audio_debug_session is not None:
                    audio_debug_session.record_event("end_cue_finished", {})
            elif audio_debug_session is not None:
                audio_debug_session.record_event("silent_first_turn_timeout", {})
            if audio_debug_session is not None:
                audio_debug_session.record_event("conversation_context_preserved", {})
            # Normal local session boundaries must not call fresh_session().
            # Keeping the persistent Live session (and its resumption handle)
            # preserves conversational context across Stop/idle/semantic close
            # and removes the multi-second reconnect delay before the next wake.
            voice_gate.finish_standby()
            semantic_close_pending = False
            if (
                semantic_close_watchdog_task is not None
                and semantic_close_watchdog_task is not asyncio.current_task()
                and not semantic_close_watchdog_task.done()
            ):
                semantic_close_watchdog_task.cancel()
            semantic_close_watchdog_task = None
            tv_gate_armed = False
            VOICE_RUNTIME_STATE["tv_gate_armed"] = False
            local_speech_detected = False
            local_speech_stop_grace_until = 0.0
            speech_active.clear()
            await send_session_state()
            if audio_debug_session is not None:
                VOICE_RUNTIME_STATE["last_audio_debug_dir"] = str(audio_debug_session.dir)
                await audio_debug_session.close(reason)
                audio_debug_session = None
            logger.info("session=%s returned to standby reason=%s", session_id, reason)

    def ensure_finalize(reason: str, *, clear_playback: bool) -> None:
        nonlocal close_task
        if close_task is not None and not close_task.done():
            return
        close_task = asyncio.create_task(
            finalize_session(reason, clear_playback=clear_playback),
            name=f"ada-finalize-{reason}",
        )

    async def semantic_close_watchdog() -> None:
        try:
            await asyncio.sleep(semantic_close_timeout)
        except asyncio.CancelledError:
            return
        if not semantic_close_pending or voice_gate.state == "standby":
            return
        if close_task is not None and not close_task.done():
            return
        logger.warning(
            "session=%s semantic close timed out after %.1fs; forcing standby",
            session_id,
            semantic_close_timeout,
        )
        if audio_debug_session is not None:
            audio_debug_session.record_event(
                "semantic_close_watchdog_timeout",
                {
                    "timeout_seconds": semantic_close_timeout,
                    "assistant_response_active": assistant_response_active,
                    "manual_activity_active": manual_activity_active,
                },
            )
        ensure_finalize("semantic", clear_playback=True)

    try:
        await provider.wait_connected()
    except Exception as exc:
        logger.exception("could not open AI session")
        with suppress(Exception):
            await send_json(browser, "error", message=str(exc))
            await browser.close(code=1011)
        VOICE_RUNTIME_STATE["browser_connected"] = False
        return

    try:
        await send_json(browser, "ready")
        await send_session_state()
    except (WebSocketDisconnect, RuntimeError):
        VOICE_RUNTIME_STATE["browser_connected"] = False
        return

    async def browser_to_provider() -> None:
        nonlocal speech_clear_task, audio_debug_session, local_speech_detected, local_speech_stop_grace_until, tv_gate_armed, manual_activity_active, assistant_playback_locked, playback_drain_wait_started, assistant_echo_guard_until
        raw_packet_count = 0
        forwarded_packet_count = 0
        last_audio_log = time.monotonic()
        tv_gate_suppressing = False
        tv_probable_only = False
        self_echo_suppressing = False
        tv_delay_buffer: deque[tuple[bytes, bool]] = deque()
        tv_delay_bytes = 0
        manual_preroll: deque[bytes] = deque()
        manual_preroll_bytes = 0
        manual_preroll_limit = int(0.30 * 16000 * 2)
        browser_speech_active = False
        barge_candidate_since = 0.0
        local_interrupt_listen_until = 0.0
        local_interrupt_window_seconds = max(
            0.8, float(os.environ.get("ADA_LOCAL_INTERRUPT_WINDOW_SECONDS", "2.5"))
        )
        overlap_speech_active = False
        overlap_turn_pending = False
        overlap_audio: deque[bytes] = deque()
        overlap_audio_bytes = 0
        overlap_audio_limit = int(
            max(2.0, float(os.environ.get("ADA_OVERLAP_BUFFER_SECONDS", "20"))) * 16000 * 2
        )
        barge_confirm_seconds = max(
            0.20, float(os.environ.get("ADA_BARGE_CONFIRM_MS", "300")) / 1000.0
        )
        playback_reopen_delay_seconds = max(
            0.20, float(os.environ.get("ADA_PLAYBACK_REOPEN_DELAY_MS", "500")) / 1000.0
        )

        async def forward_gemini_audio(chunk: bytes) -> None:
            nonlocal forwarded_packet_count
            await provider.send_audio(chunk)
            VOICE_RUNTIME_STATE["forwarded_gemini_chunks"] = int(
                VOICE_RUNTIME_STATE.get("forwarded_gemini_chunks") or 0
            ) + 1
            VOICE_RUNTIME_STATE["forwarded_gemini_bytes"] = int(
                VOICE_RUNTIME_STATE.get("forwarded_gemini_bytes") or 0
            ) + len(chunk)
            if audio_debug_session is not None:
                audio_debug_session.write_gemini_input(chunk)
            forwarded_packet_count += 1

        async def begin_manual_activity(reason: str) -> None:
            nonlocal manual_activity_active, manual_preroll_bytes
            if manual_activity_active:
                return
            # Local VAD segments are transport hints only. Gemini Live may merge
            # multiple segments into one semantic user turn, so never infer queue
            # depth or response staleness from these boundaries.
            await provider.send_activity_start()
            manual_activity_active = True
            VOICE_RUNTIME_STATE["manual_activity_active"] = True
            logger.info(
                "session=%s Gemini manual activity_start reason=%s preroll_chunks=%d",
                session_id, reason, len(manual_preroll),
            )
            if audio_debug_session is not None:
                audio_debug_session.record_event(
                    "manual_activity_start", {"reason": reason, "preroll_chunks": len(manual_preroll)}
                )
            while manual_preroll:
                await forward_gemini_audio(manual_preroll.popleft())
            manual_preroll_bytes = 0

        async def end_manual_activity(reason: str) -> None:
            nonlocal manual_activity_active, manual_preroll_bytes
            if not manual_activity_active:
                manual_preroll.clear()
                manual_preroll_bytes = 0
                return
            await provider.send_activity_end()
            manual_activity_active = False
            VOICE_RUNTIME_STATE["manual_activity_active"] = False
            VOICE_RUNTIME_STATE["pending_user_turns"] = 0
            manual_preroll.clear()
            manual_preroll_bytes = 0
            logger.info("session=%s Gemini manual activity_end reason=%s", session_id, reason)
            if audio_debug_session is not None:
                audio_debug_session.record_event("manual_activity_end", {"reason": reason})

        async def flush_overlap_turn(reason: str) -> None:
            """Replay speech heard during ADA TTS without making it interrupt TTS."""
            nonlocal manual_activity_active, manual_preroll_bytes, overlap_turn_pending, overlap_audio_bytes
            if not overlap_turn_pending or not overlap_audio or manual_activity_active:
                return
            await provider.send_activity_start()
            manual_activity_active = True
            VOICE_RUNTIME_STATE["manual_activity_active"] = True
            buffered_chunks = len(overlap_audio)
            buffered_bytes = overlap_audio_bytes
            while overlap_audio:
                await forward_gemini_audio(overlap_audio.popleft())
            overlap_audio_bytes = 0
            overlap_turn_pending = False
            VOICE_RUNTIME_STATE["overlap_turn_pending"] = False
            manual_preroll.clear()
            manual_preroll_bytes = 0
            logger.info(
                "session=%s replaying overlap speech after ADA playback reason=%s chunks=%d bytes=%d still_speaking=%s",
                session_id, reason, buffered_chunks, buffered_bytes, overlap_speech_active,
            )
            if audio_debug_session is not None:
                audio_debug_session.record_event(
                    "overlap_turn_replayed",
                    {
                        "reason": reason,
                        "chunks": buffered_chunks,
                        "bytes": buffered_bytes,
                        "still_speaking": overlap_speech_active,
                    },
                )
            if not overlap_speech_active:
                await provider.send_activity_end()
                manual_activity_active = False
                VOICE_RUNTIME_STATE["manual_activity_active"] = False
                logger.info("session=%s overlap speech replay completed as closed turn", session_id)

        async def confirm_local_speech(reason: str) -> None:
            nonlocal local_speech_detected, local_speech_stop_grace_until
            nonlocal speech_clear_task, barge_candidate_since
            barge_candidate_since = 0.0
            VOICE_RUNTIME_STATE["barge_candidate_active"] = False
            local_speech_detected = True
            local_speech_stop_grace_until = 0.0
            voice_gate.confirm_first_turn()
            voice_gate.note_activity()
            if speech_clear_task is not None:
                speech_clear_task.cancel()
                speech_clear_task = None
            speech_active.set()
            await begin_manual_activity(reason)
            if audio_debug_session is not None:
                audio_debug_session.record_event("local_speech_confirmed", {"reason": reason})
            if video_mode == "activity":
                await forward_video_frame(camera.latest_frame)

        def reject_barge_candidate(reason: str) -> None:
            nonlocal barge_candidate_since, manual_preroll_bytes
            if not barge_candidate_since:
                return
            logger.info("session=%s barge-in candidate rejected reason=%s", session_id, reason)
            if audio_debug_session is not None:
                audio_debug_session.record_event("barge_candidate_rejected", {"reason": reason})
            barge_candidate_since = 0.0
            VOICE_RUNTIME_STATE["barge_candidate_active"] = False
            manual_preroll.clear()
            manual_preroll_bytes = 0
        try:
            while True:
                message = await browser.receive()
                if message["type"] == "websocket.disconnect":
                    break
                pcm16 = message.get("bytes")
                if pcm16:
                    raw_packet_count += 1
                    VOICE_RUNTIME_STATE["raw_mic_chunks"] = int(
                        VOICE_RUNTIME_STATE.get("raw_mic_chunks") or 0
                    ) + 1
                    VOICE_RUNTIME_STATE["raw_mic_bytes"] = int(
                        VOICE_RUNTIME_STATE.get("raw_mic_bytes") or 0
                    ) + len(pcm16)
                    if voice_gate.state == "standby" and tv_reference_gate.wake_guard_enabled:
                        tv_decision = tv_reference_gate.classify_wake_candidate(pcm16)
                    elif tv_gate_armed and voice_gate.active:
                        tv_decision = tv_reference_gate.classify_active(pcm16)
                    else:
                        tv_decision = tv_reference_gate.observe(
                            pcm16, learn=not voice_gate.active
                        )
                    self_echo_decision = self_echo_gate.classify(
                        pcm16,
                        assistant_active=bool(
                            voice_gate.active
                            and (
                                assistant_playback_locked
                                or assistant_response_active
                                or time.monotonic() < assistant_echo_guard_until
                            )
                        ),
                    )
                    if self_echo_decision.suppress != self_echo_suppressing:
                        self_echo_suppressing = self_echo_decision.suppress
                        match_score = (
                            self_echo_decision.match_score
                            if self_echo_decision.match_score is not None
                            else -1.0
                        )
                        logger.info(
                            "session=%s self-echo gate %s mic=%.5f ref=%.5f match=%.3f "
                            "trigger=%.3f delay=%.0fms ready=%s near_end=%s",
                            session_id,
                            "suppressing" if self_echo_suppressing else "open",
                            self_echo_decision.mic_rms,
                            self_echo_decision.reference_rms,
                            match_score,
                            self_echo_gate.match_threshold,
                            self_echo_decision.match_delay_ms,
                            self_echo_decision.ready,
                            self_echo_decision.near_end_hold,
                        )
                        if audio_debug_session is not None:
                            audio_debug_session.record_event(
                                "self_echo_gate", {
                                    "suppressing": self_echo_suppressing,
                                    "mic_rms": round(self_echo_decision.mic_rms, 6),
                                    "reference_rms": round(self_echo_decision.reference_rms, 6),
                                    "match_score": (
                                        round(self_echo_decision.match_score, 4)
                                        if self_echo_decision.match_score is not None else None
                                    ),
                                    "match_threshold": self_echo_gate.match_threshold,
                                    "match_delay_ms": round(self_echo_decision.match_delay_ms, 1),
                                    "near_end_hold": self_echo_decision.near_end_hold,
                                },
                            )
                    filtered_pcm16 = (
                        b"\x00" * len(pcm16) if self_echo_decision.suppress else pcm16
                    )
                    if overlap_speech_active and (assistant_response_active or assistant_playback_locked):
                        overlap_audio.append(filtered_pcm16)
                        overlap_audio_bytes += len(filtered_pcm16)
                        while overlap_audio and overlap_audio_bytes > overlap_audio_limit:
                            dropped = overlap_audio.popleft()
                            overlap_audio_bytes -= len(dropped)
                    if tv_decision.suppress != tv_gate_suppressing:
                        tv_gate_suppressing = tv_decision.suppress
                        match_score = (
                            tv_decision.match_score
                            if tv_decision.match_score is not None
                            else -1.0
                        )
                        logger.info(
                            "session=%s TV content gate %s mic=%.5f physical=%.5f "
                            "ref=%.5f match=%.3f trigger=%.3f delay=%.0fms ready=%s",
                            session_id,
                            "suppressing" if tv_gate_suppressing else "open",
                            tv_decision.mic_rms,
                            tv_decision.physical_rms,
                            tv_decision.ref_rms,
                            match_score,
                            tv_decision.match_threshold,
                            tv_decision.match_delay_ms,
                            tv_decision.ready,
                        )
                        if (
                            not tv_gate_suppressing
                            and tv_decision.near_end_hold
                            and not assistant_response_active
                            and not self_echo_decision.suppress
                        ):
                            # The content matcher is an independent near-end VAD only
                            # while ADA is quiet. During assistant playback, barge-in
                            # must pass the render-aware confirmation path below; a
                            # transient near-end classification must never authorize
                            # Stop against ADA's own speaker echo.
                            local_speech_detected = True
                            local_speech_stop_grace_until = 0.0
                            voice_gate.note_activity()
                            if speech_clear_task is not None:
                                speech_clear_task.cancel()
                                speech_clear_task = None
                            speech_active.set()
                        elif tv_gate_suppressing and local_speech_detected:
                            local_speech_detected = False
                            local_speech_stop_grace_until = time.monotonic() + 1.0
                            if speech_clear_task is not None:
                                speech_clear_task.cancel()
                            speech_clear_task = asyncio.create_task(
                                clear_speech_activity_after_grace()
                            )
                        if audio_debug_session is not None:
                            audio_debug_session.record_event(
                                "tv_reference_gate", {
                                    "suppressing": tv_gate_suppressing,
                                    "mic_rms": round(tv_decision.mic_rms, 6),
                                    "physical_rms": round(tv_decision.physical_rms, 6),
                                    "ref_rms": round(tv_decision.ref_rms, 6),
                                    "match_score": (
                                        round(tv_decision.match_score, 4)
                                        if tv_decision.match_score is not None else None
                                    ),
                                    "match_threshold": round(tv_decision.match_threshold, 4),
                                    "match_delay_ms": round(tv_decision.match_delay_ms, 1),
                                    "near_end_hold": tv_decision.near_end_hold,
                                }
                            )

                    if tv_decision.probable_tv_only != tv_probable_only:
                        tv_probable_only = tv_decision.probable_tv_only
                        logger.info(
                            "session=%s TV reference observation probable_tv_only=%s "
                            "suppression_enabled=%s mic=%.5f physical=%.5f ref=%.5f "
                            "match=%s trigger=%.3f delay=%.0fms ready=%s",
                            session_id,
                            tv_probable_only,
                            tv_reference_gate.enabled,
                            tv_decision.mic_rms,
                            tv_decision.physical_rms,
                            tv_decision.ref_rms,
                            (
                                f"{tv_decision.match_score:.3f}"
                                if tv_decision.match_score is not None else "n/a"
                            ),
                            tv_decision.match_threshold,
                            tv_decision.match_delay_ms,
                            tv_decision.ready,
                        )
                        if audio_debug_session is not None:
                            audio_debug_session.record_event(
                                "tv_reference_observation", {
                                    "probable_tv_only": tv_probable_only,
                                    "suppression_enabled": tv_reference_gate.enabled,
                                    "mic_rms": round(tv_decision.mic_rms, 6),
                                    "physical_rms": round(tv_decision.physical_rms, 6),
                                    "ref_rms": round(tv_decision.ref_rms, 6),
                                    "match_score": (
                                        round(tv_decision.match_score, 4)
                                        if tv_decision.match_score is not None else None
                                    ),
                                    "match_threshold": round(tv_decision.match_threshold, 4),
                                    "match_delay_ms": round(tv_decision.match_delay_ms, 1),
                                    "near_end_hold": tv_decision.near_end_hold,
                                },
                            )

                    # A browser VAD start that occurs while ADA is speaking is only
                    # a candidate.  Require a sustained interval that the render-aware
                    # matcher does not classify as echo before authorizing Gemini's
                    # activityStart.  This prevents a single jittery matcher frame from
                    # making ADA interrupt herself, while keeping real barge-in with a
                    # small (~300 ms) confirmation delay.
                    if barge_candidate_since and browser_speech_active and not manual_activity_active:
                        now_barge = time.monotonic()
                        reference_active = (
                            self_echo_decision.reference_rms
                            >= self_echo_gate.reference_active_rms
                        )
                        score = self_echo_decision.match_score
                        independent_speech = bool(
                            not self_echo_decision.suppress
                            and not (
                                tv_gate_armed
                                and tv_reference_gate.activity_guard_enabled
                                and tv_decision.probable_tv_only
                            )
                            and (
                                not reference_active
                                or (
                                    score is not None
                                    and score <= self_echo_gate.match_threshold - 0.10
                                )
                            )
                        )
                        if not independent_speech:
                            # Restart the clean-speech timer every time echo is seen.
                            barge_candidate_since = now_barge
                        elif now_barge - barge_candidate_since >= barge_confirm_seconds:
                            logger.info(
                                "session=%s barge-in confirmed after %.0fms mic=%.5f ref=%.5f match=%s",
                                session_id,
                                (now_barge - barge_candidate_since) * 1000.0,
                                self_echo_decision.mic_rms,
                                self_echo_decision.reference_rms,
                                (
                                    f"{score:.3f}"
                                    if score is not None else "none"
                                ),
                            )
                            await confirm_local_speech("barge_confirmed")
                    was_active = voice_gate.active
                    if was_active and audio_debug_session is not None:
                        audio_debug_session.write_raw_mic(pcm16)
                    if voice_gate.state == "standby":
                        voice_gate.remember_pre_roll(pcm16)
                    interrupt_listen_active = time.monotonic() < local_interrupt_listen_until
                    event = voice_gate.process_audio(
                        filtered_pcm16 if voice_gate.active else pcm16,
                        # During ADA playback, ordinary speech remains locked out of Gemini,
                        # but a browser-VAD near-end window is allowed to feed the dedicated
                        # local Stop model. Echo-classified frames are passed as silence
                        # instead of resetting the model on every matcher wobble.
                        allow_stop=(
                            activation_cue_done.is_set()
                            and (
                                interrupt_listen_active
                                or local_speech_detected
                                or time.monotonic() < local_speech_stop_grace_until
                                or (
                                    tv_decision.near_end_hold
                                    and not assistant_response_active
                                )
                                or (
                                    not assistant_response_active
                                    and not (
                                        tv_reference_gate.activity_guard_enabled
                                        and tv_decision.probable_tv_only
                                    )
                                    and voice_gate.normalized_rms(pcm16) >= 0.006
                                )
                            )
                        ),
                        close_on_stop=not (assistant_response_active or assistant_playback_locked),
                        # Keep the wake-word threshold sensitive, but reject a wake
                        # candidate when the independent HDMI/physical-mic matcher says
                        # the same content is coming from the TV/system output.
                        allow_wake=not (
                            tv_reference_gate.wake_guard_enabled
                            and tv_decision.probable_tv_only
                            and not tv_decision.near_end_hold
                        ),
                    )
                    if event == "wake_rejected":
                        _, wake_preroll_seconds = voice_gate.pre_roll_snapshot()
                        _note_voice_event("wake_rejected_tv_only")
                        logger.warning(
                            "session=%s rejected false wake from TV/system output: "
                            "probability=%.4f threshold=%.4f match=%s ref=%.5f mic=%.5f preroll=%.2fs",
                            session_id,
                            voice_gate.last_rejected_wake_probability,
                            voice_gate.wake_threshold,
                            (
                                f"{tv_decision.match_score:.3f}"
                                if tv_decision.match_score is not None else "n/a"
                            ),
                            tv_decision.ref_rms,
                            tv_decision.mic_rms,
                            wake_preroll_seconds,
                        )
                        await send_session_state()
                        continue
                    if event == "wake":
                        wake_preroll, wake_preroll_seconds = voice_gate.pre_roll_snapshot()
                        wake_probability = voice_gate.last_wake_probability
                        tv_gate_armed = False
                        VOICE_RUNTIME_STATE["tv_gate_armed"] = False
                        tv_delay_buffer.clear()
                        tv_delay_bytes = 0
                        voice_gate.begin_activation_buffer()
                        # Close the transmit gate synchronously before scheduling
                        # the cue task, so no mic chunk can race through first.
                        activation_cue_done.clear()
                        if audio_debug_session is not None:
                            await audio_debug_session.close("superseded")
                        audio_debug_session = AudioDebugSession(ROOT, session_id)
                        await audio_debug_session.start()
                        for buffered in wake_preroll:
                            audio_debug_session.write_wake_preroll(buffered)
                        audio_debug_session.record_event("wake", {
                            "word": "Okay Nabu",
                            "probability": round(wake_probability, 4),
                            "threshold": voice_gate.wake_threshold,
                            "preroll_seconds": round(wake_preroll_seconds, 3),
                            "command_overlap_seconds": voice_gate.command_overlap_seconds,
                        })
                        VOICE_RUNTIME_STATE["current_audio_debug_dir"] = str(audio_debug_session.dir)
                        _note_voice_event("wake")
                        logger.info(
                            "session=%s local wake detected: Okay Nabu probability=%.4f threshold=%.4f preroll=%.2fs",
                            session_id, wake_probability, voice_gate.wake_threshold, wake_preroll_seconds,
                        )
                        await send_session_state()
                        asyncio.create_task(play_wake_cue(), name="ada-wake-cue")
                        continue
                    if event == "stop":
                        reject_barge_candidate("stop")
                        if assistant_response_active or assistant_playback_locked:
                            _note_voice_event("local_interrupt:stop")
                            local_interrupt_listen_until = 0.0
                            overlap_speech_active = False
                            overlap_turn_pending = False
                            VOICE_RUNTIME_STATE["overlap_speech_active"] = False
                            VOICE_RUNTIME_STATE["overlap_turn_pending"] = False
                            overlap_audio.clear()
                            overlap_audio_bytes = 0
                            logger.info("session=%s local Stop detected during ADA speech; interrupting response only", session_id)
                            if audio_debug_session is not None:
                                audio_debug_session.record_event("local_interrupt", {"command": "stop"})
                            with suppress(Exception):
                                await send_json(browser, "clear_audio")
                            with suppress(Exception):
                                await provider.interrupt_response()
                            continue
                        _note_voice_event("stop")
                        logger.info("session=%s local Stop detected while ADA is quiet; closing conversation", session_id)
                        await send_session_state()
                        ensure_finalize("stop", clear_playback=True)
                        continue
                    if not voice_gate.active:
                        reject_barge_candidate("session_inactive")
                        continue

                    provider_ready = bool(live_manager.snapshot()["connected"])
                    if (
                        provider_ready
                        and overlap_turn_pending
                        and not assistant_response_active
                        and not assistant_playback_locked
                    ):
                        await flush_overlap_turn("provider_recovered")
                    if not activation_cue_done.is_set():
                        # Deliberately discard everything heard during the wake
                        # cue. The user asked for: wake -> cue -> listen.
                        continue
                    if not provider_ready:
                        # Once listening has actually started, keep the earliest
                        # command audio if Gemini reconnects briefly.
                        voice_gate.buffer_activation_frame(pcm16)
                        continue

                    if voice_gate.status()["buffered_input_frames"]:
                        voice_gate.buffer_activation_frame(filtered_pcm16)
                        pending = voice_gate.drain_activation_buffer()
                        logger.info(
                            "session=%s moving %d reconnect-buffered chunks into manual VAD preroll",
                            session_id, len(pending),
                        )
                        for buffered in pending:
                            manual_preroll.append(buffered)
                            manual_preroll_bytes += len(buffered)

                    if not manual_activity_active:
                        # With Gemini automatic VAD disabled, do not stream audio
                        # outside an explicitly delimited user activity. Keep only a
                        # short filtered look-back so the first syllable survives the
                        # local VAD's detection latency.
                        manual_preroll.append(filtered_pcm16)
                        manual_preroll_bytes += len(filtered_pcm16)
                        while manual_preroll and manual_preroll_bytes > manual_preroll_limit:
                            dropped = manual_preroll.popleft()
                            manual_preroll_bytes -= len(dropped)
                    else:
                        if tv_gate_armed:
                            # Delay only post-first-response audio.  A low content-match
                            # score can therefore retroactively open the queued pre-roll,
                            # preserving the beginning of a real near-end utterance.
                            tv_activity_block = bool(
                                tv_reference_gate.activity_guard_enabled
                                and tv_decision.probable_tv_only
                                and not tv_decision.near_end_hold
                            )
                            if tv_delay_buffer:
                                tv_delay_buffer = deque(
                                    (buffered, not tv_activity_block)
                                    for buffered, _ in tv_delay_buffer
                                )
                            tv_delay_buffer.append((filtered_pcm16, not tv_activity_block))
                            tv_delay_bytes += len(filtered_pcm16)
                            outgoing: list[bytes] = []
                            while (
                                tv_delay_buffer
                                and tv_delay_bytes > tv_reference_gate.decision_delay_bytes
                            ):
                                delayed_pcm, pass_audio = tv_delay_buffer.popleft()
                                tv_delay_bytes -= len(delayed_pcm)
                                outgoing.append(
                                    delayed_pcm if pass_audio else b"\x00" * len(delayed_pcm)
                                )
                        else:
                            tv_delay_buffer.clear()
                            tv_delay_bytes = 0
                            outgoing = [filtered_pcm16]

                        for gemini_pcm in outgoing:
                            await forward_gemini_audio(gemini_pcm)

                    now = time.monotonic()
                    if now - last_audio_log >= 5:
                        logger.info(
                            "audio chunks raw=%d forwarded=%d gate=%s in last interval",
                            raw_packet_count, forwarded_packet_count, voice_gate.state,
                        )
                        raw_packet_count = 0
                        forwarded_packet_count = 0
                        last_audio_log = now
                    continue

                text = message.get("text")
                if text:
                    with suppress(ValueError, TypeError):
                        control = json.loads(text)
                        if control.get("type") == "local_speech_started":
                            if not voice_gate.active or not activation_cue_done.is_set():
                                continue
                            browser_speech_active = True
                            logger.info(
                                "session=%s duplex local_speech_started rms=%s threshold=%s "
                                "noise_floor=%s assistant_active=%s playback_active=%s",
                                session_id, control.get("rms"), control.get("threshold"),
                                control.get("noise_floor"), assistant_response_active,
                                control.get("playback_active"),
                            )
                            assistant_playback_window = bool(
                                assistant_playback_locked
                                or assistant_response_active
                                or bool(control.get("playback_active"))
                            )
                            if assistant_playback_window:
                                # Ordinary overlap must not interrupt ADA, but it must not be
                                # lost either. Capture the user's utterance while TTS is playing
                                # and replay it into Gemini immediately after the physical queue
                                # drains. Dedicated local interrupt words remain a separate path.
                                local_interrupt_listen_until = max(
                                    local_interrupt_listen_until,
                                    time.monotonic() + local_interrupt_window_seconds,
                                )
                                if not overlap_speech_active:
                                    overlap_speech_active = True
                                    overlap_turn_pending = True
                                    VOICE_RUNTIME_STATE["overlap_speech_active"] = True
                                    VOICE_RUNTIME_STATE["overlap_turn_pending"] = True
                                    voice_gate.note_activity()
                                    if not overlap_audio:
                                        for buffered in manual_preroll:
                                            overlap_audio.append(buffered)
                                            overlap_audio_bytes += len(buffered)
                                browser_speech_active = False
                                reject_barge_candidate("assistant_playback")
                                logger.info(
                                    "session=%s local speech during assistant playback: Gemini barge blocked, local interrupt window %.0fms rms=%s threshold=%s",
                                    session_id, local_interrupt_window_seconds * 1000.0,
                                    control.get("rms"), control.get("threshold"),
                                )
                                if audio_debug_session is not None:
                                    audio_debug_session.record_event(
                                        "barge_ignored_during_playback", {
                                            "rms": control.get("rms"),
                                            "threshold": control.get("threshold"),
                                            "assistant_response_active": assistant_response_active,
                                            "browser_playback_active": bool(control.get("playback_active")),
                                            "local_interrupt_window_ms": round(local_interrupt_window_seconds * 1000.0),
                                        }
                                    )
                                continue
                            if time.monotonic() < assistant_echo_guard_until:
                                browser_speech_active = False
                                logger.info(
                                    "session=%s ignored local_speech_started during post-playback reopen delay",
                                    session_id,
                                )
                                if audio_debug_session is not None:
                                    audio_debug_session.record_event("barge_ignored_reopen_delay", {})
                                continue
                            if self_echo_gate.current_echo_only():
                                logger.info(
                                    "session=%s ignored local_speech_started: self-echo gate active",
                                    session_id,
                                )
                                browser_speech_active = False
                                continue
                            if (
                                tv_gate_armed
                                and tv_reference_gate.activity_guard_enabled
                                and tv_reference_gate.recent_probable_tv_only()
                            ):
                                logger.info(
                                    "session=%s ignored local_speech_started: probable TV-only input",
                                    session_id,
                                )
                                browser_speech_active = False
                                continue
                            await confirm_local_speech("local_vad")
                        elif control.get("type") == "playback_telemetry":
                            telemetry = {key: value for key, value in control.items() if key != "type"}
                            playback_event = str(telemetry.get("event") or "unknown")
                            if playback_event == "underrun":
                                logger.warning(
                                    "session=%s playback underrun threshold_ms=%s max_chunk_gap_ms=%s underruns=%s queue_ms=%s",
                                    session_id, telemetry.get("threshold_ms"), telemetry.get("max_chunk_gap_ms"),
                                    telemetry.get("underruns"), telemetry.get("queue_ms"),
                                )
                            elif playback_event in {"environment", "resume", "flush", "drained"}:
                                logger.info("session=%s playback %s %s", session_id, playback_event, telemetry)
                            if playback_event == "drained":
                                assistant_playback_locked = False
                                playback_drain_wait_started = 0.0
                                VOICE_RUNTIME_STATE["assistant_playback_locked"] = False
                                live_manager.set_playback_busy(False)
                                assistant_echo_guard_until = (
                                    time.monotonic() + playback_reopen_delay_seconds
                                )
                                browser_speech_active = False
                                reject_barge_candidate("playback_drained")
                                VOICE_RUNTIME_STATE["barge_candidate_active"] = False
                                logger.info(
                                    "session=%s playback physically drained; reopening VAD after %.0fms",
                                    session_id,
                                    playback_reopen_delay_seconds * 1000.0,
                                )
                                if overlap_turn_pending:
                                    await flush_overlap_turn("playback_drained")
                                if audio_debug_session is not None:
                                    audio_debug_session.record_event(
                                        "playback_drained",
                                        {"reopen_delay_ms": round(playback_reopen_delay_seconds * 1000.0)},
                                    )
                            if audio_debug_session is not None:
                                audio_debug_session.record_event("playback_telemetry", telemetry)
                                if playback_event == "underrun":
                                    audio_debug_session.record_event("playback_underrun", telemetry)
                                    audio_debug_session.record_system_snapshot("playback_underrun", telemetry)
                        elif control.get("type") == "local_speech_stopped":
                            if not voice_gate.active:
                                continue
                            browser_speech_active = False
                            if overlap_speech_active:
                                overlap_speech_active = False
                                VOICE_RUNTIME_STATE["overlap_speech_active"] = False
                            if barge_candidate_since and not manual_activity_active:
                                reject_barge_candidate("vad_stopped_before_confirmation")
                            local_speech_detected = False
                            local_speech_stop_grace_until = time.monotonic() + 1.5
                            logger.info(
                                "session=%s duplex local_speech_stopped rms=%s threshold=%s "
                                "assistant_active=%s playback_active=%s",
                                session_id, control.get("rms"), control.get("threshold"),
                                assistant_response_active, control.get("playback_active"),
                            )
                            if speech_clear_task is not None:
                                speech_clear_task.cancel()
                            if manual_activity_active:
                                await end_manual_activity("local_vad")
                            speech_clear_task = asyncio.create_task(clear_speech_activity_after_grace())
        except WebSocketDisconnect:
            pass
        finally:
            reject_barge_candidate("browser_disconnect")
            if manual_activity_active:
                with suppress(Exception):
                    await end_manual_activity("browser_disconnect")
            closed.set()

    async def forward_video_frame(frame: bytes | None) -> None:
        """Forward at most one frame/sec, and only during an active conversation."""
        nonlocal last_video_sent, forwarded_video_count
        if frame is None or not voice_gate.active:
            return
        async with video_send_lock:
            now = time.monotonic()
            if now - last_video_sent < 0.95:
                return
            await provider.send_video(frame)
            last_video_sent = time.monotonic()
            forwarded_video_count += 1
            if forwarded_video_count == 1 or forwarded_video_count % 10 == 0:
                logger.info(
                    "video frames forwarded (%d total, latest=%d bytes)",
                    forwarded_video_count,
                    len(frame),
                )

    async def clear_speech_activity_after_grace() -> None:
        await asyncio.sleep(1.5)
        speech_active.clear()

    async def provider_to_browser() -> None:
        nonlocal speech_clear_task, assistant_response_active, assistant_playback_locked, playback_drain_wait_started, last_assistant_audio_at, assistant_echo_guard_until, semantic_close_pending, audio_debug_session, tv_gate_armed, semantic_close_watchdog_task, manual_activity_active
        forward_audio = False
        try:
            async for event in provider.events():
                if event.type == "live_reconnecting":
                    had_stale_response = assistant_response_active or assistant_playback_locked
                    assistant_response_active = False
                    assistant_playback_locked = False
                    playback_drain_wait_started = 0.0
                    last_assistant_audio_at = 0.0
                    assistant_echo_guard_until = time.monotonic() + playback_reopen_delay_seconds
                    VOICE_RUNTIME_STATE["assistant_response_active"] = False
                    VOICE_RUNTIME_STATE["assistant_playback_locked"] = False
                    live_manager.set_playback_busy(False)
                    if manual_activity_active:
                        manual_activity_active = False
                        VOICE_RUNTIME_STATE["manual_activity_active"] = False
                    forward_audio = False
                    assistant_memory_parts.clear()
                    voice_gate.note_activity()
                    if had_stale_response:
                        VOICE_RUNTIME_STATE["provider_reconnect_recovery_count"] = int(
                            VOICE_RUNTIME_STATE.get("provider_reconnect_recovery_count") or 0
                        ) + 1
                        logger.error(
                            "session=%s Gemini Live reconnect aborted unfinished assistant response; releasing playback lock message=%r",
                            session_id, event.data.get("message", ""),
                        )
                        if audio_debug_session is not None:
                            audio_debug_session.record_event(
                                "provider_reconnect_response_recovery",
                                {"message": event.data.get("message", "")},
                            )
                        await send_json(browser, "clear_audio")
                        await send_json(browser, "response_interrupted", reason="provider_reconnecting")
                    continue
                if event.type == "live_connected":
                    logger.info(
                        "session=%s Gemini Live connected/reconnected count=%s overlap_pending=%s",
                        session_id, event.data.get("reconnect_count"),
                        VOICE_RUNTIME_STATE.get("overlap_turn_pending"),
                    )
                    continue
                if event.type == "audio":
                    last_assistant_audio_at = time.monotonic()
                    pcm_out = event.data["pcm16"]
                    chunk_sequence = int(event.data.get("sequence") or 0)
                    chunk_trace = {
                        "sequence": chunk_sequence,
                        "bytes": len(pcm_out),
                        "provider_gap_ms": event.data.get("provider_gap_ms"),
                        "provider_jitter_deficit_ms": event.data.get("provider_jitter_deficit_ms"),
                        "chunk_audio_ms": event.data.get("chunk_audio_ms"),
                        "provider_elapsed_ms": event.data.get("provider_elapsed_ms"),
                    }
                    if audio_debug_session is not None:
                        audio_debug_session.write_assistant_raw(pcm_out)
                        audio_debug_session.record_event("assistant_chunk_received", chunk_trace)
                    if forward_audio and (voice_gate.active or semantic_close_pending):
                        send_started = time.monotonic()
                        await browser.send_bytes(pcm_out)
                        send_ms = (time.monotonic() - send_started) * 1000.0
                        if send_ms >= 50.0:
                            logger.warning(
                                "session=%s websocket audio backpressure seq=%d send_ms=%.1f bytes=%d",
                                session_id, chunk_sequence, send_ms, len(pcm_out),
                            )
                        if audio_debug_session is not None:
                            audio_debug_session.write_assistant_sent(pcm_out)
                            audio_debug_session.record_event(
                                "assistant_chunk_sent",
                                {**chunk_trace, "websocket_send_ms": round(send_ms, 1)},
                            )
                elif event.type == "speech_started":
                    if not voice_gate.active:
                        continue
                    if assistant_playback_locked:
                        logger.warning(
                            "session=%s ignored provider speech_started while playback lock is active",
                            session_id,
                        )
                        if audio_debug_session is not None:
                            audio_debug_session.record_event("provider_speech_started_ignored_playback_lock", {})
                        continue
                    if tv_gate_armed and tv_reference_gate.activity_guard_enabled and tv_reference_gate.recent_probable_tv_only():
                        logger.info(
                            "session=%s ignored provider speech_started: probable TV-only input",
                            session_id,
                        )
                        continue
                    voice_gate.note_activity()
                    if speech_clear_task is not None:
                        speech_clear_task.cancel()
                        speech_clear_task = None
                    speech_active.set()
                    forward_audio = False
                    await send_json(browser, "speech_started")
                    await send_json(browser, "clear_audio")
                    if video_mode == "activity":
                        await forward_video_frame(camera.latest_frame)
                elif event.type == "response_interrupted":
                    assistant_memory_parts.clear()
                    if not voice_gate.active:
                        continue
                    logger.warning(
                        "session=%s duplex response_interrupted local_speech_active=%s",
                        session_id, speech_active.is_set(),
                    )
                    VOICE_RUNTIME_STATE["interruption_count"] = int(
                        VOICE_RUNTIME_STATE.get("interruption_count") or 0
                    ) + 1
                    if audio_debug_session is not None:
                        audio_debug_session.record_event("response_interrupted", {})
                    assistant_response_active = False
                    assistant_playback_locked = False
                    playback_drain_wait_started = 0.0
                    last_assistant_audio_at = 0.0
                    assistant_echo_guard_until = time.monotonic() + 0.5
                    VOICE_RUNTIME_STATE["assistant_response_active"] = False
                    VOICE_RUNTIME_STATE["assistant_playback_locked"] = False
                    live_manager.set_playback_busy(False)
                    forward_audio = False
                    await send_json(browser, "clear_audio")
                    await send_json(browser, event.type)
                elif event.type == "speech_stopped":
                    if not voice_gate.active:
                        continue
                    if speech_clear_task is not None:
                        speech_clear_task.cancel()
                    speech_clear_task = asyncio.create_task(clear_speech_activity_after_grace())
                    await send_json(browser, "speech_stopped")
                elif event.type == "assistant_transcript_delta":
                    if forward_audio and (voice_gate.active or semantic_close_pending):
                        text = str(event.data.get("text") or "")
                        if text:
                            assistant_memory_parts.append(text)
                        await send_json(browser, event.type, text=text)
                elif event.type == "user_transcript":
                    if voice_gate.active or semantic_close_pending:
                        text = str(event.data.get("text") or "")
                        logger.info("session=%s user transcript=%r", session_id, text)
                        if audio_debug_session is not None:
                            audio_debug_session.record_event("user_transcript", {"text": text})
                        if self_echo_gate.recent_echo_only(1.5) and len(text.strip()) <= 6:
                            logger.info(
                                "session=%s skipping short transcript in persistent memory after self-echo: %r",
                                session_id,
                                text,
                            )
                        else:
                            remember_conversation_turn("user", text)
                        voice_gate.note_activity()
                        await send_json(browser, event.type, text=text)
                elif event.type == "semantic_end_requested":
                    if voice_gate.active:
                        semantic_close_pending = True
                        voice_gate.begin_close("semantic")
                        _note_voice_event("semantic_end_requested")
                        if audio_debug_session is not None:
                            audio_debug_session.record_event(
                                "semantic_end_requested", {"reason": event.data.get("reason", "")}
                            )
                        logger.info(
                            "session=%s semantic end requested reason=%r; closing silently",
                            session_id, event.data.get("reason", ""),
                        )
                        # end_conversation is authoritative once Gemini has classified
                        # the current utterance as end intent. Do not wait for or generate
                        # a farewell response; clear any queued speech and play only the
                        # normal local session-end cue.
                        await send_session_state()
                        ensure_finalize("semantic", clear_playback=True)
                elif event.type == "control_feedback":
                    kind = str(event.data.get("kind") or "")
                    if kind in {"on", "off"}:
                        cue_path = CONTROL_ON_CUE if kind == "on" else CONTROL_OFF_CUE
                        _note_voice_event(f"control_feedback:{kind}")
                        if audio_debug_session is not None:
                            audio_debug_session.record_event("control_feedback", {"kind": kind})
                        logger.info("session=%s control feedback cue kind=%s", session_id, kind)
                        asyncio.create_task(
                            _play_voice_cue(cue_path, f"control_{kind}"),
                            name=f"ada-control-{kind}-cue",
                        )
                elif event.type == "silent_turn_completed":
                    # A SILENT tool result closes the model's control turn, but
                    # local VAD segment counts are deliberately not tracked as
                    # semantic turns. Keep only the activity/diagnostic signal.
                    VOICE_RUNTIME_STATE["pending_user_turns"] = 0
                    voice_gate.note_activity()
                    _note_voice_event("silent_turn_completed")
                    if audio_debug_session is not None:
                        audio_debug_session.record_event("silent_turn_completed", {})
                    logger.info("session=%s silent control turn completed", session_id)
                elif event.type == "thinking_signal":
                    if voice_gate.active:
                        voice_gate.note_activity()
                        _note_voice_event("thinking_signal")
                        if audio_debug_session is not None:
                            audio_debug_session.record_event("thinking_signal", {})
                        asyncio.create_task(
                            _play_voice_cue(THINKING_CUE, "thinking"),
                            name="ada-thinking-cue",
                        )
                elif event.type == "response_started":
                    assistant_memory_parts.clear()
                    if not (voice_gate.active or semantic_close_pending):
                        forward_audio = False
                        continue
                    # Gemini Live is authoritative for response ordering. Local
                    # activity_start/end boundaries are not semantic turns and
                    # must never suppress an otherwise valid model response.
                    VOICE_RUNTIME_STATE["pending_user_turns"] = 0
                    tv_gate_armed = True
                    VOICE_RUNTIME_STATE["tv_gate_armed"] = True
                    assistant_response_active = True
                    assistant_playback_locked = True
                    playback_drain_wait_started = 0.0
                    last_assistant_audio_at = time.monotonic()
                    assistant_echo_guard_until = 0.0
                    VOICE_RUNTIME_STATE["assistant_response_active"] = True
                    VOICE_RUNTIME_STATE["assistant_playback_locked"] = True
                    live_manager.set_playback_busy(True)
                    if voice_gate.active:
                        voice_gate.note_activity()
                    logger.info("session=%s duplex response_started", session_id)
                    if audio_debug_session is not None:
                        audio_debug_session.record_event("response_started", {})
                    forward_audio = True
                    await send_json(browser, event.type)
                elif event.type == "response_completed":
                    assistant_response_active = False
                    last_assistant_audio_at = 0.0
                    if assistant_playback_locked:
                        playback_drain_wait_started = time.monotonic()
                    assistant_echo_guard_until = time.monotonic() + 0.8
                    VOICE_RUNTIME_STATE["assistant_response_active"] = False
                    logger.info("session=%s duplex response_completed", session_id)
                    if audio_debug_session is not None:
                        audio_debug_session.record_event("response_completed", event.data)
                    if assistant_memory_parts:
                        if voice_gate.active or semantic_close_pending:
                            remember_conversation_turn("assistant", "".join(assistant_memory_parts))
                        assistant_memory_parts.clear()
                    if voice_gate.active:
                        voice_gate.note_activity()
                    await send_json(browser, event.type)
                    if semantic_close_pending:
                        ensure_finalize("semantic", clear_playback=False)
                elif event.type == "expression":
                    expression = str(event.data.get("name", "neutral"))
                    asyncio.create_task(sync_expression_lighting(expression))
                    await send_json(browser, event.type, **event.data)
                else:
                    await send_json(browser, event.type, **event.data)
        finally:
            closed.set()

    async def idle_watch() -> None:
        nonlocal assistant_response_active, assistant_playback_locked, playback_drain_wait_started, last_assistant_audio_at, assistant_echo_guard_until
        while not closed.is_set():
            await asyncio.sleep(0.5)
            if not voice_gate.active:
                continue
            if voice_gate.first_turn_expired():
                elapsed = voice_gate.first_turn_wait_elapsed()
                voice_gate.begin_close("first_turn_timeout")
                _note_voice_event("first_turn_timeout")
                logger.info(
                    "session=%s no first user turn after %.1fs; returning silently to standby",
                    session_id,
                    elapsed,
                )
                if audio_debug_session is not None:
                    audio_debug_session.record_event(
                        "first_turn_timeout",
                        {"elapsed_seconds": round(elapsed, 3)},
                    )
                await send_session_state()
                ensure_finalize("first_turn_timeout", clear_playback=True)
                continue
            if (
                assistant_response_active
                and last_assistant_audio_at > 0.0
                and time.monotonic() - last_assistant_audio_at >= response_stall_timeout_seconds
            ):
                stalled_for = time.monotonic() - last_assistant_audio_at
                assistant_response_active = False
                assistant_playback_locked = False
                playback_drain_wait_started = 0.0
                last_assistant_audio_at = 0.0
                assistant_echo_guard_until = time.monotonic() + playback_reopen_delay_seconds
                VOICE_RUNTIME_STATE["assistant_response_active"] = False
                VOICE_RUNTIME_STATE["assistant_playback_locked"] = False
                VOICE_RUNTIME_STATE["response_stall_timeout_count"] = int(
                    VOICE_RUNTIME_STATE.get("response_stall_timeout_count") or 0
                ) + 1
                live_manager.set_playback_busy(False)
                voice_gate.note_activity()
                _note_voice_event("response_stall_timeout")
                logger.error(
                    "session=%s Gemini response produced no audio for %.1fs; clearing stale speaking state",
                    session_id, stalled_for,
                )
                if audio_debug_session is not None:
                    audio_debug_session.record_event(
                        "response_stall_timeout",
                        {"stalled_seconds": round(stalled_for, 3)},
                    )
                with suppress(Exception):
                    await send_json(browser, "clear_audio")
                    await send_json(browser, "response_interrupted", reason="response_stall_timeout")
                if live_manager.snapshot().get("connected"):
                    with suppress(Exception):
                        await asyncio.wait_for(provider.interrupt_response(), timeout=1.5)
            if (
                assistant_playback_locked
                and not assistant_response_active
                and playback_drain_wait_started > 0.0
                and time.monotonic() - playback_drain_wait_started >= playback_drain_timeout_seconds
            ):
                waited = time.monotonic() - playback_drain_wait_started
                assistant_playback_locked = False
                playback_drain_wait_started = 0.0
                VOICE_RUNTIME_STATE["assistant_playback_locked"] = False
                VOICE_RUNTIME_STATE["playback_drain_timeout_count"] = int(
                    VOICE_RUNTIME_STATE.get("playback_drain_timeout_count") or 0
                ) + 1
                live_manager.set_playback_busy(False)
                assistant_echo_guard_until = time.monotonic() + playback_reopen_delay_seconds
                reject_barge_candidate("playback_drain_timeout")
                VOICE_RUNTIME_STATE["barge_candidate_active"] = False
                _note_voice_event("playback_drain_timeout")
                logger.error(
                    "session=%s playback drained telemetry missing for %.1fs; fail-safe reopening VAD",
                    session_id,
                    waited,
                )
                if audio_debug_session is not None:
                    audio_debug_session.record_event(
                        "playback_drain_timeout",
                        {"waited_seconds": round(waited, 3)},
                    )
            if assistant_response_active or assistant_playback_locked or speech_active.is_set() or VOICE_RUNTIME_STATE.get("cue"):
                continue
            if voice_gate.idle_expired():
                elapsed = voice_gate.idle_elapsed()
                voice_gate.begin_close("idle")
                _note_voice_event("idle_timeout")
                logger.info("session=%s idle timeout after %.1fs; returning to standby", session_id, elapsed)
                await send_session_state()
                ensure_finalize("idle", clear_playback=True)

    async def camera_to_provider() -> None:
        try:
            async for frame in camera.frames():
                if closed.is_set():
                    break
                if not voice_gate.active:
                    continue
                if video_mode == "activity" and not speech_active.is_set():
                    continue
                await forward_video_frame(frame)
        except RuntimeError as exc:
            logger.warning("camera unavailable; continuing audio-only: %s", exc)

    tasks = {
        asyncio.create_task(browser_to_provider(), name="ada-browser-audio"),
        asyncio.create_task(provider_to_browser(), name="ada-provider-events"),
        asyncio.create_task(camera_to_provider(), name="ada-camera"),
        asyncio.create_task(idle_watch(), name="ada-idle-watch"),
    }
    try:
        await closed.wait()
    finally:
        if speech_clear_task is not None:
            speech_clear_task.cancel()
        if close_task is not None and not close_task.done():
            close_task.cancel()
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await task
        if voice_gate.state != "standby":
            if voice_gate.state == "active":
                voice_gate.begin_close("browser_disconnect")
            with suppress(Exception):
                await live_manager.fresh_session(timeout=6.0)
            voice_gate.finish_standby()
        if audio_debug_session is not None:
            VOICE_RUNTIME_STATE["last_audio_debug_dir"] = str(audio_debug_session.dir)
            with suppress(Exception):
                await audio_debug_session.close("browser_disconnect")
            audio_debug_session = None
        with suppress(Exception):
            await browser.close()
        live_manager.set_playback_busy(False)
        VOICE_RUNTIME_STATE.update(
            browser_connected=False,
            session_id=None,
            assistant_response_active=False,
            assistant_playback_locked=False,
        )
        _note_voice_event("browser_disconnected")
        logger.info("session=%s closed", session_id)
