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


live_manager = LiveSessionManager(
    lambda: posture_store.system_prompt(DEFAULT_ADA_INSTRUCTIONS),
    office_state_getter=lambda: office_light_monitor.snapshot(),
    habit_state_getter=habit_tool_snapshot,
)
voice_gate = WakeSessionGate(ROOT)
VOICE_RUNTIME_STATE: dict[str, object] = {
    "browser_connected": False,
    "session_id": None,
    "assistant_response_active": False,
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
    local_speech_detected = False
    local_speech_stop_grace_until = 0.0
    assistant_response_active = False
    semantic_close_pending = False
    audio_debug_session: AudioDebugSession | None = None

    VOICE_RUNTIME_STATE.update(
        browser_connected=True,
        session_id=session_id,
        assistant_response_active=False,
    )
    _note_voice_event("browser_connected")

    async def send_session_state() -> None:
        with suppress(WebSocketDisconnect, RuntimeError):
            await send_json(
                browser,
                "session_state",
                **voice_gate.status(),
                provider=live_manager.snapshot(),
            )

    async def play_wake_cue() -> None:
        activation_cue_done.clear()
        if audio_debug_session is not None:
            audio_debug_session.record_event("wake_cue_started", {})
        try:
            await _play_voice_cue(WAKE_CUE, "wake")
        finally:
            if audio_debug_session is not None:
                audio_debug_session.record_event("wake_cue_finished", {})
            voice_gate.reset_after_wake_cue()
            activation_cue_done.set()
            await send_session_state()

    async def finalize_session(reason: str, *, clear_playback: bool) -> None:
        nonlocal assistant_response_active, semantic_close_pending, audio_debug_session, local_speech_detected, local_speech_stop_grace_until
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
            VOICE_RUNTIME_STATE["assistant_response_active"] = False
            if clear_playback:
                with suppress(Exception):
                    await send_json(browser, "clear_audio")
            # A semantic end is finalized only after the farewell response has
            # completed. Give the browser's small jitter queue time to drain.
            if reason == "semantic" and not clear_playback:
                await asyncio.sleep(0.35)
            fresh_task = asyncio.create_task(live_manager.fresh_session())
            if audio_debug_session is not None:
                audio_debug_session.record_event("end_cue_started", {})
            await _play_voice_cue(END_CUE, "end")
            if audio_debug_session is not None:
                audio_debug_session.record_event("end_cue_finished", {})
            try:
                await fresh_task
                if audio_debug_session is not None:
                    audio_debug_session.record_event("fresh_session", {})
            except Exception as exc:
                logger.warning("fresh Gemini standby was not ready after %s: %s", reason, exc)
                if audio_debug_session is not None:
                    audio_debug_session.record_event("fresh_session_failed", {"error": str(exc)})
            voice_gate.finish_standby()
            semantic_close_pending = False
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
        nonlocal speech_clear_task, audio_debug_session, local_speech_detected, local_speech_stop_grace_until
        raw_packet_count = 0
        forwarded_packet_count = 0
        last_audio_log = time.monotonic()
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
                    was_active = voice_gate.active
                    if was_active and audio_debug_session is not None:
                        audio_debug_session.write_raw_mic(pcm16)
                    if voice_gate.state == "standby":
                        voice_gate.remember_pre_roll(pcm16)
                    event = voice_gate.process_audio(
                        pcm16,
                        allow_stop=(
                            local_speech_detected
                            or time.monotonic() < local_speech_stop_grace_until
                        ),
                    )
                    if event == "wake":
                        voice_gate.begin_activation_buffer()
                        # Close the transmit gate synchronously before scheduling
                        # the cue task, so no mic chunk can race through first.
                        activation_cue_done.clear()
                        if audio_debug_session is not None:
                            await audio_debug_session.close("superseded")
                        audio_debug_session = AudioDebugSession(ROOT, session_id)
                        await audio_debug_session.start()
                        for buffered in voice_gate.activation_buffer_snapshot():
                            audio_debug_session.write_raw_mic(buffered)
                        audio_debug_session.record_event("wake", {
                            "word": "Okay Nabu",
                            "command_overlap_seconds": voice_gate.command_overlap_seconds,
                        })
                        VOICE_RUNTIME_STATE["current_audio_debug_dir"] = str(audio_debug_session.dir)
                        _note_voice_event("wake")
                        logger.info("session=%s local wake detected: Okay Nabu", session_id)
                        await send_session_state()
                        asyncio.create_task(play_wake_cue(), name="ada-wake-cue")
                        continue
                    if event == "stop":
                        _note_voice_event("stop")
                        logger.info("session=%s local Stop detected; closing conversation", session_id)
                        await send_session_state()
                        ensure_finalize("stop", clear_playback=True)
                        continue
                    if not voice_gate.active:
                        continue

                    provider_ready = bool(live_manager.snapshot()["connected"])
                    if not activation_cue_done.is_set() or not provider_ready:
                        voice_gate.buffer_activation_frame(pcm16)
                        continue

                    if voice_gate.status()["buffered_input_frames"]:
                        voice_gate.buffer_activation_frame(pcm16)
                        pending = voice_gate.drain_activation_buffer()
                        logger.info(
                            "session=%s flushing %d buffered microphone chunks after wake",
                            session_id, len(pending),
                        )
                        for buffered in pending:
                            await provider.send_audio(buffered)
                            VOICE_RUNTIME_STATE["forwarded_gemini_chunks"] = int(
                                VOICE_RUNTIME_STATE.get("forwarded_gemini_chunks") or 0
                            ) + 1
                            VOICE_RUNTIME_STATE["forwarded_gemini_bytes"] = int(
                                VOICE_RUNTIME_STATE.get("forwarded_gemini_bytes") or 0
                            ) + len(buffered)
                            if audio_debug_session is not None:
                                audio_debug_session.write_gemini_input(buffered)
                            forwarded_packet_count += 1
                    else:
                        await provider.send_audio(pcm16)
                        VOICE_RUNTIME_STATE["forwarded_gemini_chunks"] = int(
                            VOICE_RUNTIME_STATE.get("forwarded_gemini_chunks") or 0
                        ) + 1
                        VOICE_RUNTIME_STATE["forwarded_gemini_bytes"] = int(
                            VOICE_RUNTIME_STATE.get("forwarded_gemini_bytes") or 0
                        ) + len(pcm16)
                        if audio_debug_session is not None:
                            audio_debug_session.write_gemini_input(pcm16)
                        forwarded_packet_count += 1

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
                            if not voice_gate.active:
                                continue
                            local_speech_detected = True
                            local_speech_stop_grace_until = 0.0
                            logger.info(
                                "session=%s duplex local_speech_started rms=%s threshold=%s "
                                "noise_floor=%s assistant_active=%s playback_active=%s",
                                session_id, control.get("rms"), control.get("threshold"),
                                control.get("noise_floor"), assistant_response_active,
                                control.get("playback_active"),
                            )
                            voice_gate.note_activity()
                            if speech_clear_task is not None:
                                speech_clear_task.cancel()
                                speech_clear_task = None
                            speech_active.set()
                            if video_mode == "activity":
                                await forward_video_frame(camera.latest_frame)
                        elif control.get("type") == "local_speech_stopped":
                            if not voice_gate.active:
                                continue
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
                            speech_clear_task = asyncio.create_task(clear_speech_activity_after_grace())
        except WebSocketDisconnect:
            pass
        finally:
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
        nonlocal speech_clear_task, assistant_response_active, semantic_close_pending, audio_debug_session
        forward_audio = False
        try:
            async for event in provider.events():
                if event.type == "audio":
                    pcm_out = event.data["pcm16"]
                    if audio_debug_session is not None:
                        audio_debug_session.write_assistant_raw(pcm_out)
                    if forward_audio and (voice_gate.active or semantic_close_pending):
                        await browser.send_bytes(pcm_out)
                        if audio_debug_session is not None:
                            audio_debug_session.write_assistant_sent(pcm_out)
                elif event.type == "speech_started":
                    if not voice_gate.active:
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
                    VOICE_RUNTIME_STATE["assistant_response_active"] = False
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
                        await send_json(browser, event.type, text=event.data["text"])
                elif event.type == "user_transcript":
                    if voice_gate.active or semantic_close_pending:
                        logger.info("session=%s user transcript=%r", session_id, event.data["text"])
                        if audio_debug_session is not None:
                            audio_debug_session.record_event("user_transcript", {"text": event.data["text"]})
                        voice_gate.note_activity()
                        await send_json(browser, event.type, text=event.data["text"])
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
                            "session=%s semantic end requested reason=%r",
                            session_id, event.data.get("reason", ""),
                        )
                        await send_session_state()
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
                    if not (voice_gate.active or semantic_close_pending):
                        forward_audio = False
                        continue
                    assistant_response_active = True
                    VOICE_RUNTIME_STATE["assistant_response_active"] = True
                    if voice_gate.active:
                        voice_gate.note_activity()
                    logger.info("session=%s duplex response_started", session_id)
                    if audio_debug_session is not None:
                        audio_debug_session.record_event("response_started", {})
                    forward_audio = True
                    await send_json(browser, event.type)
                elif event.type == "response_completed":
                    assistant_response_active = False
                    VOICE_RUNTIME_STATE["assistant_response_active"] = False
                    logger.info("session=%s duplex response_completed", session_id)
                    if audio_debug_session is not None:
                        audio_debug_session.record_event("response_completed", {})
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
        while not closed.is_set():
            await asyncio.sleep(0.5)
            if not voice_gate.active:
                continue
            if assistant_response_active or speech_active.is_set() or VOICE_RUNTIME_STATE.get("cue"):
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
        VOICE_RUNTIME_STATE.update(
            browser_connected=False,
            session_id=None,
            assistant_response_active=False,
        )
        _note_voice_event("browser_disconnected")
        logger.info("session=%s closed", session_id)
