#!/usr/bin/env python3
"""Layer-aware watchdog for the Home Server ADA Pi voice stack.

Checks ADA API, Chromium voice websocket state, P610/AEC PipeWire nodes and the
Gemini provider. It repairs only the narrow layer that is safely repairable:
AEC is restarted independently; ADA itself is restarted only after repeated API
or browser failures. Home Assistant and unrelated services are never touched.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_URL = "http://127.0.0.1:18765/api/voice/status"
STATE_DIR = Path("/home/renat/.local/state/ada-pi")
STATE_FILE = STATE_DIR / "voice-watchdog-state.json"
HEALTH_FILE = STATE_DIR / "voice-health.json"
XDG_RUNTIME_DIR = "/run/user/1000"
ADA_SERVICE = "ada-pi.service"
AEC_SERVICE = "ada-p610-aec.service"
TV_AEC_SERVICE = "ada-tv-aec.service"
FAILURE_THRESHOLD = 2
ADA_RESTART_COOLDOWN = 300.0
AEC_RESTART_COOLDOWN = 120.0
P610_SINK_FRAGMENT = "alsa_output.usb-Plantronics_Plantronics_P610_"
AEC_PLAYBACK_NODE = 'node.name = "echo-cancel-playback"'


def now() -> float:
    return time.time()


def load_state() -> dict[str, Any]:
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def api_status() -> tuple[bool, dict[str, Any]]:
    try:
        with urllib.request.urlopen(API_URL, timeout=5) as response:
            payload = json.load(response)
        return isinstance(payload, dict), payload if isinstance(payload, dict) else {}
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return False, {"error": str(exc)}


def run(command: list[str], timeout: float = 8.0) -> tuple[bool, str]:
    env = os.environ.copy()
    env["XDG_RUNTIME_DIR"] = XDG_RUNTIME_DIR
    try:
        result = subprocess.run(
            command, env=env, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def ensure_aec_playback_on_p610() -> dict[str, Any]:
    """Keep only ADA's AEC render stream pinned to the physical P610 sink."""
    sinks_ok, sinks = run(["pactl", "list", "short", "sinks"])
    inputs_ok, sink_inputs = run(["pactl", "list", "sink-inputs"])
    if not sinks_ok or not inputs_ok:
        return {"ok": False, "repaired": False, "detail": "pactl unavailable"}

    p610_sink_id: str | None = None
    for line in sinks.splitlines():
        parts = line.split()
        if len(parts) >= 2 and P610_SINK_FRAGMENT in parts[1]:
            p610_sink_id = parts[0]
            break

    if not p610_sink_id:
        return {"ok": False, "repaired": False, "detail": "P610 sink not found"}

    for block in re.split(r"\n\s*\n", sink_inputs):
        if AEC_PLAYBACK_NODE not in block:
            continue
        input_match = re.search(r"Sink Input #(\d+)", block)
        sink_match = re.search(r"^\s*Sink:\s*(\d+)\s*$", block, re.MULTILINE)
        if not input_match or not sink_match:
            return {"ok": False, "repaired": False, "detail": "AEC sink-input parse failed"}
        input_id = input_match.group(1)
        current_sink_id = sink_match.group(1)
        if current_sink_id == p610_sink_id:
            return {
                "ok": True,
                "repaired": False,
                "sink_input_id": input_id,
                "p610_sink_id": p610_sink_id,
            }
        ok, output = run(["pactl", "move-sink-input", input_id, p610_sink_id])
        return {
            "ok": ok,
            "repaired": ok,
            "sink_input_id": input_id,
            "previous_sink_id": current_sink_id,
            "p610_sink_id": p610_sink_id,
            "detail": output[-500:],
        }

    return {"ok": False, "repaired": False, "detail": "AEC playback stream not found"}


def audio_nodes() -> dict[str, bool]:
    sinks_ok, sinks = run(["pactl", "list", "short", "sinks"])
    sources_ok, sources = run(["pactl", "list", "short", "sources"])
    joined = f"{sinks}\n{sources}"
    return {
        "pipewire_reachable": sinks_ok and sources_ok,
        "p610_present": "Plantronics_Plantronics_P610" in joined,
        "aec_sink_present": "ada_aec_sink" in sinks,
        "aec_source_present": "ada_aec_source" in sources,
        "tv_aec_source_present": "ada_tv_aec_source" in sources,
    }


def restart_user_service(name: str) -> tuple[bool, str]:
    return run(["systemctl", "--user", "restart", name], timeout=20)


def main() -> int:
    state = load_state()
    checked_at = now()
    api_ok, status = api_status()
    aec_route = ensure_aec_playback_on_p610()
    audio = audio_nodes()

    runtime = status.get("runtime", {}) if api_ok else {}
    provider = status.get("provider", {}) if api_ok else {}
    gate = status.get("gate", {}) if api_ok else {}
    browser_ok = bool(runtime.get("browser_connected")) if api_ok else False
    provider_ok = bool(provider.get("connected")) if api_ok else False
    web_search_ok = bool(provider.get("web_search_enabled")) if api_ok else False
    ha_mcp_tool_count = int(provider.get("ha_mcp_tool_count") or 0) if api_ok else 0
    smart_tools_ok = bool(web_search_ok and ha_mcp_tool_count >= 13)
    aec_ok = bool(audio["aec_source_present"] and audio["aec_sink_present"])
    tv_aec_ok = bool(audio["tv_aec_source_present"])
    current_raw = runtime.get("raw_mic_chunks") if api_ok else None
    previous_raw = state.get("previous_raw_mic_chunks")
    mic_expected = bool(api_ok and browser_ok and audio["p610_present"] and aec_ok)
    mic_stale = bool(
        mic_expected
        and isinstance(current_raw, int)
        and isinstance(previous_raw, int)
        and current_raw == previous_raw
    )

    def bump(key: str, failed: bool) -> int:
        value = int(state.get(key, 0) or 0)
        value = value + 1 if failed else 0
        state[key] = value
        return value

    api_failures = bump("api_failures", not api_ok)
    browser_failures = bump("browser_failures", api_ok and not browser_ok)
    aec_failures = bump("aec_failures", audio["p610_present"] and not aec_ok)
    tv_aec_failures = bump("tv_aec_failures", audio["p610_present"] and not tv_aec_ok)
    provider_failures = bump("provider_failures", api_ok and not provider_ok)
    mic_failures = bump("mic_failures", mic_stale)
    if isinstance(current_raw, int):
        state["previous_raw_mic_chunks"] = current_raw

    actions: list[dict[str, Any]] = []
    if aec_route.get("repaired"):
        actions.append({"action": "repin_aec_playback_to_p610", "ok": True, "detail": aec_route})

    # The TV-reference AEC is the first capture stage. If it disappears, restore
    # it first and then rebuild the P610 self-AEC so its source_master points at
    # the newly-created ada_tv_aec_source instead of the physical fallback.
    if tv_aec_failures >= FAILURE_THRESHOLD:
        last = float(state.get("last_tv_aec_restart_at", 0) or 0)
        if checked_at - last >= AEC_RESTART_COOLDOWN:
            tv_ok, tv_output = restart_user_service(TV_AEC_SERVICE)
            p610_ok = False
            p610_output = ""
            ada_ok = False
            ada_output = ""
            if tv_ok:
                p610_ok, p610_output = restart_user_service(AEC_SERVICE)
            if tv_ok and p610_ok:
                # Recreating the capture sources invalidates Chromium's old
                # recording stream. Reconnect ADA immediately instead of
                # waiting for a later browser-stale watchdog cycle.
                ada_ok, ada_output = restart_user_service(ADA_SERVICE)
            state["last_tv_aec_restart_at"] = checked_at
            if tv_ok and p610_ok and ada_ok:
                state["tv_aec_failures"] = 0
                state["aec_failures"] = 0
                state["browser_failures"] = 0
                state["mic_failures"] = 0
                tv_aec_failures = 0
                aec_failures = 0
                browser_failures = 0
                mic_failures = 0
            actions.append({
                "action": "restart_tv_aec_cascade",
                "ok": bool(tv_ok and p610_ok and ada_ok),
                "detail": {
                    "tv_aec": tv_output[-500:],
                    "p610_aec": p610_output[-500:],
                    "ada": ada_output[-500:],
                },
            })

    # P610 self-AEC has its own isolated service, so repair it independently.
    if aec_failures >= FAILURE_THRESHOLD:
        last = float(state.get("last_aec_restart_at", 0) or 0)
        if checked_at - last >= AEC_RESTART_COOLDOWN:
            ok, output = restart_user_service(AEC_SERVICE)
            state["last_aec_restart_at"] = checked_at
            state["aec_failures"] = 0 if ok else aec_failures
            actions.append({"action": "restart_aec", "ok": ok, "detail": output[-500:]})

    # ADA restart is deliberately conservative and never happens for a transient
    # Gemini reconnect. API/browser/capture failures must repeat and respect cooldown.
    ada_failed = (
        api_failures >= FAILURE_THRESHOLD
        or browser_failures >= FAILURE_THRESHOLD
        or mic_failures >= FAILURE_THRESHOLD
    )
    if ada_failed:
        last = float(state.get("last_ada_restart_at", 0) or 0)
        if checked_at - last >= ADA_RESTART_COOLDOWN:
            ok, output = restart_user_service(ADA_SERVICE)
            state["last_ada_restart_at"] = checked_at
            if ok:
                state["api_failures"] = 0
                state["browser_failures"] = 0
                state["mic_failures"] = 0
            actions.append({"action": "restart_ada", "ok": ok, "detail": output[-500:]})

    # A provider failure is exposed but left to LiveSessionManager's own reconnect
    # and resumption loop; restarting ADA would be less narrow and can lose context.
    mic_ok = bool(mic_expected and not mic_stale)
    health = {
        "checked_at": checked_at,
        "overall_ok": bool(
            api_ok
            and browser_ok
            and audio["p610_present"]
            and tv_aec_ok
            and aec_ok
            and bool(aec_route.get("ok"))
            and provider_ok
            and mic_ok
            and smart_tools_ok
        ),
        "layers": {
            "ada_api": {"ok": api_ok, "failures": api_failures},
            "browser": {"ok": browser_ok, "failures": browser_failures},
            "p610": {"ok": audio["p610_present"]},
            "tv_aec": {"ok": tv_aec_ok, "failures": tv_aec_failures},
            "aec": {"ok": aec_ok, "failures": aec_failures},
            "aec_playback_route": aec_route,
            "mic_capture": {
                "ok": mic_ok,
                "stale": mic_stale,
                "failures": mic_failures,
                "raw_mic_chunks": current_raw,
                "previous_raw_mic_chunks": previous_raw,
            },
            "provider": {
                "ok": provider_ok,
                "status": provider.get("status") if api_ok else None,
                "failures": provider_failures,
                "self_healing": True,
            },
            "smart_tools": {
                "ok": smart_tools_ok,
                "web_search_enabled": web_search_ok,
                "google_grounding_enabled": bool(provider.get("google_grounding_enabled")) if api_ok else False,
                "ha_mcp_tool_count": ha_mcp_tool_count,
                "web_tool_calls": int(provider.get("web_tool_calls") or 0) if api_ok else 0,
                "ha_tool_calls": int(provider.get("ha_tool_calls") or 0) if api_ok else 0,
                "self_healing": False,
            },
        },
        "gate_state": gate.get("state") if api_ok else None,
        "raw_mic_chunks": runtime.get("raw_mic_chunks") if api_ok else None,
        "forwarded_gemini_chunks": runtime.get("forwarded_gemini_chunks") if api_ok else None,
        "actions": actions,
    }
    save_json(HEALTH_FILE, health)
    save_json(STATE_FILE, state)
    print(json.dumps(health, ensure_ascii=False, separators=(",", ":")))
    return 0 if health["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
