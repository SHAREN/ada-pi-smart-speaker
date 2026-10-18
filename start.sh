#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
ENV_FILE="$PROJECT_DIR/.env"
HOST="${ADA_HOST:-0.0.0.0}"
PORT="${ADA_PORT:-8000}"
VIDEO_MODE="${ADA_VIDEO_MODE:-activity}"
UI_MODE="${ADA_UI_MODE:-}"
KIOSK_URL=""
CHROMIUM_PROFILE="${ADA_CHROMIUM_PROFILE:-$PROJECT_DIR/.chromium-kiosk}"
START_HOME_ASSISTANT="${ADA_START_HOME_ASSISTANT:-}"
HOME_ASSISTANT_COMPOSE_FILE="${ADA_HOME_ASSISTANT_COMPOSE_FILE:-}"
HOME_ASSISTANT_URL="${HOME_ASSISTANT_URL:-}"

if systemctl is-enabled --quiet ada-fan-max.service 2>/dev/null; then
  systemctl start ada-fan-max.service 2>/dev/null || true
else
  echo "WARNING: Maximum-fan service is not installed." >&2
  echo "Run once: sudo '$PROJECT_DIR/scripts/install_max_fan_service.sh'" >&2
fi

env_value() {
  "$VENV_PYTHON" -c \
    'from dotenv import dotenv_values; import sys; print(dotenv_values(sys.argv[1]).get(sys.argv[2], "") or "")' \
    "$ENV_FILE" "$1"
}

start_home_assistant() {
  if [[ "$START_HOME_ASSISTANT" != "true" ]]; then
    return
  fi
  if [[ ! -f "$HOME_ASSISTANT_COMPOSE_FILE" ]]; then
    echo "Home Assistant Compose file not found: $HOME_ASSISTANT_COMPOSE_FILE" >&2
    echo "Set ADA_HOME_ASSISTANT_COMPOSE_FILE or ADA_START_HOME_ASSISTANT=false." >&2
    exit 1
  fi
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required to start Home Assistant but is not installed." >&2
    exit 1
  fi

  local -a docker_command=(docker)
  if ! docker info >/dev/null 2>&1; then
    if sudo -n docker info >/dev/null 2>&1; then
      docker_command=(sudo -n docker)
    else
      echo "Cannot access Docker. Add '$USER' to the docker group or allow passwordless Docker startup." >&2
      echo "Temporary workaround: sudo docker compose -f '$HOME_ASSISTANT_COMPOSE_FILE' up -d" >&2
      exit 1
    fi
  fi

  echo "Starting Home Assistant..."
  "${docker_command[@]}" compose -f "$HOME_ASSISTANT_COMPOSE_FILE" up -d

  local ready=false
  for _ in {1..90}; do
    if "$VENV_PYTHON" -c "import urllib.request; urllib.request.urlopen('$HOME_ASSISTANT_URL', timeout=.5)" >/dev/null 2>&1; then
      ready=true
      break
    fi
    sleep 1
  done
  if [[ "$ready" != true ]]; then
    echo "Home Assistant did not become reachable at $HOME_ASSISTANT_URL within 90 seconds." >&2
    exit 1
  fi
  echo "Home Assistant is ready at $HOME_ASSISTANT_URL."

  if ! grep -Eq '^HOME_ASSISTANT_TOKEN=.+$' "$ENV_FILE"; then
    echo "WARNING: HOME_ASSISTANT_TOKEN is missing or empty in $ENV_FILE." >&2
    echo "ADA can open Home Assistant, but office-light monitoring will remain disabled until a long-lived access token is added." >&2
  fi
}

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "ADA Pi virtual environment is missing." >&2
  echo "Run: python3 -m venv '$PROJECT_DIR/.venv'" >&2
  echo "Then: '$PROJECT_DIR/.venv/bin/python' -m pip install -r '$PROJECT_DIR/backend/requirements.txt'" >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ADA Pi configuration is missing: $ENV_FILE" >&2
  echo "Copy .env.example to .env and add GEMINI_API_KEY." >&2
  exit 1
fi

START_HOME_ASSISTANT="${START_HOME_ASSISTANT:-$(env_value ADA_START_HOME_ASSISTANT)}"
START_HOME_ASSISTANT="${START_HOME_ASSISTANT:-true}"
HOME_ASSISTANT_COMPOSE_FILE="${HOME_ASSISTANT_COMPOSE_FILE:-$(env_value ADA_HOME_ASSISTANT_COMPOSE_FILE)}"
HOME_ASSISTANT_COMPOSE_FILE="${HOME_ASSISTANT_COMPOSE_FILE:-$PROJECT_DIR/../homeassistant/compose.yaml}"
HOME_ASSISTANT_URL="${HOME_ASSISTANT_URL:-$(env_value HOME_ASSISTANT_URL)}"
HOME_ASSISTANT_URL="${HOME_ASSISTANT_URL:-http://127.0.0.1:8123}"
UI_MODE="${ADA_UI_MODE:-$(env_value ADA_UI_MODE)}"
UI_MODE="${UI_MODE:-full}"
KIOSK_URL="http://localhost:$PORT/?ui=$UI_MODE"

start_home_assistant

if command -v chromium >/dev/null 2>&1; then
  CHROMIUM_BIN="$(command -v chromium)"
elif command -v chromium-browser >/dev/null 2>&1; then
  CHROMIUM_BIN="$(command -v chromium-browser)"
else
  echo "Chromium is not installed or is not on PATH." >&2
  exit 1
fi

cd "$PROJECT_DIR"
ADA_VIDEO_MODE="$VIDEO_MODE" "$VENV_PYTHON" -m uvicorn backend.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --env-file "$ENV_FILE" &
SERVER_PID=$!
BROWSER_PID=""
WINDOW_GUARD_PID=""

cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$WINDOW_GUARD_PID" ]] && kill -0 "$WINDOW_GUARD_PID" 2>/dev/null; then
    kill "$WINDOW_GUARD_PID" 2>/dev/null || true
  fi
  if [[ -n "$BROWSER_PID" ]] && kill -0 "$BROWSER_PID" 2>/dev/null; then
    kill "$BROWSER_PID" 2>/dev/null || true
  fi
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
  [[ -z "$WINDOW_GUARD_PID" ]] || wait "$WINDOW_GUARD_PID" 2>/dev/null || true
  [[ -z "$BROWSER_PID" ]] || wait "$BROWSER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Do not open Chromium until the backend is accepting requests.
READY=false
for _ in {1..100}; do
  if "$VENV_PYTHON" -c "import urllib.request; urllib.request.urlopen('$KIOSK_URL', timeout=.2)" >/dev/null 2>&1; then
    READY=true
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ADA Pi backend stopped before Chromium could start." >&2
    exit 1
  fi
  sleep .1
done
if [[ "$READY" != true ]]; then
  echo "Timed out waiting for ADA Pi backend at $KIOSK_URL." >&2
  exit 1
fi

ACTIVE_CHROMIUM_PROFILE="$CHROMIUM_PROFILE"
if [[ "$UI_MODE" == "voice-only" ]]; then
  ACTIVE_CHROMIUM_PROFILE="${CHROMIUM_PROFILE}-voice-only"
fi

CHROMIUM_ARGS=(
  --no-first-run
  --disable-session-crashed-bubble
  --autoplay-policy=no-user-gesture-required
  --use-fake-ui-for-media-stream
  --disable-background-timer-throttling
  --disable-renderer-backgrounding
  --user-data-dir="$ACTIVE_CHROMIUM_PROFILE"
)
if [[ "$UI_MODE" == "voice-only" ]]; then
  CHROMIUM_ARGS+=(
    --app="$KIOSK_URL"
    --class=ada-pi-voice
    --ozone-platform=x11
    --start-minimized
    --window-size=1,1
    --window-position=-32000,-32000
  )
else
  CHROMIUM_ARGS+=(--kiosk "$KIOSK_URL")
fi

"$CHROMIUM_BIN" "${CHROMIUM_ARGS[@]}" &
BROWSER_PID=$!

# labwc 0.8.x can still map/focus Chromium's tiny app window even when the
# voice-only launch asks to start minimized.  For the XWayland voice window we
# therefore enforce the hidden state directly.  xprop --spy is event-driven: if
# Chromium later removes _NET_WM_STATE_HIDDEN while requesting activation, the
# window is immediately hidden again without a polling loop.
if [[ "$UI_MODE" == "voice-only" ]] && command -v wmctrl >/dev/null 2>&1 && command -v xprop >/dev/null 2>&1; then
  (
    export DISPLAY="${DISPLAY:-:0}"
    export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

    WINDOW_ID=""
    for _ in {1..100}; do
      WINDOW_ID="$(wmctrl -lx 2>/dev/null | awk '$3 ~ /(^|\.)ada-pi-voice$/ {print $1; exit}')"
      [[ -n "$WINDOW_ID" ]] && break
      kill -0 "$BROWSER_PID" 2>/dev/null || exit 0
      sleep 0.05
    done

    if [[ -n "$WINDOW_ID" ]]; then
      hide_voice_window() {
        wmctrl -ir "$WINDOW_ID" -b add,hidden,skip_taskbar,skip_pager,below >/dev/null 2>&1 || true
      }
      hide_voice_window
      while kill -0 "$BROWSER_PID" 2>/dev/null && IFS= read -r _; do
        hide_voice_window
      done < <(xprop -spy -id "$WINDOW_ID" _NET_WM_STATE 2>/dev/null)
    fi
  ) &
  WINDOW_GUARD_PID=$!
fi

# Closing either side closes the other. This also handles the on-screen Exit
# button, which asks the backend to stop and lets this supervisor close Chromium.
set +e
wait -n "$SERVER_PID" "$BROWSER_PID"
STATUS=$?
set -e
exit "$STATUS"
