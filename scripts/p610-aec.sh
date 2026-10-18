#!/usr/bin/env bash
set -euo pipefail

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
MODULE_FILE="$XDG_RUNTIME_DIR/ada-p610-aec.module"

find_existing_module() {
  pactl list short modules 2>/dev/null | awk '$2 == "module-echo-cancel" && $0 ~ /source_name=ada_aec_source/ {print $1; exit}'
}

unload_existing() {
  local module_id=""
  if [[ -f "$MODULE_FILE" ]]; then
    module_id="$(cat "$MODULE_FILE" 2>/dev/null || true)"
  fi
  if [[ -z "$module_id" ]]; then
    module_id="$(find_existing_module || true)"
  fi
  if [[ -n "$module_id" ]]; then
    pactl unload-module "$module_id" 2>/dev/null || true
  fi
  rm -f "$MODULE_FILE"
}

case "${1:-start}" in
  start)
    # Wait briefly for pipewire-pulse and the USB profile to appear after login/restart.
    for _ in $(seq 1 40); do
      if pactl info >/dev/null 2>&1; then
        P610_SOURCE="$(pactl list short sources 2>/dev/null | awk '$2 ~ /^alsa_input\.usb-Plantronics_Plantronics_P610_/ && $2 ~ /\.analog-stereo$/ {print $2; exit}')"
        P610_SINK="$(pactl list short sinks 2>/dev/null | awk '$2 ~ /^alsa_output\.usb-Plantronics_Plantronics_P610_/ && $2 ~ /\.analog-stereo$/ {print $2; exit}')"
        if [[ -n "$P610_SOURCE" && -n "$P610_SINK" ]]; then
          break
        fi
      fi
      sleep 0.5
    done

    if [[ -z "${P610_SOURCE:-}" || -z "${P610_SINK:-}" ]]; then
      echo "Plantronics P610 source/sink not found" >&2
      exit 1
    fi

    unload_existing

    # Do not change the user's normal system playback device here. ADA is
    # pinned separately to ada_aec_sink, while the AEC render master is P610.
    MODULE_ID="$(pactl load-module module-echo-cancel \
      aec_method=webrtc \
      source_master="$P610_SOURCE" \
      sink_master="$P610_SINK" \
      source_name=ada_aec_source \
      sink_name=ada_aec_sink \
      use_master_format=1)"

    printf '%s\n' "$MODULE_ID" > "$MODULE_FILE"
    # Keep the physical P610 at a usable speech level. The old 40% setting
    # mapped to a very low USB hardware PCM level and made Gemini replies
    # effectively inaudible even though playback was active.
    pactl set-sink-volume "$P610_SINK" 80%
    pactl set-sink-mute "$P610_SINK" 0
    pactl set-sink-volume ada_aec_sink 100%
    pactl set-sink-mute ada_aec_sink 0
    pactl set-default-source ada_aec_source
    ;;
  stop)
    unload_existing
    ;;
  restart)
    unload_existing
    exec "$0" start
    ;;
  *)
    echo "Usage: $0 {start|stop|restart}" >&2
    exit 2
    ;;
esac
