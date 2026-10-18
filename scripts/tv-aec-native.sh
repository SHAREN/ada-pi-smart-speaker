#!/usr/bin/env bash
set -euo pipefail

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

P610_SOURCE=""

# WirePlumber creates physical nodes after PipeWire starts. Wait until the P610
# capture source exists; monitor.mode will then follow the user's current
# default system sink (normally HDMI/TV) without changing where system audio
# plays.
for _ in $(seq 1 60); do
  if pactl info >/dev/null 2>&1; then
    P610_SOURCE="$(pactl list short sources 2>/dev/null | awk '$2 ~ /^alsa_input\.usb-Plantronics_Plantronics_P610_/ && $2 ~ /\.analog-stereo$/ {print $2; exit}')"
    if [[ -n "$P610_SOURCE" ]]; then
      break
    fi
  fi
  sleep 0.5
done

if [[ -z "$P610_SOURCE" ]]; then
  echo "TV AEC P610 capture source not found" >&2
  exit 1
fi

PW_COMMAND="load-module libpipewire-module-echo-cancel {
  library.name = aec/libspa-aec-webrtc
  monitor.mode = true
  audio.rate = 48000
  audio.channels = 2
  audio.position = [ FL FR ]
  aec.args = {
    webrtc.extended_filter = true
    webrtc.delay_agnostic = true
    webrtc.high_pass_filter = true
    webrtc.noise_suppression = false
    webrtc.gain_control = false
  }
  capture.props = {
    node.name = \"ada-tv-aec-capture\"
    target.object = \"${P610_SOURCE}\"
    node.passive = true
  }
  source.props = {
    node.name = \"ada_tv_aec_source\"
    node.description = \"ADA TV Echo-Cancel Source\"
  }
}"

# pw-cli uses line breaks as command separators. Submit the module as one
# command and keep the client alive because modules loaded through pw-cli are
# owned by that client.
PW_COMMAND="$(printf '%s' "$PW_COMMAND" | tr '\n' ' ')"
producer() {
  printf '%s\n' "$PW_COMMAND"
  while sleep 3600; do :; done
}
producer | pw-cli
