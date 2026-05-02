#!/usr/bin/env bash
# run_lumi.sh — launch Brain + Tauri dev with clean shutdown on Ctrl+C

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/.run_logs"
mkdir -p "${LOG_DIR}"

declare -a PIDS=()

cleanup() {
  echo ""
  echo "[lumi] shutting down..."
  for pid in "${PIDS[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  sleep 1
  for pid in "${PIDS[@]}"; do
    kill -KILL "${pid}" 2>/dev/null || true
  done
  fuser -k 1420/tcp 2>/dev/null || true
  kill -- -$$ 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Re-exec under setsid so kill -- -$$ is safe and doesn't kill the terminal
if command -v setsid >/dev/null 2>&1 && [[ -z "${_LUMI_REEXEC:-}" ]]; then
  export _LUMI_REEXEC=1
  exec setsid "$0" "$@"
fi

echo "[lumi] starting Brain (WebSocket server on ws://127.0.0.1:5556)..."
uv run python -m src.main >"${LOG_DIR}/brain.log" 2>&1 &
PIDS+=($!)
echo "[lumi]   brain pid=${PIDS[-1]} -> ${LOG_DIR}/brain.log"

# Wait for the Brain's WebSocket server to bind before Tauri connects.
sleep 1

echo "[lumi] starting Tauri dev..."
fuser -k 1420/tcp 2>/dev/null || true
GDK_BACKEND=x11 WEBKIT_DISABLE_DMABUF_RENDERER=1 npm run tauri dev --prefix app >"${LOG_DIR}/tauri.log" 2>&1 &
PIDS+=($!)
echo "[lumi]   tauri pid=${PIDS[-1]} -> ${LOG_DIR}/tauri.log"

echo ""
echo "[lumi] all running. Ctrl+C to stop."
echo "       tail -F ${LOG_DIR}/brain.log ${LOG_DIR}/tauri.log"
echo ""

# If any child exits, bring everything down
while true; do
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[lumi] pid ${pid} exited unexpectedly; stopping."
      exit 1
    fi
  done
  sleep 1
done
