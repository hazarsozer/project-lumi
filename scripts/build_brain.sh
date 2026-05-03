#!/usr/bin/env bash
# build_brain.sh — Builds the Lumi Brain sidecar binary and copies it to the
# Tauri binaries/ directory so `npm run tauri build` can bundle it into the AppImage.
#
# Usage: bash scripts/build_brain.sh
#
# Prerequisites:
#   uv sync --extra llm --extra tts --extra ptt --extra rag
#   # pyinstaller is included in the dev extra:
#   uv sync --extra dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[build_brain] Building lumi-brain sidecar with PyInstaller..."
cd "${REPO_ROOT}"
uv run pyinstaller \
    "${SCRIPT_DIR}/brain.spec" \
    --distpath "${SCRIPT_DIR}/dist" \
    --workpath "${SCRIPT_DIR}/build" \
    --noconfirm

# Tauri expects sidecar binaries named with the Rust target triple suffix.
# For Linux x86_64 this is: lumi-brain-x86_64-unknown-linux-gnu
TAURI_BINS="${REPO_ROOT}/app/src-tauri/binaries"
mkdir -p "${TAURI_BINS}"

TARGET="lumi-brain-x86_64-unknown-linux-gnu"
echo "[build_brain] Copying lumi-brain → ${TAURI_BINS}/${TARGET}"
rm -rf "${TAURI_BINS:?}/${TARGET}"
cp -r "${SCRIPT_DIR}/dist/lumi-brain" "${TAURI_BINS}/${TARGET}"

echo "[build_brain] Done."
echo "    Binary: ${TAURI_BINS}/${TARGET}/lumi-brain"
echo ""
echo "    Next step: cd app && npm run tauri build"
