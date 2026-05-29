#!/usr/bin/env bash
# build_brain.sh — Builds the Lumi Brain sidecar binary and copies it to the
# Tauri binaries/ directory so `npm run tauri build` can bundle it into the AppImage.
#
# Usage: bash scripts/build_brain.sh [--print-target]
#
# Prerequisites:
#   uv sync --extra llm --extra tts --extra ptt --extra rag
#   # pyinstaller is included in the dev extra:
#   uv sync --extra dev
#
# Environment:
#   LUMI_TARGET_TRIPLE  Override the detected Rust target triple (e.g. for
#                       cross-compilation when a cross toolchain is available).
#                       Example: LUMI_TARGET_TRIPLE=aarch64-apple-darwin bash scripts/build_brain.sh
#
# Note on cross-compilation:
#   This script resolves a native target triple for the current host.
#   True cross-compilation (e.g. building a macOS binary on Linux) is NOT
#   provided automatically; set LUMI_TARGET_TRIPLE together with a matching
#   cross toolchain if you need it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# detect_target_triple — resolve the Rust target triple for the current host.
# Returns one of:
#   x86_64-unknown-linux-gnu
#   aarch64-unknown-linux-gnu
#   x86_64-apple-darwin
#   aarch64-apple-darwin
# Exits non-zero with a clear error on an unsupported host.
# ---------------------------------------------------------------------------
detect_target_triple() {
    local _os _arch

    _os="$(uname -s)"
    _arch="$(uname -m)"

    case "${_os}" in
        Linux)
            case "${_arch}" in
                x86_64)         echo "x86_64-unknown-linux-gnu" ;;
                aarch64|arm64)  echo "aarch64-unknown-linux-gnu" ;;
                *)
                    echo "[build_brain] ERROR: unsupported Linux architecture: ${_arch}" >&2
                    echo "[build_brain] Set LUMI_TARGET_TRIPLE to override." >&2
                    exit 1
                    ;;
            esac
            ;;
        Darwin)
            case "${_arch}" in
                arm64|aarch64)  echo "aarch64-apple-darwin" ;;
                x86_64)         echo "x86_64-apple-darwin" ;;
                *)
                    echo "[build_brain] ERROR: unsupported macOS architecture: ${_arch}" >&2
                    echo "[build_brain] Set LUMI_TARGET_TRIPLE to override." >&2
                    exit 1
                    ;;
            esac
            ;;
        *)
            echo "[build_brain] ERROR: unsupported OS: ${_os}" >&2
            echo "[build_brain] Set LUMI_TARGET_TRIPLE to override." >&2
            exit 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Resolve target triple: honour env override, otherwise detect from host.
# ---------------------------------------------------------------------------
if [[ -n "${LUMI_TARGET_TRIPLE:-}" ]]; then
    RESOLVED_TRIPLE="${LUMI_TARGET_TRIPLE}"
    echo "[build_brain] Using override target triple: ${RESOLVED_TRIPLE}"
else
    RESOLVED_TRIPLE="$(detect_target_triple)"
    echo "[build_brain] Detected host target triple: ${RESOLVED_TRIPLE}"
fi

# --print-target: just emit the resolved triple and exit (useful for tests /
# CI introspection without triggering a real build).
if [[ "${1:-}" == "--print-target" ]]; then
    echo "${RESOLVED_TRIPLE}"
    exit 0
fi

echo "[build_brain] Building lumi-brain sidecar with PyInstaller..."
cd "${REPO_ROOT}"
uv run pyinstaller \
    "${SCRIPT_DIR}/brain.spec" \
    --distpath "${SCRIPT_DIR}/dist" \
    --workpath "${SCRIPT_DIR}/build" \
    --noconfirm

# Tauri expects sidecar binaries named with the Rust target triple suffix.
# e.g. lumi-brain-x86_64-unknown-linux-gnu
TAURI_BINS="${REPO_ROOT}/app/src-tauri/binaries"
mkdir -p "${TAURI_BINS}"

TARGET="lumi-brain-${RESOLVED_TRIPLE}"
echo "[build_brain] Copying lumi-brain → ${TAURI_BINS}/${TARGET}"
rm -rf "${TAURI_BINS:?}/${TARGET}"
cp -r "${SCRIPT_DIR}/dist/lumi-brain" "${TAURI_BINS}/${TARGET}"

echo "[build_brain] Done."
echo "    Binary: ${TAURI_BINS}/${TARGET}/lumi-brain"
echo ""
echo "    Next step: cd app && npm run tauri build"
