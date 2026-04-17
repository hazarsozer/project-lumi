"""
Vision tool for Project Lumi.

Implements ScreenshotTool (name="screenshot") which:
1. Captures the screen using grim (Wayland), scrot (X11), or Pillow.
2. Optionally describes the screenshot via moondream2 GGUF loaded on demand.
3. Unloads the vision model 30 seconds after last use to free VRAM.
4. Unloads the main LLM before loading the vision model (VRAM mutual exclusion).

Security decisions:
- No shell=True anywhere; all subprocess calls use explicit argument lists.
- scrot writes to a deterministic /tmp path; no user-supplied paths accepted.
- Pillow ImageGrab.grab() captures without any subprocess at all.
- The vision model path is taken from VisionConfig (frozen dataclass), not
  from user input, so there is no path traversal risk at model load time.
"""

from __future__ import annotations

import base64
import io
import logging
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from src.core.config import VisionConfig
from src.tools.base import ToolResult

logger = logging.getLogger(__name__)

# llama_cpp is an optional heavy dependency.  Import it at module level so that
# tests can patch it via "llama_cpp.Llama"; fall back to None when the package
# is not installed (vision will still run in capture-only mode).
try:
    import llama_cpp  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    llama_cpp = None  # type: ignore[assignment]

# Temporary file used when scrot is the capture backend.
_SCROT_TMP: str = "/tmp/lumi_screenshot.png"


class ScreenshotTool:
    """Capture the current screen and describe it using moondream2 GGUF.

    Tool schema
    -----------
    name: "screenshot"

    Schema::

        {"tool": "screenshot", "args": {}}

    No arguments are required or accepted.

    Returns:
        ToolResult(success=True, output="<description>", data={"screenshot_bytes": <int>})
        on success.

        ToolResult(success=True, output="Screenshot captured but vision model not
        available.", data={"screenshot_bytes": <int>}) when the GGUF model file is
        absent or vision is disabled.

        ToolResult(success=False, output="<error>", data={}) when capture fails or
        description raises an unexpected exception.
    """

    name: str = "screenshot"
    description: str = (
        "Capture the current screen and return a text description of what is visible. "
        "Args: {} (no arguments required)."
    )

    def __init__(
        self,
        config: VisionConfig,
        llm_loader: Any | None = None,
    ) -> None:
        """Initialise the tool.

        Args:
            config:     VisionConfig frozen dataclass from LumiConfig.vision.
            llm_loader: The orchestrator's ModelLoader instance.  When provided,
                        the LLM is unloaded before the vision model is loaded so
                        the two GGUF models do not compete for VRAM.  Passing None
                        disables VRAM mutual exclusion (safe for testing without a
                        real LLM).
        """
        self._config = config
        self._llm_loader = llm_loader
        self._vision_model: Any | None = None
        self._unload_timer: threading.Timer | None = None
        # Share the VRAM lock with ModelLoader so LLM load and vision-model load
        # are mutually exclusive.  Falls back to a private lock when no LLM loader
        # is provided (e.g. during unit tests without a real LLM).
        self._model_lock: threading.Lock = (
            llm_loader._vram_lock
            if llm_loader is not None and hasattr(llm_loader, "_vram_lock")
            else threading.Lock()
        )

    # ------------------------------------------------------------------
    # Public Tool interface
    # ------------------------------------------------------------------

    def execute(self, args: dict[str, Any]) -> ToolResult:
        """Capture a screenshot and return a text description.

        Args:
            args: Ignored — the screenshot tool takes no arguments.

        Returns:
            ToolResult with output as the description string on success.
        """
        logger.info("ScreenshotTool: executing capture.")
        png_bytes = self._capture()

        if png_bytes is None:
            logger.warning("ScreenshotTool: capture returned None.")
            return ToolResult(
                success=False,
                output=(
                    "Screenshot capture failed. "
                    "Ensure grim, scrot, or Pillow is available."
                ),
                data={},
            )

        model_path = Path(self._config.model_path)
        if not self._config.enabled or not model_path.exists():
            logger.info(
                "ScreenshotTool: vision model not available "
                "(enabled=%s, model_path_exists=%s).",
                self._config.enabled,
                model_path.exists(),
            )
            return ToolResult(
                success=True,
                output="Screenshot captured but vision model not available.",
                data={"screenshot_bytes": len(png_bytes)},
            )

        try:
            description = self._describe(png_bytes)
            logger.info("ScreenshotTool: description complete (%d chars).", len(description))
            return ToolResult(
                success=True,
                output=description,
                data={"screenshot_bytes": len(png_bytes)},
            )
        except Exception as exc:
            logger.exception("ScreenshotTool: description failed")
            return ToolResult(success=False, output=str(exc), data={})

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _capture(self) -> bytes | None:
        """Capture the screen and return PNG bytes, or None on failure.

        The capture backend is chosen according to self._config.capture_method:
        - "grim"   : Wayland-native; pipes PNG to stdout.
        - "scrot"  : X11-native; writes to a temp file.
        - "pillow" : Cross-platform PIL.ImageGrab (requires X11 or XWayland).
        - "auto"   : Try grim → scrot → pillow in order.
        """
        method = self._config.capture_method

        if method == "grim":
            return self._capture_grim()
        if method == "scrot":
            return self._capture_scrot()
        if method == "pillow":
            return self._capture_pillow()
        # "auto" fallthrough
        result = self._capture_grim()
        if result is not None:
            return result
        result = self._capture_scrot()
        if result is not None:
            return result
        return self._capture_pillow()

    def _capture_grim(self) -> bytes | None:
        """Capture via grim (Wayland). Returns PNG bytes or None."""
        grim = shutil.which("grim")
        if grim is None:
            logger.debug("ScreenshotTool._capture_grim: grim not on PATH.")
            return None

        try:
            result = subprocess.run(  # noqa: S603
                [grim, "-", "-t", "png"],
                capture_output=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("ScreenshotTool._capture_grim: subprocess error: %s", exc)
            return None

        if result.returncode != 0 or not result.stdout:
            logger.warning(
                "ScreenshotTool._capture_grim: grim exited %d, stdout=%d bytes.",
                result.returncode,
                len(result.stdout),
            )
            return None

        logger.debug(
            "ScreenshotTool._capture_grim: captured %d bytes.", len(result.stdout)
        )
        return self._maybe_downscale(result.stdout)

    def _capture_scrot(self) -> bytes | None:
        """Capture via scrot (X11). Returns PNG bytes or None."""
        scrot = shutil.which("scrot")
        if scrot is None:
            logger.debug("ScreenshotTool._capture_scrot: scrot not on PATH.")
            return None

        try:
            result = subprocess.run(  # noqa: S603
                [scrot, "-o", _SCROT_TMP],
                capture_output=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("ScreenshotTool._capture_scrot: subprocess error: %s", exc)
            return None

        if result.returncode != 0:
            logger.warning(
                "ScreenshotTool._capture_scrot: scrot exited %d.", result.returncode
            )
            return None

        tmp_path = Path(_SCROT_TMP)
        try:
            png_bytes = tmp_path.read_bytes()
        except OSError as exc:
            logger.warning(
                "ScreenshotTool._capture_scrot: could not read temp file: %s", exc
            )
            return None
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

        logger.debug(
            "ScreenshotTool._capture_scrot: captured %d bytes.", len(png_bytes)
        )
        return self._maybe_downscale(png_bytes)

    def _capture_pillow(self) -> bytes | None:
        """Capture via Pillow ImageGrab. Returns PNG bytes or None."""
        try:
            from PIL import ImageGrab  # type: ignore[import-untyped]

            img = ImageGrab.grab()
            buf = io.BytesIO()
            img.save(buf, "PNG")
            png_bytes = buf.getvalue()
        except Exception as exc:
            logger.warning("ScreenshotTool._capture_pillow: error: %s", exc)
            return None

        logger.debug(
            "ScreenshotTool._capture_pillow: captured %d bytes.", len(png_bytes)
        )
        return self._maybe_downscale(png_bytes)

    def _maybe_downscale(self, png_bytes: bytes) -> bytes:
        """Downscale PNG bytes if the image exceeds max_resolution.

        The image is scaled proportionally so the longest side equals
        max_resolution.  If the image is already within bounds, the original
        bytes are returned unchanged.

        Args:
            png_bytes: Raw PNG file bytes.

        Returns:
            PNG bytes, possibly re-encoded at a smaller resolution.
        """
        max_res = self._config.max_resolution
        try:
            from PIL import Image  # type: ignore[import-untyped]

            img = Image.open(io.BytesIO(png_bytes))
            width, height = img.size
            if max(width, height) <= max_res:
                return png_bytes
            # Scale proportionally.
            if width >= height:
                new_width = max_res
                new_height = int(height * max_res / width)
            else:
                new_height = max_res
                new_width = int(width * max_res / height)
            img = img.resize((new_width, new_height))
            buf = io.BytesIO()
            img.save(buf, "PNG")
            logger.debug(
                "ScreenshotTool._maybe_downscale: %dx%d → %dx%d.",
                width,
                height,
                new_width,
                new_height,
            )
            return buf.getvalue()
        except Exception as exc:
            # If Pillow is unavailable or the image cannot be parsed, return
            # original bytes unchanged.
            logger.warning(
                "ScreenshotTool._maybe_downscale: could not downscale: %s", exc
            )
            return png_bytes

    # ------------------------------------------------------------------
    # Description (vision model)
    # ------------------------------------------------------------------

    def _describe(self, png_bytes: bytes) -> str:
        """Load the vision model and return a description of the image.

        VRAM mutual exclusion: if an LLM is loaded, unload it before loading
        the vision model.  Both models share the same GPU VRAM budget.

        Args:
            png_bytes: PNG image bytes to describe.

        Returns:
            A human-readable description string.
        """
        with self._model_lock:
            # VRAM mutual exclusion — unload LLM first.
            if self._llm_loader is not None and self._llm_loader.is_loaded:
                logger.info(
                    "ScreenshotTool._describe: unloading LLM for VRAM mutual exclusion."
                )
                self._llm_loader.unload()

            if self._vision_model is None:
                logger.info(
                    "ScreenshotTool._describe: loading vision model from %s.",
                    self._config.model_path,
                )
                self._vision_model = llama_cpp.Llama(
                    model_path=self._config.model_path,
                    n_gpu_layers=-1,
                    n_ctx=2048,
                    verbose=False,
                )
                logger.info(
                    "ScreenshotTool: vision model loaded from %s",
                    self._config.model_path,
                )

            # Schedule the 30-second idle unload timer (resets on each call).
            self._schedule_unload()

            # Encode image as base64 for the moondream2 multimodal prompt format.
            img_b64 = base64.b64encode(png_bytes).decode("ascii")
            prompt = (
                f"<image>{img_b64}</image>\n"
                "Describe what you see in this screenshot.\n"
            )

            result = self._vision_model.create_completion(
                prompt,
                max_tokens=256,
                temperature=0.1,
            )
            description: str = result["choices"][0]["text"].strip()
            return description if description else "Unable to describe the image."

    # ------------------------------------------------------------------
    # Timer helpers
    # ------------------------------------------------------------------

    def _schedule_unload(self) -> None:
        """Cancel any existing idle timer and schedule a new one for 30 s."""
        if self._unload_timer is not None:
            self._unload_timer.cancel()
        self._unload_timer = threading.Timer(30.0, self._unload_vision_model)
        self._unload_timer.daemon = True
        self._unload_timer.start()

    def _unload_vision_model(self) -> None:
        """Unload the vision model (called by the idle timer after 30 s)."""
        with self._model_lock:
            self._vision_model = None
            logger.info("ScreenshotTool: vision model unloaded (idle timeout)")
