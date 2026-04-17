"""
Unit tests for src.tools.vision.ScreenshotTool.

All subprocess, PIL, and llama_cpp calls are mocked — no real hardware,
display server, or model files are required.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.core.config import VisionConfig
from src.tools.vision import ScreenshotTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(
    enabled: bool = True,
    capture_method: str = "auto",
    model_path: str = "models/vision/moondream2.gguf",
    llm_loader: MagicMock | None = None,
) -> ScreenshotTool:
    config = VisionConfig(
        enabled=enabled,
        capture_method=capture_method,
        model_path=model_path,
    )
    return ScreenshotTool(config=config, llm_loader=llm_loader)


# ---------------------------------------------------------------------------
# Test 1: capture failure → ToolResult(success=False)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_execute_capture_fails_returns_failure() -> None:
    """When _capture returns None, execute returns success=False."""
    tool = _make_tool()

    with patch.object(tool, "_capture", return_value=None):
        result = tool.execute({})

    assert result.success is False
    assert "capture failed" in result.output.lower() or "capture" in result.output.lower()
    assert result.data == {}


# ---------------------------------------------------------------------------
# Test 2: model not present (disabled) → captured but no description
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_execute_model_not_present_returns_captured_success() -> None:
    """When vision is disabled, execute returns success=True with 'not available'."""
    tool = _make_tool(enabled=False)

    with patch.object(tool, "_capture", return_value=b"PNG_BYTES"):
        result = tool.execute({})

    assert result.success is True
    assert "not available" in result.output.lower()
    assert result.data.get("screenshot_bytes") == len(b"PNG_BYTES")


# ---------------------------------------------------------------------------
# Test 3: full path — capture + describe → description returned
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_execute_calls_describe_and_returns_description(tmp_path: Path) -> None:
    """When enabled and model exists, execute calls _describe and returns output."""
    # Create a fake model file so Path.exists() returns True.
    fake_model = tmp_path / "moondream2.gguf"
    fake_model.write_bytes(b"FAKE")

    tool = _make_tool(enabled=True, model_path=str(fake_model))

    with (
        patch.object(tool, "_capture", return_value=b"PNG_BYTES"),
        patch.object(tool, "_describe", return_value="A desktop with a browser"),
    ):
        result = tool.execute({})

    assert result.success is True
    assert result.output == "A desktop with a browser"
    assert result.data.get("screenshot_bytes") == len(b"PNG_BYTES")


# ---------------------------------------------------------------------------
# Test 4: grim capture success
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_capture_grim_success() -> None:
    """_capture with method=grim returns PNG bytes from subprocess stdout."""
    tool = _make_tool(capture_method="grim")

    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.stdout = b"PNG_BYTES_GRIM"

    with (
        patch("src.tools.vision.shutil.which", return_value="/usr/bin/grim"),
        patch("src.tools.vision.subprocess.run", return_value=fake_proc),
        # Prevent downscaling from requiring Pillow in this unit test.
        patch.object(tool, "_maybe_downscale", side_effect=lambda b: b),
    ):
        result = tool._capture()

    assert result == b"PNG_BYTES_GRIM"


# ---------------------------------------------------------------------------
# Test 5: grim not found → _capture returns None
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_capture_grim_not_found_returns_none() -> None:
    """When grim is not installed, _capture with method=grim returns None."""
    tool = _make_tool(capture_method="grim")

    with patch("src.tools.vision.shutil.which", return_value=None):
        result = tool._capture()

    assert result is None


# ---------------------------------------------------------------------------
# Test 6: _describe unloads LLM before loading vision model
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_describe_unloads_llm_before_loading_vision(tmp_path: Path) -> None:
    """_describe must call llm_loader.unload() before constructing the Llama instance."""
    # Set up mock LLM loader that reports it is loaded.
    mock_llm_loader = MagicMock()
    mock_llm_loader.is_loaded = True

    # Fake model file so the tool does not raise FileNotFoundError.
    fake_model = tmp_path / "moondream2.gguf"
    fake_model.write_bytes(b"FAKE")

    tool = _make_tool(
        enabled=True,
        model_path=str(fake_model),
        llm_loader=mock_llm_loader,
    )

    call_order: list[str] = []

    def record_unload() -> None:
        call_order.append("unload")

    mock_llm_loader.unload.side_effect = record_unload

    mock_vision_instance = MagicMock()
    mock_vision_instance.create_completion.return_value = {
        "choices": [{"text": "A test description"}]
    }

    def record_llama_init(*args: object, **kwargs: object) -> MagicMock:
        call_order.append("llama_init")
        return mock_vision_instance

    with patch("src.tools.vision.llama_cpp") as mock_llama_cpp_module:
        mock_llama_cpp_module.Llama.side_effect = record_llama_init
        # Cancel timer immediately to avoid interference with other tests.
        with patch.object(tool, "_schedule_unload"):
            tool._describe(b"PNG_BYTES")

    # unload must appear before llama_init in the call sequence.
    assert "unload" in call_order, "llm_loader.unload() was never called"
    assert "llama_init" in call_order, "llama_cpp.Llama() was never called"
    assert call_order.index("unload") < call_order.index("llama_init"), (
        "llm_loader.unload() must be called before constructing the vision model"
    )


# ---------------------------------------------------------------------------
# Test 7: _unload_timer is set after _describe
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unload_timer_scheduled_after_describe(tmp_path: Path) -> None:
    """After _describe completes, _unload_timer is a running threading.Timer."""
    fake_model = tmp_path / "moondream2.gguf"
    fake_model.write_bytes(b"FAKE")

    tool = _make_tool(enabled=True, model_path=str(fake_model))

    mock_vision_instance = MagicMock()
    mock_vision_instance.create_completion.return_value = {
        "choices": [{"text": "A desktop"}]
    }

    with patch("src.tools.vision.llama_cpp") as mock_llama_cpp_module:
        mock_llama_cpp_module.Llama.return_value = mock_vision_instance
        tool._describe(b"PNG_BYTES")

    try:
        assert tool._unload_timer is not None, "_unload_timer should not be None after _describe"
        assert isinstance(tool._unload_timer, threading.Timer), (
            "_unload_timer must be a threading.Timer instance"
        )
    finally:
        # Cancel to prevent the timer from firing during other tests.
        if tool._unload_timer is not None:
            tool._unload_timer.cancel()


# ---------------------------------------------------------------------------
# Test 8: execute() when _describe raises → ToolResult(success=False)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_execute_describe_raises_returns_failure(tmp_path: Path) -> None:
    """When _describe raises an exception, execute returns success=False."""
    fake_model = tmp_path / "moondream2.gguf"
    fake_model.write_bytes(b"FAKE")
    tool = _make_tool(enabled=True, model_path=str(fake_model))
    with (
        patch.object(tool, "_capture", return_value=b"PNG_BYTES"),
        patch.object(tool, "_describe", side_effect=RuntimeError("model boom")),
    ):
        result = tool.execute({})
    assert result.success is False
    assert "model boom" in result.output
    assert result.data == {}


# ---------------------------------------------------------------------------
# Test 9: _capture in "auto" mode — fallthrough paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_capture_auto_falls_through_grim_to_scrot() -> None:
    """In 'auto' mode, if grim returns None, scrot result is used."""
    tool = _make_tool(capture_method="auto")
    with (
        patch.object(tool, "_capture_grim", return_value=None),
        patch.object(tool, "_capture_scrot", return_value=b"SCROT_BYTES"),
    ):
        assert tool._capture() == b"SCROT_BYTES"


@pytest.mark.unit
def test_capture_auto_falls_through_grim_scrot_to_pillow() -> None:
    """In 'auto' mode, if grim and scrot both return None, pillow result is used."""
    tool = _make_tool(capture_method="auto")
    with (
        patch.object(tool, "_capture_grim", return_value=None),
        patch.object(tool, "_capture_scrot", return_value=None),
        patch.object(tool, "_capture_pillow", return_value=b"PILLOW_BYTES"),
    ):
        assert tool._capture() == b"PILLOW_BYTES"


@pytest.mark.unit
def test_capture_auto_all_fail_returns_none() -> None:
    """In 'auto' mode, all backends failing returns None."""
    tool = _make_tool(capture_method="auto")
    with (
        patch.object(tool, "_capture_grim", return_value=None),
        patch.object(tool, "_capture_scrot", return_value=None),
        patch.object(tool, "_capture_pillow", return_value=None),
    ):
        assert tool._capture() is None


@pytest.mark.unit
def test_capture_auto_grim_succeeds_skips_others() -> None:
    """In 'auto' mode, grim success means scrot and pillow are not called."""
    tool = _make_tool(capture_method="auto")
    mock_scrot = MagicMock(return_value=b"SCROT")
    mock_pillow = MagicMock(return_value=b"PILLOW")
    with (
        patch.object(tool, "_capture_grim", return_value=b"GRIM_BYTES"),
        patch.object(tool, "_capture_scrot", mock_scrot),
        patch.object(tool, "_capture_pillow", mock_pillow),
    ):
        assert tool._capture() == b"GRIM_BYTES"
    mock_scrot.assert_not_called()
    mock_pillow.assert_not_called()


@pytest.mark.unit
def test_capture_routes_to_scrot_when_method_is_scrot() -> None:
    """When capture_method='scrot', _capture delegates to _capture_scrot."""
    tool = _make_tool(capture_method="scrot")
    with patch.object(tool, "_capture_scrot", return_value=b"SCROT_BYTES") as m:
        assert tool._capture() == b"SCROT_BYTES"
    m.assert_called_once()


@pytest.mark.unit
def test_capture_routes_to_pillow_when_method_is_pillow() -> None:
    """When capture_method='pillow', _capture delegates to _capture_pillow."""
    tool = _make_tool(capture_method="pillow")
    with patch.object(tool, "_capture_pillow", return_value=b"PILLOW_BYTES") as m:
        assert tool._capture() == b"PILLOW_BYTES"
    m.assert_called_once()


# ---------------------------------------------------------------------------
# Test 10: _capture_grim failure paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_capture_grim_timeout_returns_none() -> None:
    """When grim subprocess times out, _capture_grim returns None."""
    tool = _make_tool(capture_method="grim")
    with (
        patch("src.tools.vision.shutil.which", return_value="/usr/bin/grim"),
        patch(
            "src.tools.vision.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="grim", timeout=10),
        ),
    ):
        assert tool._capture_grim() is None


@pytest.mark.unit
def test_capture_grim_oserror_returns_none() -> None:
    """When grim subprocess raises OSError, _capture_grim returns None."""
    tool = _make_tool(capture_method="grim")
    with (
        patch("src.tools.vision.shutil.which", return_value="/usr/bin/grim"),
        patch("src.tools.vision.subprocess.run", side_effect=OSError("display error")),
    ):
        assert tool._capture_grim() is None


@pytest.mark.unit
def test_capture_grim_nonzero_returncode_returns_none() -> None:
    """When grim exits non-zero, _capture_grim returns None."""
    tool = _make_tool(capture_method="grim")
    fake_proc = MagicMock()
    fake_proc.returncode = 1
    fake_proc.stdout = b""
    with (
        patch("src.tools.vision.shutil.which", return_value="/usr/bin/grim"),
        patch("src.tools.vision.subprocess.run", return_value=fake_proc),
    ):
        assert tool._capture_grim() is None


@pytest.mark.unit
def test_capture_grim_empty_stdout_returns_none() -> None:
    """When grim exits 0 but stdout is empty, _capture_grim returns None."""
    tool = _make_tool(capture_method="grim")
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.stdout = b""
    with (
        patch("src.tools.vision.shutil.which", return_value="/usr/bin/grim"),
        patch("src.tools.vision.subprocess.run", return_value=fake_proc),
    ):
        assert tool._capture_grim() is None


# ---------------------------------------------------------------------------
# Test 11: _capture_scrot — all paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_capture_scrot_success() -> None:
    """_capture_scrot reads bytes from the temp file and returns them."""
    tool = _make_tool(capture_method="scrot")
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_png = b"\x89PNG_FAKE"
    with (
        patch("src.tools.vision.shutil.which", return_value="/usr/bin/scrot"),
        patch("src.tools.vision.subprocess.run", return_value=fake_proc),
        patch("src.tools.vision.Path.read_bytes", return_value=fake_png),
        patch("src.tools.vision.Path.unlink"),
        patch.object(tool, "_maybe_downscale", side_effect=lambda b: b),
    ):
        assert tool._capture_scrot() == fake_png


@pytest.mark.unit
def test_capture_scrot_not_found_returns_none() -> None:
    """When scrot is absent from PATH, _capture_scrot returns None."""
    tool = _make_tool(capture_method="scrot")
    with patch("src.tools.vision.shutil.which", return_value=None):
        assert tool._capture_scrot() is None


@pytest.mark.unit
def test_capture_scrot_nonzero_returncode_returns_none() -> None:
    """When scrot exits non-zero, _capture_scrot returns None."""
    tool = _make_tool(capture_method="scrot")
    fake_proc = MagicMock()
    fake_proc.returncode = 1
    with (
        patch("src.tools.vision.shutil.which", return_value="/usr/bin/scrot"),
        patch("src.tools.vision.subprocess.run", return_value=fake_proc),
    ):
        assert tool._capture_scrot() is None


@pytest.mark.unit
def test_capture_scrot_read_bytes_oserror_returns_none() -> None:
    """When reading the temp file raises OSError, _capture_scrot returns None."""
    tool = _make_tool(capture_method="scrot")
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    with (
        patch("src.tools.vision.shutil.which", return_value="/usr/bin/scrot"),
        patch("src.tools.vision.subprocess.run", return_value=fake_proc),
        patch("src.tools.vision.Path.read_bytes", side_effect=OSError("no file")),
        patch("src.tools.vision.Path.unlink"),
    ):
        assert tool._capture_scrot() is None


@pytest.mark.unit
def test_capture_scrot_subprocess_timeout_returns_none() -> None:
    """When scrot subprocess times out, _capture_scrot returns None."""
    tool = _make_tool(capture_method="scrot")
    with (
        patch("src.tools.vision.shutil.which", return_value="/usr/bin/scrot"),
        patch(
            "src.tools.vision.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="scrot", timeout=10),
        ),
    ):
        assert tool._capture_scrot() is None


# ---------------------------------------------------------------------------
# Test 12: _capture_pillow — success and failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_capture_pillow_exception_returns_none() -> None:
    """When PIL raises an exception inside _capture_pillow, it returns None."""
    import sys

    tool = _make_tool(capture_method="pillow")
    mock_pil = MagicMock()
    mock_pil.ImageGrab.grab.side_effect = Exception("no display")
    with patch.dict(sys.modules, {"PIL": mock_pil, "PIL.ImageGrab": mock_pil.ImageGrab}):
        assert tool._capture_pillow() is None


# ---------------------------------------------------------------------------
# Test 13: _maybe_downscale
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_maybe_downscale_small_image_unchanged() -> None:
    """When image dimensions are within max_resolution, bytes pass through unchanged."""
    import io as _io

    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")

    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buf = _io.BytesIO()
    img.save(buf, "PNG")
    small_png = buf.getvalue()

    tool = _make_tool()
    assert tool._maybe_downscale(small_png) == small_png


@pytest.mark.unit
def test_maybe_downscale_wide_image_is_resized() -> None:
    """When image width > max_resolution, the image is scaled proportionally."""
    import io as _io

    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")

    img = Image.new("RGB", (2000, 800), color=(0, 128, 255))
    buf = _io.BytesIO()
    img.save(buf, "PNG")
    large_png = buf.getvalue()

    tool = _make_tool()
    result = tool._maybe_downscale(large_png)
    result_img = Image.open(_io.BytesIO(result))
    assert max(result_img.size) <= 1280


@pytest.mark.unit
def test_maybe_downscale_tall_image_is_resized() -> None:
    """When image height > max_resolution, the image is scaled proportionally."""
    import io as _io

    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")

    img = Image.new("RGB", (500, 2000), color=(200, 100, 50))
    buf = _io.BytesIO()
    img.save(buf, "PNG")
    tall_png = buf.getvalue()

    tool = _make_tool()
    result = tool._maybe_downscale(tall_png)
    result_img = Image.open(_io.BytesIO(result))
    assert max(result_img.size) <= 1280


@pytest.mark.unit
def test_maybe_downscale_unparseable_bytes_returns_original() -> None:
    """When Pillow cannot parse the bytes, the original bytes are returned unchanged."""
    tool = _make_tool()
    garbage = b"NOT_A_PNG"
    assert tool._maybe_downscale(garbage) == garbage


# ---------------------------------------------------------------------------
# Test 14: _describe — model already loaded skips re-init; empty text fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_describe_model_already_loaded_skips_init() -> None:
    """When _vision_model is already set, llama_cpp.Llama() is not called again."""
    fake_model_instance = MagicMock()
    fake_model_instance.create_completion.return_value = {
        "choices": [{"text": "Already loaded description"}]
    }
    tool = _make_tool()
    tool._vision_model = fake_model_instance

    with patch("src.tools.vision.llama_cpp") as mock_llama_cpp_module:
        with patch.object(tool, "_schedule_unload"):
            result = tool._describe(b"PNG_BYTES")

    mock_llama_cpp_module.Llama.assert_not_called()
    assert result == "Already loaded description"
    if tool._unload_timer is not None:
        tool._unload_timer.cancel()


@pytest.mark.unit
def test_describe_returns_fallback_when_text_empty() -> None:
    """When the model returns whitespace-only text, _describe returns the fallback."""
    fake_model_instance = MagicMock()
    fake_model_instance.create_completion.return_value = {
        "choices": [{"text": "   "}]
    }
    tool = _make_tool()
    tool._vision_model = fake_model_instance

    with patch.object(tool, "_schedule_unload"):
        result = tool._describe(b"PNG_BYTES")

    assert result == "Unable to describe the image."


# ---------------------------------------------------------------------------
# Test 15: _schedule_unload — cancels existing timer; works with no prior timer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_schedule_unload_cancels_existing_timer() -> None:
    """_schedule_unload cancels any existing timer before scheduling a new one."""
    tool = _make_tool()
    old_timer = MagicMock(spec=threading.Timer)
    tool._unload_timer = old_timer

    with patch("src.tools.vision.threading.Timer") as mock_timer_cls:
        new_timer = MagicMock()
        mock_timer_cls.return_value = new_timer
        tool._schedule_unload()

    old_timer.cancel.assert_called_once()
    new_timer.start.assert_called_once()
    assert tool._unload_timer is new_timer


@pytest.mark.unit
def test_schedule_unload_no_existing_timer() -> None:
    """_schedule_unload works when no prior timer exists."""
    tool = _make_tool()
    assert tool._unload_timer is None

    with patch("src.tools.vision.threading.Timer") as mock_timer_cls:
        new_timer = MagicMock()
        mock_timer_cls.return_value = new_timer
        tool._schedule_unload()

    new_timer.start.assert_called_once()
    assert tool._unload_timer is new_timer


# ---------------------------------------------------------------------------
# Test 16: _unload_vision_model — timer callback clears the model
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unload_vision_model_clears_model() -> None:
    """_unload_vision_model sets _vision_model to None under the lock."""
    tool = _make_tool()
    tool._vision_model = MagicMock()
    tool._unload_vision_model()
    assert tool._vision_model is None


@pytest.mark.unit
def test_unload_vision_model_idempotent_when_none() -> None:
    """Calling _unload_vision_model when model is already None does not raise."""
    tool = _make_tool()
    assert tool._vision_model is None
    tool._unload_vision_model()
    assert tool._vision_model is None
