"""
VRAM mutex concurrency tests for ModelLoader + ScreenshotTool.

This test file documents and verifies whether ModelLoader.load() and
ScreenshotTool._describe() share a common threading.Lock that prevents
concurrent VRAM use.

## Findings (discovered while writing these tests)

- ScreenshotTool owns a private self._model_lock (threading.Lock) that
  protects its own _describe() and _unload_vision_model() methods.
- ModelLoader has NO threading.Lock whatsoever. Its load() / unload() /
  is_loaded are entirely unprotected.
- The "VRAM mutual exclusion" is one-directional: ScreenshotTool acquires
  ITS OWN lock and then calls llm_loader.unload() — but if a second thread
  is simultaneously calling ModelLoader.load(), no synchronisation blocks it.

## Test outcomes

- test_model_loader_and_vision_share_same_lock:
    FAILS — the lock objects are different instances (ModelLoader has none).

- test_concurrent_load_and_screenshot_serialize:
    FAILS — ModelLoader.load() runs without acquiring any shared lock, so it
    can overlap with ScreenshotTool._describe().

- test_no_deadlock_on_sequential_use:
    PASSES — sequential use never deadlocks because ModelLoader never blocks
    on a lock that ScreenshotTool holds.

These failing tests are intentional regression markers.  The fix would be:
  1. Add a module-level (or injected) threading.Lock to ModelLoader.
  2. Acquire that same lock inside ModelLoader.load() / unload().
  3. Pass the lock to ScreenshotTool so both classes use the same instance.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import LLMConfig, VisionConfig
from src.llm.model_loader import ModelLoader
from src.tools.vision import ScreenshotTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vision_tool(
    llm_loader: ModelLoader | None = None,
    model_path: str = "/fake/moondream.gguf",
) -> ScreenshotTool:
    config = VisionConfig(
        enabled=True,
        capture_method="auto",
        model_path=model_path,
    )
    return ScreenshotTool(config=config, llm_loader=llm_loader)


def _make_llm_config() -> LLMConfig:
    return LLMConfig()


# ---------------------------------------------------------------------------
# Test 1 — Shared lock identity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_model_loader_and_vision_share_same_lock() -> None:
    """ModelLoader and ScreenshotTool must share the same threading.Lock instance.

    This test FAILS with the current implementation because:
    - ModelLoader has no threading.Lock at all.
    - ScreenshotTool creates its own private self._model_lock.

    The fix: expose a shared lock (e.g. passed via constructor or a
    module-level singleton) so both classes acquire the same object.
    """
    loader = ModelLoader()
    tool = _make_vision_tool(llm_loader=loader)

    # ModelLoader currently has no lock attribute at all.
    assert hasattr(loader, "_vram_lock"), (
        "ModelLoader must expose a '_vram_lock' attribute. "
        "Currently it has no lock, so concurrent load() calls from another "
        "thread are not serialised with ScreenshotTool._describe()."
    )

    # The lock inside ScreenshotTool must be the SAME object as the one in
    # ModelLoader, not a separately created instance.
    assert loader._vram_lock is tool._model_lock, (  # type: ignore[attr-defined]
        "ScreenshotTool._model_lock and ModelLoader._vram_lock must be the "
        "same threading.Lock instance.  Currently they are independent objects "
        "(or ModelLoader has no lock at all), which allows VRAM races."
    )


# ---------------------------------------------------------------------------
# Test 2 — Concurrent load + screenshot serialize (no overlap)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_concurrent_load_and_screenshot_serialize() -> None:
    """ModelLoader.load() and ScreenshotTool._describe() must not overlap.

    This test FAILS with the current implementation because ModelLoader.load()
    acquires no lock, so both threads proceed simultaneously.

    Mechanism:
    - Thread A calls ScreenshotTool._describe() which acquires _model_lock
      and then sleeps 0.15 s (simulating slow GGUF load).
    - Thread B calls ModelLoader.load() at the same time.
    - If the mutex is shared, thread B must wait until A's 0.15 s sleep ends.
    - We measure the start times of both critical sections; overlap means
      B's critical section started before A's ended.
    """
    loader = ModelLoader()
    tool = _make_vision_tool(llm_loader=loader)

    timeline: list[tuple[str, float]] = []

    HOLD_SECONDS = 0.15

    # Patch ModelLoader.load() to record its entry/exit times.
    original_load = ModelLoader.load

    def instrumented_load(self: ModelLoader, config: LLMConfig) -> None:
        # Acquire the shared VRAM lock exactly as the real load() does, so we
        # measure whether the critical sections truly serialize.
        with self._vram_lock:
            timeline.append(("load_start", time.monotonic()))
            time.sleep(0.05)  # simulate some work
            timeline.append(("load_end", time.monotonic()))

    # Patch ScreenshotTool._describe() to record entry/exit and hold the lock
    # for HOLD_SECONDS to give load() a window to race.
    original_describe = ScreenshotTool._describe

    def instrumented_describe(self: ScreenshotTool, png_bytes: bytes) -> str:
        # Acquire the lock the same way _describe does so we measure real
        # critical-section overlap.
        with self._model_lock:
            timeline.append(("describe_start", time.monotonic()))
            time.sleep(HOLD_SECONDS)
            timeline.append(("describe_end", time.monotonic()))
        return "mocked description"

    with (
        patch.object(ModelLoader, "load", instrumented_load),
        patch.object(ScreenshotTool, "_describe", instrumented_describe),
    ):
        # Start the screenshot thread first so it grabs the lock.
        t_describe = threading.Thread(
            target=tool._describe, args=(b"fake_png",), daemon=True
        )
        t_load = threading.Thread(
            target=loader.load, args=(_make_llm_config(),), daemon=True
        )

        t_describe.start()
        time.sleep(0.02)  # let describe acquire the lock before load starts
        t_load.start()

        t_describe.join(timeout=2.0)
        t_load.join(timeout=2.0)

    assert len(timeline) == 4, f"Expected 4 timeline events, got: {timeline}"

    events = {name: ts for name, ts in timeline}

    describe_end = events["describe_end"]
    load_start = events["load_start"]

    # If the mutex is shared, load_start must be >= describe_end (load waited).
    # If load_start < describe_end the two critical sections overlapped → FAIL.
    assert load_start >= describe_end, (
        f"VRAM race detected: ModelLoader.load() started at {load_start:.4f}s "
        f"but ScreenshotTool._describe() did not finish until {describe_end:.4f}s. "
        f"Overlap = {describe_end - load_start:.4f}s. "
        "ModelLoader.load() must acquire the shared VRAM lock before proceeding."
    )


# ---------------------------------------------------------------------------
# Test 3 — No deadlock on sequential use
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_deadlock_on_sequential_use() -> None:
    """Sequential load() then ScreenshotTool.execute() must complete within 2s.

    This test PASSES with the current implementation because ModelLoader never
    blocks on a lock.  It is included as a regression guard: once the shared
    lock is introduced, this test ensures the implementation does not deadlock
    (e.g. via nested acquisition on the same non-reentrant Lock).
    """
    loader = ModelLoader()
    tool = _make_vision_tool(llm_loader=loader)

    completed = threading.Event()

    def run() -> None:
        mock_llama = MagicMock()
        mock_llama.return_value = {"choices": [{"text": "mock response"}]}
        mock_llama.create_completion.return_value = {
            "choices": [{"text": "mocked description"}]
        }

        # llama_cpp is not installed on CI — inject a fake module so the lazy
        # import inside ModelLoader.load() and vision.py both see the mock.
        _llama_mod = types.ModuleType("llama_cpp")
        _llama_mod.Llama = mock_llama  # type: ignore[attr-defined]

        with (
            patch.dict(sys.modules, {"llama_cpp": _llama_mod}),
            patch("src.tools.vision.llama_cpp") as mock_vision_llama,
        ):
            mock_vision_instance = MagicMock()
            mock_vision_instance.create_completion.return_value = {
                "choices": [{"text": "mocked description"}]
            }
            mock_vision_llama.Llama.return_value = mock_vision_instance

            # Sequential: load LLM first.
            loader.load(_make_llm_config())
            assert loader.is_loaded

            # Patch _capture to return fake PNG so _describe is reached.
            fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

            # Patch Path.exists so vision model path appears present.
            with patch.object(Path, "exists", return_value=True):
                with patch.object(tool, "_capture", return_value=fake_png):
                    result = tool.execute({})

        assert result is not None
        completed.set()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=2.0)

    assert completed.is_set(), (
        "Sequential load() + execute() did not complete within 2 seconds — "
        "possible deadlock introduced by the shared lock implementation."
    )
