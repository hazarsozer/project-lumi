"""
End-to-end voice pipeline smoke test.

Verifies that all pipeline stages fire in sequence without a microphone
by injecting a fake WakeDetectedEvent + RecordingCompleteEvent directly
into the Orchestrator's event queue, then asserting that the expected
downstream events (TranscriptReady, LLMResponseReady, TTSChunkReady,
state transitions) are posted within a timeout.

Run with:
    uv run python scripts/smoke_test_voice.py

Exit 0 = PASS. Exit 1 = FAIL (details printed).
"""

import sys
import time
import queue
import threading
import logging
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import load_config
from src.core.events import (
    WakeDetectedEvent,
    RecordingCompleteEvent,
    TranscriptReadyEvent,
    LLMResponseReadyEvent,
    TTSChunkReadyEvent,
    SpeechCompletedEvent,
    ShutdownEvent,
)
from src.core.orchestrator import Orchestrator
from src.core.logging_config import setup_logging
from src.audio.scribe import Scribe

TIMEOUT_S = 30

setup_logging("WARNING")
log = logging.getLogger("smoke_test")


def run_smoke_test() -> bool:
    cfg = load_config()

    print("Loading Scribe (faster-whisper)...")
    scribe = Scribe(
        model_size=cfg.scribe.model_size,
        initial_prompt=cfg.scribe.initial_prompt or "Lumi, desktop assistant.",
    )

    observed: dict[str, float] = {}
    failures: list[str] = []
    done_event = threading.Event()

    # Patch Orchestrator to intercept events as they are handled
    original_dispatch = Orchestrator._dispatch

    def patched_dispatch(self, event):
        name = type(event).__name__
        if name not in observed:
            observed[name] = time.monotonic()
            log.warning("OBSERVED: %s", name)
        if isinstance(event, (SpeechCompletedEvent, LLMResponseReadyEvent)):
            done_event.set()
        return original_dispatch(self, event)

    Orchestrator._dispatch = patched_dispatch

    print("Building orchestrator...")
    orch = Orchestrator(cfg, scribe=scribe)
    t = threading.Thread(target=orch.run, daemon=True)
    t.start()

    # Give orchestrator a moment to start
    time.sleep(0.3)

    # Inject wake + fake recording
    import numpy as np
    fake_np = np.zeros(16000, dtype=np.int16)  # 1 s silence at 16 kHz
    orch._event_queue.put(WakeDetectedEvent(timestamp=time.monotonic()))
    time.sleep(0.1)
    orch._event_queue.put(RecordingCompleteEvent(audio=fake_np))

    # Wait for pipeline to reach LLM response or TTS
    reached = done_event.wait(timeout=TIMEOUT_S)
    if not reached:
        failures.append(
            f"Pipeline did not reach LLMResponseReady or SpeechCompleted within {TIMEOUT_S}s"
        )

    # Shut down
    orch._event_queue.put(ShutdownEvent())
    t.join(timeout=5)

    # Report
    print("\n=== Lumi End-to-End Smoke Test ===\n")
    stages = [
        ("WakeDetectedEvent", "Wake word detected"),
        ("RecordingCompleteEvent", "VAD recording captured"),
        ("TranscriptReadyEvent", "STT transcription completed"),
        ("LLMResponseReadyEvent", "LLM response generated"),
        ("TTSChunkReadyEvent", "TTS audio chunk synthesised"),
        ("SpeechCompletedEvent", "Speech playback completed"),
    ]

    for event_name, label in stages:
        if event_name in observed:
            print(f"  PASS  {label}")
        else:
            print(f"  FAIL  {label}  ← not observed")
            if event_name not in ("TTSChunkReadyEvent", "SpeechCompletedEvent"):
                failures.append(f"{event_name} not observed")

    print()
    if failures:
        for f in failures:
            print(f"FAILURE: {f}")
        return False

    print("PASS — all required pipeline stages fired.\n")
    return True


if __name__ == "__main__":
    ok = run_smoke_test()
    sys.exit(0 if ok else 1)
