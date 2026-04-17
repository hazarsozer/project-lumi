"""Discovery test: document the actual return format of kokoro_onnx.Kokoro.create().

This test is marked skip when the model file is absent (CI), but runs locally
to verify what the second return value from create() looks like before we
build viseme_map.py on top of it.
"""

from __future__ import annotations

import os

import pytest

MODEL_PATH = "models/tts/kokoro-v1_0.onnx"
VOICES_PATH = "models/tts/voices.bin"


@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Kokoro model not present — CI skip",
)
def test_create_returns_tuple() -> None:
    import kokoro_onnx

    k = kokoro_onnx.Kokoro(MODEL_PATH, VOICES_PATH)
    result = k.create("hello", voice="af_heart", speed=1.0, lang="en-us")
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple, got len={len(result)}"
    samples, phonemes = result
    # Document what phonemes looks like:
    print(f"\nphonemes type: {type(phonemes)}")
    print(f"phonemes value: {phonemes!r}")
