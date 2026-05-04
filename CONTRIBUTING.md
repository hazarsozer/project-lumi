# Contributing to Lumi

## Wake-Word Model Provenance

The repository ships with `models/hey_lumi.onnx` as the default wake-word model.

**Origin:** Custom model trained by the project author using [openWakeWord](https://github.com/dscripka/openWakeWord)'s training pipeline on a personal voice dataset.

**License:** Proprietary — not for redistribution. This file is excluded from the public release bundle. Third-party users must supply their own ONNX wake-word model (openWakeWord compatible) or disable wake-word detection entirely via `audio.wake_word_enabled: false` in `config.yaml` and use push-to-talk (`audio.ptt_enabled: true`) instead.

**Training reproducibility:** The training data and exact openWakeWord version used are not publicly available. To train your own model see [openWakeWord custom model training](https://github.com/dscripka/openWakeWord#training-new-models).

**Python version constraint:** `pyproject.toml` pins `openwakeword==0.4.0` exactly. Version 0.6.0 has no Python 3.12 wheels; all other versions break the monkey-patch in `src/audio/ears.py:70-88` which works around a missing `inference_framework` kwarg in that exact release. Do not upgrade without validating the patch still applies.

## Development Setup

```bash
# Install runtime + dev extras
uv sync --extra llm --extra tts --extra dev

# Run tests
uv run pytest --ignore=tests/test_rag_store.py

# Build frontend
cd app && npm install && npm run build

# Build Tauri AppImage (Linux)
cd app && npm run tauri -- build
```

### Persona LoRA Training (optional)

The LoRA → GGUF merge/quantize pipeline (`scripts/merge_and_quantize.py`) requires:

1. Install the `[qlora]` extra: `uv sync --extra qlora`
2. Build `llama.cpp` locally. The script expects bins at `<llama-cpp-dir>/build/bin/llama-quantize` and `<llama-cpp-dir>/convert_hf_to_gguf.py`. Pass the path via `--llama-cpp-dir`. See `scripts/merge_and_quantize.py --help` for full usage.

## Code Style

- Python: `black` + `ruff` (see `pyproject.toml`)
- TypeScript: `tsc --noEmit` + `eslint`
- No `print()` in production Python — use `logging.getLogger(__name__)`
- Frozen dataclasses for all events (`src/core/events.py`)
- Every public Python change must keep `uv run pytest` green at 80%+ coverage
