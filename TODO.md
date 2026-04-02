# Project Lumi: Development TODOs

## 1. Synchronous Blocking Audio Pipeline
* **Context:** The audio pipeline currently calls the wake callback synchronously, blocking the main stream for up to 13 seconds. Phase 3 (LLM) and 4 (TTS) will increase this to 38 seconds of "deafness".
* **Why it matters:** Lumi must remain interruptible. A blocking pipeline drops wake words, cannot be stopped mid-response, and creates unbounded queue growth. Without an explicit interrupt path, the pipeline is non-blocking in structure but still effectively uninterruptible.
* **Step-by-step Actions:**
  1. **Define all typed events upfront:** Create `src/core/events.py` and define the complete API surface even as stubs: `WakeDetectedEvent`, `RecordingCompleteEvent`, `TranscriptReadyEvent`, `LLMResponseReadyEvent`, `TTSChunkReadyEvent`, `InterruptEvent`, and `ShutdownEvent`.
  2. **Refactor Ears thread:** Update `ears.py` to push `WakeDetectedEvent` to an event queue rather than invoking a synchronous handler.
  3. **Build the Orchestrator with explicit Interrupt handling:** Create `src/core/orchestrator.py` that runs in a separate thread. Critically, when it receives an `InterruptEvent` while in `PROCESSING` or `SPEAKING` state, it must signal the in-progress stage to stop via a cancel flag, drain pending `TTSChunkReadyEvent`s, transition back to `IDLE`, and re-enable wake word detection.

## 2. Mixed Runtime and Training Dependencies
* **Context:** `pyproject.toml` mixes core runtime packages (Whisper, ONNX) with heavy ML training tools (PyTorch, Torchaudio), bloating the user installation by 2-5 GB.
* **Why it matters:** For the "Zero Cost" philosophy, Lumi needs to be lightweight. Users downloading Lumi shouldn't have to install heavy training modules.
* **Step-by-step Actions:**
  1. **Modify `pyproject.toml`:** Separate runtime dependencies (only those strictly needed for inference).
  2. **Define optional dependency groups:** Create `[project.optional-dependencies]` blocks for `training`, `onnx-tools`, `tts`, and `dev` environments using `uv`.

## 3. Monkey-Patching openwakeword Internals
* **Context:** `ears.py` currently relies on an unsafe monkey-patch of `openwakeword.utils.AudioFeatures` to bypass an unsupported kwarg.
* **Why it matters:** Monkey-patching third-party private attributes is extremely fragile and will silently break during minor updates to the library. A version mismatch will silently pass or fail unpredictably, leaving the app in a broken state.
* **Step-by-step Actions:**
  1. **Pin specific version:** Update `pyproject.toml` to pin `"openwakeword==0.6.0"`.
  2. **Add hard startup check:** Implement a startup check in `startup_check.py` that halts execution immediately with a `RuntimeError` if the `openwakeword` version is not exactly `0.6.0`, preventing unpredictable behavior.
  3. **Long-term fix:** Push a PR upstream to add `inference_framework` kwarg and remove the monkey patch entirely once merged.

## 4. `time` Module Shadow in `_mic_callback`
* **Context:** The `time` argument in `sounddevice`'s `_mic_callback` shadows the global `import time`, meaning any use of `time.monotonic()` in that function will crash.
* **Why it matters:** It acts as a silent ticking time bomb for any future developer maintaining the microphone callback.
* **Step-by-step Actions:**
  1. **Alias the module:** Change the import statement to `import time as _time`.
  2. **Update usages:** Replace all instances of `time.monotonic()` with `_time.monotonic()` in `ears.py`.

## 5. No Configuration System
* **Context:** Variables like VAD threshold, chunk sizes, beam size, and recording timeouts are scattered and hardcoded across `main.py`, `ears.py`, and `scribe.py`.
* **Why it matters:** Implementing "Performance Editions" or allowing user-specific tuning requires centralized configuration management.
* **Step-by-step Actions:**
  1. **Create Dataclasses:** Write `src/core/config.py` with frozen `AudioConfig`, `ScribeConfig`, and `LumiConfig` classes holding default values.
  2. **Implement YAML loader:** Write a `load_config` function to override defaults from an external `config.yaml`.
  3. **Auto-detect hardware:** Add a utility to detect system VRAM to dynamically select Light, Standard, or Pro editions at startup.

## 6. No Orchestrator or Event Bus
* **Context:** `main.py` currently acts as a 42-line god script manually wiring the `Ears` and `Scribe` components.
* **Why it matters:** Adding LLMs, TTS, and ZeroMQ without an orchestrator will result in tightly-coupled spaghetti code that's hard to scale and debug.
* **Step-by-step Actions:**
  1. **Create `Orchestrator` class:** Build `src/core/orchestrator.py` to initialize and own the lifecycle of all internal modules.
  2. **Build the Main Loop:** Implement `orchestrator.run()` to consume events and execute the system's logic path.
  3. **Refactor Entry Point:** Shrink `main.py` into a thin wrapper that initializes config and starts the `Orchestrator`.

## 7. No Explicit State Machine
* **Context:** Ad-hoc boolean flags tracking whether Lumi is listening or processing are scattered everywhere.
* **Why it matters:** Future visual avatars (Godot overlay) and complex interrupt logic rely on exact, unambiguous system states.
* **Step-by-step Actions:**
  1. **Define States:** Create `src/core/state_machine.py` defining an enum of `IDLE`, `LISTENING`, `PROCESSING`, and `SPEAKING`.
  2. **Enforce transitions:** Build a `StateMachine` class that uses a dictionary to strictly enforce valid state transitions.
  3. **Hook into Orchestrator:** Tie state changes directly into the orchestrator to publish external IPC events on every transition.

## 8. No VRAM Resource Manager
* **Context:** The core idea of offloading LLMs to system RAM to keep VRAM free for gaming is completely unhandled right now.
* **Why it matters:** Without VRAM management, Lumi will randomly spike VRAM usage, causing lag during heavy GPU tasks and violating the "Zero Cost" premise.
* **Step-by-step Actions:**
  1. **Create `ModelLoader`:** Build `src/llm/model_loader.py` wrapping `llama_cpp.Llama`.
  2. **Implement Wake/Hibernate:** Add `wake()` to load to VRAM during the `PROCESSING` state, and `hibernate()` to garbage-collect and unload upon returning to `IDLE`.
  3. **Add limits:** Incorporate config logic limiting the `n_gpu_layers` based on the autodetected VRAM budget.

## 9. No IPC Contract
* **Context:** The planned ZeroMQ integration with the Godot frontend lacks a formal schema.
* **Why it matters:** Building a separate Body component is difficult if the JSON payload formats and event types change unexpectedly during Brain development. Defining a partial contract requires breaking schema changes later.
* **Step-by-step Actions:**
  1. **Define Full Schema upfront:** Add a `ZMQMessage` dataclass in `src/core/events.py` enforcing `{event, payload, timestamp, version}` keys.
  2. **List all event types:** Document and standardize all 8 events immediately: `state_change`, `transcript`, `tts_start`, `tts_viseme`, `tts_stop`, `error`, `interrupt` (Body to Brain), and `user_text` (Body to Brain).
  3. **Internal Routing first:** Use this exact schema for internal orchestrator routing before ZeroMQ is even wired up. Dispatch `ZMQMessage`-shaped dicts to internal handlers to validate the contract.
  4. **Implement Server:** Create `src/interface/zmq_server.py` as a publish/subscribe wrapper over the IPC contract.

## 10. Naming Divergence Between Design and Code
* **Context:** Several components described in `ARCHITECTURE.md` (like `audio/listener.py`) don't map to the physical file tree (like `audio/ears.py`).
* **Why it matters:** Discrepancies between documentation and actual implementation cause confusion for future maintenance.
* **Step-by-step Actions:**
  1. **Align imports:** Ensure any future internal imports match the exact layout provided in the corrected architecture tree.
  2. **File setup:** Ensure new modules (e.g., `orchestrator.py`, `state_machine.py`) follow the updated paths mapped in the revised docs.

## 11. Zero Test Coverage
* **Context:** The system has no unit tests. Validating complex audio streaming timeouts and state transitions requires speaking into a microphone manually.
* **Why it matters:** As components become more heavily threaded and event-driven, lack of tests practically guarantees silent regression bugs. Specifically, the new Orchestrator and ModelLoader are critical to the system's stability and "Zero Cost" VRAM claims.
* **Step-by-step Actions:**
  1. **Introduce `pytest` with coverage targets:** Setup testing infrastructure with the `pytest` and `pytest-cov` development dependencies. Set a hard coverage target of 80% (e.g., `--cov-fail-under=80`) to prevent coverage erosion before Phase 3 ships.
  2. **Create mock audio:** Write fixtures in `tests/conftest.py` that yield synthetic sine waves and arrays of zeros for testing VAD.
  3. **Write Priority tests (Core and Audio):** Test `Scribe.transcribe()`, VAD silence timeouts in `Ears`, and all path branches in `StateMachine`.
  4. **Write Priority tests (VRAM and Events):** Write tests for `ModelLoader` lifecycle (`hibernate` releasing VRAM, errors on `generate` without `wake`) and `Orchestrator` event routing (verifying correct handlers trigger and `InterruptEvent` returns state to `IDLE`).

## 12. No Structured Logging
* **Context:** Debugging currently relies on raw `print()` statements scattered across the project.
* **Why it matters:** It's impossible to isolate errors or suppress noisy logs without structured log levels.
* **Step-by-step Actions:**
  1. **Setup global logging:** Add `src/core/logging_config.py` that configures Python's built-in `logging` module.
  2. **Standardize levels:** Migrate debugging arrays to `DEBUG`, and major state updates to `INFO`.
  3. **Module loggers:** Initialize `logging.getLogger(__name__)` at the top of each file.

## 13. `play_ready_sound()` Blocks the Audio Thread
* **Context:** Generating the startup sound utilizes a synchronous `sd.wait()` call, blocking the pipeline for 200ms.
* **Why it matters:** It introduces noticeable latency when trying to execute rapid commands.
* **Step-by-step Actions:**
  1. **Create Output Queue:** Create a dedicated audio output thread with its own playback queue.
  2. **Post sound events:** Refactor `play_ready_sound()` to post a `"ready"` audio playback event instead of executing directly.
  3. **Scale for Phase 4:** Expand this thread to act as the eventual handler for TTS streaming.

## 14. No Graceful Error Recovery
* **Context:** Failing to load the Wake Word ONNX model or lacking a microphone will result in an immediate fatal crash.
* **Why it matters:** A desktop assistant should gracefully notify the user about what's broken instead of abruptly failing and disappearing.
* **Step-by-step Actions:**
  1. **Create validation pipeline:** Implement `src/core/startup_check.py` to verify config paths, ONNX dependencies, and microphone statuses before the event loop starts.
  2. **Safe try-catches:** Wrap internal state cycles in try-except blocks, falling back into the `IDLE` state upon encountering localized errors (like brief VAD dropouts).
