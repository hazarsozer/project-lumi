# Project Lumi (ルミ)

> **"The Ghost in the Machine."**
>
> Lumi is a local, privacy-first desktop companion designed to be invisible until needed. It is built with a "Zero-Cost" philosophy, running on consumer hardware without interfering with high-performance tasks like gaming or rendering.

**For a deep dive into the technical design, see [ARCHITECTURE.md](ARCHITECTURE.md).**

---

## 🚧 Current Status

**Active Development (Phase 1: The Ears)**

* **Current Blocker:** Low confidence scores (0.3 - 0.5) on wake word detection due to generic model mismatch with specific microphone hardware/drivers on Linux.
* **Next Step:** Training a custom wake word model ("Hey Lumi") to overfit on the user's voice and resolve the confidence threshold issue.

---

## 🗺️ Detailed Roadmap

### Phase 0: Foundation & Architecture

- [x] **Project Initialization** (Python 3.12, `uv` package manager).
- [x] **Hardware Budgeting** (Defined <4GB VRAM constraint for Standard).
- [x] **Architecture Design** ("Split-Brain" Strategy: Python Backend + Godot Frontend).
- [x] **IPC Protocol** (Designed ZeroMQ JSON Schema for Brain-Body communication).

### Phase 1: The Ears (Audio Input)

*Goal: A low-latency, non-blocking listening system that runs on CPU.*

- [x] **Audio Driver Integration** (`sounddevice` + `libportaudio2`).
- [x] **Threaded Listener** (Producer-Consumer pattern with `queue`).
- [x] **Latency Tuning** (Implemented `latency='high'` to fix buffer underruns). ⚠️ **Unstable**
- [x] **Logic: Double Trigger Fix** (Implemented "Flush & Ignore" cooldown logic). ⚠️ **Needs Calibration**
- [x] **Wake Word Accuracy** 🔴 **Critical Issue**
    - [x] Train custom "Hey Lumi" model (`openWakeWord`).
    - [x] Verify >0.8 confidence score.
- [ ] **Voice Activity Detection (VAD)**
    - [ ] Integrate `silero-vad` to detect end-of-speech (Smart Stop).
    - [ ] Calibrate silence threshold for background noise.

### Phase 2: The Scribe (Transcription)

*Goal: Converting speech to text without touching the GPU.*

- [x] **Model Selection** (`faster-whisper` int8 quantized).
- [ ] **Context Injection** (Fix "Fire folks" vs "Firefox" using initial prompts).
- [ ] **Command Parsing** (Regex/Keyword matching for instant actions).

### Phase 3: The Brain (Intelligence)

*Goal: Smart decision making using Local LLMs.*

- [ ] **Model Loading Strategy** ("Hibernate & Wake" logic to swap RAM/VRAM).
- [ ] **LLM Integration** (Phi-3.5 or Gemma-2).
- [ ] **Orchestrator** (Router for "Chat" vs "Command").
- [ ] **Memory System** (JSON-based user profile and conversation history).

### Phase 4: The Mouth (TTS)

*Goal: High-quality voice response.*

- [ ] **TTS Engine** (Integrate `kokoro-onnx` for local synthesis).
- [ ] **Audio Output** (Non-blocking playback).
- [ ] **Viseme Generation** (Extract mouth shapes for avatar lip-sync).

### Phase 5: The Body (Visuals)

*Goal: A transparent, interactive overlay.*

- [ ] **Godot Project Setup** (Transparent window settings for Linux X11/Wayland).
- [ ] **ZeroMQ Client** (Godot script to receive Python states).
- [ ] **Avatar Rendering** (Live2D or 2D Sprite animation system).
- [ ] **State Machine** (Idle / Listening / Thinking / Speaking animations).

---

## ⚠️ Pre-Alpha Notice

This repository is currently a **work-in-progress log**. The code is experimental and broken in several places as we solve low-level Linux audio driver issues. Cloning and running this right now will likely result in crashes or silence.
