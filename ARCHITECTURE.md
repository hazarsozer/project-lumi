# NenOS: Architecture & Roadmap (v1.0)

**Project Goal:** *"Siri on Steroids"* — A local, privacy-first Desktop Assistant
**Core Philosophy:** *"Architect First"* — Logic before code. Zero Cost. Local Only.

---

## 1. System Architecture: "The Split-Brain"

To ensure the desktop remains responsive (especially during gaming), NenOS is decoupled into two independent processes communicating via ZeroMQ (ZMQ).

### The Body (Frontend)

* **Role:** Visual Avatar, UI Overlay, Animations
* **Tech:** Godot Engine (HTML5 export via WebSockets) OR Tauri (Webview)
* **State:** Always "Alive" but lightweight
* **Resource Target:** < 200MB RAM, Negligible GPU

### The Brain (Backend)

* **Role:** Intelligence, Orchestration, OS Control
* **Tech:** Python 3.12+ (Managed via uv)
* **State:** Hibernate & Wake

  * **Idle:** LLM offloaded to System RAM. Listening for Wake Word (CPU)
  * **Active:** LLM loaded to VRAM. Full processing power

### The Nerves (IPC)

* **Protocol:** ZeroMQ (PAIR socket)
* **Format:** JSON Schema (`event`, `payload`, `timestamp`)

---

## 2. Performance Editions (Scalability)

The system detects available hardware at startup and selects the appropriate "Edition."

| Feature         | NenOS Light (Low-End / Laptop)       | NenOS Standard (Gamer / Current Target) | NenOS Pro (Workstation)      |
| --------------- | ------------------------------------ | --------------------------------------- | ---------------------------- |
| **VRAM Budget** | < 2 GB                               | < 4 GB (Dynamic Offloading)             | 8 GB+ (Always Loaded)        |
| **LLM Model**   | Qwen-1.5 (1.8B) or Phi-3 Mini (Int4) | Phi-3.5 Mini or Gemma 2 (2B)            | Llama-3 (8B) or Gemma 2 (9B) |
| **TTS Engine**  | System TTS / Piper (Low Quality)     | Kokoro ONNX (High Quality)              | StyleTTS2 (Studio Quality)   |
| **Vision**      | Disabled                             | On-Demand (Screenshots)                 | Real-Time (Camera + Screen)  |
| **Avatar**      | Static / 2 Frame Animation           | Live2D (Standard)                       | 3D VRM (Full Motion)         |

---

## 3. Technology Stack (Standard Edition)

### Core

* **Language:** Python 3.12
* **Package Manager:** uv
* **Orchestration:** Custom MainLoop with:

  * **"Reflex"** (Regex-based command routing)
  * **"Reasoning"** (LLM-based processing)

---

### Audio Input (The Ears)

* **VAD:** Silero VAD v5
* **Wake Word:** openWakeWord (Custom Model)
* **STT:** faster-whisper (int8 quantized)

---

### Audio Output (The Mouth)

* **TTS:** Kokoro-82M (ONNX)
* **Playback:** sounddevice / miniaudio

---

### Inference (The Brain)

* **Engine:** llama-cpp-python (GGUF support)
* **Context:** Rolling window (Last 10 turns)

---

## 4. Development Roadmap

### Phase 1: The Foundation (Current)

* [x] Project Initialization (`uv init`)
* [ ] The Ears: Implement Wake Word → VAD → STT pipeline on CPU
* [ ] The Mouth: Implement Kokoro TTS playback
* [ ] The Spine: Setup ZMQ communication test between Python and a dummy Client

---

### Phase 2: The Brain (Intelligence)

* [ ] Implement "Reflex" routing (Regex commands for OS control)
* [ ] Implement "Hibernate" logic (Model loading/unloading from VRAM)
* [ ] Connect LLM (Phi-3.5) for chat

---

### Phase 3: The Body (Visuals)

* [ ] Setup Live2D viewer (Godot/Tauri)
* [ ] Sync Lip-sync (Visemes) from TTS to Avatar
* [ ] Implement "Idle" vs "Listening" animations

---

### Phase 4: The Hands (OS Control)

* [ ] Build "Vision" tool (Screenshot analysis)
* [ ] Build "Automation" tools (App launch, File management)
* [ ] v1.0 Release

---

## 5. Directory Structure

```plaintext
NenOS/
├── .venv/                 # Managed by uv
├── models/                # GGUF / ONNX models (Ignored)
├── src/
│   ├── audio/             # listener.py, speaker.py
│   ├── core/              # main.py, config.py, orchestrator.py
│   ├── llm/               # model_loader.py, prompt_engine.py
│   ├── tools/             # os_actions.py, vision.py
│   └── interface/         # ZMQ server code
├── ui/                    # Godot/Tauri project source
├── config.yaml            # User settings (Edition selection)
└── README.md
```
