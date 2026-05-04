# Lumi Codemaps Index

**Last Updated:** 2026-05-04 (Ring 2 closure)

This directory contains architectural maps of Project Lumi's codebase, organized by feature area.

## Maps by Topic

### Core Infrastructure

- **[training_pipeline.md](training_pipeline.md)** — QLoRA fine-tuning, GGUF export, persona evaluation
  - Ring 2 focus: Fixed gibberish from text→prompt/completion dataset format, training format sync, TRL 1.0 config, critical merge pipeline fixes
  - Key files: `scripts/train_lumi.py`, `scripts/merge_and_quantize.py`, `scripts/eval_persona.py`
  - Status: Complete & tested (eval evidence: 75% on regex criteria)

### Coming Soon

- **frontend.md** — Tauri/React UI, WebSocket client, state management
- **backend.md** — Brain (Python) audio pipeline, IPC, orchestrator
- **database.md** — SQLite schema, RAG embeddings (sqlite-vec)
- **integrations.md** — External tools (web search, calculator, file ops), OS actions
- **workers.md** — Background jobs, event loop, async patterns

## Quick Navigation

**By file modification date (most recent first):**

1. **2026-05-04:** training_pipeline.md (Ring 2 closure — major fixes)

**By development phase:**

- **Ring 2:** training_pipeline.md (persona LoRA v1, GGUF pipeline)
- **Ring 1:** (Frontend sidecar, cross-platform tools, PTT)
- **Phase 9:** (Godot UI overhaul, overlay design)
- **Phase 7:** (LightRAG custom RAG)
- **Phase 5:** (IPC transport, WebSocket)

## Using These Maps

Each codemap includes:
- **Architecture diagram** — Component relationships
- **Key modules table** — Exports, dependencies, entry points
- **External dependencies** — Versions, purposes
- **CLI interface** — Usage examples (reproducible commands from git history)
- **Data flow** — How information moves through the system
- **Known issues** — Workarounds for environment-specific bugs
- **Future work** — Backlog items for next phase

## Regeneration

To regenerate codemaps after major changes:

```bash
# (Future: automated generation script)
# For now, codemaps are manually maintained alongside code changes.
# Update them when:
#   - New major modules added/removed
#   - CLI interface changes
#   - Architecture decisions made
#   - Critical bugs/workarounds discovered
```

## Project Context

- **Two-process architecture:** Brain (Python) + Body (Tauri/React)
- **IPC transport:** WebSocket (4-byte length-prefix JSON framing)
- **LLM engine:** Phi-3.5-mini fine-tuned with LoRA, quantized to GGUF, served via llama-cpp-python
- **Current phase:** Ring 2 complete (persona LoRA v1, GGUF pipeline debugged)
- **Test coverage:** 80%+ gate in CI (912 passed, 7 skipped)

## Related Documentation

- **[CLAUDE.md](../../CLAUDE.md)** — Project core context, constraints, key paths
- **[MVP_REPORT.md](../../MVP_REPORT.md)** — Ring 2 backlog, next steps (Ring 3+)
- **[lora_api_probe.md](../lora_api_probe.md)** — Technical probe of LoRA API changes (historical)
