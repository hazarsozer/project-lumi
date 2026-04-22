# LoRA Adapter API Probe (Wave H1)

**Date:** 2026-04-22
**Probe scope:** Determine whether the installed `llama-cpp-python` exposes a
usable LoRA adapter API so Wave H6 can choose between hot-swap adapters and the
`ModelRegistry` (reload-with-`lora_path`) pattern.

## Environment

| Item | Value |
|------|-------|
| `llama-cpp-python` version | **0.3.20** |
| Upstream `llama.cpp` LoRA API | `llama_adapter_lora_*` family (post-renamed from older `llama_lora_adapter_*`) |

## 1. Constructor-level LoRA support (high-level Python API)

`llama_cpp.Llama.__init__` accepts three LoRA-related keyword arguments:

```python
Llama(
    model_path=...,
    lora_base: Optional[str] = None,   # base model (useful when main is quantized)
    lora_scale: float = 1.0,           # adapter strength, 0.0–1.0+
    lora_path: Optional[str] = None,   # path to .gguf LoRA adapter file
    ...
)
```

Internally (verified via `inspect.getsource(Llama.__init__)`), when `lora_path`
is set the constructor:

1. Disables `use_mmap` (line 188): `use_mmap if lora_path is None else False`.
2. Initialises the adapter: `self._lora_adapter = llama_cpp.llama_adapter_lora_init(model, lora_path.encode("utf-8"))`.
3. Registers a cleanup callback: `llama_adapter_lora_free(self._lora_adapter)` on context teardown.
4. Attaches the adapter to the context: `llama_cpp.llama_set_adapters_lora(ctx, adapters, 1, scales)`.

This means **loading a model with a LoRA adapter is fully supported via the
high-level constructor** — no ctypes glue required.

## 2. Hot-swap LoRA API (low-level C bindings)

The following symbols are exposed at the `llama_cpp` module level (not on the
`Llama` class — `dir(Llama)` returned `[]` for `lora`):

| Symbol | Signature | Purpose |
|--------|-----------|---------|
| `llama_adapter_lora_init(model, path_lora: bytes) -> Optional[llama_adapter_lora_p]` | Loads a LoRA adapter file into memory, bound to a model. | Allocate adapter |
| `llama_adapter_lora_free(adapter)` | Releases the adapter. | Free adapter |
| `llama_set_adapter_lora(ctx, adapter, scale: float) -> int` | Attaches a single adapter to a context with a given scale. | Activate one |
| `llama_set_adapters_lora(ctx, adapters, n_adapters: int, scales) -> int` | Batch-sets multiple adapters. "Set LoRA adapters on the context if they differ from the current adapters." | Activate many |
| `llama_adapter_get_alora_invocation_tokens` | aLoRA invocation-token accessor | aLoRA support |
| `llama_adapter_get_alora_n_invocation_tokens` | aLoRA token count | aLoRA support |
| `llama_adapter_lora_p` / `llama_adapter_lora_p_ctypes` | opaque pointer types | FFI |

**Hot-swap is possible** via:

```python
# Load adapter once
adapter = llama_cpp.llama_adapter_lora_init(loaded_model.model, b"/path/to/lora.gguf")

# Attach to the live context at runtime
rc = llama_cpp.llama_set_adapter_lora(loaded_model._ctx.ctx, adapter, 1.0)

# Detach by passing NULL / empty adapter array via llama_set_adapters_lora
# (or call llama_adapter_lora_free when done)
```

However, this requires reaching into `_ctx.ctx` (a private attribute of the
high-level `Llama` wrapper), which is fragile across `llama-cpp-python`
versions.

## 3. Architectural options for Wave H6

### Option A — Hot-swap adapters (low-level C API)

- **Pros**
  - No model reload: VRAM budget unchanged; one-time base-model load cost.
  - Adapter switch latency in the milliseconds range (adapter files are small).
  - Multiple adapters can be layered via `llama_set_adapters_lora`.
- **Cons**
  - Requires ctypes-level code against `_ctx.ctx`; brittle to
    `llama-cpp-python` internal refactors (no public Python wrapper).
  - No official API to *remove* a currently attached adapter cleanly; detach
    semantics rely on `llama_set_adapters_lora(ctx, NULL, 0, NULL)` pattern
    (not yet verified in 0.3.20).
  - Requires careful lifecycle management — adapter pointers must be freed
    before the base model is released.
  - Violates Lumi's "Zero Cost VRAM" boundary only mildly (base model still
    hibernates), but adapter pointers must be freed in `ModelLoader.unload()`.

### Option B — `ModelRegistry` pattern (reload with `lora_path`)

- **Pros**
  - Uses only the stable high-level `Llama(..., lora_path=...)` constructor.
  - Clean integration with existing `ModelLoader.load()` / `unload()` — just
    thread `lora_path` through `LLMConfig`.
  - Registry can cache adapter *paths* (not loaded instances), keyed by
    persona/task, and hand them to `load()`.
  - Safer: no private-attribute access, no ctypes.
- **Cons**
  - Switching adapters requires a full model reload (~seconds for 2B models,
    longer for 8B). Poor fit for per-turn adapter switching.
  - `use_mmap` is forced off when `lora_path` is set (line 188), so the base
    model re-reads from disk on each swap — noticeably slower than a pure
    reload of the same base.

## 4. Recommendation for Wave H6

**Use the `ModelRegistry` pattern (Option B).**

Rationale:

1. **Stability over speed.** Lumi's "Zero Cost" VRAM lifecycle already accepts
   a cold-start cost on every PROCESSING transition. Adding ~1–3 s of
   additional load time when the persona/task adapter changes is acceptable,
   and falls well within the user-perceived latency budget already established
   by the model wake cycle.
2. **Adapter switches are infrequent.** Persona/task LoRA changes happen at
   session boundaries or major context switches, not per token or per turn. The
   hot-swap API's main selling point (sub-ms switching) is wasted here.
3. **`llama-cpp-python` internal API drift is a known risk.** Wave I1 already
   had to guard `cache_type_k` / `cache_type_v` behind a `TypeError` fallback
   because upstream PR #21089 hasn't shipped in all builds. Depending on
   `_ctx.ctx` for hot-swap would expose us to the same class of regression
   with no public fallback.
4. **Integration surface is smaller.** `LLMConfig` already has `model_path`;
   adding `lora_path: Optional[str] = None` and `lora_scale: float = 1.0`
   plus a `ModelRegistry` that resolves `(persona, task) -> lora_path` is a
   clean extension. No new ctypes lifecycle to manage.

### Suggested Wave H6 API sketch

```python
# src/llm/model_registry.py
@dataclass(frozen=True)
class AdapterSpec:
    persona: str
    task: str | None
    lora_path: str
    lora_scale: float = 1.0

class ModelRegistry:
    def resolve(self, persona: str, task: str | None) -> AdapterSpec | None: ...

# src/core/config.py  (extend LLMConfig)
@dataclass(frozen=True)
class LLMConfig:
    ...
    lora_path: str | None = None
    lora_scale: float = 1.0

# src/llm/model_loader.py  (extend load())
kwargs["lora_path"] = config.lora_path
kwargs["lora_scale"] = config.lora_scale
```

If, during Wave H6 implementation, profiling shows that adapter-switch latency
dominates the PROCESSING budget, we can revisit Option A as an optimisation —
but only after the `ModelRegistry` path is proven end-to-end.

## 5. References

- `llama-cpp-python` 0.3.20 source: `llama_cpp/llama.py`, `Llama.__init__`
  lines 42–44 (signature), 124–125 (docstring), 188 (mmap guard),
  362–388 (adapter lifecycle).
- Upstream `llama.cpp` header rename: `llama_lora_adapter_*` →
  `llama_adapter_lora_*` (completed prior to llama-cpp-python 0.3.x).
