"""HuggingFace Phi3ForCausalLM wrapper with persona-vector steering.

Implements the same ``create_completion()`` interface as ``llama_cpp.Llama``
so ``reasoning_router.py`` needs zero changes when persona_steering_backend="hf_hooks".

Steering mechanism:
    On every forward pass, for each layer i in ``hooked_layers``:
        h = h - alpha * phi_prior_direction[i]
    where ``phi_prior_direction`` is a per-layer unit vector pointing from the
    Lumi-voice residual-stream mean toward the Phi-prior residual-stream mean.
    Subtracting it pushes the residual stream away from the Phi prior.

Inference strategy:
    Because llama_cpp exposes a stateful token-by-token API (create_completion
    called with max_tokens=1 in a loop against the same static prompt), this
    wrapper uses lazy batch generation: the first call for a given prompt runs
    the full HF generate() to produce a token buffer; subsequent calls drain
    it one token at a time.  This is O(n) instead of O(n²).

Usage (via model_loader.py — not instantiated directly):
    loader = ModelLoader()
    loader.load(config)        # returns HFSteeredModel when steering enabled
    model = loader.model       # HFSteeredModel instance
    chunk = model.create_completion(prompt, max_tokens=1, temperature=0.5, ...)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class HFSteeredModel:
    """Phi3ForCausalLM with per-layer residual-stream persona-vector steering."""

    def __init__(
        self,
        hf_model_path: str | Path,
        vector_path: str | Path,
        hooked_layers: tuple[int, ...] | list[int],
        alpha: float,
        device: str = "cuda",
    ) -> None:
        from transformers import AutoTokenizer, Phi3Config, Phi3ForCausalLM

        hf_model_path = Path(hf_model_path)
        vector_path = Path(vector_path)

        logger.info("HFSteeredModel: loading tokenizer from %s", hf_model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(str(hf_model_path), trust_remote_code=True)

        logger.info("HFSteeredModel: loading Phi3ForCausalLM fp16 from %s", hf_model_path)
        config = Phi3Config.from_pretrained(str(hf_model_path))
        self._model = Phi3ForCausalLM.from_pretrained(
            str(hf_model_path),
            config=config,
            torch_dtype=torch.float16,
            device_map=device,
            ignore_mismatched_sizes=False,
        )
        self._model.eval()

        self._device = torch.device(device)
        self._alpha = alpha
        self._hooked_layers = list(hooked_layers)

        logger.info("HFSteeredModel: loading direction vectors from %s", vector_path)
        payload = torch.load(str(vector_path), map_location="cpu", weights_only=False)
        self._directions: torch.Tensor = payload["directions"]  # (n_layers, hidden_size)

        self._hooks: list[Any] = []
        self._register_hooks()

        # Lazy generation buffer
        self._current_prompt: str | None = None
        self._token_iter: Iterator[str] | None = None
        self._finished: bool = False

        logger.info(
            "HFSteeredModel ready: layers=%s alpha=%.1f device=%s",
            self._hooked_layers, self._alpha, device,
        )

    def _register_hooks(self) -> None:
        def make_hook(layer_idx: int) -> Any:
            dir_vec = self._directions[layer_idx].to(device=self._device, dtype=torch.float16)

            def hook(module: Any, input: Any, output: Any) -> Any:
                hs = output[0] if isinstance(output, tuple) else output
                hs = hs - self._alpha * dir_vec
                return (hs,) + output[1:] if isinstance(output, tuple) else hs
            return hook

        for layer_idx in self._hooked_layers:
            h = self._model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            self._hooks.append(h)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def create_completion(
        self,
        prompt: str,
        max_tokens: int = 1,
        temperature: float = 0.5,
        top_p: float = 0.9,
        top_k: int = 30,
        min_p: float = 0.05,
        repeat_penalty: float = 1.05,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate up to max_tokens tokens for prompt.

        First call for a given prompt triggers full batch generation.
        Subsequent calls with the same prompt drain the token buffer.

        Non-streaming callers (max_tokens > 1) use max_tokens as the
        generation budget.  Streaming callers (max_tokens=1) generate a
        512-token batch so the full response can be drained token-by-token.
        """
        if kwargs:
            logger.debug(
                "HFSteeredModel.create_completion: ignoring unknown kwargs %s",
                sorted(kwargs),
            )

        if prompt != self._current_prompt:
            self._current_prompt = prompt
            # Streaming callers pass max_tokens=1 per call; use 512 so the full
            # response is generated once and drained token by token.
            batch_size = max_tokens if max_tokens > 1 else 512
            self._token_iter = self._generate_tokens(
                prompt=prompt,
                max_new_tokens=batch_size,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repeat_penalty=repeat_penalty,
            )
            self._finished = False

        if self._finished or self._token_iter is None:
            return {"choices": [{"text": "", "finish_reason": "stop"}]}

        collected: list[str] = []
        for _i in range(max_tokens):
            try:
                tok = next(self._token_iter)
                collected.append(tok)
            except StopIteration:
                self._finished = True
                break

        text = "".join(collected)
        finish_reason: str | None = "stop" if self._finished else None
        return {"choices": [{"text": text, "finish_reason": finish_reason}]}

    # A4: expose the same callable interface as llama_cpp.Llama so callers can
    # use model(prompt, max_tokens=128, ...) without special-casing the backend.
    __call__ = create_completion

    def _generate_tokens(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repeat_penalty: float,
    ) -> Iterator[str]:
        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(self._device)

        do_sample = temperature > 0.0
        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
            "repetition_penalty": repeat_penalty,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
            generate_kwargs["top_k"] = top_k
            generate_kwargs["min_p"] = min_p  # transformers >= 4.42
        else:
            generate_kwargs["temperature"] = None
            generate_kwargs["top_p"] = None

        with torch.inference_mode():
            output_ids = self._model.generate(**generate_kwargs)

        new_ids = output_ids[0, input_ids.shape[1]:]
        for token_id in new_ids:
            token_id_int = int(token_id)
            if token_id_int == self._tokenizer.eos_token_id:
                return
            text = self._tokenizer.decode([token_id_int], skip_special_tokens=True)
            yield text
