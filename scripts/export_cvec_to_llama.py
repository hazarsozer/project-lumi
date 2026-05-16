"""Convert a persona-vector .pt file to a llama.cpp control-vector binary.

llama_set_adapter_cvec expects a flat float32 buffer of shape
[n_layers_total * n_embd], where entry i*n_embd corresponds to transformer
layer i (0-indexed, full model depth).

Sign convention: the HF steering hook subtracts the direction vector
  (hs = hs - alpha * direction[layer]).
llama.cpp ADDS the buffer to the residual stream.
So the exported buffer must be negated: buf[layer] = -alpha * directions[layer].

Usage
-----
    uv run python scripts/export_cvec_to_llama.py \\
        --vector models/persona_vectors/phi_prior_v1.pt \\
        --layer-start 12 --layer-end 28 --alpha 8.0 \\
        --output models/persona_vectors/phi_prior_v1_alpha8_l12-28.cvec.bin
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def export(
    vector_path: Path,
    output_path: Path,
    layer_start: int,
    layer_end: int,
    alpha: float,
) -> dict:
    """Export direction vectors to a llama.cpp cvec binary buffer.

    Returns the sidecar metadata dict.
    """
    import numpy as np
    import torch

    payload = torch.load(str(vector_path), map_location="cpu", weights_only=False)
    directions: torch.Tensor = payload["directions"]  # (n_layers, hidden_size)
    n_layers_total, n_embd = directions.shape
    assert directions.dtype == torch.float32, f"Expected float32, got {directions.dtype}"

    if layer_start < 0 or layer_end >= n_layers_total or layer_start > layer_end:
        raise ValueError(
            f"Invalid layer range [{layer_start}, {layer_end}] for model with "
            f"{n_layers_total} layers."
        )

    # Allocate zero buffer: layers outside [layer_start, layer_end] are zero
    # and have no effect on inference.
    buf = np.zeros((n_layers_total, n_embd), dtype=np.float32)
    for layer in range(layer_start, layer_end + 1):
        # Negate because HF hook subtracts but llama.cpp adds.
        buf[layer] = -alpha * directions[layer].numpy()

    flat = np.ascontiguousarray(buf.reshape(-1), dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flat.tofile(output_path)

    src_sha256 = hashlib.sha256(vector_path.read_bytes()).hexdigest()
    bin_sha256 = hashlib.sha256(flat.tobytes()).hexdigest()

    meta = {
        "source_vector": str(vector_path),
        "source_sha256": src_sha256,
        "alpha": alpha,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "n_layers_total": n_layers_total,
        "n_embd": n_embd,
        "buffer_elements": int(flat.size),
        "buffer_dtype": "float32",
        "sign_convention": "negated — HF hook subtracts; llama.cpp adds",
        "bin_sha256": bin_sha256,
    }
    sidecar = output_path.with_suffix(".json")  # replaces .bin → .json
    sidecar.write_text(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export persona vectors to a llama.cpp control-vector binary.",
    )
    parser.add_argument("--vector", type=Path, required=True,
                        help="Input .pt file (must contain 'directions' key).")
    parser.add_argument("--layer-start", type=int, default=12,
                        help="First layer to apply cvec (inclusive, 0-indexed).")
    parser.add_argument("--layer-end", type=int, default=28,
                        help="Last layer to apply cvec (inclusive, 0-indexed).")
    parser.add_argument("--alpha", type=float, default=8.0,
                        help="Steering coefficient (negated before export).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output .cvec.bin path.")
    args = parser.parse_args(argv)

    if not args.vector.exists():
        print(f"ERROR: {args.vector} not found", file=sys.stderr)
        return 1

    meta = export(args.vector, args.output, args.layer_start, args.layer_end, args.alpha)
    sidecar = args.output.with_suffix(".json")
    print(f"Exported {meta['buffer_elements']:,} floats → {args.output}")
    print(f"  layers  : [{args.layer_start}, {args.layer_end}] (inclusive)")
    print(f"  alpha   : {args.alpha}")
    print(f"  n_embd  : {meta['n_embd']}")
    print(f"  checksum: {meta['bin_sha256'][:16]}")
    print(f"  sidecar : {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
