"""Unit tests for scripts/export_cvec_to_llama.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="qlora extra not installed (uv sync --extra qlora)")
np = pytest.importorskip("numpy", reason="numpy not installed")

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.export_cvec_to_llama import export


N_LAYERS = 32
N_EMBD = 3072
ALPHA = 8.0
IL_START, IL_END = 12, 28


@pytest.fixture()
def fake_pt(tmp_path: Path) -> Path:
    """Minimal .pt file with a (32, 3072) directions tensor."""
    directions = torch.randn(N_LAYERS, N_EMBD, dtype=torch.float32)
    path = tmp_path / "test_vectors.pt"
    torch.save({"directions": directions, "meta": {}}, path)
    return path


@pytest.fixture()
def exported(tmp_path: Path, fake_pt: Path):
    """Run export() and return (meta, flat_buf, directions)."""
    out = tmp_path / "test.cvec.bin"
    payload = torch.load(str(fake_pt), map_location="cpu", weights_only=False)
    directions = payload["directions"].numpy()
    meta = export(fake_pt, out, IL_START, IL_END, ALPHA)
    flat = np.fromfile(out, dtype=np.float32)
    return meta, flat, directions


class TestBufferShape:
    def test_total_elements(self, exported) -> None:
        meta, flat, _ = exported
        assert flat.size == N_LAYERS * N_EMBD

    def test_dtype_float32(self, exported) -> None:
        _, flat, _ = exported
        assert flat.dtype == np.float32

    def test_contiguous(self, exported) -> None:
        _, flat, _ = exported
        assert flat.flags["C_CONTIGUOUS"]


class TestLayerValues:
    def test_layers_inside_range_negated_and_scaled(self, exported) -> None:
        meta, flat, directions = exported
        buf = flat.reshape(N_LAYERS, N_EMBD)
        for layer in range(IL_START, IL_END + 1):
            expected = -ALPHA * directions[layer]
            np.testing.assert_allclose(buf[layer], expected, rtol=1e-5)

    def test_layers_outside_range_are_zero(self, exported) -> None:
        meta, flat, _ = exported
        buf = flat.reshape(N_LAYERS, N_EMBD)
        for layer in list(range(0, IL_START)) + list(range(IL_END + 1, N_LAYERS)):
            assert np.all(buf[layer] == 0.0), f"Layer {layer} should be zero"

    def test_sign_negation(self, exported) -> None:
        """Buffer value sign must be opposite to alpha * direction value."""
        meta, flat, directions = exported
        buf = flat.reshape(N_LAYERS, N_EMBD)
        layer = IL_START
        dir_val = directions[layer, 0]
        buf_val = buf[layer, 0]
        # sign must be opposite (allow near-zero with tolerance)
        if abs(dir_val) > 1e-6:
            assert dir_val * buf_val < 0, (
                f"Expected opposite signs: dir={dir_val:.4f} buf={buf_val:.4f}"
            )


class TestSidecar:
    def test_sidecar_written(self, tmp_path: Path, fake_pt: Path) -> None:
        out = tmp_path / "test.cvec.bin"
        export(fake_pt, out, IL_START, IL_END, ALPHA)
        sidecar = out.with_suffix(".json")
        assert sidecar.exists()

    def test_sidecar_contents(self, tmp_path: Path, fake_pt: Path) -> None:
        out = tmp_path / "test.cvec.bin"
        meta = export(fake_pt, out, IL_START, IL_END, ALPHA)
        assert meta["n_layers_total"] == N_LAYERS
        assert meta["n_embd"] == N_EMBD
        assert meta["layer_start"] == IL_START
        assert meta["layer_end"] == IL_END
        assert meta["alpha"] == ALPHA
        assert meta["buffer_elements"] == N_LAYERS * N_EMBD
        assert len(meta["bin_sha256"]) == 64

    def test_sidecar_json_valid(self, tmp_path: Path, fake_pt: Path) -> None:
        out = tmp_path / "test.cvec.bin"
        export(fake_pt, out, IL_START, IL_END, ALPHA)
        sidecar = out.with_suffix(".json")
        data = json.loads(sidecar.read_text())
        assert "sign_convention" in data


class TestValidation:
    def test_invalid_layer_range_raises(self, tmp_path: Path, fake_pt: Path) -> None:
        out = tmp_path / "test.cvec.bin"
        with pytest.raises(ValueError, match="Invalid layer range"):
            export(fake_pt, out, layer_start=28, layer_end=12, alpha=ALPHA)

    def test_layer_end_out_of_bounds_raises(self, tmp_path: Path, fake_pt: Path) -> None:
        out = tmp_path / "test.cvec.bin"
        with pytest.raises(ValueError, match="Invalid layer range"):
            export(fake_pt, out, layer_start=0, layer_end=99, alpha=ALPHA)
