from __future__ import annotations

import pytest
import torch

from dynamode.model.adapters import make_spectral_batch_adapter
from dynamode.model.frequency_bands import default_band_edges, parse_band_edges
from dynamode.model.wrapper import (
    MODEL_REGISTRY,
    SpectralConvBlockMixAmplitudeConfig,
    SpectralDiTConfig,
    UnifiedWrapper,
)
from dynamode.spectral.adapters import DCT, denormalize_adaptive, normalize_adaptive
from dynamode.utils import validate_batch


def _spectral_batch(batch_size: int = 2, length: int = 5, top_k: int = 4) -> dict:
    return {
        "x": torch.randn(batch_size, length, top_k * 3),
        "t": torch.zeros(batch_size, dtype=torch.long),
        "temp": torch.full((batch_size,), 0.5, dtype=torch.float32),
        "native_coords": torch.randn(batch_size, length, 3),
        "mask": torch.ones(batch_size, length, dtype=torch.bool),
        "win_pos": torch.zeros(batch_size, dtype=torch.float32),
        "cond_drop_mask": torch.zeros(batch_size, dtype=torch.bool),
    }


def test_supported_registry_is_public_surface_only():
    assert sorted(MODEL_REGISTRY) == [
        "spectral_conv_block_mix_amplitude",
        "spectral_dit_low_k",
    ]


def test_spectral_batch_adapter_accepts_current_batch_shape():
    adapter = make_spectral_batch_adapter(top_k_freqs=4, in_channels=3, cond_channels=3)
    adapted = adapter(_spectral_batch(top_k=4))

    assert adapted["x"].shape == (2, 5, 12)
    assert adapted["native_coords"].shape == (2, 5, 3)
    assert adapted["mask"].dtype == torch.bool
    assert adapted["return_aux"] is False


def test_spectral_batch_adapter_rejects_bad_x_shape():
    adapter = make_spectral_batch_adapter(top_k_freqs=4, in_channels=3, cond_channels=3)
    batch = _spectral_batch(top_k=4)
    batch["x"] = batch["x"][..., :-1]

    with pytest.raises(ValueError, match="last dim"):
        adapter(batch)


def test_validate_batch_reports_missing_keys():
    with pytest.raises(KeyError, match="Missing batch keys"):
        validate_batch({"x": torch.zeros(1)}, ("x", "t"))


@pytest.mark.parametrize(
    ("spec", "top_k", "expected"),
    [
        (None, 64, (0, 1, 9, 33, 64)),
        ("block_mix", 64, (0, 1, 9, 33, 64)),
        ("DC,1-8,9+", 16, (0, 1, 9, 16)),
        ((0, 1, 9, 16), 16, (0, 1, 9, 16)),
    ],
)
def test_frequency_band_parser(spec, top_k, expected):
    assert parse_band_edges(spec, top_k) == expected


def test_default_frequency_bands_keep_dc_separate():
    assert default_band_edges(256) == (0, 1, 9, 33, 129, 256)


def test_dct_roundtrip_reconstruction():
    x_time = torch.randn(2, 8, 5, 3)
    x_spec = DCT.time_to_spectral(x_time, top_k=None)
    x_recon = DCT.spectral_to_time(x_spec, n_time_steps=8, n_channels=3)

    assert torch.allclose(x_time, x_recon, atol=5e-5, rtol=5e-5)


def test_spectral_normalization_roundtrip():
    x_spec = torch.randn(2, 5, 12)
    scales = torch.linspace(1.0, 2.0, 12)

    x_norm = normalize_adaptive(x_spec, scales)
    x_back = denormalize_adaptive(x_norm, scales)

    assert torch.allclose(x_spec, x_back, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("model_name", "config"),
    [
        (
            "spectral_dit_low_k",
            SpectralDiTConfig(
                top_k_freqs=4,
                in_channels=3,
                cond_channels=3,
                freq_hidden_size=2,
                depth=1,
                num_heads=1,
                cond_dim=16,
                cfg_dropout=False,
                prediction_target="x_0",
            ),
        ),
        (
            "spectral_conv_block_mix_amplitude",
            SpectralConvBlockMixAmplitudeConfig(
                top_k_freqs=4,
                in_channels=3,
                cond_channels=3,
                freq_hidden_size=2,
                depth=1,
                num_heads=1,
                spectral_modes=4,
                cond_dim=16,
                cfg_dropout=False,
                prediction_target="x_0",
                amp_head_context_modes=2,
                amp_head_target_modes=1,
                amp_head_d_model=8,
                amp_head_depth=1,
                amp_head_num_heads=1,
            ),
        ),
    ],
)
def test_unified_wrapper_forward_for_supported_models(model_name, config):
    torch.manual_seed(0)
    wrapper = UnifiedWrapper(model_name=model_name, config=config)
    wrapper.eval()

    with torch.no_grad():
        out = wrapper(_spectral_batch(top_k=4))

    assert out.shape == (2, 5, 12)
    assert torch.isfinite(out).all()
