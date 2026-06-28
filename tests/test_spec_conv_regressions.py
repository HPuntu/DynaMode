from __future__ import annotations

import pytest
import torch

from dynamode.model.diffusion import SpectralDiffusion
from dynamode.model.spec_conv.hilbert import (
    HilbertSpatialEnvelope,
    HilbertSpatialEnvelopeDCT,
)
from dynamode.model.spec_conv.spectral_conv import (
    SpectralConvDiT,
)


@pytest.mark.parametrize(
    ("envelope_cls", "length"),
    [
        (HilbertSpatialEnvelope, 8),
        (HilbertSpatialEnvelope, 9),
        (HilbertSpatialEnvelopeDCT, 8),
        (HilbertSpatialEnvelopeDCT, 9),
    ],
)
def test_hilbert_envelope_constant_signal_is_identity(envelope_cls, length):
    x = torch.ones(2, 3, 4, length, dtype=torch.float32)
    env = envelope_cls._hilbert_envelope(x)
    assert torch.allclose(env, x.abs(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("prediction_target", "expect_same"),
    [
        ("v", True),
        ("x_0", False),
    ],
)
def test_rmsf_prior_gain_only_acts_directly_in_x0_space(prediction_target, expect_same):
    torch.manual_seed(0)

    model = SpectralConvDiT(
        top_k_freqs=2,
        in_channels=3,
        cond_channels=3,
        freq_hidden_size=2,
        depth=1,
        num_heads=1,
        spectral_modes=2,
        cfg_dropout=False,
        prediction_target=prediction_target,
        use_rmsf_prior_gain=True,
    )
    with torch.no_grad():
        model.final_proj.weight.fill_(0.25)
        model.final_proj.bias.zero_()
        model.rmsf_gate.fill_(1.0)

    B, L, D = 1, 3, 6
    x = torch.randn(B, L, D)
    t = torch.tensor([3], dtype=torch.long)
    temp = torch.tensor([0.5], dtype=torch.float32)
    native = torch.randn(B, L, 3)
    mask = torch.ones(B, L, dtype=torch.bool)
    win_pos = torch.tensor([0.25], dtype=torch.float32)

    rmsf_prior_a = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32)
    rmsf_prior_b = torch.tensor([[4.0, 2.0, 1.0]], dtype=torch.float32)

    out_a = model(
        x, t, temp, native, mask=mask, win_pos=win_pos, rmsf_prior=rmsf_prior_a
    )
    out_b = model(
        x, t, temp, native, mask=mask, win_pos=win_pos, rmsf_prior=rmsf_prior_b
    )

    if expect_same:
        assert torch.allclose(out_a, out_b, atol=1e-5, rtol=1e-5)
    else:
        assert not torch.allclose(out_a, out_b)


def test_v_space_roundtrip_matches_gained_x0():
    diffusion = SpectralDiffusion(T=10, device="cpu", schedule="linear")
    model = SpectralConvDiT(
        top_k_freqs=2,
        in_channels=3,
        cond_channels=3,
        freq_hidden_size=2,
        depth=1,
        num_heads=1,
        spectral_modes=2,
        cfg_dropout=False,
        prediction_target="v",
        use_rmsf_prior_gain=True,
    )
    with torch.no_grad():
        model.rmsf_gate.fill_(0.7)

    x_t = torch.randn(2, 4, 6)
    v_pred = torch.randn_like(x_t)
    t = torch.tensor([1, 7], dtype=torch.long)
    mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]], dtype=torch.bool
    )
    rmsf_prior = torch.tensor(
        [[1.0, 2.0, 4.0, 0.0], [1.0, 1.5, 2.0, 3.0]], dtype=torch.float32
    )

    x0_base, _ = diffusion.extract_x0_eps_from_prediction(v_pred, x_t, t, "v")
    x0_gained = model.apply_rmsf_prior_gain(x0_base, rmsf_prior, mask=mask)
    v_gained = diffusion.prediction_from_x0(x0_gained, x_t, t, "v")
    x0_roundtrip, _ = diffusion.extract_x0_eps_from_prediction(v_gained, x_t, t, "v")

    assert torch.allclose(x0_roundtrip, x0_gained, atol=1e-5, rtol=1e-5)


def test_low_k_correction_head_is_zero_init_identity():
    torch.manual_seed(7)
    base = SpectralConvDiT(
        top_k_freqs=4,
        in_channels=3,
        cond_channels=3,
        freq_hidden_size=2,
        depth=1,
        num_heads=1,
        spectral_modes=4,
        cfg_dropout=False,
        prediction_target="x_0",
        use_low_k_correction_head=False,
    )
    torch.manual_seed(7)
    headed = SpectralConvDiT(
        top_k_freqs=4,
        in_channels=3,
        cond_channels=3,
        freq_hidden_size=2,
        depth=1,
        num_heads=1,
        spectral_modes=4,
        cfg_dropout=False,
        prediction_target="x_0",
        use_low_k_correction_head=True,
        low_k_correction_modes=2,
    )

    x = torch.randn(2, 5, 12)
    t = torch.tensor([2, 5], dtype=torch.long)
    temp = torch.tensor([0.25, 0.75], dtype=torch.float32)
    native = torch.randn(2, 5, 3)
    mask = torch.ones(2, 5, dtype=torch.bool)
    win_pos = torch.tensor([0.1, 0.6], dtype=torch.float32)

    base_out = base(x, t, temp, native, mask=mask, win_pos=win_pos)
    headed_out = headed(
        x, t, temp, native, mask=mask, win_pos=win_pos, return_aux=True
    )

    assert torch.allclose(headed_out["pred"], base_out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        headed_out["low_k_correction"],
        torch.zeros_like(headed_out["low_k_correction"]),
        atol=1e-7,
        rtol=1e-7,
    )
    assert headed_out["low_k_correction_modes"] == 2


def test_low_k_correction_head_requires_x0_prediction():
    with pytest.raises(ValueError, match="prediction_target='x_0'"):
        SpectralConvDiT(
            top_k_freqs=2,
            in_channels=3,
            cond_channels=3,
            freq_hidden_size=2,
            depth=1,
            num_heads=1,
            spectral_modes=2,
            cfg_dropout=False,
            prediction_target="v",
            use_low_k_correction_head=True,
        )


def test_multi_band_low_k_correction_spec_is_zero_init_identity():
    torch.manual_seed(11)
    base = SpectralConvDiT(
        top_k_freqs=5,
        in_channels=3,
        cond_channels=3,
        freq_hidden_size=2,
        depth=1,
        num_heads=1,
        spectral_modes=5,
        cfg_dropout=False,
        prediction_target="x_0",
        use_low_k_correction_head=False,
    )
    torch.manual_seed(11)
    headed = SpectralConvDiT(
        top_k_freqs=5,
        in_channels=3,
        cond_channels=3,
        freq_hidden_size=2,
        depth=1,
        num_heads=1,
        spectral_modes=5,
        cfg_dropout=False,
        prediction_target="x_0",
        use_low_k_correction_head=True,
        low_k_correction_modes="DC,1-4",
    )

    x = torch.randn(2, 4, 15)
    t = torch.tensor([1, 6], dtype=torch.long)
    temp = torch.tensor([0.2, 0.8], dtype=torch.float32)
    native = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    win_pos = torch.tensor([0.0, 0.5], dtype=torch.float32)

    base_out = base(x, t, temp, native, mask=mask, win_pos=win_pos)
    headed_out = headed(
        x, t, temp, native, mask=mask, win_pos=win_pos, return_aux=True
    )

    assert torch.allclose(headed_out["pred"], base_out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        headed_out["low_k_correction"],
        torch.zeros_like(headed_out["low_k_correction"]),
        atol=1e-7,
        rtol=1e-7,
    )
    assert torch.allclose(
        headed_out["low_k_correction_dc"],
        torch.zeros_like(headed_out["low_k_correction_dc"]),
        atol=1e-7,
        rtol=1e-7,
    )
    assert headed_out["low_k_correction_modes"] == 5
    assert headed_out["low_k_correction_specs"] == [(0, 0), (1, 4)]


def test_multi_band_low_k_correction_dc_uses_dc_scale_slice():
    torch.manual_seed(13)
    freq_scale = torch.arange(1, 16, dtype=torch.float32)
    model = SpectralConvDiT(
        top_k_freqs=5,
        in_channels=3,
        cond_channels=3,
        freq_hidden_size=2,
        depth=1,
        num_heads=1,
        spectral_modes=5,
        cfg_dropout=False,
        prediction_target="x_0",
        use_low_k_correction_head=True,
        low_k_correction_modes="DC,1-4",
        freq_scale=freq_scale,
    )

    x = torch.randn(2, 4, 15)
    t = torch.tensor([1, 6], dtype=torch.long)
    temp = torch.tensor([0.2, 0.8], dtype=torch.float32)
    native = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    win_pos = torch.tensor([0.0, 0.5], dtype=torch.float32)

    out = model(
        x, t, temp, native, mask=mask, win_pos=win_pos, return_aux=True
    )

    assert out["low_k_correction_dc"].shape == (2, 4, 3)
    assert torch.allclose(
        out["low_k_correction_dc"],
        torch.zeros_like(out["low_k_correction_dc"]),
        atol=1e-7,
        rtol=1e-7,
    )


@pytest.mark.parametrize(
    ("hilbert_mode", "expected_blocks"),
    [
        ("every_block", [True, True, True, True]),
        ("every_3_blocks", [True, False, False, True]),
        ("input_only", [True, False, False, False]),
        ("off", [False, False, False, False]),
    ],
)
def test_hilbert_mode_selects_expected_blocks(hilbert_mode, expected_blocks):
    model = SpectralConvDiT(
        top_k_freqs=2,
        in_channels=3,
        cond_channels=3,
        freq_hidden_size=2,
        depth=4,
        num_heads=1,
        spectral_modes=2,
        cfg_dropout=False,
        prediction_target="x_0",
        use_hilbert_dct=True,
        hilbert_mode=hilbert_mode,
    )

    actual = [getattr(block, "hilbert", None) is not None for block in model.blocks]
    assert actual == expected_blocks
