from __future__ import annotations

import torch

from dynamode.model.adapters import make_spectral_batch_adapter
from dynamode.spectral.conditioned_freq_scale import (
    ConditionedFreqScaleLookup,
    build_conditioned_freq_scale_payload,
)
from dynamode.model.spec_conv.spectral_conv import SpectralConvDiT
from dynamode.model.transformer.transformer import SpectralDiT
from dynamode.spectral.adapters import DCT, SpectralAdapter


def _stats_payload():
    top_k = 2
    n_channels = 3
    feature_dim = top_k * n_channels

    def feature(scale, mean):
        return {
            "abs_q75": torch.as_tensor(scale, dtype=torch.float32),
            "mean": torch.as_tensor(mean, dtype=torch.float32),
        }

    return {
        "metadata": {
            "top_k": top_k,
            "n_channels": n_channels,
            "feature_dim": feature_dim,
            "size_bins": [100, 200],
            "temps_seen": [320, 450],
            "coords_type": "ca",
            "coords_only": True,
            "displacement": True,
            "subset_tag": "train",
        },
        "buckets": {
            "overall": {"feature": feature([10, 20, 30, 40, 50, 60], [1, 2, 3, 0, 0, 0])},
            "temp:450": {"feature": feature([20, 30, 40, 50, 60, 70], [3, 4, 5, 0, 0, 0])},
            "size:le_100": {"feature": feature([12, 22, 32, 42, 52, 62], [2, 3, 4, 0, 0, 0])},
            "temp_size:450|le_100": {"feature": feature([30, 40, 50, 60, 70, 80], [5, 6, 7, 0, 0, 0])},
        },
    }


def _conditioned_payload():
    return build_conditioned_freq_scale_payload(
        _stats_payload(),
        alpha=0.75,
        stat_name="abs_q75",
        coord_channels=3,
        scale_condition_modes=1,
        scale_condition_channels=3,
    )


def test_conditioned_lookup_roundtrip_and_bucket_selection():
    payload = _conditioned_payload()
    lookup = ConditionedFreqScaleLookup(payload)

    temp = torch.tensor([1.0], dtype=torch.float32)  # normalized -> 450K
    mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)

    scales = lookup.lookup_scales(temp, mask)
    dc = lookup.lookup_dc_baselines(temp, mask, coord_channels=3)
    scale_features = lookup.lookup_scale_features(temp, mask)

    expected_scale = 0.25 * torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.float32) + 0.75 * torch.tensor(
        [30, 40, 50, 60, 70, 80], dtype=torch.float32
    )
    expected_dc = 0.25 * torch.tensor([1, 2, 3], dtype=torch.float32) + 0.75 * torch.tensor(
        [5, 6, 7], dtype=torch.float32
    )

    assert torch.allclose(scales[0], expected_scale)
    assert torch.allclose(dc[0], expected_dc)
    assert scale_features.shape == (1, 6)
    assert torch.allclose(scale_features[0], expected_scale.view(2, 3).reshape(-1))

    x = torch.arange(18, dtype=torch.float32).view(1, 3, 6)
    x_resid, baseline = lookup.residualise_dc(x, temp, mask, coord_channels=3)
    x_restored = lookup.restore_dc(x_resid, baseline, coord_channels=3)
    assert torch.allclose(x, x_restored)


def test_spectral_adapter_adds_scale_override_and_features():
    payload = _conditioned_payload()
    adapter = make_spectral_batch_adapter(
        top_k_freqs=2,
        in_channels=3,
        cond_channels=3,
        conditioned_freq_scale=payload,
    )
    batch = {
        "x": torch.zeros(1, 3, 6),
        "t": torch.tensor([1], dtype=torch.long),
        "temp": torch.tensor([1.0], dtype=torch.float32),
        "native_coords": torch.zeros(1, 3, 3),
        "mask": torch.ones(1, 3, dtype=torch.bool),
    }
    out = adapter(batch)
    assert out["freq_scale_override"] is not None
    assert out["scale_cond"] is not None
    assert out["freq_scale_override"].shape == (1, 6)
    assert out["scale_cond"].shape == (1, 6)


def test_unified_spectral_adapter_roundtrip_and_conditioned_helpers():
    payload = _conditioned_payload()
    adapter = SpectralAdapter(
        transform_engine=DCT,
        scale_factors=torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.float32),
        conditioned_freq_scale=payload,
        channels=3,
        coord_channels=3,
    )

    x_time = torch.randn(1, 4, 3, 3)
    x_spec = adapter.time_to_spectral(x_time, top_k=2)
    x_time_rt = adapter.spectral_to_time(x_spec, n_time_steps=4, n_channels=3)
    assert x_spec.shape == (1, 3, 6)
    assert x_time_rt.shape == x_time.shape

    x_norm = adapter.normalise(x_spec)
    x_recon = adapter.denormalise(x_norm)
    assert torch.allclose(x_spec, x_recon, atol=1e-5, rtol=1e-5)

    temp = torch.tensor([1.0], dtype=torch.float32)
    mask = torch.ones(1, 3, dtype=torch.bool)
    freq_scale_override, scale_cond = adapter.lookup_model_conditioning(temp, mask)
    assert freq_scale_override.shape == (1, 6)
    assert scale_cond.shape == (1, 6)


def _build_scale_model(model_cls):
    payload = _conditioned_payload()
    kwargs = dict(
        top_k_freqs=2,
        in_channels=3,
        cond_channels=3,
        freq_hidden_size=2,
        depth=0,
        num_heads=1,
        cfg_dropout=False,
        prediction_target="x_0",
        cond_dim=4,
        conditioned_freq_scale=payload,
    )
    if model_cls is SpectralConvDiT:
        kwargs["spectral_modes"] = 2
    return model_cls(**kwargs)


def _assert_scale_conditioning_changes_output(model_cls):
    model = _build_scale_model(model_cls)
    with torch.no_grad():
        model.freq_pos_embed.zero_()
        model.x_embed.weight.zero_()
        model.x_embed.bias.zero_()
        model.scale_global_proj.weight.zero_()
        model.scale_global_proj.bias.zero_()
        model.scale_global_proj.weight[0, 0] = 1.0
        model.final_adaLN[-1].weight.zero_()
        model.final_adaLN[-1].bias.zero_()
        model.final_adaLN[-1].weight[: model.d_model, 0] = 1.0
        model.final_proj.weight.zero_()
        model.final_proj.bias.zero_()
        model.final_proj.weight[:, 0] = 1.0

    x = torch.zeros(1, 3, 6)
    t = torch.tensor([1], dtype=torch.long)
    temp = torch.tensor([0.5], dtype=torch.float32)
    native = torch.zeros(1, 3, 3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    win_pos = torch.tensor([0.2], dtype=torch.float32)
    scale_a = torch.ones(1, 6)
    scale_b = torch.full((1, 6), 2.0)

    out_a = model(x, t, temp, native, mask=mask, win_pos=win_pos, scale_cond=scale_a)
    out_b = model(x, t, temp, native, mask=mask, win_pos=win_pos, scale_cond=scale_b)
    assert not torch.allclose(out_a, out_b)


def test_transformer_scale_conditioning_changes_output():
    _assert_scale_conditioning_changes_output(SpectralDiT)


def test_spectral_conv_scale_conditioning_changes_output():
    _assert_scale_conditioning_changes_output(SpectralConvDiT)
