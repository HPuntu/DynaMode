from __future__ import annotations

import pytest
import torch

from dynamode.model.spec_conv.spectral_conv import SpectralConvDiT
from dynamode.model.transformer.transformer import SpectralDiT


def _build_model(model_cls):
    kwargs = dict(
        top_k_freqs=2,
        in_channels=3,
        cond_channels=3,
        freq_hidden_size=2,
        depth=0,
        num_heads=1,
        cfg_dropout=False,
        prediction_target="x_0",
        use_seq_conditioning=True,
        seq_embed_dim=2,
        cond_dim=4,
    )
    if model_cls is SpectralConvDiT:
        kwargs["spectral_modes"] = 2
    return model_cls(**kwargs)


def _build_ss_model(model_cls):
    kwargs = dict(
        top_k_freqs=2,
        in_channels=3,
        cond_channels=3,
        freq_hidden_size=2,
        depth=0,
        num_heads=1,
        cfg_dropout=False,
        prediction_target="x_0",
        use_ss_conditioning=True,
        ss_embed_dim=2,
        cond_dim=4,
    )
    if model_cls is SpectralConvDiT:
        kwargs["spectral_modes"] = 2
    return model_cls(**kwargs)


def _build_size_model(model_cls):
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
    )
    if model_cls is SpectralConvDiT:
        kwargs["spectral_modes"] = 2
    return model_cls(**kwargs)


@pytest.mark.parametrize("model_cls", [SpectralDiT, SpectralConvDiT])
@pytest.mark.parametrize("one_hot_input", [False, True])
def test_sequence_local_embedding_changes_output(model_cls, one_hot_input):
    model = _build_model(model_cls)
    with torch.no_grad():
        model.freq_pos_embed.zero_()
        model.x_embed.weight.zero_()
        model.x_embed.bias.zero_()
        model.x_embed.weight[0, -2] = 1.0
        model.x_embed.weight[1, -1] = 1.0
        model.residue_embed.weight.zero_()
        model.residue_embed.weight[0, 0] = 1.0
        model.residue_embed.weight[1, 1] = 1.0
        model.seq_global_proj.weight.zero_()
        model.seq_global_proj.bias.zero_()
        model.final_adaLN[-1].weight.zero_()
        model.final_adaLN[-1].bias.zero_()
        model.final_proj.weight.zero_()
        model.final_proj.bias.zero_()
        model.final_proj.weight[:, 0] = 1.0

    x = torch.zeros(1, 3, 6)
    t = torch.tensor([1], dtype=torch.long)
    temp = torch.tensor([0.5], dtype=torch.float32)
    native = torch.zeros(1, 3, 3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    win_pos = torch.tensor([0.2], dtype=torch.float32)

    res_type_a = torch.zeros(1, 3, dtype=torch.long)
    res_type_b = torch.ones(1, 3, dtype=torch.long)
    if one_hot_input:
        res_type_a = torch.nn.functional.one_hot(res_type_a, num_classes=21).float()
        res_type_b = torch.nn.functional.one_hot(res_type_b, num_classes=21).float()

    out_a = model(x, t, temp, native, mask=mask, win_pos=win_pos, res_type=res_type_a)
    out_b = model(x, t, temp, native, mask=mask, win_pos=win_pos, res_type=res_type_b)

    assert not torch.allclose(out_a, out_b)


@pytest.mark.parametrize("model_cls", [SpectralDiT, SpectralConvDiT])
def test_sequence_global_embedding_changes_output(model_cls):
    model = _build_model(model_cls)
    with torch.no_grad():
        model.freq_pos_embed.zero_()
        model.x_embed.weight.zero_()
        model.x_embed.bias.zero_()
        model.residue_embed.weight.zero_()
        model.residue_embed.weight[0, 0] = 1.0
        model.residue_embed.weight[1, 0] = 2.0
        model.seq_global_proj.weight.zero_()
        model.seq_global_proj.bias.zero_()
        model.seq_global_proj.weight[0, 0] = 1.0
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

    res_type_a = torch.zeros(1, 3, dtype=torch.long)
    res_type_b = torch.ones(1, 3, dtype=torch.long)

    out_a = model(x, t, temp, native, mask=mask, win_pos=win_pos, res_type=res_type_a)
    out_b = model(x, t, temp, native, mask=mask, win_pos=win_pos, res_type=res_type_b)

    assert not torch.allclose(out_a, out_b)


@pytest.mark.parametrize("model_cls", [SpectralDiT, SpectralConvDiT])
@pytest.mark.parametrize("one_hot_input", [False, True])
def test_secondary_structure_local_embedding_changes_output(model_cls, one_hot_input):
    model = _build_ss_model(model_cls)
    with torch.no_grad():
        model.freq_pos_embed.zero_()
        model.x_embed.weight.zero_()
        model.x_embed.bias.zero_()
        model.x_embed.weight[0, -2] = 1.0
        model.x_embed.weight[1, -1] = 1.0
        model.ss_local_proj.weight.zero_()
        model.ss_local_proj.bias.zero_()
        model.ss_local_proj.weight[0, 0] = 1.0
        model.ss_local_proj.weight[1, 1] = 1.0
        model.ss_global_proj.weight.zero_()
        model.ss_global_proj.bias.zero_()
        model.final_adaLN[-1].weight.zero_()
        model.final_adaLN[-1].bias.zero_()
        model.final_proj.weight.zero_()
        model.final_proj.bias.zero_()
        model.final_proj.weight[:, 0] = 1.0

    x = torch.zeros(1, 3, 6)
    t = torch.tensor([1], dtype=torch.long)
    temp = torch.tensor([0.5], dtype=torch.float32)
    native = torch.zeros(1, 3, 3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    win_pos = torch.tensor([0.2], dtype=torch.float32)

    dssp_a = torch.zeros(1, 3, dtype=torch.long)
    dssp_b = torch.ones(1, 3, dtype=torch.long)
    if one_hot_input:
        dssp_a = torch.nn.functional.one_hot(dssp_a, num_classes=8).float()
        dssp_b = torch.nn.functional.one_hot(dssp_b, num_classes=8).float()

    out_a = model(x, t, temp, native, mask=mask, win_pos=win_pos, dssp=dssp_a)
    out_b = model(x, t, temp, native, mask=mask, win_pos=win_pos, dssp=dssp_b)

    assert not torch.allclose(out_a, out_b)


@pytest.mark.parametrize("model_cls", [SpectralDiT, SpectralConvDiT])
def test_secondary_structure_global_embedding_changes_output(model_cls):
    model = _build_ss_model(model_cls)
    with torch.no_grad():
        model.freq_pos_embed.zero_()
        model.x_embed.weight.zero_()
        model.x_embed.bias.zero_()
        model.ss_local_proj.weight.zero_()
        model.ss_local_proj.bias.zero_()
        model.ss_local_proj.weight[0, 0] = 1.0
        model.ss_local_proj.weight[0, 1] = 2.0
        model.ss_global_proj.weight.zero_()
        model.ss_global_proj.bias.zero_()
        model.ss_global_proj.weight[0, 0] = 1.0
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

    dssp_a = torch.zeros(1, 3, dtype=torch.long)
    dssp_b = torch.ones(1, 3, dtype=torch.long)

    out_a = model(x, t, temp, native, mask=mask, win_pos=win_pos, dssp=dssp_a)
    out_b = model(x, t, temp, native, mask=mask, win_pos=win_pos, dssp=dssp_b)

    assert not torch.allclose(out_a, out_b)


@pytest.mark.parametrize("model_cls", [SpectralDiT, SpectralConvDiT])
def test_size_global_embedding_changes_output(model_cls):
    model = _build_size_model(model_cls)
    with torch.no_grad():
        model.freq_pos_embed.zero_()
        model.x_embed.weight.zero_()
        model.x_embed.bias.zero_()
        model.time_embedder.mlp[0].weight.zero_()
        model.time_embedder.mlp[0].bias.zero_()
        model.time_embedder.mlp[2].weight.zero_()
        model.time_embedder.mlp[2].bias.zero_()
        model.temp_embedder.net[0].weight.zero_()
        model.temp_embedder.net[0].bias.zero_()
        model.temp_embedder.net[2].weight.zero_()
        model.temp_embedder.net[2].bias.zero_()
        model.pos_embedder.net[0].weight.zero_()
        model.pos_embedder.net[0].bias.zero_()
        model.pos_embedder.net[2].weight.zero_()
        model.pos_embedder.net[2].bias.zero_()
        model.size_embed.weight.zero_()
        model.size_embed.weight[0, 0] = 1.0
        model.size_embed.weight[-1, 0] = 2.0
        model.size_global_proj.weight.zero_()
        model.size_global_proj.bias.zero_()
        model.size_global_proj.weight[0, 0] = 1.0
        model.final_adaLN[-1].weight.zero_()
        model.final_adaLN[-1].bias.zero_()
        model.final_adaLN[-1].weight[: model.d_model, 0] = 1.0
        model.final_proj.weight.zero_()
        model.final_proj.bias.zero_()
        model.final_proj.weight[:, 0] = 1.0

    x = torch.zeros(2, 700, 6)
    t = torch.tensor([1, 1], dtype=torch.long)
    temp = torch.tensor([0.5, 0.5], dtype=torch.float32)
    native = torch.zeros(2, 700, 3)
    mask = torch.zeros(2, 700, dtype=torch.bool)
    mask[0, :80] = True
    mask[1, :650] = True
    win_pos = torch.tensor([0.2, 0.2], dtype=torch.float32)

    out = model(x, t, temp, native, mask=mask, win_pos=win_pos)

    assert not torch.allclose(out[0, :80], out[1, :80])
