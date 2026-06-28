'''Batch adapters for supported DynaMode model wrappers.'''


from __future__ import annotations
from typing import Any

from dynamode.spectral.adapters import DCT, DFT, SpectralAdapter
from dynamode.utils import (
    as_tensor,
    check_1d,
    check_2d,
    check_3d,
    check_res_type,
    check_residue_feature,
    validate_batch,
)



def make_spectral_batch_adapter(
    *,
    top_k_freqs: int,
    in_channels: int,
    cond_channels: int,
    is_dct: bool = True,
    conditioned_freq_scale: dict[str, Any] | None = None,
):
    expected_x_last = top_k_freqs * in_channels
    spectral_adapter = (
        SpectralAdapter(
            transform_engine=DCT if is_dct else DFT,
            scale_factors=None,
            conditioned_freq_scale=conditioned_freq_scale,
            channels=in_channels,
            coord_channels=min(cond_channels, in_channels),
        )
        if conditioned_freq_scale is not None
        else None
    )

    def adapter(batch: dict[str, Any]) -> dict[str, Any]:
        validate_batch(batch, ("x", "t", "temp", "native_coords"))

        x = as_tensor(batch["x"], "x")
        t = as_tensor(batch["t"], "t")
        temp = as_tensor(batch["temp"], "temp")
        native_coords = as_tensor(batch["native_coords"], "native_coords")

        if x.ndim != 3:
            raise ValueError(
                f"batch['x'] must have shape (B, L, {expected_x_last}), got {tuple(x.shape)}"
            )

        B, L, Cx = x.shape
        if Cx != expected_x_last:
            raise ValueError(
                f"batch['x'] last dim must be {expected_x_last} = top_k_freqs * in_channels, got {Cx}"
            )

        check_1d(t, "t", B)
        check_1d(temp, "temp", B)
        check_3d(native_coords, "native_coords", B, L, cond_channels)

        mask = batch.get("mask", None)
        if mask is not None:
            mask = as_tensor(mask, "mask")
            check_2d(mask, "mask", B, L)

        win_pos = batch.get("win_pos", None)
        if win_pos is not None:
            win_pos = as_tensor(win_pos, "win_pos")
            check_1d(win_pos, "win_pos", B)

        cond_drop_mask = batch.get("cond_drop_mask", None)
        if cond_drop_mask is not None:
            cond_drop_mask = as_tensor(cond_drop_mask, "cond_drop_mask")
            check_1d(cond_drop_mask, "cond_drop_mask", B)

        native_angles = batch.get("native_angles", None)
        if native_angles is not None:
            native_angles = as_tensor(native_angles, "native_angles")
            if native_angles.shape[0] != B:
                raise ValueError(
                    f"batch['native_angles'] first dim must be {B}, got {native_angles.shape[0]}"
                )

        res_type = batch.get("res_type", None)
        if res_type is not None:
            res_type = as_tensor(res_type, "res_type")
            check_res_type(res_type, "res_type", B, L)

        dssp = batch.get("dssp", None)
        if dssp is not None:
            dssp = as_tensor(dssp, "dssp")
            check_residue_feature(dssp, "dssp", B, L)

        rmsf_prior = batch.get("rmsf_prior", None)
        if rmsf_prior is not None:
            rmsf_prior = as_tensor(rmsf_prior, "rmsf_prior")
            check_2d(rmsf_prior, "rmsf_prior", B, L)

        freq_scale_override = None
        scale_cond = None
        if spectral_adapter is not None:
            freq_scale_override, scale_cond = spectral_adapter.lookup_model_conditioning(
                temp, mask, seq_len=L, device=x.device
            )

        return {
            "x": x,
            "t": t,
            "temp": temp,
            "native_coords": native_coords,
            "mask": mask,
            "win_pos": win_pos,
            "cond_drop_mask": cond_drop_mask,
            "native_angles": native_angles,
            "res_type": res_type,
            "dssp": dssp,
            "rmsf_prior": rmsf_prior,
            "freq_scale_override": freq_scale_override,
            "scale_cond": scale_cond,
            "return_aux": bool(batch.get("return_aux", False)),
        }

    return adapter


__all__ = ["make_spectral_batch_adapter"]
