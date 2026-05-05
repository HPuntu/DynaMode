from __future__ import annotations

from typing import Any
import torch

from src.spectral.adapters import SpectralAdapter, DCT, DFT


def validate_batch(batch: dict[str, Any], required: tuple[str, ...]) -> None:
    missing = [k for k in required if k not in batch]
    if missing:
        raise KeyError(f"Missing batch keys: {missing}")


def _as_tensor(x: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"batch['{name}'] must be a torch.Tensor, got {type(x).__name__}")
    return x


def _check_1d(t: torch.Tensor, name: str, B: int | None = None) -> None:
    if t.ndim != 1:
        raise ValueError(f"batch['{name}'] must be 1D, got shape {tuple(t.shape)}")
    if B is not None and t.shape[0] != B:
        raise ValueError(f"batch['{name}'] must have shape ({B},), got {tuple(t.shape)}")


def _check_2d(t: torch.Tensor, name: str, B: int | None = None, L: int | None = None) -> None:
    if t.ndim != 2:
        raise ValueError(f"batch['{name}'] must be 2D, got shape {tuple(t.shape)}")
    if B is not None and t.shape[0] != B:
        raise ValueError(f"batch['{name}'] first dim must be {B}, got {t.shape[0]}")
    if L is not None and t.shape[1] != L:
        raise ValueError(f"batch['{name}'] second dim must be {L}, got {t.shape[1]}")


def _check_3d(
    t: torch.Tensor,
    name: str,
    B: int | None = None,
    L: int | None = None,
    C: int | None = None,
) -> None:
    if t.ndim != 3:
        raise ValueError(f"batch['{name}'] must be 3D, got shape {tuple(t.shape)}")
    if B is not None and t.shape[0] != B:
        raise ValueError(f"batch['{name}'] first dim must be {B}, got {t.shape[0]}")
    if L is not None and t.shape[1] != L:
        raise ValueError(f"batch['{name}'] second dim must be {L}, got {t.shape[1]}")
    if C is not None and t.shape[2] != C:
        raise ValueError(f"batch['{name}'] last dim must be {C}, got {t.shape[2]}")


def _check_res_type(t: torch.Tensor, name: str, B: int, L: int) -> None:
    if t.ndim == 2:
        if t.shape[0] != B or t.shape[1] != L:
            raise ValueError(f"batch['{name}'] must have shape ({B}, {L}), got {tuple(t.shape)}")
        return
    if t.ndim == 3:
        if t.shape[0] != B or t.shape[1] != L:
            raise ValueError(
                f"batch['{name}'] first two dims must be ({B}, {L}), got {tuple(t.shape)}"
            )
        return
    raise ValueError(
        f"batch['{name}'] must be residue indices (B, L) or one-hot (B, L, C), got {tuple(t.shape)}"
    )


def _check_residue_feature(t: torch.Tensor, name: str, B: int, L: int) -> None:
    if t.ndim == 2:
        if t.shape[0] != B or t.shape[1] != L:
            raise ValueError(f"batch['{name}'] must have shape ({B}, {L}), got {tuple(t.shape)}")
        return
    if t.ndim == 3:
        if t.shape[0] != B or t.shape[1] != L:
            raise ValueError(
                f"batch['{name}'] first two dims must be ({B}, {L}), got {tuple(t.shape)}"
            )
        return
    raise ValueError(f"batch['{name}'] must be 2D or 3D, got {tuple(t.shape)}")


def make_spectral_batch_adapter(
    *,
    top_k_freqs: int,
    in_channels: int,
    cond_channels: int,
    is_dct: bool = True,
    conditioned_freq_scale: dict[str, Any] | None = None,
    include_frequency_masks: bool = False,
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

        x = _as_tensor(batch["x"], "x")
        t = _as_tensor(batch["t"], "t")
        temp = _as_tensor(batch["temp"], "temp")
        native_coords = _as_tensor(batch["native_coords"], "native_coords")

        if x.ndim != 3:
            raise ValueError(f"batch['x'] must have shape (B, L, {expected_x_last}), got {tuple(x.shape)}")

        B, L, Cx = x.shape
        if Cx != expected_x_last:
            raise ValueError(
                f"batch['x'] last dim must be {expected_x_last} = top_k_freqs * in_channels, got {Cx}"
            )

        _check_1d(t, "t", B)
        _check_1d(temp, "temp", B)
        _check_3d(native_coords, "native_coords", B, L, cond_channels)

        mask = batch.get("mask", None)
        if mask is not None:
            mask = _as_tensor(mask, "mask")
            _check_2d(mask, "mask", B, L)

        win_pos = batch.get("win_pos", None)
        if win_pos is not None:
            win_pos = _as_tensor(win_pos, "win_pos")
            _check_1d(win_pos, "win_pos", B)

        cond_drop_mask = batch.get("cond_drop_mask", None)
        if cond_drop_mask is not None:
            cond_drop_mask = _as_tensor(cond_drop_mask, "cond_drop_mask")
            _check_1d(cond_drop_mask, "cond_drop_mask", B)

        native_angles = batch.get("native_angles", None)
        if native_angles is not None and torch.is_tensor(native_angles):
            if native_angles.shape[0] != B:
                raise ValueError(
                    f"batch['native_angles'] first dim must be {B}, got {native_angles.shape[0]}"
                )

        res_type = batch.get("res_type", None)
        if res_type is not None:
            res_type = _as_tensor(res_type, "res_type")
            _check_res_type(res_type, "res_type", B, L)

        dssp = batch.get("dssp", None)
        if dssp is not None:
            dssp = _as_tensor(dssp, "dssp")
            _check_residue_feature(dssp, "dssp", B, L)

        # Optional NMA RMSF prior — per-residue unitless ANM fluctuation scale.
        rmsf_prior = batch.get("rmsf_prior", None)
        if rmsf_prior is not None:
            rmsf_prior = _as_tensor(rmsf_prior, "rmsf_prior")
            _check_2d(rmsf_prior, "rmsf_prior", B, L)

        freq_scale_override = None
        scale_cond = None
        if spectral_adapter is not None:
            freq_scale_override, scale_cond = spectral_adapter.lookup_model_conditioning(
                temp, mask, seq_len=L, device=x.device
            )

        return_aux = bool(batch.get("return_aux", False))

        adapted = {
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
            "return_aux": return_aux,
        }
        if include_frequency_masks:
            adapted.update({
                "freq_keep_groups": batch.get("freq_keep_groups", None),
                "freq_mask_groups": batch.get("freq_mask_groups", None),
                "freq_mask": batch.get("freq_mask", None),
                "freq_target_groups": batch.get("freq_target_groups", None),
            })
        return adapted

    return adapter


def make_fno_batch_adapter(
    *,
    window_size: int,
    in_channels: int,
    cond_channels: int,
):
    expected_x_last = window_size * in_channels

    def adapter(batch: dict[str, Any]) -> dict[str, Any]:
        validate_batch(batch, ("x", "t", "temp", "native_coords"))

        x = _as_tensor(batch["x"], "x")
        t = _as_tensor(batch["t"], "t")
        temp = _as_tensor(batch["temp"], "temp")
        native_coords = _as_tensor(batch["native_coords"], "native_coords")

        if x.ndim != 3:
            raise ValueError(f"batch['x'] must have shape (B, L, {expected_x_last}), got {tuple(x.shape)}")

        B, L, Cx = x.shape
        if Cx != expected_x_last:
            raise ValueError(
                f"batch['x'] last dim must be {expected_x_last} = window_size * in_channels, got {Cx}"
            )

        _check_1d(t, "t", B)
        _check_1d(temp, "temp", B)
        _check_3d(native_coords, "native_coords", B, L, cond_channels)

        mask = batch.get("mask", None)
        if mask is not None:
            mask = _as_tensor(mask, "mask")
            _check_2d(mask, "mask", B, L)

        win_pos = batch.get("win_pos", None)
        if win_pos is not None:
            win_pos = _as_tensor(win_pos, "win_pos")
            _check_1d(win_pos, "win_pos", B)

        cond_drop_mask = batch.get("cond_drop_mask", None)
        if cond_drop_mask is not None:
            cond_drop_mask = _as_tensor(cond_drop_mask, "cond_drop_mask")
            _check_1d(cond_drop_mask, "cond_drop_mask", B)

        native_angles = batch.get("native_angles", None)
        if native_angles is not None and torch.is_tensor(native_angles):
            if native_angles.shape[0] != B:
                raise ValueError(
                    f"batch['native_angles'] first dim must be {B}, got {native_angles.shape[0]}"
                )

        res_type = batch.get("res_type", None)
        if res_type is not None:
            res_type = _as_tensor(res_type, "res_type")
            _check_res_type(res_type, "res_type", B, L)

        dssp = batch.get("dssp", None)
        if dssp is not None:
            dssp = _as_tensor(dssp, "dssp")
            _check_residue_feature(dssp, "dssp", B, L)

        rmsf_prior = batch.get("rmsf_prior", None)
        if rmsf_prior is not None:
            rmsf_prior = _as_tensor(rmsf_prior, "rmsf_prior")
            _check_2d(rmsf_prior, "rmsf_prior", B, L)

        slow_plan = batch.get("slow_plan", None)
        if slow_plan is not None:
            slow_plan = _as_tensor(slow_plan, "slow_plan")
            if slow_plan.ndim == 3:
                _check_3d(slow_plan, "slow_plan", B, L, expected_x_last)
            elif slow_plan.ndim == 4:
                if slow_plan.shape[0] != B or slow_plan.shape[1] != window_size or slow_plan.shape[2] != L or slow_plan.shape[3] != in_channels:
                    raise ValueError(
                        f"batch['slow_plan'] must have shape ({B}, {window_size}, {L}, {in_channels}) or "
                        f"({B}, {L}, {expected_x_last}), got {tuple(slow_plan.shape)}"
                    )
            else:
                raise ValueError(
                    f"batch['slow_plan'] must be 3D or 4D, got shape {tuple(slow_plan.shape)}"
                )

        frame_mask = batch.get("frame_mask", None)
        if frame_mask is not None:
            frame_mask = _as_tensor(frame_mask, "frame_mask")
            _check_2d(frame_mask, "frame_mask", B, window_size)

        delta_t = batch.get("delta_t", None)
        if delta_t is not None:
            delta_t = _as_tensor(delta_t, "delta_t")
            _check_1d(delta_t, "delta_t", B)

        slow_plan_noise_scale = batch.get("slow_plan_noise_scale", None)
        if slow_plan_noise_scale is not None and torch.is_tensor(slow_plan_noise_scale):
            if slow_plan_noise_scale.ndim > 1:
                raise ValueError(
                    f"batch['slow_plan_noise_scale'] must be scalar or 1D, got {tuple(slow_plan_noise_scale.shape)}"
                )

        return_aux = bool(batch.get("return_aux", False))

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
            "slow_plan": slow_plan,
            "frame_mask": frame_mask,
            "delta_t": delta_t,
            "slow_plan_noise_scale": slow_plan_noise_scale,
            "return_aux": return_aux,
        }

    return adapter


def make_manifold_batch_adapter(
    *,
    window_size: int,
    latent_dim: int,
    cond_channels: int,
):
    expected_x_last = window_size * latent_dim

    def adapter(batch: dict[str, Any]) -> dict[str, Any]:
        validate_batch(batch, ("x", "t", "temp", "native_coords"))

        x = _as_tensor(batch["x"], "x")
        t = _as_tensor(batch["t"], "t")
        temp = _as_tensor(batch["temp"], "temp")
        native_coords = _as_tensor(batch["native_coords"], "native_coords")

        if x.ndim != 3:
            raise ValueError(f"batch['x'] must have shape (B, L, {expected_x_last}), got {tuple(x.shape)}")
        B, L, Cx = x.shape
        if Cx != expected_x_last:
            raise ValueError(
                f"batch['x'] last dim must be {expected_x_last} = window_size * latent_dim, got {Cx}"
            )

        _check_1d(t, "t", B)
        _check_1d(temp, "temp", B)
        _check_3d(native_coords, "native_coords", B, L, cond_channels)

        mask = batch.get("mask", None)
        if mask is not None:
            mask = _as_tensor(mask, "mask")
            _check_2d(mask, "mask", B, L)

        win_pos = batch.get("win_pos", None)
        if win_pos is not None:
            win_pos = _as_tensor(win_pos, "win_pos")
            _check_1d(win_pos, "win_pos", B)

        cond_drop_mask = batch.get("cond_drop_mask", None)
        if cond_drop_mask is not None:
            cond_drop_mask = _as_tensor(cond_drop_mask, "cond_drop_mask")
            _check_1d(cond_drop_mask, "cond_drop_mask", B)

        res_type = batch.get("res_type", None)
        if res_type is not None:
            res_type = _as_tensor(res_type, "res_type")
            _check_res_type(res_type, "res_type", B, L)

        dssp = batch.get("dssp", None)
        if dssp is not None:
            dssp = _as_tensor(dssp, "dssp")
            _check_residue_feature(dssp, "dssp", B, L)

        rmsf_prior = batch.get("rmsf_prior", None)
        if rmsf_prior is not None:
            rmsf_prior = _as_tensor(rmsf_prior, "rmsf_prior")
            _check_2d(rmsf_prior, "rmsf_prior", B, L)

        return {
            "x": x,
            "t": t,
            "temp": temp,
            "native_coords": native_coords,
            "mask": mask,
            "win_pos": win_pos,
            "cond_drop_mask": cond_drop_mask,
            "native_angles": batch.get("native_angles", None),
            "res_type": res_type,
            "dssp": dssp,
            "rmsf_prior": rmsf_prior,
        }

    return adapter


def identity_output(output: Any) -> Any:
    return output
