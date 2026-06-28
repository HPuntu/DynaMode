"""CA-only geometry minimiser inspired by aSAM2's postprocessor.

aSAM2 minimises all-atom atom14 structures with force-field bonds, angles,
dihedrals and non-bonded terms.  Pancake often works with CA trajectories, so
this module keeps the same optimiser shape but replaces the energy with CA-only
terms:

* adjacent CA-CA bond restraints;
* CA bend-angle preservation;
* CA pseudo-dihedral preservation;
* non-bonded CA-CA clash repulsion for sequence separation >= ``min_sep``.
* weak segment-intersection repulsion.

Coordinates are in Angstroms.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_PARAMS = {
    "data": {"batch_size": 25},
    "top": {
        "min_sep": 2,
        "min_segment_sep": 2,
        "nb_centers_threshold": 12.0,
    },
    "opt_ini": {
        "opt": "adam",
        "step_size": 0.001,
        "steps": 80,
        "beta1": 0.5,
        "beta2": 0.9,
        "nb_update_freq": 10,
        "nb_centers_threshold": 12.0,
        "energy_params": {
            "bond_const": 1000.0,
            "angle_const": 100.0,
            "dihedral_const": 10.0,
            "nb_const": 10.0,
            "nb_form": "l2",
            "nb_threshold": 4.0,
            "nb_normalize_by": "active",
            "segment_const": 10.0,
            "segment_form": "l2",
            "segment_threshold": 2.0,
            "segment_normalize_by": "active",
            "early_stopping_clash_score": None,
            "early_stopping_clash_thresh": 3.5,
            "normalize_energy": True,
        },
        "bond_init_range": [3.20, 4.60],
        "bond_target": 3.8,
        "bond_target_mode": "initial_in_range_else_ignore",
        "gradient_clip": 100.0,
        "gradient_clip_mode": "norm",
        "max_step_displacement": 0.35,
        "rollback_on_energy_increase": True,
        "energy_increase_tolerance": 10.0,
        "max_abs_coord": 1000.0,
        "min_caca_bond": 1.0,
        "max_caca_bond": 10.0,
        "caca_bond_guard_tolerance": 0.0,
    },
    "opt": {
        "opt": "lbfgs",
        "step_size": 0.1,
        "steps": 15,
        "max_iter": 5,
        "history_size": 100,
        "line_search_fn": "strong_wolfe",
        "nb_update_freq": 10,
        "nb_centers_threshold": 12.0,
        "energy_params": {
            "bond_const": 1000.0,
            "angle_const": 100.0,
            "dihedral_const": 10.0,
            "nb_const": 25.0,
            "nb_form": "l2",
            "nb_threshold": 4.0,
            "nb_normalize_by": "active",
            "segment_const": 50.0,
            "segment_form": "l2",
            "segment_threshold": 2.0,
            "segment_normalize_by": "active",
            "early_stopping_clash_score": None,
            "early_stopping_clash_thresh": 3.5,
            "normalize_energy": True,
        },
        "bond_init_range": [3.20, 4.60],
        "bond_target": 3.8,
        "bond_target_mode": "initial_in_range_else_ignore",
        "gradient_clip": 100.0,
        "gradient_clip_mode": "norm",
        "max_step_displacement": 0.35,
        "rollback_on_energy_increase": True,
        "energy_increase_tolerance": 10.0,
        "max_abs_coord": 1000.0,
        "min_caca_bond": 1.0,
        "max_caca_bond": 10.0,
        "caca_bond_guard_tolerance": 0.0,
    },
}


class CAMinimizerEarlyStopping(Exception):
    """Raised internally when the clash score is already low enough."""

    def __init__(self, score: torch.Tensor, thresh: float):
        super().__init__(f"CA minimizer early stopping: {score.item():.4g} <= {thresh}")
        self.score = score
        self.thresh = thresh


@dataclass
class _ShapeInfo:
    original_shape: torch.Size
    flat_shape: torch.Size


def _deep_update(base: dict[str, Any], updates: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    if not updates:
        return out
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_params(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    params = copy.deepcopy(DEFAULT_PARAMS)
    if path is not None:
        import yaml

        with open(path) as handle:
            params = _deep_update(params, yaml.safe_load(handle) or {})
    return _deep_update(params, overrides)


def _as_flat_positions(ca: torch.Tensor) -> tuple[torch.Tensor, _ShapeInfo]:
    if ca.ndim < 3 or ca.shape[-1] != 3:
        raise ValueError(f"Expected CA coordinates with shape (..., L, 3), got {tuple(ca.shape)}")
    original_shape = ca.shape
    flat = ca.reshape(-1, ca.shape[-2], 3)
    return flat, _ShapeInfo(original_shape=original_shape, flat_shape=flat.shape)


def _restore_shape(ca_flat: torch.Tensor, info: _ShapeInfo) -> torch.Tensor:
    return ca_flat.reshape(info.original_shape)


def _empty_long(device: torch.device, width: int) -> torch.Tensor:
    return torch.empty((0, width), dtype=torch.long, device=device)


def _prepare_mask(mask: torch.Tensor | None, n_frames: int, n_residues: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(n_frames, n_residues, dtype=torch.bool, device=device)
    mask = mask.to(device=device, dtype=torch.bool)
    if mask.shape[-1] != n_residues:
        raise ValueError(f"Mask has incompatible residue dimension: {tuple(mask.shape)} vs L={n_residues}")
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand(n_frames, -1)
    elif mask.ndim > 2:
        mask = mask.reshape(-1, n_residues)
    if mask.shape[0] == 1 and n_frames != 1:
        mask = mask.expand(n_frames, -1)
    if mask.shape != (n_frames, n_residues):
        raise ValueError(f"Mask has incompatible shape {tuple(mask.shape)} for flattened coords {(n_frames, n_residues)}")
    return mask


def _caca_bond_range_per_frame(
    positions: torch.Tensor,
    topology: dict[str, Any],
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return min/max adjacent CA-CA bond lengths for each flattened frame."""
    bond_ids = topology["bonds"]["ids"].to(device=positions.device)
    n_frames = positions.shape[0]
    if bond_ids.numel() == 0:
        return (
            positions.new_full((n_frames,), float("inf")),
            positions.new_zeros((n_frames,)),
        )
    mask = topology["mask"].to(device=positions.device)
    valid = _valid_pair_mask(mask, bond_ids)
    distances = calc_distances(positions.detach(), bond_ids, eps=eps)
    any_valid = valid.any(dim=1)
    max_dist = distances.masked_fill(~valid, float("-inf")).amax(dim=1)
    min_dist = distances.masked_fill(~valid, float("inf")).amin(dim=1)
    max_dist = torch.where(any_valid, max_dist, positions.new_zeros((n_frames,)))
    min_dist = torch.where(any_valid, min_dist, positions.new_full((n_frames,), float("inf")))
    return min_dist, max_dist


def get_topology(
    n_residues: int,
    *,
    mask: torch.Tensor | None = None,
    n_frames: int = 1,
    min_sep: int = 2,
    min_segment_sep: int = 2,
    nb_centers_threshold: float = 10.0,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Build a CA-only topology for a fixed residue count."""
    device = torch.device(device) if device is not None else torch.device("cpu")
    n_residues = int(n_residues)
    if n_residues <= 0:
        raise ValueError(f"n_residues must be positive, got {n_residues}")

    bond_ids = (
        torch.stack([
            torch.arange(n_residues - 1, device=device),
            torch.arange(1, n_residues, device=device),
        ], dim=1)
        if n_residues >= 2 else _empty_long(device, 2)
    )
    angle_ids = (
        torch.stack([
            torch.arange(n_residues - 2, device=device),
            torch.arange(1, n_residues - 1, device=device),
            torch.arange(2, n_residues, device=device),
        ], dim=1)
        if n_residues >= 3 else _empty_long(device, 3)
    )
    dihedral_ids = (
        torch.stack([
            torch.arange(n_residues - 3, device=device),
            torch.arange(1, n_residues - 2, device=device),
            torch.arange(2, n_residues - 1, device=device),
            torch.arange(3, n_residues, device=device),
        ], dim=1)
        if n_residues >= 4 else _empty_long(device, 4)
    )
    if n_residues > int(min_sep):
        pair_i, pair_j = torch.triu_indices(n_residues, n_residues, offset=int(min_sep), device=device)
        nb_ids = torch.stack([pair_i, pair_j], dim=1)
    else:
        nb_ids = _empty_long(device, 2)
    segment_ids = (
        torch.stack([
            torch.arange(n_residues - 1, device=device),
            torch.arange(1, n_residues, device=device),
        ], dim=1)
        if n_residues >= 2 else _empty_long(device, 2)
    )
    n_segments = int(segment_ids.shape[0])
    if n_segments > int(min_segment_sep):
        seg_i, seg_j = torch.triu_indices(n_segments, n_segments, offset=int(min_segment_sep), device=device)
        segment_pair_ids = torch.stack([seg_i, seg_j], dim=1)
    else:
        segment_pair_ids = _empty_long(device, 2)

    return {
        "n_residues": n_residues,
        "mask": _prepare_mask(mask, n_frames, n_residues, device),
        "bonds": {"ids": bond_ids},
        "angles": {"ids": angle_ids},
        "dihedrals": {"ids": dihedral_ids},
        "nb_centers": {"ids": nb_ids},
        "segments": {"ids": segment_ids},
        "segment_pairs": {"ids": segment_pair_ids},
        "nb_cache": {"ids": nb_ids},
        "nb_centers_threshold": float(nb_centers_threshold),
        "min_sep": int(min_sep),
        "min_segment_sep": int(min_segment_sep),
    }


def calc_bond_lengths(pos_i: torch.Tensor, pos_j: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.sum(torch.square(pos_i - pos_j), dim=-1) + eps)


def calc_angles(pos_i: torch.Tensor, pos_j: torch.Tensor, pos_k: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    u = pos_i - pos_j
    v = pos_k - pos_j
    u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True).clamp_min(eps)
    v = v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)
    cos_theta = torch.sum(u * v, dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.arccos(cos_theta)


def calc_dihedrals(
    pos_i: torch.Tensor,
    pos_j: torch.Tensor,
    pos_k: torch.Tensor,
    pos_l: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    b0 = -(pos_j - pos_i)
    b1 = pos_k - pos_j
    b2 = pos_l - pos_k
    b0xb1 = torch.cross(b0, b1, dim=-1)
    b1xb2 = torch.cross(b2, b1, dim=-1)
    y = torch.sum(torch.cross(b0xb1, b1xb2, dim=-1) * b1, dim=-1) / torch.linalg.vector_norm(b1, dim=-1).clamp_min(eps)
    x = torch.sum(b0xb1 * b1xb2, dim=-1)
    return torch.atan2(y, x)


def calc_distances(positions: torch.Tensor, pair_ids: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if pair_ids.numel() == 0:
        return positions.new_zeros((positions.shape[0], 0))
    return calc_bond_lengths(positions[:, pair_ids[:, 0], :], positions[:, pair_ids[:, 1], :], eps=eps)


def calc_segment_segment_distances(
    p1: torch.Tensor,
    q1: torch.Tensor,
    p2: torch.Tensor,
    q2: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Closest distances between batched 3D line segments."""
    u = q1 - p1
    v = q2 - p2
    w = p1 - p2
    a = torch.sum(u * u, dim=-1)
    b = torch.sum(u * v, dim=-1)
    c = torch.sum(v * v, dim=-1)
    d = torch.sum(u * w, dim=-1)
    e = torch.sum(v * w, dim=-1)
    denom = a * c - b * b

    small = float(eps)
    s_den = denom
    t_den = denom
    s_num = b * e - c * d
    t_num = a * e - b * d

    parallel = denom < small
    s_num = torch.where(parallel, torch.zeros_like(s_num), s_num)
    s_den = torch.where(parallel, torch.ones_like(s_den), s_den)
    t_num = torch.where(parallel, e, t_num)
    t_den = torch.where(parallel, c, t_den)

    before_start = s_num < 0.0
    s_num = torch.where(before_start, torch.zeros_like(s_num), s_num)
    t_num = torch.where(before_start, e, t_num)
    t_den = torch.where(before_start, c, t_den)

    after_end = s_num > s_den
    s_num = torch.where(after_end, s_den, s_num)
    t_num = torch.where(after_end, e + b, t_num)
    t_den = torch.where(after_end, c, t_den)

    before_t = t_num < 0.0
    t_num = torch.where(before_t, torch.zeros_like(t_num), t_num)
    before_t_s = torch.clamp(-d, min=0.0)
    before_t_s = torch.minimum(before_t_s, a)
    s_num = torch.where(before_t, before_t_s, s_num)
    s_den = torch.where(before_t, a, s_den)

    after_t = t_num > t_den
    t_num = torch.where(after_t, t_den, t_num)
    after_t_s = torch.clamp(-d + b, min=0.0)
    after_t_s = torch.minimum(after_t_s, a)
    s_num = torch.where(after_t, after_t_s, s_num)
    s_den = torch.where(after_t, a, s_den)

    sc = torch.where(torch.abs(s_num) < small, torch.zeros_like(s_num), s_num / s_den.clamp_min(small))
    tc = torch.where(torch.abs(t_num) < small, torch.zeros_like(t_num), t_num / t_den.clamp_min(small))
    delta = w + sc.unsqueeze(-1) * u - tc.unsqueeze(-1) * v
    return torch.sqrt(torch.sum(delta * delta, dim=-1) + eps)


def _valid_pair_mask(mask: torch.Tensor, pair_ids: torch.Tensor) -> torch.Tensor:
    if pair_ids.numel() == 0:
        return mask.new_zeros((mask.shape[0], 0), dtype=torch.bool)
    return mask[:, pair_ids[:, 0]] & mask[:, pair_ids[:, 1]]


def _valid_segment_pair_mask(mask: torch.Tensor, segment_ids: torch.Tensor, segment_pair_ids: torch.Tensor) -> torch.Tensor:
    if segment_pair_ids.numel() == 0:
        return mask.new_zeros((mask.shape[0], 0), dtype=torch.bool)
    first = segment_ids[segment_pair_ids[:, 0]]
    second = segment_ids[segment_pair_ids[:, 1]]
    return (
        mask[:, first[:, 0]] & mask[:, first[:, 1]]
        & mask[:, second[:, 0]] & mask[:, second[:, 1]]
    )


def _hinge_energy(overlap: torch.Tensor, form: str) -> torch.Tensor:
    if form == "l2":
        return torch.square(overlap)
    if form == "l1":
        return torch.abs(overlap)
    raise KeyError(form)


def _angle_delta(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value - target), torch.cos(value - target))


def calc_energy(
    positions: torch.Tensor,
    topology: dict[str, Any],
    *,
    bond_const: float = 1000.0,
    angle_const: float = 100.0,
    dihedral_const: float = 10.0,
    nb_const: float = 2.5,
    nb_threshold: float = 3.5,
    nb_form: str = "l2",
    nb_normalize_by: str = "valid",
    segment_const: float = 2.5,
    segment_threshold: float = 1.0,
    segment_form: str = "l2",
    segment_normalize_by: str = "valid",
    early_stopping_clash_score: float | None = None,
    early_stopping_clash_thresh: float = 3.5,
    normalize_energy: bool = True,
    eps: float = 1e-12,
    verbose: bool = False,
) -> torch.Tensor:
    """Return summed CA minimisation energy over flattened frames."""
    del verbose
    mask = topology["mask"].to(device=positions.device)
    total = positions.new_zeros(positions.shape[0])

    bond_ids = topology["bonds"]["ids"].to(device=positions.device)
    if bond_ids.numel() > 0:
        bond_valid = _valid_pair_mask(mask, bond_ids)
        if "mask" in topology["bonds"]:
            bond_valid = bond_valid & topology["bonds"]["mask"].to(device=positions.device, dtype=torch.bool)
        bond_len = calc_distances(positions, bond_ids, eps=eps)
        bond_target = topology["bonds"]["params"].to(device=positions.device, dtype=positions.dtype)
        bond_energy = torch.square(bond_len - bond_target) * bond_valid.float()
        bond_sum = bond_energy.sum(dim=1)
        if normalize_energy:
            bond_sum = bond_sum / bond_valid.float().sum(dim=1).clamp_min(1.0)
        total = total + float(bond_const) * bond_sum

    angle_ids = topology["angles"]["ids"].to(device=positions.device)
    if angle_ids.numel() > 0:
        angle_valid = mask[:, angle_ids[:, 0]] & mask[:, angle_ids[:, 1]] & mask[:, angle_ids[:, 2]]
        angle_vals = calc_angles(
            positions[:, angle_ids[:, 0], :],
            positions[:, angle_ids[:, 1], :],
            positions[:, angle_ids[:, 2], :],
            eps=eps,
        )
        angle_target = topology["angles"]["params"].to(device=positions.device, dtype=positions.dtype)
        angle_energy = torch.square(_angle_delta(angle_vals, angle_target)) * angle_valid.float()
        angle_sum = angle_energy.sum(dim=1)
        if normalize_energy:
            angle_sum = angle_sum / angle_valid.float().sum(dim=1).clamp_min(1.0)
        total = total + float(angle_const) * angle_sum

    dihedral_ids = topology["dihedrals"]["ids"].to(device=positions.device)
    if dihedral_ids.numel() > 0:
        dihedral_valid = (
            mask[:, dihedral_ids[:, 0]] & mask[:, dihedral_ids[:, 1]]
            & mask[:, dihedral_ids[:, 2]] & mask[:, dihedral_ids[:, 3]]
        )
        dihedral_vals = calc_dihedrals(
            positions[:, dihedral_ids[:, 0], :],
            positions[:, dihedral_ids[:, 1], :],
            positions[:, dihedral_ids[:, 2], :],
            positions[:, dihedral_ids[:, 3], :],
            eps=eps,
        )
        dihedral_target = topology["dihedrals"]["params"].to(device=positions.device, dtype=positions.dtype)
        dihedral_energy = torch.square(_angle_delta(dihedral_vals, dihedral_target)) * dihedral_valid.float()
        dihedral_sum = dihedral_energy.sum(dim=1)
        if normalize_energy:
            dihedral_sum = dihedral_sum / dihedral_valid.float().sum(dim=1).clamp_min(1.0)
        total = total + float(dihedral_const) * dihedral_sum

    nb_ids = topology["nb_cache"]["ids"].to(device=positions.device)
    if nb_ids.numel() > 0:
        nb_valid = _valid_pair_mask(mask, nb_ids)
        nb_dist = calc_distances(positions, nb_ids, eps=eps)
        clashes = (nb_dist < float(early_stopping_clash_thresh)) & nb_valid
        if early_stopping_clash_score is not None:
            clash_score = torch.mean(clashes.sum(dim=1).float(), dim=0)
            if clash_score <= float(early_stopping_clash_score):
                raise CAMinimizerEarlyStopping(clash_score.detach(), float(early_stopping_clash_score))
        nb_overlap = torch.clamp(float(nb_threshold) - nb_dist, min=0.0)
        nb_energy = _hinge_energy(nb_overlap, nb_form)
        nb_sum = (nb_energy * nb_valid.float()).sum(dim=1)
        if normalize_energy:
            if nb_normalize_by == "valid":
                nb_denom = nb_valid.float().sum(dim=1)
            elif nb_normalize_by == "active":
                nb_denom = ((nb_overlap > 0.0) & nb_valid).float().sum(dim=1)
            else:
                raise KeyError(nb_normalize_by)
            nb_sum = nb_sum / nb_denom.clamp_min(1.0)
        total = total + float(nb_const) * nb_sum

    segment_pair_ids = topology["segment_pairs"]["ids"].to(device=positions.device)
    if segment_pair_ids.numel() > 0 and float(segment_const) > 0:
        segment_ids = topology["segments"]["ids"].to(device=positions.device)
        first = segment_ids[segment_pair_ids[:, 0]]
        second = segment_ids[segment_pair_ids[:, 1]]
        segment_valid = _valid_segment_pair_mask(mask, segment_ids, segment_pair_ids)
        segment_dist = calc_segment_segment_distances(
            positions[:, first[:, 0], :],
            positions[:, first[:, 1], :],
            positions[:, second[:, 0], :],
            positions[:, second[:, 1], :],
            eps=eps,
        )
        segment_overlap = torch.clamp(float(segment_threshold) - segment_dist, min=0.0)
        segment_energy = _hinge_energy(segment_overlap, segment_form)
        segment_sum = (segment_energy * segment_valid.float()).sum(dim=1)
        if normalize_energy:
            if segment_normalize_by == "valid":
                segment_denom = segment_valid.float().sum(dim=1)
            elif segment_normalize_by == "active":
                segment_denom = ((segment_overlap > 0.0) & segment_valid).float().sum(dim=1)
            else:
                raise KeyError(segment_normalize_by)
            segment_sum = segment_sum / segment_denom.clamp_min(1.0)
        total = total + float(segment_const) * segment_sum

    return total.mean(dim=0) if normalize_energy else total.sum(dim=0)


def _set_initial_restraints(
    positions: torch.Tensor,
    topology: dict[str, Any],
    *,
    bond_target: float = 3.8,
    bond_target_mode: str = "initial_in_range_else_ignore",
    bond_init_range: tuple[float, float] | list[float] | None = (3.20, 4.60),
    eps: float = 1e-12,
) -> None:
    mask = topology["mask"].to(device=positions.device)
    bond_ids = topology["bonds"]["ids"].to(device=positions.device)
    if bond_ids.numel() > 0:
        init = calc_distances(positions.detach(), bond_ids, eps=eps)
        if bond_target_mode == "ideal":
            target = torch.full_like(init, float(bond_target))
            active = torch.ones_like(init, dtype=torch.bool)
        elif bond_target_mode == "initial":
            target = init
            active = torch.ones_like(init, dtype=torch.bool)
        elif bond_target_mode == "initial_in_range_else_ideal":
            target = torch.full_like(init, float(bond_target))
            active = torch.ones_like(init, dtype=torch.bool)
            if bond_init_range is not None:
                lo, hi = float(bond_init_range[0]), float(bond_init_range[1])
                use_init = (init >= lo) & (init <= hi)
                target = torch.where(use_init, init, target)
        elif bond_target_mode == "initial_in_range_else_ignore":
            target = init
            if bond_init_range is not None:
                lo, hi = float(bond_init_range[0]), float(bond_init_range[1])
                active = (init >= lo) & (init <= hi)
            else:
                active = torch.ones_like(init, dtype=torch.bool)
        else:
            raise KeyError(bond_target_mode)
        pair_valid = _valid_pair_mask(mask, bond_ids)
        active = active & pair_valid
        target = torch.where(pair_valid, target, torch.full_like(target, float(bond_target)))
        topology["bonds"]["params"] = target
        topology["bonds"]["mask"] = active

    angle_ids = topology["angles"]["ids"].to(device=positions.device)
    if angle_ids.numel() > 0:
        topology["angles"]["params"] = calc_angles(
            positions.detach()[:, angle_ids[:, 0], :],
            positions.detach()[:, angle_ids[:, 1], :],
            positions.detach()[:, angle_ids[:, 2], :],
            eps=eps,
        )
    else:
        topology["angles"]["params"] = positions.new_zeros((positions.shape[0], 0))

    dihedral_ids = topology["dihedrals"]["ids"].to(device=positions.device)
    if dihedral_ids.numel() > 0:
        topology["dihedrals"]["params"] = calc_dihedrals(
            positions.detach()[:, dihedral_ids[:, 0], :],
            positions.detach()[:, dihedral_ids[:, 1], :],
            positions.detach()[:, dihedral_ids[:, 2], :],
            positions.detach()[:, dihedral_ids[:, 3], :],
            eps=eps,
        )
    else:
        topology["dihedrals"]["params"] = positions.new_zeros((positions.shape[0], 0))


def _update_nb_cache(
    positions: torch.Tensor,
    topology: dict[str, Any],
    *,
    nb_centers_threshold: float,
    eps: float = 1e-12,
) -> None:
    all_ids = topology["nb_centers"]["ids"].to(device=positions.device)
    if all_ids.numel() == 0:
        topology["nb_cache"]["ids"] = all_ids
        return
    mask = topology["mask"].to(device=positions.device)
    distances = calc_distances(positions.detach(), all_ids, eps=eps)
    valid = _valid_pair_mask(mask, all_ids)
    close_in_any_frame = ((distances < float(nb_centers_threshold)) & valid).any(dim=0)
    selected = all_ids[close_in_any_frame]
    topology["nb_cache"]["ids"] = selected if selected.numel() > 0 else all_ids[:0]


def minimize(
    positions: torch.Tensor,
    topology: dict[str, Any],
    *,
    opt: str = "lbfgs",
    step_size: float = 1.0,
    steps: int = 30,
    max_iter: int = 10,
    history_size: int = 100,
    line_search_fn: str | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    nb_centers_threshold: float | None = None,
    nb_update_freq: int = 10,
    bond_init_range: tuple[float, float] | list[float] | None = (3.20, 4.60),
    bond_target: float = 3.8,
    bond_target_mode: str = "initial_in_range_else_ideal",
    energy_params: dict[str, Any] | None = None,
    gradient_clip: float | None = None,
    gradient_clip_mode: str = "value",
    max_step_displacement: float | None = None,
    rollback_on_energy_increase: bool = False,
    energy_increase_tolerance: float = 10.0,
    max_abs_coord: float | None = None,
    min_caca_bond: float | None = None,
    max_caca_bond: float | None = None,
    caca_bond_guard_tolerance: float = 0.0,
    eps: float = 1e-12,
    return_early_stopping: bool = False,
    verbose: int | bool = 1,
) -> torch.Tensor | tuple[torch.Tensor, bool]:
    """Run one optimisation stage on flattened CA positions."""
    if isinstance(verbose, bool):
        verbose = int(verbose)
    energy_params = dict(energy_params or {})
    nb_centers_threshold = float(
        topology.get("nb_centers_threshold", 10.0) if nb_centers_threshold is None else nb_centers_threshold
    )

    positions = torch.autograd.Variable(positions.detach().clone())
    positions.requires_grad = True
    _set_initial_restraints(
        positions,
        topology,
        bond_target=bond_target,
        bond_target_mode=bond_target_mode,
        bond_init_range=bond_init_range,
        eps=eps,
    )

    if opt == "sgd":
        optimizer = torch.optim.SGD([positions], lr=float(step_size), momentum=0.9)
    elif opt == "gd":
        optimizer = torch.optim.SGD([positions], lr=float(step_size), momentum=0.0)
    elif opt == "adam":
        optimizer = torch.optim.Adam([positions], betas=(float(beta1), float(beta2)), lr=float(step_size))
    elif opt == "lbfgs":
        resolved_line_search = None if line_search_fn in (None, "", "none", "None") else str(line_search_fn)
        optimizer = torch.optim.LBFGS(
            [positions],
            lr=float(step_size),
            max_iter=int(max_iter),
            max_eval=None,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=int(history_size),
            line_search_fn=resolved_line_search,
        )
    else:
        raise KeyError(opt)

    step_idx = 0
    initial_bond_min, initial_bond_max = _caca_bond_range_per_frame(positions, topology, eps=eps)
    frozen_frame_mask = torch.zeros(positions.shape[0], dtype=torch.bool, device=positions.device)
    frozen_positions = positions.detach().clone()

    def closure():
        if bool(frozen_frame_mask.any()):
            with torch.no_grad():
                positions[frozen_frame_mask] = frozen_positions[frozen_frame_mask]
        optimizer.zero_grad()
        energy = calc_energy(
            positions=positions,
            topology=topology,
            eps=eps,
            verbose=verbose > 1,
            **energy_params,
        )
        energy.backward()
        if gradient_clip is not None:
            if gradient_clip_mode == "norm":
                torch.nn.utils.clip_grad_norm_(positions, float(gradient_clip))
            elif gradient_clip_mode == "value":
                torch.nn.utils.clip_grad_value_(positions, float(gradient_clip))
            else:
                raise KeyError(gradient_clip_mode)
        return energy

    early_stopped = False
    guard_energy_params = dict(energy_params)
    guard_energy_params["early_stopping_clash_score"] = None
    for step_idx in range(int(steps)):
        t0 = time.perf_counter()
        if step_idx % max(int(nb_update_freq), 1) == 0:
            _update_nb_cache(
                positions,
                topology,
                nb_centers_threshold=nb_centers_threshold,
                eps=eps,
            )
        prev_positions = positions.detach().clone()
        prev_energy = None
        if rollback_on_energy_increase:
            try:
                prev_energy = calc_energy(positions=positions, topology=topology, eps=eps, **guard_energy_params).detach()
            except Exception:
                prev_energy = None
        try:
            optimizer.step(closure)
        except CAMinimizerEarlyStopping:
            early_stopped = True
            if verbose:
                print("[ca_minimiser] early stopping: nonbonded clash score already below threshold")
            break
        with torch.no_grad():
            if max_step_displacement is not None and float(max_step_displacement) > 0:
                delta = positions - prev_positions
                delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                scale = torch.clamp(float(max_step_displacement) / delta_norm.clamp_min(eps), max=1.0)
                positions.copy_(prev_positions + delta * scale)

            if bool(frozen_frame_mask.any()):
                positions[frozen_frame_mask] = frozen_positions[frozen_frame_mask]

            frame_invalid = ~torch.isfinite(positions).all(dim=(1, 2))
            frame_reasons = []
            if bool(frame_invalid.any()):
                frame_reasons.append(f"nonfinite_positions={int(frame_invalid.sum().item())}")
            if max_abs_coord is not None:
                frame_max_abs = torch.amax(torch.abs(positions), dim=(1, 2))
                bad_abs = frame_max_abs > float(max_abs_coord)
                if bool(bad_abs.any()):
                    frame_invalid |= bad_abs
                    frame_reasons.append(
                        f"max_abs_coord={float(frame_max_abs[bad_abs].max().item()):.3g}>{float(max_abs_coord):.3g}"
                    )
            if min_caca_bond is not None or max_caca_bond is not None:
                bond_min, bond_max = _caca_bond_range_per_frame(positions, topology, eps=eps)
                guard_tol = max(float(caca_bond_guard_tolerance), 0.0)
                if max_caca_bond is not None:
                    max_allowed = torch.maximum(
                        torch.full_like(initial_bond_max, float(max_caca_bond)),
                        initial_bond_max,
                    ) + guard_tol
                    bad_max = bond_max > max_allowed
                    if bool(bad_max.any()):
                        frame_invalid |= bad_max
                        frame_reasons.append(
                            f"max_caca_bond={float(bond_max[bad_max].max().item()):.3g}>"
                            f"{float(max_allowed[bad_max].max().item()):.3g}"
                        )
                if min_caca_bond is not None:
                    min_allowed = torch.minimum(
                        torch.full_like(initial_bond_min, float(min_caca_bond)),
                        initial_bond_min,
                    ) - guard_tol
                    bad_min = bond_min < min_allowed
                    if bool(bad_min.any()):
                        frame_invalid |= bad_min
                        frame_reasons.append(
                            f"min_caca_bond={float(bond_min[bad_min].min().item()):.3g}<"
                            f"{float(min_allowed[bad_min].min().item()):.3g}"
                        )

            if bool(frame_invalid.any()):
                positions[frame_invalid] = prev_positions[frame_invalid]
                frozen_positions[frame_invalid] = prev_positions[frame_invalid]
                frozen_frame_mask |= frame_invalid
                if verbose:
                    reason = ", ".join(frame_reasons) if frame_reasons else "geometry_guard"
                    print(
                        f"[ca_minimiser] froze {int(frame_invalid.sum().item())}/"
                        f"{positions.shape[0]} frame(s) at step {step_idx}: {reason}"
                    )
                if bool(frozen_frame_mask.all()):
                    break

            rollback_reasons = []
            invalid = False
            if rollback_on_energy_increase and prev_energy is not None:
                try:
                    new_energy = calc_energy(positions=positions, topology=topology, eps=eps, **guard_energy_params).detach()
                    if not torch.isfinite(new_energy).all():
                        invalid = True
                        rollback_reasons.append("nonfinite_energy")
                    if torch.isfinite(prev_energy).all() and torch.isfinite(new_energy).all():
                        tol = max(float(energy_increase_tolerance), 1.0)
                        if bool(new_energy.item() > prev_energy.item() * tol + 1e-6):
                            invalid = True
                            rollback_reasons.append(
                                f"energy_increase={new_energy.item():.4g}>{tol:.4g}x{prev_energy.item():.4g}"
                            )
                except Exception as exc:
                    invalid = True
                    rollback_reasons.append(f"energy_guard_failed={type(exc).__name__}")
            if invalid:
                positions.copy_(prev_positions)
                if verbose:
                    reason = ", ".join(rollback_reasons) if rollback_reasons else "unknown"
                    print(f"[ca_minimiser] rollback at step {step_idx}: {reason}")
                break
        if verbose > 1:
            print(f"[ca_minimiser] step {step_idx} took {time.perf_counter() - t0:.4f}s")

    out = positions.detach()
    if return_early_stopping:
        return out, early_stopped
    return out


class CAMinimizer:
    """Small runner class matching the aSAM2 minimizer shape for CA traces."""

    def __init__(
        self,
        *,
        protocol: str = "mdcath",
        params_fp: str | Path | None = None,
        params: dict[str, Any] | None = None,
    ):
        if protocol not in {"mdcath", "custom"}:
            raise KeyError(protocol)
        if protocol == "custom" and params_fp is None and params is None:
            raise ValueError("Custom CA minimization requires params_fp or params.")
        if params_fp is None and protocol == "mdcath":
            params_fp = Path(__file__).resolve().parent / "params" / "mdcath_ca.yaml"
        self.params = load_params(params_fp, params)
        self.data_params = self.params.get("data", {"batch_size": 25})
        self.top_params = self.params.get("top", {})
        self.opt_ini_params = self.params.get("opt_ini")
        self.opt_params = self.params.get("opt", {})

    def run(
        self,
        ca_coords: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
        verbose: int | bool = 1,
    ) -> torch.Tensor:
        """Minimise a CA trajectory tensor with shape ``(..., L, 3)``."""
        if device is None:
            device = ca_coords.device
        device = torch.device(device)
        ca = ca_coords.to(device=device, dtype=torch.float32)
        flat, info = _as_flat_positions(ca)
        mask_flat = _prepare_mask(mask, flat.shape[0], flat.shape[1], device)
        batch_size = int(batch_size or self.data_params.get("batch_size", 25))

        out_batches = []
        for start in range(0, flat.shape[0], batch_size):
            batch = flat[start:start + batch_size]
            batch_mask = mask_flat[start:start + batch.shape[0]]
            topology = get_topology(
                batch.shape[1],
                mask=batch_mask,
                n_frames=batch.shape[0],
                device=device,
                **self.top_params,
            )
            if self.opt_ini_params is not None:
                batch, early_stopped = minimize(
                    batch,
                    topology,
                    return_early_stopping=True,
                    verbose=verbose,
                    **self.opt_ini_params,
                )
            else:
                early_stopped = False
            if not early_stopped:
                batch = minimize(batch, topology, verbose=verbose, **self.opt_params)
            out_batches.append(batch)

        out = torch.cat(out_batches, dim=0)
        return _restore_shape(out, info).to(device=ca_coords.device, dtype=ca_coords.dtype)


def minimise_ca(
    ca_coords: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    protocol: str = "mdcath",
    params_fp: str | Path | None = None,
    params: dict[str, Any] | None = None,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
    verbose: int | bool = 1,
) -> torch.Tensor:
    """Convenience wrapper around :class:`CAMinimizer`."""
    runner = CAMinimizer(protocol=protocol, params_fp=params_fp, params=params)
    return runner.run(ca_coords, mask=mask, batch_size=batch_size, device=device, verbose=verbose)


minimize_ca = minimise_ca
