"""CA-only geometry minimiser inspired by aSAM2's postprocessor.

aSAM2 minimises all-atom atom14 structures with force-field bonds, angles,
dihedrals and non-bonded terms.  Pancake often works with CA trajectories, so
this module keeps the same optimiser shape but replaces the energy with CA-only
terms:

* adjacent CA-CA bond restraints;
* CA bend-angle preservation;
* CA pseudo-dihedral preservation;
* non-bonded CA-CA clash repulsion for sequence separation >= ``min_sep``.

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
    "data": {"batch_size": 50},
    "top": {
        "min_sep": 2,
        "min_segment_sep": 2,
        "nb_centers_threshold": 10.0,
    },
    "opt_ini": {
        "opt": "adam",
        "step_size": 0.001,
        "steps": 50,
        "beta1": 0.5,
        "beta2": 0.9,
        "nb_update_freq": 10,
        "energy_params": {
            "bond_const": 10000.0,
            "angle_const": 100.0,
            "dihedral_const": 10.0,
            "nb_const": 100.0,
            "nb_form": "l2",
            "nb_threshold": 3.5,
            "segment_const": 0.0,
            "segment_form": "l2",
            "segment_threshold": 1.0,
            "early_stopping_clash_score": 0.7,
            "early_stopping_clash_thresh": 3.5,
        },
        "bond_init_range": [3.57, 4.11],
        "bond_target": 3.8,
        "bond_target_mode": "initial_in_range_else_ideal",
    },
    "opt": {
        "opt": "lbfgs",
        "step_size": 1.0,
        "steps": 30,
        "max_iter": 10,
        "history_size": 100,
        "nb_update_freq": 10,
        "nb_centers_threshold": 10.0,
        "energy_params": {
            "bond_const": 10000.0,
            "angle_const": 100.0,
            "dihedral_const": 10.0,
            "nb_const": 250.0,
            "nb_form": "l2",
            "nb_threshold": 3.5,
            "segment_const": 0.0,
            "segment_form": "l2",
            "segment_threshold": 1.0,
            "early_stopping_clash_score": 0.7,
            "early_stopping_clash_thresh": 3.5,
        },
        "bond_init_range": [3.57, 4.11],
        "bond_target": 3.8,
        "bond_target_mode": "initial_in_range_else_ideal",
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
    bond_const: float = 10000.0,
    angle_const: float = 100.0,
    dihedral_const: float = 10.0,
    nb_const: float = 250.0,
    nb_threshold: float = 3.5,
    nb_form: str = "l2",
    segment_const: float = 1000.0,
    segment_threshold: float = 1.0,
    segment_form: str = "l2",
    early_stopping_clash_score: float | None = None,
    early_stopping_clash_thresh: float = 3.5,
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
        bond_len = calc_distances(positions, bond_ids, eps=eps)
        bond_target = topology["bonds"]["params"].to(device=positions.device, dtype=positions.dtype)
        bond_energy = torch.square(bond_len - bond_target) * bond_valid.float()
        total = total + float(bond_const) * bond_energy.sum(dim=1)

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
        total = total + float(angle_const) * angle_energy.sum(dim=1)

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
        total = total + float(dihedral_const) * dihedral_energy.sum(dim=1)

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
        total = total + float(nb_const) * (nb_energy * nb_valid.float()).sum(dim=1)

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
        total = total + float(segment_const) * (segment_energy * segment_valid.float()).sum(dim=1)

    return total.sum(dim=0)


def _set_initial_restraints(
    positions: torch.Tensor,
    topology: dict[str, Any],
    *,
    bond_target: float = 3.8,
    bond_target_mode: str = "initial_in_range_else_ideal",
    bond_init_range: tuple[float, float] | list[float] | None = (3.57, 4.11),
    eps: float = 1e-12,
) -> None:
    mask = topology["mask"].to(device=positions.device)
    bond_ids = topology["bonds"]["ids"].to(device=positions.device)
    if bond_ids.numel() > 0:
        init = calc_distances(positions.detach(), bond_ids, eps=eps)
        if bond_target_mode == "ideal":
            target = torch.full_like(init, float(bond_target))
        elif bond_target_mode == "initial":
            target = init
        elif bond_target_mode == "initial_in_range_else_ideal":
            target = torch.full_like(init, float(bond_target))
            if bond_init_range is not None:
                lo, hi = float(bond_init_range[0]), float(bond_init_range[1])
                use_init = (init >= lo) & (init <= hi)
                target = torch.where(use_init, init, target)
        else:
            raise KeyError(bond_target_mode)
        target = torch.where(_valid_pair_mask(mask, bond_ids), target, torch.full_like(target, float(bond_target)))
        topology["bonds"]["params"] = target

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
    beta1: float = 0.9,
    beta2: float = 0.999,
    nb_centers_threshold: float | None = None,
    nb_update_freq: int = 10,
    bond_init_range: tuple[float, float] | list[float] | None = (3.57, 4.11),
    bond_target: float = 3.8,
    bond_target_mode: str = "initial_in_range_else_ideal",
    energy_params: dict[str, Any] | None = None,
    gradient_clip: float | None = None,
    gradient_clip_mode: str = "value",
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
        optimizer = torch.optim.LBFGS(
            [positions],
            lr=float(step_size),
            max_iter=int(max_iter),
            max_eval=None,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=int(history_size),
            line_search_fn=None,
        )
    else:
        raise KeyError(opt)

    step_idx = 0

    def closure():
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
    for step_idx in range(int(steps)):
        t0 = time.perf_counter()
        if step_idx % max(int(nb_update_freq), 1) == 0:
            _update_nb_cache(
                positions,
                topology,
                nb_centers_threshold=nb_centers_threshold,
                eps=eps,
            )
        try:
            optimizer.step(closure)
        except CAMinimizerEarlyStopping:
            early_stopped = True
            if verbose:
                print("[ca_minimiser] early stopping: nonbonded clash score already below threshold")
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
        self.data_params = self.params.get("data", {"batch_size": 50})
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
        batch_size = int(batch_size or self.data_params.get("batch_size", 50))

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
