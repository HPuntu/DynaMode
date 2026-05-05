"""Unified Pancake inference interface.

This module is deliberately library-first.  It reuses the model/runtime builder
from ``src.evaluation2`` and the sampler from ``src.train`` so standalone
inference, notebooks, validation, and evaluation all pass through the same
runtime contracts.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    import mdtraj as md
except ModuleNotFoundError:  # Keep the module importable in minimal test envs.
    md = None

from disco.minimiser import minimise_ca


def _evaluation2():
    from src import evaluation2

    return evaluation2


def _features():
    from src.features import features

    return features


def run_inference(*args: Any, **kwargs: Any):
    """Lazy re-export of ``src.train.run_inference`` for compatibility."""
    from disco.train import run_inference as _run_inference

    return _run_inference(*args, **kwargs)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(cpu: bool = False, device: str | None = None, device_index: int = 0) -> torch.device:
    if device:
        out = torch.device(device)
    elif torch.cuda.is_available() and not cpu:
        out = torch.device(f"cuda:{int(device_index)}")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() and not cpu:
        out = torch.device("mps")
    else:
        out = torch.device("cpu")
    if out.type == "cuda":
        torch.cuda.set_device(out)
    return out


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def pad_tensor(tensor: torch.Tensor, target_len: int, dim: int = 0, value: float = 0.0) -> torch.Tensor:
    current_len = tensor.shape[dim]
    if current_len >= target_len:
        return tensor[:target_len]
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_len - current_len
    padding = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=dim)


def create_protein_torsion_mask(batch_size: int, seq_len: int, device: torch.device | str = "cpu") -> torch.Tensor:
    mask = torch.ones((batch_size, seq_len, 4), dtype=torch.bool, device=device)
    if seq_len > 0:
        mask[:, 0, 0] = False
        mask[:, 0, 1] = False
        mask[:, -1, 0] = False
        mask[:, -1, 1] = False
    return mask


def _require_mdtraj():
    if md is None:
        raise ModuleNotFoundError("mdtraj is required for PDB loading/export in src.inference.")
    return md


def compute_native_angle_features(pdb_obj: Any, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    md_mod = _require_mdtraj()
    n_res = pdb_obj.topology.n_residues
    try:
        _, phi = md_mod.compute_phi(pdb_obj)
        _, psi = md_mod.compute_psi(pdb_obj)
    except Exception:
        return None, None

    phi_pad = torch.zeros(n_res, dtype=torch.float32, device=device)
    psi_pad = torch.zeros(n_res, dtype=torch.float32, device=device)
    if phi.shape[1] > 0:
        phi_pad[1:1 + phi.shape[1]] = torch.from_numpy(phi[0]).to(device=device, dtype=torch.float32)
    if psi.shape[1] > 0:
        psi_pad[:psi.shape[1]] = torch.from_numpy(psi[0]).to(device=device, dtype=torch.float32)
    angles = torch.stack([
        torch.sin(phi_pad), torch.cos(phi_pad),
        torch.sin(psi_pad), torch.cos(psi_pad),
    ], dim=-1)
    torsion_mask = create_protein_torsion_mask(1, n_res, device=device).squeeze(0)
    return angles, torsion_mask


def resolve_checkpoint_dir(config: dict[str, Any]) -> str:
    if config.get("checkpoint_dir"):
        return str(config["checkpoint_dir"])
    if config.get("checkpoint_path"):
        return str(Path(config["checkpoint_path"]).resolve().parent)
    return str(Path.cwd())


def load_inference_config(
    config_path: str | None = None,
    *,
    checkpoint_dir: str | None = None,
    checkpoint_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and normalise an inference config using evaluation2 conventions."""
    evaluation2 = _evaluation2()
    config: dict[str, Any] = {}
    if config_path:
        config.update(evaluation2.flatten_yaml_config(config_path))
    elif checkpoint_dir:
        candidate = Path(checkpoint_dir) / "run_config.yaml"
        if candidate.exists():
            config.update(evaluation2.flatten_yaml_config(str(candidate)))

    config = evaluation2.coerce_config_types(config)
    if checkpoint_dir is not None:
        config["checkpoint_dir"] = str(checkpoint_dir)
    if checkpoint_path is not None:
        config["checkpoint_path"] = str(checkpoint_path)
    if overrides:
        config.update({k: v for k, v in overrides.items() if v is not None})
    config = evaluation2.coerce_config_types(config)

    if not config.get("checkpoint_path") and config.get("checkpoint_dir"):
        candidate = Path(config["checkpoint_dir"]) / "best_model.pt"
        if candidate.exists():
            config["checkpoint_path"] = str(candidate)
    if not config.get("checkpoint_path"):
        raise ValueError("checkpoint_path is required, either directly or via checkpoint_dir/best_model.pt.")

    config.setdefault("checkpoint_dir", resolve_checkpoint_dir(config))
    config.setdefault("coords_type", "ca")
    config.setdefault("window_size", 256)
    config.setdefault("top_k_freqs", config.get("k", 64))
    config.setdefault("use_DCT", not bool(config.get("no_dct", False)))
    config.setdefault("num_steps", 1000)
    config.setdefault("num_ode_steps", config.get("num_steps_sampler", config.get("num_ode_steps", 200)))
    config.setdefault("guidance_scale", 1.0)
    config.setdefault("batch_size", 4)
    config.setdefault("crop_size", config.get("max_len", 1024))
    config.setdefault("temperature", config.get("t", 300.0))
    config.setdefault("include_angles", False)

    if config.get("representation") is None:
        config["representation"] = "displacement" if bool(config.get("displacement", True)) else "raw_coords"
    config["representation"] = evaluation2.canonical_representation(config["representation"])
    config["displacement"] = config["representation"] == "displacement"
    config["freq_normalization"] = evaluation2.canonical_freq_normalization(config.get("freq_normalization", "auto"))
    config["dc_residualization"] = evaluation2.canonical_dc_residualization(config.get("dc_residualization", "auto"))
    config["aniso_source"] = evaluation2.canonical_aniso_source(config.get("aniso_source", "auto"))
    return config


@dataclass
class InferenceSample:
    name: str
    native_coords: torch.Tensor
    res_type: torch.Tensor
    dssp: torch.Tensor
    mask: torch.Tensor
    top_ca: Any
    length: int
    temp: float
    native_angles: torch.Tensor | None = None
    torsion_mask: torch.Tensor | None = None
    pdb_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    name: str
    coords: torch.Tensor
    ca: torch.Tensor
    topology: Any
    sample: InferenceSample
    window_ranges: list[tuple[int, int]]
    raw_coords: torch.Tensor | None = None
    window_outputs: list[dict[str, torch.Tensor]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_mdtraj(self, align: bool = False) -> md.Trajectory:
        md_mod = _require_mdtraj()
        traj = md_mod.Trajectory(xyz=self.ca.detach().cpu().numpy() / 10.0, topology=self.topology)
        if align and len(traj) > 0:
            traj.center_coordinates()
            traj.superpose(traj, 0)
        return traj

    def save(self, output_prefix: str | os.PathLike[str], align: bool = True) -> None:
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        traj = self.to_mdtraj(align=align)
        if len(traj) > 0:
            traj[0].save_pdb(str(prefix.with_suffix(".pdb")))
        traj.save_xtc(str(prefix.with_suffix(".xtc")))


def coords_to_ca(coords: torch.Tensor, coords_type: str) -> torch.Tensor:
    if coords_type == "bb":
        if coords.shape[-1] != 12:
            raise ValueError(f"Backbone coords expected 12 channels, got {coords.shape[-1]}")
        return coords.reshape(*coords.shape[:-1], 4, 3)[..., 1, :]
    if coords.shape[-1] != 3:
        raise ValueError(f"CA coords expected 3 channels, got {coords.shape[-1]}")
    return coords


class Inference:
    """Unified high-level inference wrapper.

    Example:
        ``runner = Inference(config_path="run_config.yaml", checkpoint_path="best_model.pt")``
        ``result = runner.generate_from_pdb("input.pdb", frames=750, post_minimise=True)[0]``
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        config_path: str | None = None,
        checkpoint_dir: str | None = None,
        checkpoint_path: str | None = None,
        device: torch.device | str | None = None,
        cpu: bool = False,
        device_index: int = 0,
        no_amp: bool = False,
        **overrides: Any,
    ):
        if config is None:
            config = load_inference_config(
                config_path,
                checkpoint_dir=checkpoint_dir,
                checkpoint_path=checkpoint_path,
                overrides=overrides,
            )
        else:
            config = load_inference_config(
                config_path,
                checkpoint_dir=checkpoint_dir or config.get("checkpoint_dir"),
                checkpoint_path=checkpoint_path or config.get("checkpoint_path"),
                overrides={**config, **overrides},
            )

        self.config = config
        self.device = select_device(cpu=cpu, device=str(device) if device is not None else None, device_index=device_index)
        self.runtime = _evaluation2().build_model_stack(self.config, self.device)
        self.dtype_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported() and not no_amp
            else contextlib.nullcontext()
        )
        features = _features()
        self.featurizer = features.Featurizer(aligner=features.Aligner(), device=self.device, max_alignment_iters=2)

    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir: str, **kwargs: Any) -> "Inference":
        return cls(checkpoint_dir=checkpoint_dir, **kwargs)

    @property
    def coords_type(self) -> str:
        return str(self.config.get("coords_type", "ca"))

    @property
    def window_size(self) -> int:
        return int(self.runtime["window_size"])

    def load_pdb(self, pdb_path: str | os.PathLike[str], *, temperature: float | None = None, crop_size: int | None = None) -> InferenceSample:
        pdb_path = str(pdb_path)
        md_mod = _require_mdtraj()
        pdb_obj = md_mod.load_pdb(pdb_path)
        name = Path(pdb_path).stem
        native_coords, top_ca, length = self._extract_native_coords(pdb_obj)
        crop_size = int(crop_size or self.config.get("crop_size", 1024))
        if length > crop_size:
            raise ValueError(f"{name}: length {length} exceeds crop_size={crop_size}.")

        features = _features()
        res_type = self.featurizer._get_sequence_onehot(pdb_obj.topology).to(self.device, dtype=torch.float32)
        try:
            dssp = features.compute_native_dssp_onehot(pdb_obj, pdb_obj.topology.n_residues).to(self.device, dtype=torch.float32)
        except Exception:
            dssp = torch.zeros(length, len(features.DSSP_STATES), dtype=torch.float32, device=self.device)
        if res_type.shape[0] != length:
            res_type = res_type[:length]
        if dssp.shape[0] != length:
            dssp = dssp[:length]

        native_angles = None
        torsion_mask = None
        if bool(self.config.get("include_angles", False)):
            native_angles, torsion_mask = compute_native_angle_features(pdb_obj, self.device)
            if native_angles is not None:
                native_angles = native_angles[:length]
            if torsion_mask is not None:
                torsion_mask = torsion_mask[:length]

        return InferenceSample(
            name=name,
            native_coords=native_coords,
            res_type=res_type,
            dssp=dssp,
            mask=torch.ones(length, dtype=torch.bool, device=self.device),
            top_ca=top_ca,
            length=length,
            temp=float(temperature if temperature is not None else self.config.get("temperature", 300.0)),
            native_angles=native_angles,
            torsion_mask=torsion_mask,
            pdb_path=pdb_path,
        )

    def load_pdbs(self, path_or_paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]], *, temperature: float | None = None) -> list[InferenceSample]:
        paths = self._resolve_pdb_paths(path_or_paths)
        return [self.load_pdb(path, temperature=temperature) for path in paths]

    def _extract_native_coords(self, pdb_obj: Any) -> tuple[torch.Tensor, Any, int]:
        topology = pdb_obj.topology
        if self.coords_type == "ca":
            ca_indices = topology.select("name CA")
            if len(ca_indices) == 0:
                raise ValueError("No CA atoms found in PDB.")
            coords = torch.tensor(pdb_obj.xyz[0, ca_indices, :], dtype=torch.float32, device=self.device) * 10.0
            coords = coords - coords.mean(dim=0, keepdim=True)
            return coords, topology.subset(ca_indices), len(ca_indices)

        if self.coords_type != "bb":
            raise ValueError(f"Unsupported coords_type={self.coords_type!r}; expected 'ca' or 'bb'.")
        ordered_indices = []
        ca_indices = []
        for residue in topology.residues:
            atom_map = {atom.name: atom.index for atom in residue.atoms}
            oxygen = "O" if "O" in atom_map else "OXT" if "OXT" in atom_map else None
            required = ["N", "CA", "C"]
            if not all(name in atom_map for name in required) or oxygen is None:
                raise ValueError(f"Residue {residue} is missing N/CA/C/O backbone atoms.")
            ordered_indices.extend([atom_map["N"], atom_map["CA"], atom_map["C"], atom_map[oxygen]])
            ca_indices.append(atom_map["CA"])
        coords = torch.tensor(pdb_obj.xyz[0, ordered_indices, :], dtype=torch.float32, device=self.device) * 10.0
        coords = coords.reshape(len(ca_indices), 4, 3)
        coords = coords - coords[:, 1, :].mean(dim=0, keepdim=True).unsqueeze(1)
        coords = coords.reshape(len(ca_indices), 12)
        return coords, topology.subset(ca_indices), len(ca_indices)

    def _resolve_pdb_paths(self, path_or_paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]]) -> list[str]:
        if isinstance(path_or_paths, (str, os.PathLike)):
            path = Path(path_or_paths)
            if path.is_dir():
                return sorted(str(p) for p in path.glob("*.pdb"))
            if path.exists():
                return [str(path)]
            matches = sorted(glob.glob(str(path_or_paths)))
            if matches:
                return matches
            raise FileNotFoundError(str(path_or_paths))
        paths = [str(Path(p)) for p in path_or_paths]
        if not paths:
            raise FileNotFoundError("No PDB paths provided.")
        return paths

    def infer_window(
        self,
        samples: list[InferenceSample],
        *,
        conditioning_coords: list[torch.Tensor] | None = None,
        conditioning_angles: list[torch.Tensor | None] | None = None,
        win_pos_values: torch.Tensor | None = None,
        num_ode_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        if not samples:
            return []
        conditioning_coords = conditioning_coords or [s.native_coords for s in samples]
        conditioning_angles = conditioning_angles or [s.native_angles for s in samples]
        lengths = [int(s.length) for s in samples]
        l_max = max(lengths)
        batch_size = len(samples)

        def pad_l(t: torch.Tensor) -> torch.Tensor:
            gap = l_max - t.shape[0]
            return t if gap == 0 else torch.cat([t, t.new_zeros((gap,) + t.shape[1:])], dim=0)

        native_coords = torch.stack([pad_l(c) for c in conditioning_coords]).to(self.device, dtype=torch.float32)
        mask = torch.zeros(batch_size, l_max, dtype=torch.bool, device=self.device)
        for b, length in enumerate(lengths):
            mask[b, :length] = True

        angle_channels = int(self.runtime["angle_channels"])
        native_angles = None
        torsion_mask = None
        if angle_channels > 0:
            angle_rows = []
            torsion_rows = []
            for sample, angles in zip(samples, conditioning_angles):
                if angles is None:
                    angle_rows.append(torch.zeros(l_max, angle_channels, dtype=torch.float32, device=self.device))
                    torsion_rows.append(torch.zeros(l_max, angle_channels, dtype=torch.bool, device=self.device))
                else:
                    angle_rows.append(pad_l(angles.to(self.device, dtype=torch.float32)))
                    torsion = sample.torsion_mask
                    if torsion is None:
                        torsion = torch.ones(sample.length, angle_channels, dtype=torch.bool, device=self.device)
                    torsion_rows.append(pad_l(torsion.to(self.device, dtype=torch.bool)))
            native_angles = torch.stack(angle_rows)
            torsion_mask = torch.stack(torsion_rows)

        res_type = torch.stack([pad_l(s.res_type.to(self.device, dtype=torch.float32)) for s in samples])
        dssp = torch.stack([pad_l(s.dssp.to(self.device, dtype=torch.float32)) for s in samples])
        temps = torch.tensor([float(s.temp) for s in samples], dtype=torch.float32, device=self.device)
        if win_pos_values is None:
            win_pos_values = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        else:
            win_pos_values = win_pos_values.to(self.device, dtype=torch.float32)

        coord_channels = int(self.runtime["coord_channels"])
        channels = int(self.runtime["total_channels"])
        complex_mult = 1 if bool(self.runtime["is_dct"]) else 2
        d = (
            self.runtime["window_size"] * coord_channels
            if bool(self.runtime.get("is_time_domain", False))
            else channels * int(self.runtime["top_k_freqs"]) * complex_mult
        )

        with torch.inference_mode(), self.dtype_ctx:
            pred = run_inference(
                self.runtime["model"],
                self.runtime["diffusion"],
                self.runtime["transform_engine"],
                (batch_size, l_max, d),
                native_coords,
                native_angles,
                temps,
                int(self.runtime["window_size"]),
                mask=mask,
                torsion_mask=torsion_mask,
                device=self.device,
                guidance_scale=float(guidance_scale if guidance_scale is not None else self.config.get("guidance_scale", 1.0)),
                num_ode_steps=int(num_ode_steps if num_ode_steps is not None else self.config.get("num_ode_steps", 200)),
                displacement=bool(self.config.get("displacement", True)),
                representation=self.runtime.get("representation"),
                win_pos=win_pos_values,
                rmsf_prior=None,
                res_type=res_type,
                dssp=dssp,
            )

        sync_device(self.device)
        out = []
        for b, length in enumerate(lengths):
            row = {
                "coords": pred["coords"][b, :, :length, :].detach().float(),
                "spectral": pred.get("spectral", torch.empty(0, device=self.device))[b:b + 1].detach().float()
                if pred.get("spectral") is not None else None,
            }
            if pred.get("angles") is not None:
                row["angles"] = pred["angles"][b, :, :length, :].detach().float()
            out.append(row)
        return out

    def generate(
        self,
        samples: list[InferenceSample],
        *,
        frames: int | None = None,
        ns: int | None = None,
        num_windows: int | None = None,
        chain: bool | None = None,
        batch_size: int | None = None,
        post_minimise: bool | None = None,
        post_minimize: bool | None = None,
        minimise_stage: str = "trajectory",
        minimise_batch_size: int | None = None,
        minimise_params_fp: str | None = None,
        num_ode_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
    ) -> list[InferenceResult]:
        if seed is not None:
            seed_everything(int(seed))
        if frames is None and ns is not None:
            frames = int(ns)
        if frames is None:
            frames = int(num_windows or 1) * self.window_size
        frames = int(frames)
        if frames <= 0:
            raise ValueError(f"frames/ns must be positive, got {frames}")
        if num_windows is None:
            num_windows = int(math.ceil(frames / self.window_size))
        num_windows = int(num_windows)
        chain = bool(num_windows > 1) if chain is None else bool(chain)
        post_minimise = bool(post_minimise if post_minimise is not None else post_minimize if post_minimize is not None else False)
        if minimise_stage not in {"window", "trajectory"}:
            raise ValueError("minimise_stage must be 'window' or 'trajectory'.")

        batch_size = int(batch_size or self.config.get("batch_size", 4))
        results_by_sample: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(samples))}
        raw_by_sample: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(samples))}
        window_outputs_by_sample: dict[int, list[dict[str, torch.Tensor]]] = {i: [] for i in range(len(samples))}
        conditioning_coords = [s.native_coords for s in samples]
        conditioning_angles = [s.native_angles for s in samples]
        ranges: list[tuple[int, int]] = []

        for window_idx in range(num_windows):
            start_frame = window_idx * self.window_size
            end_frame = start_frame + self.window_size
            ranges.append((start_frame, end_frame))
            denom = max(frames - 1, 1)
            win_pos_all = torch.full(
                (len(samples),),
                float(start_frame) / float(denom),
                dtype=torch.float32,
                device=self.device,
            )
            next_conditioning_coords = list(conditioning_coords)
            next_conditioning_angles = list(conditioning_angles)
            for start in range(0, len(samples), batch_size):
                end = min(start + batch_size, len(samples))
                batch_samples = samples[start:end]
                batch_outputs = self.infer_window(
                    batch_samples,
                    conditioning_coords=conditioning_coords[start:end],
                    conditioning_angles=conditioning_angles[start:end],
                    win_pos_values=win_pos_all[start:end],
                    num_ode_steps=num_ode_steps,
                    guidance_scale=guidance_scale,
                )
                for local_idx, out in enumerate(batch_outputs):
                    sample_idx = start + local_idx
                    coords = out["coords"]
                    raw_by_sample[sample_idx].append(coords.detach().cpu())
                    if post_minimise and minimise_stage == "window":
                        coords = self._minimise_coords(coords, batch_size=minimise_batch_size, params_fp=minimise_params_fp)
                    results_by_sample[sample_idx].append(coords.detach().cpu())
                    window_outputs_by_sample[sample_idx].append({
                        k: v.detach().cpu() for k, v in out.items() if torch.is_tensor(v)
                    })
                    if chain:
                        next_conditioning_coords[sample_idx] = coords[-1].detach().to(self.device)
                        if out.get("angles") is not None:
                            next_conditioning_angles[sample_idx] = out["angles"][-1].detach().to(self.device)
            if chain:
                conditioning_coords = next_conditioning_coords
                conditioning_angles = next_conditioning_angles

        results = []
        for idx, sample in enumerate(samples):
            raw = torch.cat(raw_by_sample[idx], dim=0)[:frames]
            coords = torch.cat(results_by_sample[idx], dim=0)[:frames]
            if post_minimise and minimise_stage == "trajectory":
                coords = self._minimise_coords(coords.to(self.device), batch_size=minimise_batch_size, params_fp=minimise_params_fp).cpu()
            ca = coords_to_ca(coords, self.coords_type)
            results.append(InferenceResult(
                name=sample.name,
                coords=coords,
                ca=ca,
                topology=sample.top_ca,
                sample=sample,
                window_ranges=ranges,
                raw_coords=raw,
                window_outputs=window_outputs_by_sample[idx],
                metadata={
                    "frames_requested": frames,
                    "num_windows": num_windows,
                    "chain": chain,
                    "post_minimise": post_minimise,
                    "minimise_stage": minimise_stage if post_minimise else None,
                },
            ))
        return results

    def generate_from_pdb(
        self,
        path_or_paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
        **kwargs: Any,
    ) -> list[InferenceResult]:
        temperature = kwargs.pop("temperature", None)
        samples = self.load_pdbs(path_or_paths, temperature=temperature)
        return self.generate(samples, **kwargs)

    def _minimise_coords(self, coords: torch.Tensor, *, batch_size: int | None = None, params_fp: str | None = None) -> torch.Tensor:
        coords = coords.to(self.device, dtype=torch.float32)
        if self.coords_type == "ca":
            return minimise_ca(
                coords,
                batch_size=batch_size,
                device=self.device,
                params_fp=params_fp,
                verbose=0,
            )
        ca = coords_to_ca(coords, self.coords_type)
        ca_min = minimise_ca(ca, batch_size=batch_size, device=self.device, params_fp=params_fp, verbose=0)
        out = coords.clone()
        out_view = out.reshape(*out.shape[:-1], 4, 3)
        out_view[..., 1, :] = ca_min
        return out_view.reshape_as(out)


def export_trajectory(coords: torch.Tensor, topology: Any, output_prefix: str = "denoised", align: bool = True) -> None:
    md_mod = _require_mdtraj()
    ca = coords_to_ca(coords.detach().cpu(), "bb" if coords.shape[-1] == 12 else "ca")
    traj = md_mod.Trajectory(xyz=ca.numpy() / 10.0, topology=topology)
    if align and len(traj) > 0:
        traj.center_coordinates()
        traj.superpose(traj, 0)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if len(traj) > 0:
        traj[0].save_pdb(str(prefix.with_suffix(".pdb")))
    traj.save_xtc(str(prefix.with_suffix(".xtc")))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Pancake inference from PDB input(s).")
    parser.add_argument("-f", "--input", required=True, help="PDB file, directory, or glob.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("-t", "--temperature", type=float, default=None)
    parser.add_argument("--frames", type=int, default=None, help="Generated frames to keep. One frame is treated as 1 ns for --ns.")
    parser.add_argument("--ns", type=int, default=None, help="Alias for --frames, e.g. 750 gives three 256-frame windows trimmed to 750.")
    parser.add_argument("--num_windows", type=int, default=None)
    parser.add_argument("--chain", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--coords_type", choices=["ca", "bb"], default=None)
    parser.add_argument("--include_angles", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num_ode_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--post_minimise", "--post_minimize", action="store_true")
    parser.add_argument("--minimise_stage", "--minimize_stage", choices=["window", "trajectory"], default="trajectory")
    parser.add_argument("--minimise_batch_size", "--minimize_batch_size", type=int, default=None)
    parser.add_argument("--minimise_params_fp", "--minimize_params_fp", default=None)
    parser.add_argument("--no_export", action="store_true")
    parser.add_argument("--no_align_export", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    overrides = {
        key: value for key, value in vars(args).items()
        if value is not None and key not in {
            "input", "outdir", "config", "checkpoint_dir", "checkpoint_path",
            "frames", "ns", "num_windows", "chain", "seed", "device",
            "device_index", "cpu", "no_amp", "post_minimise", "minimise_stage",
            "minimise_batch_size", "minimise_params_fp", "no_export",
            "no_align_export",
        }
    }
    runner = Inference(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        device_index=args.device_index,
        cpu=args.cpu,
        no_amp=args.no_amp,
        **overrides,
    )
    start = time.perf_counter()
    results = runner.generate_from_pdb(
        args.input,
        temperature=args.temperature,
        frames=args.frames,
        ns=args.ns,
        num_windows=args.num_windows,
        chain=args.chain,
        post_minimise=args.post_minimise,
        minimise_stage=args.minimise_stage,
        minimise_batch_size=args.minimise_batch_size,
        minimise_params_fp=args.minimise_params_fp,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - start
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not args.no_export:
        for result in results:
            prefix = outdir / f"{result.name}_{int(result.sample.temp)}K_pancake"
            result.save(prefix, align=not args.no_align_export)
    print(
        f"Generated {sum(r.ca.shape[0] for r in results)} frames for {len(results)} target(s) "
        f"in {elapsed:.2f}s."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
