"""Unified DynaMode inference interface.

This module is deliberately library-first. It owns the public sampling path used
by standalone inference, notebooks, training validation/test passes, and
evaluation.
"""

from __future__ import annotations

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

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    import mdtraj as md
except ModuleNotFoundError:  # Keep the module importable in minimal test envs.
    md = None

from dynamode.minimiser import minimise_ca
from dynamode.model.stack import build_model_stack
from dynamode.model.shake import shake_caca
from dynamode.model.wrapper import CFGModelWrapper
from dynamode.spectral.representation import CoordinateRepresentation


def _evaluation():
    from dynamode.eval import evaluation

    return evaluation


def _features():
    from dynamode.dataloader import features

    return features


def maybe_restore_dc(
    spectral_adapter: Any,
    x: torch.Tensor,
    dc_baseline: torch.Tensor | None,
    coord_channels: int,
) -> torch.Tensor:
    if (
        spectral_adapter is None
        or x is None
        or dc_baseline is None
        or not hasattr(spectral_adapter, "restore_dc")
    ):
        return x
    return spectral_adapter.restore_dc(x, dc_baseline, coord_channels=coord_channels)


def build_spectral_mask(
    mask: torch.Tensor,
    torsion_mask: torch.Tensor | None,
    top_k: int,
    is_dct: bool,
    *,
    coord_channels: int = 3,
    representation: CoordinateRepresentation | None = None,
) -> torch.Tensor:
    """Build the flattened spectral mask used by the sampler and validation."""
    if representation is not None:
        return representation.spectral_mask(mask, torsion_mask, top_k, is_dct)
    mask_coords = mask.unsqueeze(-1).expand(-1, -1, coord_channels)
    if torsion_mask is not None:
        feature_mask = torch.cat([mask_coords, torsion_mask], dim=-1)
    else:
        feature_mask = mask_coords

    if is_dct:
        full = feature_mask.unsqueeze(2).expand(-1, -1, top_k, -1)
        return full.reshape(feature_mask.shape[0], feature_mask.shape[1], -1)
    full = feature_mask.unsqueeze(2).unsqueeze(-1).expand(-1, -1, top_k, -1, 2)
    return full.reshape(feature_mask.shape[0], feature_mask.shape[1], -1)


def run_inference(
    model: torch.nn.Module,
    diffusion: Any,
    transform_engine: Any,
    shape: tuple[int, int, int],
    native_coords: torch.Tensor,
    native_angles: torch.Tensor | None,
    temps: torch.Tensor,
    window_size: int,
    mask: torch.Tensor | None = None,
    torsion_mask: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    guidance_scale: float = 1.0,
    num_ode_steps: int = 20,
    displacement: bool = True,
    representation: CoordinateRepresentation | None = None,
    win_pos: torch.Tensor | None = None,
    rmsf_prior: torch.Tensor | None = None,
    res_type: torch.Tensor | None = None,
    dssp: torch.Tensor | None = None,
    dc_baseline_per_res: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    """Sample a public spectral model and decode the result back to coordinates."""
    model.eval()
    real_model = model.module if hasattr(model, "module") else model
    inner = getattr(real_model, "model", real_model)
    is_dct = getattr(real_model, "is_dct", True)

    needs_prior = getattr(inner, "use_rmsf_prior_gain", False)
    if needs_prior and rmsf_prior is None:
        raise ValueError(
            "This model was configured to use an RMSF prior, but the batch did "
            "not include rmsf_prior. Provide rmsf_prior_path or disable the prior."
        )

    coord_channels = native_coords.shape[-1]
    representation = representation or CoordinateRepresentation(
        displacement=displacement,
        coord_channels=coord_channels,
    )
    repr_coord_channels = representation.model_coord_channels
    angle_channels = native_angles.shape[-1] if native_angles is not None else 0
    channels = repr_coord_channels + angle_channels

    _batch_size, _length, feature_dim = shape
    complex_mult = 1 if is_dct else 2
    top_k = feature_dim // (channels * complex_mult)

    if native_angles is not None:
        if torsion_mask is None and mask is not None:
            torsion_mask = mask.unsqueeze(-1).expand(-1, -1, angle_channels)
        elif torsion_mask is not None and torsion_mask.dim() == 2:
            torsion_mask = torsion_mask.unsqueeze(-1).expand(-1, -1, angle_channels)
    else:
        torsion_mask = None

    input_noise = diffusion.sample_initial_noise(shape, device=device)
    spec_mask = None
    if mask is not None:
        spec_mask = build_spectral_mask(
            mask,
            torsion_mask,
            top_k,
            is_dct,
            coord_channels=repr_coord_channels,
            representation=representation,
        )
        input_noise = input_noise * spec_mask

    norm_temps = torch.clamp((temps - 250.0) / 200.0, 0.0, 1.0)
    cfg_model = CFGModelWrapper(real_model, guidance_scale=guidance_scale)
    x_t = diffusion.denoise_ode(
        cfg_model,
        input_noise,
        native_coords,
        native_angles,
        norm_temps,
        mask,
        torsion_mask=torsion_mask,
        is_dct=is_dct,
        num_steps=num_ode_steps,
        win_pos=win_pos,
        rmsf_prior=rmsf_prior,
        res_type=res_type,
        dssp=dssp,
        feature_dim=None,
        spectral_mask=spec_mask,
    )

    dc_baseline = None
    if dc_baseline_per_res is not None:
        dc_baseline = dc_baseline_per_res.to(
            device=x_t.device,
            dtype=x_t.dtype,
        )[..., :repr_coord_channels]
    if hasattr(transform_engine, "lookup_dc_baselines") and dc_baseline is None:
        dc_baseline = transform_engine.lookup_dc_baselines(
            temps,
            mask,
            coord_channels=repr_coord_channels,
            device=x_t.device,
        )
    x_t = maybe_restore_dc(transform_engine, x_t, dc_baseline, repr_coord_channels)

    x_time = transform_engine.spectral_to_time(
        x_t,
        n_time_steps=window_size,
        n_channels=channels,
    )
    if mask is not None:
        x_time = x_time * mask.unsqueeze(1).unsqueeze(-1)
    coord_repr_time = x_time[..., :repr_coord_channels]
    coords_abs = representation.inverse(coord_repr_time, native_coords, mask=mask)

    if getattr(inner, "use_shake", False):
        if coords_abs.shape[-1] == 3:
            coords_abs = shake_caca(
                coords_abs,
                mask=mask,
                target=getattr(inner, "shake_target", 3.8),
                n_iter=getattr(inner, "shake_n_iter", 20),
            )
        elif coords_abs.shape[-1] == 12:
            ca_shaken = shake_caca(
                coords_abs[..., 3:6],
                mask=mask,
                target=getattr(inner, "shake_target", 3.8),
                n_iter=getattr(inner, "shake_n_iter", 20),
            )
            coords_abs = coords_abs.clone()
            coords_abs[..., 3:6] = ca_shaken

    pred_dict: dict[str, torch.Tensor | None] = {"coords": coords_abs, "spectral": x_t}
    if angle_channels > 0:
        pred_dict["angles"] = x_time[..., repr_coord_channels:]
        if torsion_mask is not None:
            pred_dict["angles"] = pred_dict["angles"] * torsion_mask.unsqueeze(1)
    return pred_dict


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
        raise ModuleNotFoundError("mdtraj is required for PDB loading/export in dynamode.inference.")
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
    """Load and normalise an inference config using public evaluation conventions."""
    evaluation = _evaluation()
    config: dict[str, Any] = {}
    if config_path:
        config.update(evaluation.flatten_yaml_config(config_path))
    elif checkpoint_dir:
        candidate = Path(checkpoint_dir) / "run_config.yaml"
        if candidate.exists():
            config.update(evaluation.flatten_yaml_config(str(candidate)))

    config = evaluation.coerce_config_types(config)
    if checkpoint_dir is not None:
        config["checkpoint_dir"] = str(checkpoint_dir)
    if checkpoint_path is not None:
        config["checkpoint_path"] = str(checkpoint_path)
    if overrides:
        config.update({k: v for k, v in overrides.items() if v is not None})
    config = evaluation.coerce_config_types(config)

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
    config["representation"] = evaluation.canonical_representation(config["representation"])
    config["displacement"] = config["representation"] == "displacement"
    config["freq_normalization"] = evaluation.canonical_freq_normalization(config.get("freq_normalization", "auto"))
    config["dc_residualization"] = evaluation.canonical_dc_residualization(config.get("dc_residualization", "auto"))
    config["aniso_source"] = evaluation.canonical_aniso_source(config.get("aniso_source", "auto"))
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
        self.runtime = build_model_stack(
            self.config,
            self.device,
            load_weights_path=self.config["checkpoint_path"],
            set_eval=True,
        )
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
        minimise_verbose: int | bool | None = None,
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
                        coords = self._minimise_coords(
                            coords,
                            batch_size=minimise_batch_size,
                            params_fp=minimise_params_fp,
                            verbose=minimise_verbose,
                        )
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
                coords = self._minimise_coords(
                    coords.to(self.device),
                    batch_size=minimise_batch_size,
                    params_fp=minimise_params_fp,
                    verbose=minimise_verbose,
                ).cpu()
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
                    "minimise_verbose": minimise_verbose,
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

    def _minimise_coords(
        self,
        coords: torch.Tensor,
        *,
        batch_size: int | None = None,
        params_fp: str | None = None,
        verbose: int | bool | None = None,
    ) -> torch.Tensor:
        coords = coords.to(self.device, dtype=torch.float32)
        verbose = 1 if verbose is None else verbose
        if self.coords_type == "ca":
            return minimise_ca(
                coords,
                batch_size=batch_size,
                device=self.device,
                params_fp=params_fp,
                verbose=verbose,
            )
        ca = coords_to_ca(coords, self.coords_type)
        ca_min = minimise_ca(ca, batch_size=batch_size, device=self.device, params_fp=params_fp, verbose=verbose)
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


INFERENCE_RUNTIME_KEYS = {
    "batch_size",
    "crop_size",
    "coords_type",
    "include_angles",
    "num_ode_steps",
    "guidance_scale",
}


def flatten_hydra_config(cfg: DictConfig | dict) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a Hydra config into model/runtime config and inference controls."""
    raw = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
    raw = raw or {}
    inference_cfg = dict(raw.get("inference") or {})
    model_config: dict[str, Any] = {}
    for key, value in raw.items():
        if key in {"hydra", "inference"}:
            continue
        if isinstance(value, dict):
            model_config.update(value)
        else:
            model_config[key] = value

    for key in INFERENCE_RUNTIME_KEYS:
        value = inference_cfg.get(key)
        if value is not None:
            model_config[key] = value
    return model_config, inference_cfg


def _required_inference_path(value: str | None, key: str) -> str:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Provide inference.{key}=... in the Hydra config or override.")
    return str(value)


@hydra.main(version_base=None, config_path="../../configs", config_name="spec_conv_displacement_ca_unit_var")
def main(cfg: DictConfig) -> None:
    """Hydra inference entrypoint.

    Example:
        python -m dynamode.inference inference.input=target.pdb inference.outdir=outputs
    """
    model_config, inference_cfg = flatten_hydra_config(cfg)
    input_path = _required_inference_path(inference_cfg.get("input"), "input")
    outdir = Path(_required_inference_path(inference_cfg.get("outdir"), "outdir"))

    checkpoint_dir = inference_cfg.get("checkpoint_dir", model_config.get("checkpoint_dir"))
    checkpoint_path = inference_cfg.get("checkpoint_path", model_config.get("checkpoint_path"))
    if checkpoint_dir is not None:
        model_config["checkpoint_dir"] = checkpoint_dir
    if checkpoint_path is not None:
        model_config["checkpoint_path"] = checkpoint_path

    runner = Inference(
        config=model_config,
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        device=inference_cfg.get("device"),
        device_index=int(inference_cfg.get("device_index", 0) or 0),
        cpu=bool(inference_cfg.get("cpu", False)),
        no_amp=bool(inference_cfg.get("no_amp", False)),
    )
    start = time.perf_counter()
    results = runner.generate_from_pdb(
        input_path,
        temperature=inference_cfg.get("temperature"),
        frames=inference_cfg.get("frames"),
        ns=inference_cfg.get("ns"),
        num_windows=inference_cfg.get("num_windows"),
        chain=inference_cfg.get("chain"),
        post_minimise=bool(inference_cfg.get("post_minimise", False)),
        minimise_stage=str(inference_cfg.get("minimise_stage", "trajectory")),
        minimise_batch_size=inference_cfg.get("minimise_batch_size"),
        minimise_params_fp=inference_cfg.get("minimise_params_fp"),
        minimise_verbose=inference_cfg.get("minimise_verbose"),
        num_ode_steps=model_config.get("num_ode_steps"),
        guidance_scale=model_config.get("guidance_scale"),
        seed=inference_cfg.get("seed"),
    )
    elapsed = time.perf_counter() - start
    outdir.mkdir(parents=True, exist_ok=True)
    if not bool(inference_cfg.get("no_export", False)):
        for result in results:
            prefix = outdir / f"{result.name}_{int(result.sample.temp)}K_dynamode"
            result.save(prefix, align=not bool(inference_cfg.get("no_align_export", False)))
    print(
        f"Generated {sum(r.ca.shape[0] for r in results)} frames for {len(results)} target(s) "
        f"in {elapsed:.2f}s."
    )


if __name__ == "__main__":
    main()
