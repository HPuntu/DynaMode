from __future__ import annotations
import os
import tempfile
from dataclasses import dataclass
from typing import Any
import mdtraj as md
import numpy as np
import torch
import torch.nn.functional as F

from dynamode.dataloader.features import (
    Aligner,
    Featurizer,
    compute_native_dssp_onehot,
    dssp_to_onehot,
)


AA_TO_INT = Featurizer.AA_MAP
ATOM_MAPPING = {"N": 0, "CA": 1, "C": 2, "O": 3}
STATIC_FEATURE_KEYS = ("res_type", "native_angles", "torsion_mask", "dssp")
DYNAMIC_FEATURE_KEYS = ("bb_coords_native_aligned", "traj_angles")


@dataclass
class NativeFeatureBundle:
    topology: md.Topology
    n_res: int
    bb_indices: np.ndarray
    native_bb: torch.Tensor
    native_ca: torch.Tensor
    res_type: np.ndarray
    native_angles: np.ndarray
    torsion_mask: np.ndarray
    dssp: np.ndarray


def decode_text(value: Any) -> str:
    """Decode HDF5/Zarr scalar text payloads into a Python string."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_pdb_from_string(pdb_str: str) -> md.Trajectory:
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp.write(pdb_str.encode("utf-8"))
        tmp.flush()
        tmp_name = tmp.name
    try:
        return md.load_pdb(tmp_name)
    finally:
        os.unlink(tmp_name)


def get_backbone_indices(topology: md.Topology) -> np.ndarray:
    """Return per-residue backbone atom indices in N, CA, C, O order."""
    bb_indices = []
    for res in topology.residues:
        atom_dict = {atom.name: atom.index for atom in res.atoms}
        o_idx = atom_dict.get("O")
        if o_idx is None:
            o_idx = atom_dict.get("OXT", atom_dict.get("O1", atom_dict.get("OT1", -1)))
        bb_indices.append([
            atom_dict.get("N", -1),
            atom_dict.get("CA", -1),
            atom_dict.get("C", -1),
            o_idx,
        ])
    return np.asarray(bb_indices, dtype=np.int64)


def get_torsion_angles(traj: md.Trajectory, n_res: int) -> torch.Tensor:
    """Return sin/cos phi/psi features with terminal padding."""
    try:
        _, phi = md.compute_phi(traj)
        _, psi = md.compute_psi(traj)
    except Exception as exc:
        raise RuntimeError("Failed to compute phi/psi torsion angles") from exc

    phi_pad = torch.zeros((traj.n_frames, n_res), dtype=torch.float32)
    psi_pad = torch.zeros((traj.n_frames, n_res), dtype=torch.float32)
    phi_pad[:, 1:] = torch.from_numpy(phi).float()
    psi_pad[:, :-1] = torch.from_numpy(psi).float()
    return torch.stack(
        [
            torch.sin(phi_pad),
            torch.cos(phi_pad),
            torch.sin(psi_pad),
            torch.cos(psi_pad),
        ],
        dim=-1,
    )


def get_sequence_onehot(topology: md.Topology) -> torch.Tensor:
    indices = []
    for res in topology.residues:
        name = res.name if res.name in AA_TO_INT else "UNK"
        indices.append(AA_TO_INT[name])
    return F.one_hot(torch.tensor(indices, dtype=torch.long), num_classes=21).float()


def get_torsion_mask(n_res: int) -> torch.Tensor:
    mask = torch.ones((n_res, 4), dtype=torch.float32)
    mask[0, 0:2] = 0.0
    mask[-1, 2:4] = 0.0
    return mask


def build_native_feature_bundle(
    native_traj: md.Trajectory,
    raw_dssp: Any | None = None,
) -> NativeFeatureBundle:
    topology = native_traj.topology
    n_res = topology.n_residues
    bb_indices = get_backbone_indices(topology)
    if (bb_indices == -1).any():
        raise ValueError("Native structure is missing one or more backbone atoms")

    native_angstrom = torch.tensor(native_traj.xyz, dtype=torch.float32) * 10.0
    native_bb = native_angstrom[:, bb_indices, :].float().squeeze(0)
    native_ca = native_bb[:, 1, :]

    if raw_dssp is not None:
        dssp = dssp_to_onehot(raw_dssp, n_res=n_res).numpy()
    else:
        dssp = compute_native_dssp_onehot(native_traj, n_res).numpy()

    return NativeFeatureBundle(
        topology=topology,
        n_res=n_res,
        bb_indices=bb_indices,
        native_bb=native_bb,
        native_ca=native_ca,
        res_type=get_sequence_onehot(topology).numpy(),
        native_angles=get_torsion_angles(native_traj, n_res).squeeze(0).numpy(),
        torsion_mask=get_torsion_mask(n_res).numpy(),
        dssp=dssp,
    )


def write_dataset(group, name: str, array: np.ndarray, force: bool = False, chunks=False) -> bool:
    if name in group:
        if not force:
            return False
        del group[name]
    group.require_dataset(
        name=name,
        shape=array.shape,
        data=array.astype(np.float32),
        dtype="float32",
        chunks=chunks,
    )
    return True


def write_text_dataset(group, name: str, text: str, force: bool = False) -> bool:
    if name in group:
        if not force:
            return False
        del group[name]
    group.require_dataset(name=name, shape=(), data=text, dtype=str, chunks=False)
    return True


def write_static_features(
    domain_group,
    pdb_str: str,
    features: NativeFeatureBundle,
    force: bool = False,
) -> None:
    write_text_dataset(domain_group, "pdbProteinAtoms", pdb_str, force=force)
    for name in STATIC_FEATURE_KEYS:
        write_dataset(domain_group, name, getattr(features, name), force=force, chunks=False)
    domain_group.attrs["atom_mapping"] = ATOM_MAPPING


def prepare_dynamic_recompute(domain_group, force: bool = False) -> None:
    if force and "native_bb_coords" in domain_group:
        del domain_group["native_bb_coords"]


def write_centered_native_backbone(domain_group, features: NativeFeatureBundle, core_idx) -> None:
    core_idx = torch.as_tensor(core_idx, dtype=torch.long)
    native_core_mean = features.native_ca[core_idx].mean(dim=0, keepdim=True)
    native_bb_coords = (features.native_bb - native_core_mean.unsqueeze(-2)).numpy()
    write_dataset(domain_group, "native_bb_coords", native_bb_coords, force=False, chunks=False)


def process_backbone_replica(
    domain_group,
    temp_name: str,
    rep_name: str,
    raw_coords_angstrom: np.ndarray,
    features: NativeFeatureBundle,
    aligner: Aligner,
    device: torch.device,
    window_size: int = 256,
    force: bool = False,
) -> str:
    temp_group = domain_group.require_group(str(temp_name))
    rep_group = temp_group.require_group(str(rep_name))

    rep_done = all(name in rep_group for name in DYNAMIC_FEATURE_KEYS)
    native_done = "native_bb_coords" in domain_group
    if rep_done and native_done and not force:
        return "skipped"

    if force:
        for name in DYNAMIC_FEATURE_KEYS:
            if name in rep_group:
                del rep_group[name]

    raw_coords_angstrom = np.asarray(raw_coords_angstrom)
    if raw_coords_angstrom.ndim != 3 or raw_coords_angstrom.shape[-1] != 3:
        raise ValueError(f"Expected coords with shape (T, N, 3), got {raw_coords_angstrom.shape}")

    t_frames = raw_coords_angstrom.shape[0]
    full_traj = md.Trajectory(raw_coords_angstrom / 10.0, features.topology)
    traj_angles = get_torsion_angles(full_traj, features.n_res).numpy()

    bb_coords_angstrom = torch.tensor(
        raw_coords_angstrom[:, features.bb_indices, :],
        dtype=torch.float32,
    )
    ca_coords_angstrom = bb_coords_angstrom[:, :, 1, :]

    with torch.no_grad():
        _, core_idx, bb_aligned = aligner.align(
            ca_coords_angstrom.to(device),
            ref_coords=features.native_ca.to(device),
            core_fraction=0.5,
            max_iters=2,
            apply_coords=bb_coords_angstrom.to(device),
        )
        core_idx = core_idx.cpu()
        bb_aligned = bb_aligned.cpu().numpy()

    if "native_bb_coords" not in domain_group:
        write_centered_native_backbone(domain_group, features, core_idx)

    write_dataset(
        rep_group,
        "bb_coords_native_aligned",
        bb_aligned,
        force=force,
        chunks=(min(window_size, t_frames), features.n_res, 4, 3),
    )
    write_dataset(
        rep_group,
        "traj_angles",
        traj_angles,
        force=force,
        chunks=(min(window_size, t_frames), features.n_res, 4),
    )
    return "written"


def make_alignment_runtime() -> tuple[torch.device, Aligner]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, Aligner()


def shard_items(items: list[str], shard_idx: int, num_shards: int) -> list[str]:
    if num_shards <= 1:
        return items
    chunk = len(items) // num_shards + int(len(items) % num_shards != 0)
    start = shard_idx * chunk
    end = min(start + chunk, len(items))
    return items[start:end]


def infer_shard_args(args) -> tuple[int, int]:
    shard_idx = getattr(args, "shard_idx", None)
    num_shards = getattr(args, "num_shards", None)
    if shard_idx is None:
        shard_idx = getattr(args, "shard_idx_pos", None)
    if num_shards is None:
        num_shards = getattr(args, "num_shards_pos", None)
    if shard_idx is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        shard_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if num_shards is None and os.environ.get("SLURM_ARRAY_TASK_COUNT") is not None:
        num_shards = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    return shard_idx or 0, num_shards or 1
