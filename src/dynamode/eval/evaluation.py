'''
Unified test-set evaluation (mdCATH + ATLAS).

Implements the protocol described in evaluation.md: for each test protein
we generate n_repeats predicted trajectories (default 5), compute metrics
against the full pooled MD ground-truth ensemble, and aggregate them.
Most metrics are still averaged over per-repeat comparisons, but RMSF profile
correlation is handled specially to match the MarS-FM analysis code: the
reported per-target RMSF Pearson/Spearman are computed from the full generated
ensemble RMSF profile against the pooled MD reference RMSF profile.
Also computes leave-one-out oracle baselines (5 folds for mdCATH, 3 for ATLAS).

Based on the AlphaFlow / MDGen / MarS-FM protocols. The trajectory construction
follows MarS-FM (two 256-frame windows concatenated then trimmed to 500 for
mdCATH; single 256-frame window trimmed to 100 for ATLAS).

HPC-friendly: single-GPU or torchrun/srun-launched DDP (target-level sharding).
'''

from __future__ import annotations
import argparse
import contextlib
import csv
import json
import math
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pprint
import concurrent.futures
import numpy as np
import torch
import torch.distributed as dist
import yaml

from dynamode.dataloader.features import (
    DSSP_STATES,
    dssp_to_onehot,
)
from dynamode.inference import run_inference
from dynamode.model.stack import build_model_stack as build_shared_model_stack
from dynamode.model.wrapper import SUPPORTED_MODEL_TYPES
from dynamode.spectral.representation import (
    canonical_aniso_source,
    canonical_dc_residualization,
    canonical_freq_normalization,
    canonical_representation,
)


# Environment setup (matches src/test.py)
BACKEND = "nccl"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MDTRAJ_NUM_THREADS", "1")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# Protocol constants (MarS-FM / MDGen defaults)
MDCATH_TOTAL_FRAMES = 500
MDCATH_PRED_FRAMES = 500
MDCATH_WINDOW_START_FRAMES = (0, 256)
ATLAS_PRED_FRAMES = 100
ATLAS_TOTAL_FRAMES = 100


# Generic helpers
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_process():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    is_distributed = world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_distributed:
        kwargs = dict(
            backend=BACKEND,
            timeout=timedelta(seconds=3600),
            world_size=world_size,
            rank=rank,
        )
        if device.type == "cuda":
            kwargs["device_id"] = device
        dist.init_process_group(**kwargs)
    return rank, local_rank, world_size, is_distributed, device

def is_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0

def log(msg: str, verbose=True) -> None:
    if is_rank0() and verbose:
        print(msg, flush=True)

def maybe_barrier() -> None:
    if dist.is_initialized():
        dist.barrier()

def shard_items(items, rank: int, world_size: int):
    return items[rank::world_size]


def flatten_yaml_config(path: str) -> dict:
    raw = yaml.safe_load(open(path)) or {}
    config = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            config.update(value)
        else:
            config[key] = value
    return config

def coerce_config_types(config: dict) -> dict:
    float_keys = [
        "min_snr_gamma", "shift_value", "guidance_scale", "aniso_gamma",
        "amp_head_mlp_ratio", "amp_head_attn_dropout", "shake_target",
        "representation_length_min", "representation_length_max",
        "representation_length_residual_max", "caca_bond_target",
        "caca_bond_tolerance", "caca_bond_step",
    ]
    int_keys = [
        "batch_size", "num_workers", "top_k_freqs", "freq_hidden_size",
        "spectral_modes", "seq_embed_dim", "ss_embed_dim", "num_layers",
        "num_heads", "num_steps", "num_ode_steps", "max_domains", "atlas_stride",
        "n_repeats", "pairwise_rmsd_samples", "hist_bins", "cond_dim",
        "mdcath_total_frames", "mdcath_pred_frames", "atlas_pred_frames",
        "atlas_total_frames", "spectral_volume_hist_bins", "amp_head_context_modes",
        "amp_head_target_modes", "amp_head_d_model", "amp_head_depth",
        "amp_head_num_heads", "shake_n_iter", "caca_bond_n_iter",
    ]
    bool_keys = [
        "use_DCT", "displacement", "use_zarr", "include_angles",
        "use_seq_conditioning", "use_ss_conditioning",
        "conditioning_dropout", "use_hilbert_spatial", "use_hilbert_spatial_dct",
        "use_rmsf_prior_gain", "use_low_k_correction_head", "save_trajectories",
        "save_caca_pair_distributions", "normalize_caca_bonds",
        "amp_head_use_rmsf_prior", "use_shake", "compute_spectral_volume_metrics",
        "compute_spectral_metrics", "eval_mdcath", "eval_atlas",
        "noise_auto_shift",
    ]
    for key in float_keys:
        if key in config and config[key] is not None:
            if key == "shift_value" and isinstance(config[key], str) and config[key].strip().lower() == "auto":
                continue
            config[key] = float(config[key])
    for key in int_keys:
        if key in config and config[key] is not None:
            config[key] = int(config[key])
    for key in bool_keys:
        if key in config and config[key] is not None:
            config[key] = bool(config[key])
    if "low_k_correction_modes" in config and config["low_k_correction_modes"] is not None:
        value = config["low_k_correction_modes"]
        if isinstance(value, str):
            value = value.strip()
            if value.isdigit():
                value = int(value)
        config["low_k_correction_modes"] = value
    return config

def resolve_checkpoint_dir(config: dict) -> str:
    if config.get("checkpoint_dir"):
        return config["checkpoint_dir"]
    if config.get("checkpoint_path"):
        return os.path.dirname(config["checkpoint_path"])
    return os.getcwd()

def resolve_base_config_path(args: argparse.Namespace) -> str | None:
    if args.config is not None:
        return args.config
    if args.checkpoint_dir is None and args.checkpoint_path is None:
        return None
    checkpoint_dir = resolve_checkpoint_dir({
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_path": args.checkpoint_path,
    })
    candidate = os.path.join(checkpoint_dir, "run_config.yaml")
    if os.path.exists(candidate):
        return candidate
    return None

def read_id_set(path: str | None):
    if path is None:
        return None
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}

def merge_id_sets(*paths: str | None):
    '''Read one or more newline-delimited id files into a single filter set.'''
    merged = set()
    for path in paths:
        ids = read_id_set(path)
        if ids:
            merged.update(ids)
    return merged or None

def mean_or_nan(values) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=np.float64)))

def median_or_nan(values) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.median(np.asarray(vals, dtype=np.float64)))

def std_or_nan(values) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return float("nan")
    return float(np.std(np.asarray(vals, dtype=np.float64), ddof=1))


# Numerical / geometry helpers
# =============================================================================

def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return float("nan")
    x = x[valid] - x[valid].mean()
    y = y[valid] - y[valid].mean()
    denom = math.sqrt(float((x * x).sum() * (y * y).sum()))
    if denom <= 0:
        return float("nan")
    return float((x * y).sum() / denom)

def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return float("nan")
    xr = np.argsort(np.argsort(x[valid])).astype(np.float64)
    yr = np.argsort(np.argsort(y[valid])).astype(np.float64)
    return pearson_corr(xr, yr)

def kabsch_align_traj(traj: np.ndarray, ref_frame: np.ndarray) -> np.ndarray:
    '''Rigid-body superpose every frame of traj onto ref_frame using batched operations.'''
    # Center reference
    ref_c0 = ref_frame.mean(axis=0)
    ref_c = ref_frame - ref_c0
    
    # Center all trajectory frames (batched)
    traj_c0 = traj.mean(axis=1, keepdims=True)
    traj_c = traj - traj_c0
    
    # Cross-covariance matrix H: (T, 3, 3)
    H = np.einsum('tli,lj->tij', traj_c, ref_c)
    
    # Batched SVD
    U, _, Vt = np.linalg.svd(H)
    
    # Resolve reflection sign for each frame
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    
    # Construct rotation matrix R
    I = np.zeros((traj.shape[0], 3, 3), dtype=H.dtype)
    I[:, 0, 0] = 1.0; I[:, 1, 1] = 1.0; I[:, 2, 2] = d
    R = np.matmul(np.matmul(Vt.transpose(0, 2, 1), I), U.transpose(0, 2, 1))
    
    # Apply rotation and translate back
    return np.matmul(traj_c, R.transpose(0, 2, 1)) + ref_c0

def compute_rmsf(ca_coords: np.ndarray) -> np.ndarray:
    '''Per-residue RMSF from (T, L, 3) CA trajectory.'''
    mean = ca_coords.mean(axis=0, keepdims=True)
    sq = ((ca_coords - mean) ** 2).sum(axis=-1)
    return np.sqrt(sq.mean(axis=0) + 1e-8)

def estimate_pairwise_rmsd(
    ca_coords: np.ndarray, n_pairs: int, rng: np.random.Generator
) -> np.ndarray:
    '''Return a sample of pairwise-RMSD values (length = n_pairs or fewer).'''
    T = ca_coords.shape[0]
    if T < 2:
        return np.array([], dtype=np.float64)
    i = rng.integers(0, T, size=n_pairs)
    j = rng.integers(0, T, size=n_pairs)
    same = i == j
    while np.any(same):
        j[same] = rng.integers(0, T, size=int(same.sum()))
        same = i == j
    diff = ca_coords[i] - ca_coords[j]
    return np.sqrt((diff ** 2).sum(axis=-1).mean(axis=-1) + 1e-8)


# Distribution metrics (JSD / KL with binning) 
# ----------------------------
def _histogram_probs(
    samples_a: np.ndarray, samples_b: np.ndarray, bins: int, eps: float = 1e-5
):
    samples_a = np.asarray(samples_a, dtype=np.float64).reshape(-1)
    samples_b = np.asarray(samples_b, dtype=np.float64).reshape(-1)
    samples_a = samples_a[np.isfinite(samples_a)]
    samples_b = samples_b[np.isfinite(samples_b)]
    if samples_a.size == 0 or samples_b.size == 0:
        return None, None
    lo = min(samples_a.min(), samples_b.min())
    hi = max(samples_a.max(), samples_b.max())
    span = hi - lo
    scale = max(abs(lo), abs(hi), 1.0)

    # NumPy can reject a nominally positive range when it is too narrow to
    # support the requested number of finite-width bins at float64 precision.
    # In that regime these samples are effectively delta-distributed anyway, so
    # we fall back to a single shared bin instead of aborting the whole target.
    if not np.isfinite(span) or span <= np.finfo(np.float64).eps * scale * max(int(bins), 1):
        return np.ones(1, dtype=np.float64), np.ones(1, dtype=np.float64)

    try:
        ha, _ = np.histogram(samples_a, bins=bins, range=(lo, hi))
        hb, _ = np.histogram(samples_b, bins=bins, range=(lo, hi))
    except ValueError as e:
        if "Too many bins for data range" not in str(e):
            raise
        ha, _ = np.histogram(samples_a, bins=1, range=(lo, hi))
        hb, _ = np.histogram(samples_b, bins=1, range=(lo, hi))
    pa = ha.astype(np.float64) + eps
    pb = hb.astype(np.float64) + eps
    pa /= pa.sum()
    pb /= pb.sum()
    return pa, pb

def jsd_1d(p_samples: np.ndarray, q_samples: np.ndarray, bins: int = 100) -> float:
    from scipy.spatial.distance import jensenshannon
    pa, pb = _histogram_probs(p_samples, q_samples, bins=bins)
    if pa is None:
        return float("nan")
    return float(jensenshannon(pa, pb) ** 2)

def fwd_kl_1d(p_samples: np.ndarray, q_samples: np.ndarray, bins: int = 100) -> float:
    pa, pb = _histogram_probs(p_samples, q_samples, bins=bins)
    if pa is None:
        return float("nan")
    mask = pa > 0
    return float(np.sum(pa[mask] * (np.log(pa[mask]) - np.log(np.maximum(pb[mask], 1e-5)))))


# Wasserstein (atom-wise gaussian)
# ----------------------------------------
def _covariance(x: np.ndarray) -> np.ndarray:
    if x.shape[0] < 2:
        return np.eye(x.shape[1], dtype=np.float64) * 1e-8
    xc = x - x.mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / max(x.shape[0] - 1, 1)
    cov = 0.5 * (cov + cov.T)
    return cov + np.eye(cov.shape[0], dtype=np.float64) * 1e-8

def _gaussian_w2_sq(
    mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray
) -> tuple[float, float, float]:
    delta_sq = float(np.sum((mu1 - mu2) ** 2))
    
    # Compute sqrtm via eigendecomposition
    evals, evecs = np.linalg.eigh(cov1)
    sqrt_cov1 = evecs @ np.diag(np.sqrt(np.maximum(evals, 0))) @ evecs.T
    
    inner = sqrt_cov1 @ cov2 @ sqrt_cov1
    evals_in, evecs_in = np.linalg.eigh(inner)
    sqrt_inner = evecs_in @ np.diag(np.sqrt(np.maximum(evals_in, 0))) @ evecs_in.T
    
    var_term = float(np.trace(cov1 + cov2 - 2.0 * sqrt_inner))
    var_term = max(var_term, 0.0)
    return delta_sq + var_term, delta_sq, var_term

def compute_rmwd(pred_ca: np.ndarray, ref_ca: np.ndarray) -> tuple[float, float, float]:
    '''
    Root Mean Wasserstein Distance (per-atom Gaussian W2, then sqrt-mean).
    Fully vectorized over atoms to eliminate Python loops.
    '''
    T1, L, _ = pred_ca.shape
    T2 = ref_ca.shape[0]

    mu1 = pred_ca.mean(axis=0)
    mu2 = ref_ca.mean(axis=0)
    delta_sq = np.sum((mu1 - mu2) ** 2, axis=-1)  # (L,)

    xc1 = pred_ca - mu1
    xc2 = ref_ca - mu2
    cov1 = np.einsum('tli,tlj->lij', xc1, xc1) / max(T1 - 1, 1)
    cov2 = np.einsum('tli,tlj->lij', xc2, xc2) / max(T2 - 1, 1)

    eye = np.eye(3) * 1e-8
    cov1 += eye
    cov2 += eye

    # Batched Eigh for sqrtm
    evals1, evecs1 = np.linalg.eigh(cov1)
    evals1 = np.clip(evals1, a_min=0.0, a_max=None)
    sqrt_cov1 = np.einsum('lij,lj,lkj->lik', evecs1, np.sqrt(evals1), evecs1)

    inner = np.matmul(np.matmul(sqrt_cov1, cov2), sqrt_cov1)
    evals_in, evecs_in = np.linalg.eigh(inner)
    evals_in = np.clip(evals_in, a_min=0.0, a_max=None)
    sqrt_inner = np.einsum('lij,lj,lkj->lik', evecs_in, np.sqrt(evals_in), evecs_in)

    trace1 = np.trace(cov1, axis1=1, axis2=2)
    trace2 = np.trace(cov2, axis1=1, axis2=2)
    trace_in = np.trace(sqrt_inner, axis1=1, axis2=2)

    var_term = np.maximum(trace1 + trace2 - 2.0 * trace_in, 0.0)
    total_w2 = delta_sq + var_term

    return (
        float(math.sqrt(np.mean(total_w2))),
        float(math.sqrt(np.mean(delta_sq))),
        float(math.sqrt(np.mean(var_term))),
    )


# PCA W2 
# ------------------------------------------------------------------
def _top_pcs(x: torch.Tensor, n_components: int = 2):
    q = min(max(n_components, 2), x.shape[1], x.shape[0])
    mean = x.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(x, q=q, center=True)
    basis = v[:, :n_components]
    return mean.squeeze(0), basis

def _weighted_joint_pcs(ref_flat: torch.Tensor, pred_flat: torch.Tensor, n_components: int = 2):
    ref_w = 0.5 / max(ref_flat.shape[0], 1)
    pred_w = 0.5 / max(pred_flat.shape[0], 1)
    mean = ref_flat.mul(ref_w).sum(0) + pred_flat.mul(pred_w).sum(0)
    ref_c = ref_flat - mean
    pred_c = pred_flat - mean
    cov = ref_w * ref_c.T @ ref_c + pred_w * pred_c.T @ pred_c
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    basis = evecs[:, order[:n_components]]
    return mean, basis

def compute_pca_metrics(pred_ca: np.ndarray, ref_ca: np.ndarray) -> tuple[float, float, float]:
    pred_flat = torch.from_numpy(pred_ca.reshape(pred_ca.shape[0], -1).astype(np.float64))
    ref_flat = torch.from_numpy(ref_ca.reshape(ref_ca.shape[0], -1).astype(np.float64))
    ref_mean, ref_basis = _top_pcs(ref_flat)
    pred_mean, pred_basis = _top_pcs(pred_flat)
    joint_mean, joint_basis = _weighted_joint_pcs(ref_flat, pred_flat)

    ref_md = (ref_flat - ref_mean) @ ref_basis
    pred_md = (pred_flat - ref_mean) @ ref_basis
    ref_j = (ref_flat - joint_mean) @ joint_basis
    pred_j = (pred_flat - joint_mean) @ joint_basis

    pred_md_np = pred_md.float().cpu().numpy()
    ref_md_np = ref_md.float().cpu().numpy()

    md_w2, _, _ = _gaussian_w2_sq(
        pred_md.float().mean(0).cpu().numpy(), _covariance(pred_md_np),
        ref_md.float().mean(0).cpu().numpy(), _covariance(ref_md_np),
    )

    pred_j_np = pred_j.float().cpu().numpy()
    ref_j_np = ref_j.float().cpu().numpy()

    joint_w2, _, _ = _gaussian_w2_sq(
        pred_j.float().mean(0).cpu().numpy(), _covariance(pred_j_np),
        ref_j.float().mean(0).cpu().numpy(), _covariance(ref_j_np),
    )

    pc_sim = float(abs(torch.nn.functional.cosine_similarity(
        ref_basis[:, 0], pred_basis[:, 0], dim=0
    ).item()))
    return float(math.sqrt(md_w2)), float(math.sqrt(joint_w2)), pc_sim

def _pca_fit_transform_np(x: np.ndarray, n_components: int = 2):
    '''
    Lightweight PCA helper matching AlphaFlow-style projected coordinates.

    Returns the mean, component matrix of shape (D, K), and transformed
    coordinates (N, K). Also returns the leading explained variances so we
    can track whether the first PC is well separated from the second one.
    Uses an SVD backend to avoid introducing a hard sklearn dependency into the
    evaluation script.
    '''
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {x.shape}")
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    if xc.shape[0] == 0 or xc.shape[1] == 0:
        return (
            mean.squeeze(0),
            np.zeros((xc.shape[1], 0), dtype=np.float64),
            np.zeros((xc.shape[0], 0), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(max(int(n_components), 1), vt.shape[0], vt.shape[1])
    basis = vt[:k].T.copy()
    coords = xc @ basis
    if xc.shape[0] > 1:
        explained_variance = (s[:k] ** 2) / (xc.shape[0] - 1)
    else:
        explained_variance = np.zeros(k, dtype=np.float64)
    return mean.squeeze(0), basis, coords, explained_variance

def _pca_transform_np(x: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if basis.size == 0:
        return np.zeros((x.shape[0], 0), dtype=np.float64)
    return (x - mean[None, :]) @ basis

def _alphaflow_wasserstein(distmat: np.ndarray, p: int = 2) -> float:
    '''AlphaFlow-style assignment Wasserstein on a pairwise distance matrix.'''
    from scipy.optimize import linear_sum_assignment

    distmat = np.asarray(distmat, dtype=np.float64)
    if distmat.ndim != 2 or distmat.shape[0] == 0 or distmat.shape[1] == 0:
        return float("nan")
    cost = distmat ** p
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean() ** (1.0 / p))

def _pc1_pc2_var_ratio(explained_variance: np.ndarray) -> float:
    '''
    Return a simple eigengap diagnostic for the first two PCs.

    Values close to 1 indicate that the leading direction is poorly separated
    from the second mode, making signed first-PC cosine much less stable.
    '''
    explained_variance = np.asarray(explained_variance, dtype=np.float64)
    if explained_variance.size < 2:
        return float("nan")
    denom = max(float(explained_variance[1]), 1e-12)
    return float(explained_variance[0] / denom)

def compute_alphaflow_pca_metrics(
    pred_ca: np.ndarray,
    ref_ca: np.ndarray,
    rng: np.random.Generator,
    n_components: int = 2,
) -> tuple[float, float, float, float, float]:
    '''
    AlphaFlow-style PCA ensemble metrics.

    This follows the AlphaFlow ATLAS evaluation pattern more closely than the
    default Gaussian PCA-W2 metric above:
      - fit PCA on the reference ensemble alone, and on an equal-weighted
        ref/pred joint sample;
      - compare projected point clouds via assignment-based Wasserstein/EMD;
      - compute raw signed cosine similarity between the first independently
        fitted PCs of the reference and predicted ensembles.

    Coordinates in this evaluation pipeline are already in Angstroms, so we
    keep AlphaFlow's per-atom normalisation by sqrt(n_atoms) but omit the
    extra *10 factor used there for MDTraj's nanometre units.
    '''
    pred_flat = np.asarray(pred_ca, dtype=np.float64).reshape(pred_ca.shape[0], -1)
    ref_flat = np.asarray(ref_ca, dtype=np.float64).reshape(ref_ca.shape[0], -1)
    if pred_flat.shape[0] < 1 or ref_flat.shape[0] < 2:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    n_atoms = max(int(pred_ca.shape[1]), 1)
    n_pred = pred_flat.shape[0]
    rand1 = rng.integers(0, ref_flat.shape[0], size=n_pred)

    ref_mean, ref_basis, ref_coords, ref_explained_variance = _pca_fit_transform_np(
        ref_flat, n_components=n_components
    )
    _, pred_basis, _, pred_explained_variance = _pca_fit_transform_np(
        pred_flat, n_components=n_components
    )
    pred_coords_ref = _pca_transform_np(pred_flat, ref_mean, ref_basis)

    joint_input = np.concatenate([ref_flat[rand1], pred_flat], axis=0)
    joint_mean, joint_basis, _, _ = _pca_fit_transform_np(
        joint_input, n_components=n_components
    )
    ref_coords_joint = _pca_transform_np(ref_flat, joint_mean, joint_basis)
    pred_coords_joint = _pca_transform_np(pred_flat, joint_mean, joint_basis)

    def _pairwise_normed_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.size == 0 or b.size == 0:
            return np.zeros((0, 0), dtype=np.float64)
        diff = a[:, None, :] - b[None, :, :]
        return np.sqrt(np.square(diff).sum(axis=-1)) / math.sqrt(n_atoms)

    md_emd = _alphaflow_wasserstein(
        _pairwise_normed_dist(ref_coords[rand1], pred_coords_ref),
        p=2,
    )
    joint_emd = _alphaflow_wasserstein(
        _pairwise_normed_dist(ref_coords_joint[rand1], pred_coords_joint),
        p=2,
    )

    if ref_basis.shape[1] < 1 or pred_basis.shape[1] < 1:
        pc_sim = float("nan")
    else:
        pc_sim = float(np.sum(ref_basis[:, 0] * pred_basis[:, 0]))

    return (
        md_emd,
        joint_emd,
        pc_sim,
        _pc1_pc2_var_ratio(ref_explained_variance),
        _pc1_pc2_var_ratio(pred_explained_variance),
    )


# Contacts / FNC / ΔG_fold / Rg 
# -------------------------------------------
def _upper_triangle_mask(n: int) -> np.ndarray:
    return np.triu(np.ones((n, n), dtype=bool), k=1)

def contact_probability(ca_coords: np.ndarray, threshold: float = 8.0, chunk_size: int = 128) -> np.ndarray:
    '''Computes contact probability matrix via batched dist_sq formulation.'''
    T, L, _ = ca_coords.shape
    counts = np.zeros((L, L), dtype=np.int64)
    thresh_sq = threshold ** 2
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = ca_coords[start:end]
        
        # dist^2 = a^2 + b^2 - 2ab
        sq_norm = np.sum(chunk ** 2, axis=-1)
        dist_sq = sq_norm[:, :, None] + sq_norm[:, None, :] - 2 * np.einsum('tli,tji->tlj', chunk, chunk)
        counts += (dist_sq < thresh_sq).sum(axis=0)
        
    return counts.astype(np.float64) / max(T, 1)

def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(inter / union)

def compute_contact_metrics(native_ca: np.ndarray, pred_ca: np.ndarray, ref_ca: np.ndarray) -> dict:
    L = native_ca.shape[0]
    tri = _upper_triangle_mask(L)
    native_dist = np.linalg.norm(native_ca[:, None, :] - native_ca[None, :, :], axis=-1)
    native_contact = native_dist < 8.0
    pred_prob = contact_probability(pred_ca)
    ref_prob = contact_probability(ref_ca)

    weak_pred = np.logical_and(native_contact, pred_prob < 0.9)
    weak_ref = np.logical_and(native_contact, ref_prob < 0.9)
    transient_pred = np.logical_and(~native_contact, pred_prob > 0.1)
    transient_ref = np.logical_and(~native_contact, ref_prob > 0.1)
    return {
        "weak_contacts_j": _jaccard(weak_pred[tri], weak_ref[tri]),
        "transient_contacts_j": _jaccard(transient_pred[tri], transient_ref[tri]),
    }

def compute_native_contacts_ca(
    native_ca: np.ndarray, threshold: float = 8.0, min_seq_sep: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    L = native_ca.shape[0]
    diff = native_ca[:, None, :] - native_ca[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    rows = np.arange(L)[:, None]
    cols = np.arange(L)[None, :]
    mask = (dists < threshold) & ((cols - rows) > min_seq_sep)
    i_idx, j_idx = np.where(mask)
    return np.stack([i_idx, j_idx], axis=1), dists[i_idx, j_idx]

def compute_fnc_traj(
    ca_traj: np.ndarray, pairs: np.ndarray, ref_dists: np.ndarray,
    beta: float = 5.0, lam: float = 1.2,
) -> np.ndarray:
    if len(pairs) == 0:
        return np.zeros(ca_traj.shape[0])
    i_idx, j_idx = pairs[:, 0], pairs[:, 1]
    diff = ca_traj[:, i_idx, :] - ca_traj[:, j_idx, :]
    dists = np.linalg.norm(diff, axis=-1)
    q = 1.0 / (1.0 + np.exp(beta * (dists - lam * ref_dists[None, :])))
    return q.mean(axis=1)

def compute_folding_free_energy(
    fnc_pred: np.ndarray, fnc_ref: np.ndarray, temp_K: float, threshold: float = 0.5
) -> dict:
    kT = 0.0019872 * temp_K
    eps = 1e-6
    p_fold_p = float(np.clip(np.mean(fnc_pred > threshold), eps, 1 - eps))
    p_fold_r = float(np.clip(np.mean(fnc_ref > threshold), eps, 1 - eps))
    dG_p = -kT * math.log(p_fold_p / (1.0 - p_fold_p))
    dG_r = -kT * math.log(p_fold_r / (1.0 - p_fold_r))
    return {
        "dG_fold_pred": dG_p, "dG_fold_ref": dG_r,
        "dG_fold_error": abs(dG_p - dG_r),
        "p_fold_pred": p_fold_p, "p_fold_ref": p_fold_r,
    }

def compute_rg_traj(ca_coords: np.ndarray) -> np.ndarray:
    center = ca_coords.mean(axis=1, keepdims=True)
    return np.sqrt(((ca_coords - center) ** 2).sum(axis=-1).mean(axis=-1))

# GDT-TS 
# ------------------------------------------------------------------
def compute_gdt_ts(pred_ca: np.ndarray, ref_native_ca: np.ndarray) -> np.ndarray:
    '''Per-frame GDT-TS with broadcasted comparisons rather than a temporal loop.'''
    aligned = kabsch_align_traj(pred_ca.astype(np.float64), ref_native_ca.astype(np.float64))
    
    # Calculate distance squared to avoid sqrt (d_sq shape: [T, L])
    d_sq = np.sum((aligned - ref_native_ca) ** 2, axis=-1)
    cutoffs_sq = np.array([1.0, 4.0, 16.0, 64.0], dtype=np.float64)
    
    # Broadcast cutoffs (hits shape: [T, L, 4])
    hits = d_sq[:, :, None] < cutoffs_sq
    
    # Mean over residues (L), then mean over the 4 cutoffs
    return hits.mean(axis=1).mean(axis=-1)

# LDDT
# ------------------------------------------------------------------
def compute_lddt_ca(
    ca_traj: np.ndarray,
    ref_ca: np.ndarray,
    cutoff: float = 15.0,
    tolerances: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    chunk_size: int = 128,
) -> np.ndarray:
    '''
    Per-frame CA-only lDDT against a static reference structure.

    This mirrors the project validation lDDT definition but avoids materialising
    a full T x L x L distance tensor. Native/reference CA pairs within
    cutoff Angstrom are scored by the usual four tolerance bands.
    '''
    ca = np.asarray(ca_traj, dtype=np.float32)
    ref = np.asarray(ref_ca, dtype=np.float32)
    T, L, _ = ca.shape
    if L < 2:
        return np.full(T, np.nan, dtype=np.float64)

    ref_diff = ref[:, None, :] - ref[None, :, :]
    ref_d = np.linalg.norm(ref_diff, axis=-1)
    tri = np.triu(np.ones((L, L), dtype=bool), k=1)
    pair_mask = tri & (ref_d < cutoff) & (ref_d > 0.0)
    i_idx, j_idx = np.where(pair_mask)
    if i_idx.size == 0:
        return np.full(T, np.nan, dtype=np.float64)

    ref_pair_d = ref_d[i_idx, j_idx].astype(np.float32)
    out = np.empty(T, dtype=np.float64)
    tol = np.asarray(tolerances, dtype=np.float32)
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        d = np.linalg.norm(ca[start:end, i_idx, :] - ca[start:end, j_idx, :], axis=-1)
        err = np.abs(d - ref_pair_d[None, :])
        out[start:end] = (err[..., None] < tol[None, None, :]).mean(axis=(1, 2))
    return out

# Ca-Ca distances and DSSP fractions 
# --------------------------------------
def compute_caca_distances(ca_coords: np.ndarray) -> np.ndarray:
    diff = ca_coords[:, 1:, :] - ca_coords[:, :-1, :]
    return np.linalg.norm(diff, axis=-1).mean(axis=-1)


def compute_caca_pair_distances(ca_coords: np.ndarray) -> np.ndarray:
    '''All adjacent CA-CA bond distances, flattened over frames and residues.'''
    diff = ca_coords[:, 1:, :] - ca_coords[:, :-1, :]
    return np.linalg.norm(diff, axis=-1).reshape(-1)


def compute_min_nonbonded_caca(
    ca_coords: np.ndarray, min_sep: int = 2
) -> np.ndarray:
    '''Per-frame minimum CA-CA distance between residues separated by
    >=min_sep positions along the chain.

    A clash proxy: values below ~3.5 A indicate likely steric overlap since
    two CA atoms further than one bond apart cannot physically be closer
    than the van-der-Waals-allowed minimum.

    Returns:
        (T,) float array. Frames with L < min_sep+1 return +inf.
    '''
    T, L, _ = ca_coords.shape
    if L < min_sep + 1:
        return np.full(T, np.inf, dtype=np.float64)
    out = np.full(T, np.inf, dtype=np.float64)
    for sep in range(min_sep, L):
        d = np.linalg.norm(
            ca_coords[:, sep:, :] - ca_coords[:, :-sep, :], axis=-1
        )                                                           # (T, L-sep)
        out = np.minimum(out, d.min(axis=-1))
    return out


def compute_nonbonded_caca_metrics(
    ca_coords: np.ndarray,
    min_sep: int = 2,
    thresholds: tuple[float, ...] = (2.5, 3.0, 3.5),
) -> dict:
    '''
    Per-frame non-bonded CA-CA minimum distances and clash counts.

    Counts are over all residue pairs separated by at least min_sep along
    sequence. A threshold of 3.5 A is the broad steric-overlap warning used
    elsewhere in this evaluator; 3.0/2.5 A are stricter severe-clash cutoffs.
    '''
    T, L, _ = ca_coords.shape
    out = {"pair_min": np.full(T, np.inf, dtype=np.float64)}
    for thr in thresholds:
        tag = str(thr).replace(".", "p")
        out[f"pair_count_lt_{tag}"] = np.zeros(T, dtype=np.float64)

    if L < min_sep + 1:
        return out

    for sep in range(min_sep, L):
        d = np.linalg.norm(
            ca_coords[:, sep:, :] - ca_coords[:, :-sep, :], axis=-1
        )                                                           # (T, L-sep)
        out["pair_min"] = np.minimum(out["pair_min"], d.min(axis=-1))
        for thr in thresholds:
            tag = str(thr).replace(".", "p")
            out[f"pair_count_lt_{tag}"] += (d < thr).sum(axis=-1)
    return out

def segment_segment_distances_np(
    p1: np.ndarray,
    q1: np.ndarray,
    p2: np.ndarray,
    q2: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    '''
    Closest distances between batched 3D line segments.

    Inputs broadcast over leading dimensions and end in (..., 3).
    Implementation follows the standard closest-points-on-two-segments
    formulation, with clamped segment parameters.
    '''
    u = q1 - p1
    v = q2 - p2
    w = p1 - p2
    a = np.sum(u * u, axis=-1)
    b = np.sum(u * v, axis=-1)
    c = np.sum(v * v, axis=-1)
    d = np.sum(u * w, axis=-1)
    e = np.sum(v * w, axis=-1)
    denom = a * c - b * b

    s = np.where(np.abs(denom) > eps, (b * e - c * d) / np.maximum(denom, eps), 0.0)
    s = np.clip(s, 0.0, 1.0)
    t = np.where(c > eps, (b * s + e) / np.maximum(c, eps), 0.0)
    t = np.clip(t, 0.0, 1.0)

    s = np.where(a > eps, (b * t - d) / np.maximum(a, eps), 0.0)
    s = np.clip(s, 0.0, 1.0)

    closest_1 = p1 + s[..., None] * u
    closest_2 = p2 + t[..., None] * v
    return np.linalg.norm(closest_1 - closest_2, axis=-1)

def compute_chain_segment_metrics(
    ca_coords: np.ndarray,
    min_segment_sep: int = 2,
    thresholds: tuple[float, ...] = (0.5, 1.0),
    frame_chunk: int = 32,
    pair_chunk: int = 32768,
) -> dict:
    '''
    Per-frame non-adjacent CA-chain segment self-intersection proxies.

    Segment i connects residue i to i+1. Segment pairs with
    |i-j| < min_segment_sep are excluded because neighbouring segments
    share local chain geometry by construction. Exact intersections are rare in
    floating point 3D, so thresholds report near-intersections.
    '''
    ca = np.asarray(ca_coords, dtype=np.float32)
    T, L, _ = ca.shape
    S = L - 1
    if S < min_segment_sep + 1:
        base = {
            "segment_min": np.full(T, np.inf, dtype=np.float64),
        }
        for thr in thresholds:
            tag = str(thr).replace(".", "p")
            base[f"segment_count_lt_{tag}"] = np.zeros(T, dtype=np.float64)
            base[f"segment_frac_pairs_lt_{tag}"] = np.zeros(T, dtype=np.float64)
        return base

    pair_i, pair_j = np.triu_indices(S, k=min_segment_sep)
    n_pairs = pair_i.size
    seg_start = ca[:, :-1, :]
    seg_end = ca[:, 1:, :]
    per_frame_min = np.full(T, np.inf, dtype=np.float64)
    counts = {
        thr: np.zeros(T, dtype=np.float64)
        for thr in thresholds
    }

    for fs in range(0, T, frame_chunk):
        fe = min(fs + frame_chunk, T)
        chunk_min = np.full(fe - fs, np.inf, dtype=np.float64)
        chunk_counts = {
            thr: np.zeros(fe - fs, dtype=np.float64)
            for thr in thresholds
        }
        for ps in range(0, n_pairs, pair_chunk):
            pe = min(ps + pair_chunk, n_pairs)
            i = pair_i[ps:pe]
            j = pair_j[ps:pe]
            d = segment_segment_distances_np(
                seg_start[fs:fe, i, :],
                seg_end[fs:fe, i, :],
                seg_start[fs:fe, j, :],
                seg_end[fs:fe, j, :],
            )
            chunk_min = np.minimum(chunk_min, d.min(axis=1))
            for thr in thresholds:
                chunk_counts[thr] += (d < thr).sum(axis=1)
        per_frame_min[fs:fe] = chunk_min
        for thr in thresholds:
            counts[thr][fs:fe] = chunk_counts[thr]

    out = {"segment_min": per_frame_min}
    for thr in thresholds:
        tag = str(thr).replace(".", "p")
        out[f"segment_count_lt_{tag}"] = counts[thr]
        out[f"segment_frac_pairs_lt_{tag}"] = counts[thr] / max(float(n_pairs), 1.0)
    return out

def normalize_caca_bonds(
    ca_coords: np.ndarray,
    target: float = 3.8,
    n_iter: int = 20,
    tolerance: float = 0.0,
    step: float = 0.5,
) -> np.ndarray:
    '''
    SHAKE-style symmetric CA-CA bond projection, with softening knobs.

    Each iteration shifts the two bonded atoms symmetrically toward the
    target length. tolerance disables correction for bonds whose length
    already lies within [target - tolerance, target + tolerance] — this
    leaves thermally-reasonable bonds alone and only collapses far outliers
    toward the upper/lower edge of the tolerance band. step scales the
    per-iteration shift (0.5 = full symmetric correction, as in classical
    SHAKE; smaller values produce softer / partial projection).

    ca_coords = (T, L, 3) array of CA coordinates.
    target = ideal CA-CA bond length in Angstroms.
    n_iter = number of iterations.
    tolerance = half-width of the "don't touch" band around target.
                A value of ~0.03–0.05 Å reproduces the MD thermal spread (σ ≈
                0.009, IQR ≈ ±0.02).
    step = per-iteration correction fraction (0 < step ≤ 0.5).

    Returns (T, L, 3) array with consecutive CA-CA distances inside
    [target - tolerance, target + tolerance] (when converged).
    '''
    out = ca_coords.copy()
    T, L, _ = out.shape
    if L < 2:
        return out
    for _ in range(n_iter):
        for i in range(L - 1):
            vec = out[:, i + 1] - out[:, i]                         # (T, 3)
            length = np.linalg.norm(vec, axis=-1, keepdims=True)    # (T, 1)
            unit = vec / np.maximum(length, 1e-8)
            # Deficit relative to the tolerance band (0 inside the band).
            over  = np.maximum(length - (target + tolerance), 0.0)
            under = np.maximum((target - tolerance) - length, 0.0)
            correction = (over - under) * float(step)               # (T, 1)
            out[:, i]     += correction * unit
            out[:, i + 1] -= correction * unit
    return out

def compute_dssp_fractions_from_traj(ca_coords: np.ndarray, native_dssp_onehot: np.ndarray) -> np.ndarray:
    return np.full(ca_coords.shape[0], np.nan, dtype=np.float64)

# Spectral recovery
# ---------------------------------------
def compute_spectral_bands(
    ca_coords: np.ndarray, low_end: int = 8, mid_end: int = 64
) -> dict:
    from scipy.fft import dct
    T, L, _ = ca_coords.shape
    # Use a mean-centred trajectory for the non-DC bands so low/mid/high
    # amplitudes reflect fluctuations around the trajectory mean, but compute
    # the DC term from the raw aligned coordinates so it still captures the
    # absolute k=0 coefficient rather than numerical noise around zero.
    spec_centered = dct(ca_coords - ca_coords.mean(axis=0, keepdims=True), axis=0, norm="ortho")
    spec_raw = dct(ca_coords, axis=0, norm="ortho")
    power = (spec_centered ** 2).sum(axis=-1)
    K = power.shape[0]
    low_end = min(max(1, low_end), K)
    mid_end = min(max(low_end + 1, mid_end), K)
    amp = np.sqrt(power + 1e-12)
    dc_coeff = spec_raw[0]
    dc_amp = np.linalg.norm(dc_coeff, axis=-1)
    return {
        "dc": dc_amp,
        "dc_signed": dc_coeff.reshape(-1),
        "low": amp[1:low_end].sum(axis=0),
        "mid": amp[low_end:mid_end].sum(axis=0),
        "high": amp[mid_end:].sum(axis=0),
        "total": amp.sum(axis=0),
        "amp_per_mode": amp,
    }

def spectral_amplitude_recovery(ref_amp: np.ndarray, pred_amp: np.ndarray) -> float:
    ref_amp = np.asarray(ref_amp, dtype=np.float64)
    pred_amp = np.asarray(pred_amp, dtype=np.float64)
    mask = np.isfinite(ref_amp) & np.isfinite(pred_amp)
    if mask.sum() == 0:
        return float("nan")
    ref_amp = ref_amp[mask]
    pred_amp = pred_amp[mask]
    denom = ref_amp + pred_amp + 1e-8
    return float(np.mean(1.0 - np.abs(pred_amp - ref_amp) / denom))

def concordance_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    '''Lin's concordance correlation coefficient: correlation plus calibration.'''
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    mx, my = float(x.mean()), float(y.mean())
    vx, vy = float(x.var()), float(y.var())
    cov = float(np.mean((x - mx) * (y - my)))
    denom = vx + vy + (mx - my) ** 2
    if denom <= 0:
        return float("nan")
    return float(2.0 * cov / denom)

def kendall_tau_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    try:
        from scipy.stats import kendalltau
        tau = kendalltau(x[mask], y[mask], nan_policy="omit").correlation
        return float(tau) if tau is not None and math.isfinite(float(tau)) else float("nan")
    except Exception:
        return float("nan")

def rmse(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return float("nan")
    diff = x[mask] - y[mask]
    return float(np.sqrt(np.mean(diff * diff)))

def mae(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(x[mask] - y[mask])))

def cosine_sim(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return float("nan")
    x = x[mask]
    y = y[mask]
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0:
        return float("nan")
    return float(np.dot(x, y) / denom)

def top_fraction_jaccard(ref_scores: np.ndarray, pred_scores: np.ndarray, fraction: float = 0.5) -> float:
    ref_scores = np.asarray(ref_scores, dtype=np.float64).reshape(-1)
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    mask = np.isfinite(ref_scores) & np.isfinite(pred_scores)
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    k = min(max(int(math.ceil(fraction * n)), 1), n)
    ref_valid = ref_scores[mask]
    pred_valid = pred_scores[mask]
    ref_top = set(np.argpartition(ref_valid, -k)[-k:].tolist())
    pred_top = set(np.argpartition(pred_valid, -k)[-k:].tolist())
    union = ref_top | pred_top
    if not union:
        return float("nan")
    return float(len(ref_top & pred_top) / len(union))


# Spectral-volume prediction metrics 
# --------------------------------------
SPECTRAL_VOLUME_BANDS = (
    ("whole", None, None),
    ("dc", 0, 0),
    ("very_low", 1, 4),
    ("low", 5, 16),
    ("mid", 17, 64),
    ("high", 65, 128),
)

def spectral_volume_band_indices(k_total: int) -> dict[str, np.ndarray]:
    '''Return clipped DCT band indices for the configured spectral volume.'''
    bands = {}
    for name, lo, hi in SPECTRAL_VOLUME_BANDS:
        if name == "whole":
            idx = np.arange(k_total, dtype=np.int64)
        else:
            if lo is None or hi is None or lo >= k_total:
                idx = np.zeros(0, dtype=np.int64)
            else:
                idx = np.arange(lo, min(hi, k_total - 1) + 1, dtype=np.int64)
        if idx.size:
            bands[name] = idx
    return bands

def spectral_flat_to_dct_volume(x_flat: np.ndarray, n_channels: int) -> np.ndarray:
    '''Reshape (L, K*C) DCT features to (L, K, C).'''
    x_flat = np.asarray(x_flat, dtype=np.float64)
    if x_flat.ndim != 2:
        raise ValueError(f"expected flat spectral array of shape (L,D), got {x_flat.shape}")
    if x_flat.shape[-1] % int(n_channels) != 0:
        raise ValueError(
            f"spectral feature dim {x_flat.shape[-1]} is not divisible by n_channels={n_channels}"
        )
    return x_flat.reshape(x_flat.shape[0], x_flat.shape[-1] // int(n_channels), int(n_channels))

def compute_spectral_volume_metrics(
    pred_volume: np.ndarray,
    ref_volumes: np.ndarray,
    *,
    window_size: int,
    hist_bins: int,
    eps: float = 1e-8,
) -> dict[str, dict[str, float]]:
    '''
    Compare one predicted DCT volume against a same-window GT replicate ensemble.

    pred_volume is (L,K,C), while ref_volumes is (R,L,K,C).
    Direct coefficient metrics use the GT ensemble mean volume as the target.
    Distributional and energy metrics use the full GT replicate pool.
    '''
    pred = np.asarray(pred_volume, dtype=np.float64)
    refs = np.asarray(ref_volumes, dtype=np.float64)
    if pred.ndim != 3 or refs.ndim != 4:
        raise ValueError(f"expected pred (L,K,C), refs (R,L,K,C); got {pred.shape}, {refs.shape}")
    if refs.shape[1:] != pred.shape:
        raise ValueError(f"reference spectral shape {refs.shape[1:]} does not match pred {pred.shape}")

    ref_mean = refs.mean(axis=0)
    total_err_sq = float(np.sum((pred - ref_mean) ** 2))
    out: dict[str, dict[str, float]] = {}

    for band, idx in spectral_volume_band_indices(pred.shape[1]).items():
        pred_b = pred[:, idx, :]
        refs_b = refs[:, :, idx, :]
        ref_mean_b = ref_mean[:, idx, :]
        diff = pred_b - ref_mean_b

        pred_flat = pred_b.reshape(-1)
        refs_flat = refs_b.reshape(-1)
        ref_mean_flat = ref_mean_b.reshape(-1)
        diff_flat = diff.reshape(-1)

        err_sq = float(np.sum(diff_flat ** 2))
        ref_norm = float(np.linalg.norm(ref_mean_flat))
        pred_norm = float(np.linalg.norm(pred_flat))
        ref_centered = ref_mean_flat - float(ref_mean_flat.mean()) if ref_mean_flat.size else ref_mean_flat
        ref_var_sq = float(np.sum(ref_centered ** 2))
        ref_energy = float(np.mean(np.sum(refs_b ** 2, axis=(1, 2, 3)))) if refs_b.size else float("nan")
        pred_energy = float(np.sum(pred_b ** 2)) if pred_b.size else float("nan")

        out[band] = {
            "coeff_rmse": float(np.sqrt(np.mean(diff_flat ** 2))) if diff_flat.size else float("nan"),
            "relative_error": float(np.linalg.norm(diff_flat) / (ref_norm + eps)),
            "explained_variance": float(1.0 - err_sq / (ref_var_sq + eps)),
            "cosine": float(np.dot(pred_flat, ref_mean_flat) / (pred_norm * ref_norm + eps)),
            "pearson": pearson_corr(ref_mean_flat, pred_flat),
            "energy_ratio": float(pred_energy / (ref_energy + eps)),
            "log_energy_error": float(abs(np.log((pred_energy + eps) / (ref_energy + eps)))),
            "energy_relative_error": float(abs(pred_energy - ref_energy) / (ref_energy + eps)),
            "abs_coeff_jsd": jsd_1d(np.abs(refs_flat), np.abs(pred_flat), bins=hist_bins),
            "power_jsd": jsd_1d(refs_flat ** 2, pred_flat ** 2, bins=hist_bins),
            "idct_rmse": float(np.sqrt(err_sq / (max(int(window_size), 1) * max(pred_b.shape[0], 1) * max(pred_b.shape[2], 1)))),
            "fraction_total_error": float(err_sq / (total_err_sq + eps)),
        }
    return out

def summarise_spectral_volume_rows(rows: list[dict], prefix: str = "spectral_volume") -> dict:
    if not rows:
        return {}
    skip = {
        "target_id", "dataset", "domain_id", "temp", "repeat", "window_start",
        "window_size", "band", "n_ref_replicates", "n_residues", "n_freqs", "n_channels",
    }
    metric_keys = sorted(k for k in rows[0].keys() if k not in skip)
    summary = {f"{prefix}/n_rows": len(rows)}
    for band in sorted({str(r["band"]) for r in rows}):
        band_rows = [r for r in rows if str(r["band"]) == band]
        summary[f"{prefix}/{band}/n_rows"] = len(band_rows)
        for key in metric_keys:
            vals = np.asarray([r.get(key, np.nan) for r in band_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            summary[f"{prefix}/{band}/{key}_mean"] = float(vals.mean())
            summary[f"{prefix}/{band}/{key}_median"] = float(np.median(vals))
            summary[f"{prefix}/{band}/{key}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
    return summary

def build_model_stack(config: dict, device: torch.device):
    stack = build_shared_model_stack(
        config,
        device,
        load_weights_path=config["checkpoint_path"],
        set_eval=True,
        log_fn=log,
    )
    log(stack.noise_diagnostics_text)
    return stack



# Data loading (zarr + raw fallback), yielding per-(domain, temp) targets
# ------------------------------------------------------------------------

def open_zarr_handle(zarr_path: str, label: str):
    import zarr
    try:
        root = zarr.open_consolidated(zarr_path, mode="r")
        return {"root": root, "store": root.store}
    except Exception:
        # Store not consolidated — fall back to DirectoryStore.
        if not os.path.isdir(zarr_path):
            raise FileNotFoundError(f"{label} zarr not found: {zarr_path!r}")
        log(f"WARNING: no .zmetadata in {zarr_path}; opening as DirectoryStore.")
        store = zarr.DirectoryStore(zarr_path)
        return {"root": None, "store": store}

def open_zarr_group(handle, path: str):
    import zarr
    if handle["root"] is not None:
        return handle["root"][path]
    return zarr.open(store=handle["store"], path=path, mode="r")

def top_level_group_keys(handle) -> list[str]:
    root = handle["root"]
    if root is not None:
        return sorted(root.group_keys())
    names = set()
    for key in handle["store"].keys():
        parts = key.split("/")
        if parts and parts[0] and not parts[0].startswith("."):
            names.add(parts[0])
    return sorted(names)

def build_conditioning_coords(coords_type: str, native_bb: torch.Tensor) -> torch.Tensor:
    if coords_type == "bb":
        return native_bb.reshape(native_bb.shape[0], 12).contiguous()
    return native_bb[:, 1, :].contiguous()

def ref_ca_from_bb_traj(bb_traj: torch.Tensor) -> torch.Tensor:
    return bb_traj[:, :, 1, :].contiguous()

def pred_ca_from_output(pred_coords: torch.Tensor, coords_type: str) -> torch.Tensor:
    if coords_type == "bb":
        T, L, _ = pred_coords.shape
        return pred_coords.view(T, L, 4, 3)[:, :, 1, :].contiguous()
    return pred_coords.contiguous()


def compute_gt_spectral_volumes_for_window(
    sample: dict,
    *,
    window_start: int,
    runtime: dict,
    device: torch.device,
) -> np.ndarray | None:
    '''Build GT DCT volumes for every available replicate in a matching window.'''
    window_size = int(runtime["window_size"])
    window_end = int(window_start) + window_size
    reps = [
        r[window_start:window_end]
        for r in sample.get("replicate_ca", [])
        if r.shape[0] >= window_end
    ]
    if not reps:
        return None

    coords = torch.stack(reps, dim=0).to(device=device, dtype=torch.float32)
    native = sample["native_coords"].to(device=device, dtype=torch.float32).unsqueeze(0).expand(coords.shape[0], -1, -1)
    mask = torch.ones(coords.shape[0], coords.shape[2], dtype=torch.bool, device=device)
    representation = runtime["representation"]
    transform_engine = runtime["transform_engine"]
    with torch.no_grad():
        repr_coords = representation.forward(coords, native, mask=mask)
        repr_coords = repr_coords * mask[:, None, :, None]
        gt_flat = transform_engine.time_to_spectral(
            repr_coords,
            top_k=int(runtime["top_k_freqs"]),
        )
    return gt_flat.detach().float().cpu().numpy()

def compute_spectral_volume_rows_for_prediction(
    sample: dict,
    pred_spectral_flat: np.ndarray,
    *,
    window_start: int,
    runtime: dict,
    config: dict,
    device: torch.device,
) -> list[dict]:
    '''Return bandwise spectral-volume rows for one generated window.'''
    if not bool(config.get("compute_spectral_volume_metrics", True)):
        return []
    if not bool(runtime.get("is_dct", True)) or bool(runtime.get("is_time_domain", False)):
        return []
    if config.get("coords_type", "ca") != "ca":
        return []

    gt_flat = compute_gt_spectral_volumes_for_window(
        sample,
        window_start=window_start,
        runtime=runtime,
        device=device,
    )
    if gt_flat is None:
        return []

    total_channels = int(runtime["total_channels"])
    repr_channels = int(runtime["repr_coord_channels"])
    pred_vol = spectral_flat_to_dct_volume(pred_spectral_flat, total_channels)[:, :, :repr_channels]
    ref_vols = np.stack([
        spectral_flat_to_dct_volume(gt_flat[r], repr_channels)
        for r in range(gt_flat.shape[0])
    ], axis=0)

    hist_bins = int(config.get("spectral_volume_hist_bins", config.get("hist_bins", 100)))
    metrics = compute_spectral_volume_metrics(
        pred_vol,
        ref_vols,
        window_size=int(runtime["window_size"]),
        hist_bins=hist_bins,
    )
    bands = spectral_volume_band_indices(pred_vol.shape[1])

    rows = []
    base = {
        "target_id": sample["target_id"],
        "dataset": sample["dataset"],
        "domain_id": sample.get("domain_id", sample["target_id"]),
        "temp": float(sample.get("temp", np.nan)),
        "window_start": int(window_start),
        "window_size": int(runtime["window_size"]),
        "n_ref_replicates": int(ref_vols.shape[0]),
        "n_residues": int(pred_vol.shape[0]),
        "n_channels": int(pred_vol.shape[2]),
    }
    for band, vals in metrics.items():
        row = dict(base)
        row["band"] = band
        row["n_freqs"] = int(len(bands.get(band, [])))
        row.update({k: float(v) for k, v in vals.items()})
        rows.append(row)
    return rows


def list_mdcath_targets_zarr(
    mdcath_zarr_path: str, test_ids: set[str] | None, exclude_ids: set[str] | None = None
):
    handle = open_zarr_handle(mdcath_zarr_path, "mdCATH")
    static_keys = {
        "pdbProteinAtoms", "native_ca_coords", "native_bb_coords",
        "native_angles", "res_type", "torsion_mask", "dssp",
    }
    targets = []
    for domain in top_level_group_keys(handle):
        if test_ids is not None and domain not in test_ids:
            continue
        if exclude_ids is not None and domain in exclude_ids:
            continue
        grp = open_zarr_group(handle, domain)
        temps = [k for k in grp.group_keys() if k not in static_keys]
        for temp in sorted(temps, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))):
            rep_ids = sorted(grp[temp].group_keys())
            if rep_ids:
                targets.append((domain, temp, rep_ids))
    return handle, targets

def list_atlas_domains_zarr(
    atlas_zarr_path: str, exclude_ids: set[str] | None = None
):
    handle = open_zarr_handle(atlas_zarr_path, "ATLAS")
    domains = []
    for domain in top_level_group_keys(handle):
        if exclude_ids is not None and domain.replace("_", "").lower() in exclude_ids:
            continue
        grp = open_zarr_group(handle, domain)
        if "300" not in grp:
            continue
        rep_ids = sorted(grp["300"].group_keys())
        if rep_ids:
            domains.append((domain, rep_ids))
    return handle, domains

def load_mdcath_target_zarr(
    handle, domain_id: str, temp: str, coords_type: str, include_angles: bool,
    total_frames: int,
) -> dict:
    '''Return target dict with per-replicate full CA trajectories.'''
    grp = open_zarr_group(handle, domain_id)
    rep_ids = sorted(grp[temp].group_keys())
    native_bb = torch.from_numpy(grp["native_bb_coords"][...]).float()
    res_type = torch.from_numpy(grp["res_type"][...]).float()
    dssp_full = grp["dssp"][...] if "dssp" in grp else None
    native_dssp = dssp_to_onehot(dssp_full, n_res=native_bb.shape[0]) if dssp_full is not None else \
        torch.zeros(native_bb.shape[0], len(DSSP_STATES))

    native_angles = None
    if include_angles:
        if "native_angles" not in grp:
            raise ValueError(f"{domain_id}: include_angles=True but native_angles missing.")
        native_angles = torch.from_numpy(grp["native_angles"][...]).float()

    replicates = []
    for rep_id in rep_ids:
        rep_grp = grp[temp][rep_id]
        bb = torch.from_numpy(rep_grp["bb_coords_native_aligned"][...]).float()
        ca = ref_ca_from_bb_traj(bb)
        if ca.shape[0] == 0:
            log(f"[warn] {domain_id}@{temp} rep {rep_id}: empty trajectory; skipping rep.")
            continue
        if ca.shape[0] < total_frames:
            log(f"[warn] {domain_id}@{temp} rep {rep_id}: only {ca.shape[0]} frames "
                f"(expected {total_frames}); using available frames.", verbose=False)
        replicates.append(ca[:total_frames].contiguous())
    if not replicates:
        raise ValueError(f"{domain_id}@{temp}: no valid replicates found.")

    return {
        "dataset": "mdcath", "domain_id": domain_id,
        "target_id": f"{domain_id}@{temp}", "temp": int(temp),
        "native_bb": native_bb,
        "native_coords": build_conditioning_coords(coords_type, native_bb),
        "native_angles": native_angles, "res_type": res_type,
        "dssp": native_dssp,
        "replicate_ca": replicates,  # list of (total_frames, L, 3)
        "rep_ids": rep_ids,
    }

def load_atlas_target_zarr(
    handle, domain_id: str, coords_type: str, include_angles: bool,
    total_frames: int, atlas_stride: int,
) -> dict:
    grp = open_zarr_group(handle, domain_id)
    rep_ids = sorted(grp["300"].group_keys())
    native_bb = torch.from_numpy(grp["native_bb_coords"][...]).float()
    res_type = torch.from_numpy(grp["res_type"][...]).float()
    dssp_full = grp["dssp"][...] if "dssp" in grp else None
    native_dssp = dssp_to_onehot(dssp_full, n_res=native_bb.shape[0]) if dssp_full is not None else \
        torch.zeros(native_bb.shape[0], len(DSSP_STATES))

    native_angles = None
    if include_angles:
        if "native_angles" not in grp:
            raise ValueError(f"{domain_id}: include_angles=True but native_angles missing.")
        native_angles = torch.from_numpy(grp["native_angles"][...]).float()

    replicates = []
    for rep_id in rep_ids:
        rep_grp = grp["300"][rep_id]
        bb = torch.from_numpy(rep_grp["bb_coords_native_aligned"][...]).float()
        ca = ref_ca_from_bb_traj(bb)
        last_needed = (total_frames - 1) * atlas_stride + 1
        if ca.shape[0] < last_needed:
            log(f"[warn] {domain_id} rep {rep_id}: only {ca.shape[0]} frames "
                f"(need {last_needed} at stride {atlas_stride}); skipping rep.")
            continue
        replicates.append(ca[:last_needed:atlas_stride].contiguous())
    if not replicates:
        raise ValueError(f"{domain_id}: no replicate long enough for ATLAS protocol.")

    return {
        "dataset": "atlas", "domain_id": domain_id,
        "target_id": domain_id, "temp": 300,
        "native_bb": native_bb,
        "native_coords": build_conditioning_coords(coords_type, native_bb),
        "native_angles": native_angles, "res_type": res_type,
        "dssp": native_dssp,
        "replicate_ca": replicates,
        "rep_ids": rep_ids,
    }


# Batched inference
# ---------------------------------------------------------------

def infer_batch_eval(
    samples: list[dict],
    win_pos_values: torch.Tensor,
    runtime: dict,
    config: dict,
    device: torch.device,
    dtype_ctx,
) -> list[dict]:
    '''
    Batch inference over B proteins in one GPU pass.

    samples = list of sample dicts (native_coords, res_type, dssp, native_angles, temp).
    win_pos_values = (B,) float32 tensor of win_pos fractions.

    Returns list of dictionaries with CA coordinates and optional spectral-volume rows.
    '''
    coords_type = config.get("coords_type", "ca")
    B = len(samples)
    coord_channels = runtime["coord_channels"]
    angle_channels = runtime["angle_channels"]
    channels = runtime["total_channels"]
    complex_mult = 1 if runtime["is_dct"] else 2
    D = (runtime["window_size"] * coord_channels) if runtime["is_time_domain"] else \
        (channels * runtime["top_k_freqs"] * complex_mult)

    lengths = [s["native_coords"].shape[0] for s in samples]
    L_max = max(lengths)

    def pad_L(t: torch.Tensor) -> torch.Tensor:
        gap = L_max - t.shape[0]
        return t if gap == 0 else torch.cat([t, t.new_zeros((gap,) + t.shape[1:])], dim=0)

    native_coords = torch.stack([pad_L(s["native_coords"]) for s in samples]).to(device)
    mask = torch.zeros(B, L_max, dtype=torch.bool, device=device)
    for b, L in enumerate(lengths):
        mask[b, :L] = True

    angle_list = [s.get("native_angles") for s in samples]
    if any(a is not None for a in angle_list):
        native_angles = torch.stack([
            pad_L(a) if a is not None else torch.zeros(L_max, angle_channels)
            for a in angle_list
        ]).to(device)
    else:
        native_angles = None

    res_type = torch.stack([pad_L(s["res_type"]) for s in samples]).to(device)
    dssp = torch.stack([pad_L(s["dssp"]) for s in samples]).to(device)
    temp = torch.tensor([float(s["temp"]) for s in samples], dtype=torch.float32, device=device)
    win_pos = win_pos_values.to(device)

    with dtype_ctx:
        pred = run_inference(
            runtime["model"], runtime["diffusion"], runtime["transform_engine"],
            (B, L_max, D), native_coords, native_angles, temp, runtime["window_size"],
            mask=mask, torsion_mask=None, device=device,
            guidance_scale=float(config.get("guidance_scale", 1.0)),
            num_ode_steps=int(config.get("num_ode_steps", config.get("num_steps", 200))),
            displacement=bool(config.get("displacement", True)),
            representation=runtime.get("representation"),
            win_pos=win_pos, rmsf_prior=None, res_type=res_type, dssp=dssp,
        )

    # Cast to float32 before leaving GPU — bfloat16 cannot be converted to numpy.
    pred_coords_batch = pred["coords"].detach().float().cpu()  # (B, T, L_max, C)
    pred_spectral_batch = (
        pred.get("spectral").detach().float().cpu()
        if pred.get("spectral") is not None else None
    )

    results = []
    for b, L_b in enumerate(lengths):
        pred_coords = pred_coords_batch[b, :, :L_b, :]  # (T, L_b, C)
        rows = []
        if pred_spectral_batch is not None:
            total_frames = samples[b]["replicate_ca"][0].shape[0] if samples[b].get("replicate_ca") else runtime["window_size"]
            window_start = int(round(float(win_pos_values[b].detach().cpu()) * max(int(total_frames) - 1, 1)))
            rows = compute_spectral_volume_rows_for_prediction(
                samples[b],
                pred_spectral_batch[b, :L_b, :].numpy(),
                window_start=window_start,
                runtime=runtime,
                config=config,
                device=device,
            )
        results.append({
            "ca": pred_ca_from_output(pred_coords, coords_type).numpy(),
            "spectral_volume_rows": rows,
        })
    return results


# Per-target metric computation (one comparison: pred vs gt ensemble)
# =============================================================================

def compute_metrics_for_comparison(
    pred_ca: np.ndarray,
    ref_ca_list: list[np.ndarray],
    native_ca: np.ndarray,
    anchor: np.ndarray,
    temp_K: float,
    pairwise_rmsd_samples: int,
    hist_bins: int,
    seed: int,
    include_spectral: bool,
) -> dict:
    '''
    Full metric panel for (single pred trajectory, pooled ref ensemble).

    - pred_ca: (T_pred, L, 3) – already raw (not yet aligned)
    - ref_ca_list: list of per-replicate (T_ref, L, 3)
    - anchor: (L, 3) first frame of first MD replicate (used for alignment)
    '''
    rng = np.random.default_rng(np.uint32(seed))

    # Alignment: both pred and each ref replicate onto anchor
    pred_aligned = kabsch_align_traj(pred_ca.astype(np.float64), anchor.astype(np.float64))
    ref_aligned_list = [
        kabsch_align_traj(r.astype(np.float64), anchor.astype(np.float64))
        for r in ref_ca_list
    ]
    ref_pooled = np.concatenate(ref_aligned_list, axis=0)

    # For metrics that expect matched sample sizes (RMWD / PCA), subsample ref
    # to the same T as pred — identical to the test_tempo_mdgen / MarS-FM path.
    n_pred = pred_aligned.shape[0]
    if ref_pooled.shape[0] != n_pred:
        idx = np.round(np.linspace(0, ref_pooled.shape[0] - 1, n_pred)).astype(int)
        ref_matched = ref_pooled[idx]
    else:
        ref_matched = ref_pooled

    # RMSF
    pred_rmsf = compute_rmsf(pred_aligned)
    ref_rmsf = compute_rmsf(ref_pooled)
    per_target_rmsf_r = pearson_corr(ref_rmsf, pred_rmsf)
    per_target_rmsf_sp = spearman_corr(ref_rmsf, pred_rmsf)

    # Pairwise RMSD samples
    pair_pred = estimate_pairwise_rmsd(pred_aligned, pairwise_rmsd_samples, rng)
    pair_ref = estimate_pairwise_rmsd(ref_pooled, pairwise_rmsd_samples, rng)
    pair_rmsd_pred_mean = float(pair_pred.mean()) if pair_pred.size else float("nan")
    pair_rmsd_ref_mean = float(pair_ref.mean()) if pair_ref.size else float("nan")
    pairwise_rmsd_jsd = jsd_1d(pair_ref, pair_pred, bins=hist_bins)

    # RMWD + PCA (subsampled ref to match pred)
    rmwd, rmwd_t, rmwd_v = compute_rmwd(pred_aligned, ref_matched)
    md_w2, joint_w2, pc_sim = compute_pca_metrics(pred_aligned, ref_matched)
    (
        alphaflow_md_emd,
        alphaflow_joint_emd,
        alphaflow_pc_sim,
        alphaflow_ref_pc1_pc2_ratio,
        alphaflow_pred_pc1_pc2_ratio,
    ) = compute_alphaflow_pca_metrics(
        pred_aligned, ref_pooled, rng
    )

    # Contacts (use raw pred / ref, not anchor-aligned, to mirror Jing convention)
    contacts = compute_contact_metrics(native_ca, pred_ca, np.concatenate(ref_ca_list, axis=0))

    # FNC + folding ΔG
    nc_pairs, nc_ref_dists = compute_native_contacts_ca(native_ca)
    fnc_pred = compute_fnc_traj(pred_aligned, nc_pairs, nc_ref_dists)
    fnc_ref = compute_fnc_traj(ref_pooled, nc_pairs, nc_ref_dists)
    fe = compute_folding_free_energy(fnc_pred, fnc_ref, temp_K=temp_K)
    fnc_jsd = jsd_1d(fnc_ref, fnc_pred, bins=hist_bins)

    # Rg
    rg_pred = compute_rg_traj(pred_aligned)
    rg_ref = compute_rg_traj(ref_pooled)
    rg_jsd = jsd_1d(rg_ref, rg_pred, bins=hist_bins)
    rg_kl = fwd_kl_1d(rg_ref, rg_pred, bins=hist_bins)

    # GDT-TS distributions (per-frame, pred vs gt)
    gdt_pred = compute_gdt_ts(pred_aligned, native_ca)
    gdt_ref = compute_gdt_ts(ref_pooled, native_ca)
    gdt_jsd = jsd_1d(gdt_ref, gdt_pred, bins=hist_bins)

    # Native-reference lDDT distributions (per-frame, pred vs gt)
    lddt_pred = compute_lddt_ca(pred_aligned, native_ca)
    lddt_ref = compute_lddt_ca(ref_pooled, native_ca)
    lddt_jsd = jsd_1d(lddt_ref, lddt_pred, bins=hist_bins)

    # Ca-Ca bond distances: per-frame mean plus all adjacent-pair bonds.
    caca_pred = compute_caca_distances(pred_aligned)
    caca_ref = compute_caca_distances(ref_pooled)
    caca_pair_pred = compute_caca_pair_distances(pred_aligned)
    caca_pair_ref = compute_caca_pair_distances(ref_pooled)
    pair_nonbonded_pred = compute_nonbonded_caca_metrics(pred_aligned, min_sep=2)
    pair_nonbonded_ref = compute_nonbonded_caca_metrics(ref_pooled, min_sep=2)
    pair_min_pred = pair_nonbonded_pred["pair_min"]
    pair_min_ref  = pair_nonbonded_ref["pair_min"]
    caca_jsd = jsd_1d(caca_ref, caca_pred, bins=hist_bins)
    caca_pair_jsd = jsd_1d(caca_pair_ref, caca_pair_pred, bins=hist_bins)

    segment_pred = compute_chain_segment_metrics(pred_aligned, min_segment_sep=2)
    segment_ref = compute_chain_segment_metrics(ref_pooled, min_segment_sep=2)
    segment_min_pred = segment_pred["segment_min"]
    segment_min_ref = segment_ref["segment_min"]
    segment_min_jsd = jsd_1d(segment_min_ref, segment_min_pred, bins=hist_bins)

    results = {
        "rmsf_pred_mean": float(pred_rmsf.mean()),
        "rmsf_ref_mean": float(ref_rmsf.mean()),
        "per_target_rmsf_r": per_target_rmsf_r,
        "per_target_rmsf_spearman": per_target_rmsf_sp,
        "pairwise_rmsd_pred_mean": pair_rmsd_pred_mean,
        "pairwise_rmsd_ref_mean": pair_rmsd_ref_mean,
        "pairwise_rmsd_jsd": pairwise_rmsd_jsd,
        "rmwd": rmwd, "rmwd_translation": rmwd_t, "rmwd_variance": rmwd_v,
        "md_pca_w2": md_w2, "joint_pca_w2": joint_w2, "pc_sim": pc_sim,
        "alphaflow_md_pca_emd": alphaflow_md_emd,
        "alphaflow_joint_pca_emd": alphaflow_joint_emd,
        "alphaflow_pc_sim": alphaflow_pc_sim,
        "alphaflow_pc_sim_abs": abs(alphaflow_pc_sim) if math.isfinite(alphaflow_pc_sim) else float("nan"),
        "alphaflow_ref_pc1_pc2_ratio": alphaflow_ref_pc1_pc2_ratio,
        "alphaflow_pred_pc1_pc2_ratio": alphaflow_pred_pc1_pc2_ratio,
        "weak_contacts_j": contacts["weak_contacts_j"],
        "transient_contacts_j": contacts["transient_contacts_j"],
        "fnc_mean_pred": float(fnc_pred.mean()),
        "fnc_mean_ref": float(fnc_ref.mean()),
        "fnc_jsd": fnc_jsd,
        "dG_fold_pred": fe["dG_fold_pred"],
        "dG_fold_ref": fe["dG_fold_ref"],
        "dG_fold_error": fe["dG_fold_error"],
        "p_fold_pred": fe["p_fold_pred"],
        "p_fold_ref": fe["p_fold_ref"],
        "rg_mean_pred": float(rg_pred.mean()),
        "rg_mean_ref": float(rg_ref.mean()),
        "rg_jsd": rg_jsd,
        "rg_fwd_kl": rg_kl,
        "gdt_ts_pred_mean": float(gdt_pred.mean()),
        "gdt_ts_ref_mean": float(gdt_ref.mean()),
        "gdt_ts_jsd": gdt_jsd,
        "lddt_pred_mean": float(np.nanmean(lddt_pred)),
        "lddt_ref_mean": float(np.nanmean(lddt_ref)),
        "lddt_jsd": lddt_jsd,
        "caca_pred_mean": float(caca_pred.mean()),
        "caca_ref_mean": float(caca_ref.mean()),
        "caca_jsd": caca_jsd,
        "caca_pair_pred_mean": float(caca_pair_pred.mean()) if caca_pair_pred.size else float("nan"),
        "caca_pair_ref_mean": float(caca_pair_ref.mean()) if caca_pair_ref.size else float("nan"),
        "caca_pair_jsd": caca_pair_jsd,
        # Clash proxy: min non-bonded CA-CA distance (sep>=2). Fractions
        # below ~3.5 A flag likely steric overlap.
        "pair_min_pred_mean":     float(pair_min_pred.mean()) if pair_min_pred.size else float("nan"),
        "pair_min_ref_mean":      float(pair_min_ref.mean())  if pair_min_ref.size  else float("nan"),
        "clash_frac_pred_3p5":    float((pair_min_pred < 3.5).mean()) if pair_min_pred.size else float("nan"),
        "clash_frac_ref_3p5":     float((pair_min_ref  < 3.5).mean()) if pair_min_ref.size  else float("nan"),
        "clash_frac_pred_3p0":    float((pair_min_pred < 3.0).mean()) if pair_min_pred.size else float("nan"),
        "clash_frac_ref_3p0":     float((pair_min_ref  < 3.0).mean()) if pair_min_ref.size  else float("nan"),
        "clash_count_pred_3p5_mean": float(pair_nonbonded_pred["pair_count_lt_3p5"].mean()),
        "clash_count_ref_3p5_mean": float(pair_nonbonded_ref["pair_count_lt_3p5"].mean()),
        "clash_count_pred_3p0_mean": float(pair_nonbonded_pred["pair_count_lt_3p0"].mean()),
        "clash_count_ref_3p0_mean": float(pair_nonbonded_ref["pair_count_lt_3p0"].mean()),
        "clash_count_pred_2p5_mean": float(pair_nonbonded_pred["pair_count_lt_2p5"].mean()),
        "clash_count_ref_2p5_mean": float(pair_nonbonded_ref["pair_count_lt_2p5"].mean()),
        # Chain segment self-intersection proxies. Counts are per frame over
        # non-adjacent CA-CA line segments.
        "segment_min_pred_mean": float(segment_min_pred.mean()) if segment_min_pred.size else float("nan"),
        "segment_min_ref_mean": float(segment_min_ref.mean()) if segment_min_ref.size else float("nan"),
        "segment_min_jsd": segment_min_jsd,
        "segment_count_pred_lt_0p5_mean": float(segment_pred["segment_count_lt_0p5"].mean()),
        "segment_count_ref_lt_0p5_mean": float(segment_ref["segment_count_lt_0p5"].mean()),
        "segment_count_pred_lt_1p0_mean": float(segment_pred["segment_count_lt_1p0"].mean()),
        "segment_count_ref_lt_1p0_mean": float(segment_ref["segment_count_lt_1p0"].mean()),
        "segment_frac_pairs_pred_lt_0p5_mean": float(segment_pred["segment_frac_pairs_lt_0p5"].mean()),
        "segment_frac_pairs_ref_lt_0p5_mean": float(segment_ref["segment_frac_pairs_lt_0p5"].mean()),
        "segment_frac_pairs_pred_lt_1p0_mean": float(segment_pred["segment_frac_pairs_lt_1p0"].mean()),
        "segment_frac_pairs_ref_lt_1p0_mean": float(segment_ref["segment_frac_pairs_lt_1p0"].mean()),
        "segment_frac_frames_pred_any_lt_0p5": float((segment_pred["segment_count_lt_0p5"] > 0).mean()),
        "segment_frac_frames_ref_any_lt_0p5": float((segment_ref["segment_count_lt_0p5"] > 0).mean()),
        "segment_frac_frames_pred_any_lt_1p0": float((segment_pred["segment_count_lt_1p0"] > 0).mean()),
        "segment_frac_frames_ref_any_lt_1p0": float((segment_ref["segment_count_lt_1p0"] > 0).mean()),
        # Raw arrays for later global-RMSF pooling, scatter-plot saving
        "_pred_rmsf": pred_rmsf,
        "_ref_rmsf": ref_rmsf,
        "_pair_pred": pair_pred,
        "_pair_ref": pair_ref,
        "_caca_pred": caca_pred,
        "_caca_ref": caca_ref,
        "_caca_pair_pred": caca_pair_pred,
        "_caca_pair_ref": caca_pair_ref,
        "_lddt_pred": lddt_pred,
        "_lddt_ref": lddt_ref,
        "_pair_min_pred": pair_min_pred,
        "_pair_min_ref":  pair_min_ref,
        "_pair_count_pred_lt_3p5": pair_nonbonded_pred["pair_count_lt_3p5"],
        "_pair_count_ref_lt_3p5": pair_nonbonded_ref["pair_count_lt_3p5"],
        "_pair_count_pred_lt_3p0": pair_nonbonded_pred["pair_count_lt_3p0"],
        "_pair_count_ref_lt_3p0": pair_nonbonded_ref["pair_count_lt_3p0"],
        "_pair_count_pred_lt_2p5": pair_nonbonded_pred["pair_count_lt_2p5"],
        "_pair_count_ref_lt_2p5": pair_nonbonded_ref["pair_count_lt_2p5"],
        "_segment_min_pred": segment_min_pred,
        "_segment_min_ref": segment_min_ref,
        "_segment_count_pred_lt_0p5": segment_pred["segment_count_lt_0p5"],
        "_segment_count_ref_lt_0p5": segment_ref["segment_count_lt_0p5"],
        "_segment_count_pred_lt_1p0": segment_pred["segment_count_lt_1p0"],
        "_segment_count_ref_lt_1p0": segment_ref["segment_count_lt_1p0"],
    }

    # Spectral (my-model only)
    if include_spectral:
        sp_pred = compute_spectral_bands(pred_aligned)
        sp_ref = compute_spectral_bands(ref_pooled)
        for band in ("dc", "low", "mid", "high", "total"):
            # Pearson r of per-residue amplitudes
            results[f"spec_{band}_r"] = pearson_corr(sp_ref[band], sp_pred[band])
            # JSD of per-residue amplitudes (treat as 1D sample distributions)
            results[f"spec_{band}_jsd"] = jsd_1d(sp_ref[band], sp_pred[band], bins=hist_bins)
        results["spec_dc_amp_recovery"] = spectral_amplitude_recovery(sp_ref["dc"], sp_pred["dc"])
        results["spec_dc_kendall_tau"] = kendall_tau_corr(sp_ref["dc"], sp_pred["dc"])
        results["spec_dc_concordance_cc"] = concordance_corrcoef(sp_ref["dc"], sp_pred["dc"])
        results["spec_dc_rmse"] = rmse(sp_ref["dc"], sp_pred["dc"])
        results["spec_dc_mae"] = mae(sp_ref["dc"], sp_pred["dc"])
        results["spec_dc_cosine"] = cosine_sim(sp_ref["dc"], sp_pred["dc"])
        results["spec_dc_top50_jaccard"] = top_fraction_jaccard(sp_ref["dc"], sp_pred["dc"], fraction=0.5)
        results["spec_dc_signed_r"] = pearson_corr(sp_ref["dc_signed"], sp_pred["dc_signed"])
        results["spec_dc_signed_jsd"] = jsd_1d(sp_ref["dc_signed"], sp_pred["dc_signed"], bins=hist_bins)
        results["spec_dc_signed_rmse"] = rmse(sp_ref["dc_signed"], sp_pred["dc_signed"])
        results["spec_dc_signed_mae"] = mae(sp_ref["dc_signed"], sp_pred["dc_signed"])
        results["spec_dc_signed_cosine"] = cosine_sim(sp_ref["dc_signed"], sp_pred["dc_signed"])

    return results


# Per-target aggregation over 5 repeats + oracle
# =============================================================================
# Fields that are scalar metrics (present in every repeat dict under a plain key)

_SCALAR_KEYS = [
    "rmsf_pred_mean", "rmsf_ref_mean", "per_target_rmsf_r", "per_target_rmsf_spearman",
    "pairwise_rmsd_pred_mean", "pairwise_rmsd_ref_mean", "pairwise_rmsd_jsd",
    "rmwd", "rmwd_translation", "rmwd_variance",
    "md_pca_w2", "joint_pca_w2", "pc_sim",
    "alphaflow_md_pca_emd", "alphaflow_joint_pca_emd", "alphaflow_pc_sim",
    "alphaflow_pc_sim_abs",
    "alphaflow_ref_pc1_pc2_ratio", "alphaflow_pred_pc1_pc2_ratio",
    "alphaflow_pc_sim_concat", "alphaflow_pc_sim_concat_abs",
    "alphaflow_ref_pc1_pc2_ratio_concat", "alphaflow_pred_pc1_pc2_ratio_concat",
    "weak_contacts_j", "transient_contacts_j",
    "fnc_mean_pred", "fnc_mean_ref", "fnc_jsd",
    "dG_fold_pred", "dG_fold_ref", "dG_fold_error", "p_fold_pred", "p_fold_ref",
    "rg_mean_pred", "rg_mean_ref", "rg_jsd", "rg_fwd_kl",
    "gdt_ts_pred_mean", "gdt_ts_ref_mean", "gdt_ts_jsd",
    "lddt_pred_mean", "lddt_ref_mean", "lddt_jsd",
    "caca_pred_mean", "caca_ref_mean", "caca_jsd",
    "caca_pair_pred_mean", "caca_pair_ref_mean", "caca_pair_jsd",
    "pair_min_pred_mean", "pair_min_ref_mean",
    "clash_frac_pred_3p5", "clash_frac_ref_3p5",
    "clash_frac_pred_3p0", "clash_frac_ref_3p0",
    "clash_count_pred_3p5_mean", "clash_count_ref_3p5_mean",
    "clash_count_pred_3p0_mean", "clash_count_ref_3p0_mean",
    "clash_count_pred_2p5_mean", "clash_count_ref_2p5_mean",
    "segment_min_pred_mean", "segment_min_ref_mean", "segment_min_jsd",
    "segment_count_pred_lt_0p5_mean", "segment_count_ref_lt_0p5_mean",
    "segment_count_pred_lt_1p0_mean", "segment_count_ref_lt_1p0_mean",
    "segment_frac_pairs_pred_lt_0p5_mean", "segment_frac_pairs_ref_lt_0p5_mean",
    "segment_frac_pairs_pred_lt_1p0_mean", "segment_frac_pairs_ref_lt_1p0_mean",
    "segment_frac_frames_pred_any_lt_0p5", "segment_frac_frames_ref_any_lt_0p5",
    "segment_frac_frames_pred_any_lt_1p0", "segment_frac_frames_ref_any_lt_1p0",
]
_SPEC_KEYS = [
    "spec_dc_r", "spec_dc_jsd", "spec_dc_amp_recovery",
    "spec_dc_kendall_tau", "spec_dc_concordance_cc", "spec_dc_rmse", "spec_dc_mae",
    "spec_dc_cosine", "spec_dc_top50_jaccard",
    "spec_dc_signed_r", "spec_dc_signed_jsd",
    "spec_dc_signed_rmse", "spec_dc_signed_mae", "spec_dc_signed_cosine",
    "spec_low_r", "spec_low_jsd",
    "spec_mid_r", "spec_mid_jsd",
    "spec_high_r", "spec_high_jsd",
    "spec_total_r", "spec_total_jsd",
]


def _average_metric_repeats(per_repeat: list[dict], include_spectral: bool) -> dict:
    '''Average scalar metrics across repeats. Non-scalars are stored under _raw.'''
    keys = list(_SCALAR_KEYS)
    if include_spectral:
        keys += _SPEC_KEYS
    out = {}
    for k in keys:
        vals = [r[k] for r in per_repeat if k in r]
        out[f"{k}_mean"] = mean_or_nan(vals)
        out[f"{k}_std"] = std_or_nan(vals)
    return out


def compute_oracle_baseline(
    target: dict, anchor: np.ndarray, config: dict, hist_bins: int, include_spectral: bool,
) -> dict:
    '''Leave-one-out oracle: each replicate vs pooled others; average folds.'''
    native_ca = target["native_bb"][:, 1, :].float().cpu().numpy()
    reps = [r.float().cpu().numpy() for r in target["replicate_ca"]]
    n_reps = len(reps)
    if n_reps < 2:
        return {}
    if target["dataset"] == "atlas":
        # Evaluation.md specifies "leave first replicate out", single fold
        fold_results = [
            compute_metrics_for_comparison(
                pred_ca=reps[0], ref_ca_list=reps[1:], native_ca=native_ca,
                anchor=anchor, temp_K=float(target["temp"]),
                pairwise_rmsd_samples=int(config["pairwise_rmsd_samples"]),
                hist_bins=hist_bins, seed=hash((target["target_id"], "oracle", 0)) & 0xFFFFFFFF,
                include_spectral=include_spectral,
            )
        ]
    else:
        fold_results = []
        for left_out in range(n_reps):
            others = [reps[i] for i in range(n_reps) if i != left_out]
            r = compute_metrics_for_comparison(
                pred_ca=reps[left_out], ref_ca_list=others, native_ca=native_ca,
                anchor=anchor, temp_K=float(target["temp"]),
                pairwise_rmsd_samples=int(config["pairwise_rmsd_samples"]),
                hist_bins=hist_bins,
                seed=hash((target["target_id"], "oracle", left_out)) & 0xFFFFFFFF,
                include_spectral=include_spectral,
            )
            fold_results.append(r)
    return _average_metric_repeats(fold_results, include_spectral=include_spectral)


def evaluate_from_trajectories(
    sample: dict,
    pred_trajectories: list[np.ndarray],
    config: dict,
    inference_sec: float = 0.0,
) -> dict:
    '''
    Compute all metrics for one protein given pre-generated trajectories.

    sample = target dict (replicate_ca, native_bb, temp, dataset, …).
    pred_trajectories = list of n_repeats (T_pred, L, 3) float32 numpy arrays.
    config = run config.
    inference_sec = total wall-clock seconds spent on inference (for logging).
    '''
    include_spectral = bool(config.get("compute_spectral_metrics", True))
    hist_bins = int(config.get("hist_bins", 100))
    n_repeats = len(pred_trajectories)
    pre_shake_trajectories = None

    if bool(config.get("normalize_caca_bonds", False)):
        caca_target = float(config.get("caca_bond_target", 3.8))
        caca_tol = float(config.get("caca_bond_tolerance", 0.05))
        caca_iter = int(config.get("caca_bond_n_iter", 20))
        caca_step = float(config.get("caca_bond_step", 0.5))
        pre_shake_trajectories = [t.copy() for t in pred_trajectories]
        pre = pred_trajectories[0]
        d_pre = np.linalg.norm(pre[:, 1:] - pre[:, :-1], axis=-1)
        pred_trajectories = [
            normalize_caca_bonds(
                t.astype(np.float64),
                target=caca_target,
                n_iter=caca_iter,
                tolerance=caca_tol,
                step=caca_step,
            ).astype(t.dtype)
            for t in pred_trajectories
        ]
        post = pred_trajectories[0]
        d_post = np.linalg.norm(post[:, 1:] - post[:, :-1], axis=-1)
        print(
            f"[SHAKE] target={caca_target:.2f} tol={caca_tol:.3f} iter={caca_iter} step={caca_step:.2f} "
            f"pre mean/std={d_pre.mean():.3f}/{d_pre.std():.3f} "
            f"post mean/std={d_post.mean():.3f}/{d_post.std():.3f} "
            f"band_frac_outside={np.mean(np.abs(d_post - caca_target) > caca_tol + 1e-6):.2%}",
            flush=True,
        )

    reps = [r.float().cpu().numpy() for r in sample["replicate_ca"]]
    anchor = reps[0][0]
    native_ca = sample["native_bb"][:, 1, :].float().cpu().numpy()

    per_repeat_metrics = []
    for idx, pred_traj in enumerate(pred_trajectories):
        r = compute_metrics_for_comparison(
            pred_ca=pred_traj, ref_ca_list=reps, native_ca=native_ca, anchor=anchor,
            temp_K=float(sample["temp"]),
            pairwise_rmsd_samples=int(config["pairwise_rmsd_samples"]),
            hist_bins=hist_bins,
            seed=hash((sample["target_id"], idx)) & 0xFFFFFFFF,
            include_spectral=include_spectral,
        )
        per_repeat_metrics.append(r)

    pred_rmsfs = np.stack([r["_pred_rmsf"] for r in per_repeat_metrics], axis=0)
    pair_pred_pool = np.concatenate([r["_pair_pred"] for r in per_repeat_metrics], axis=0) \
        if all(r["_pair_pred"].size for r in per_repeat_metrics) else np.array([])
    caca_pred_pool = np.concatenate([r["_caca_pred"] for r in per_repeat_metrics], axis=0) \
        if all(r["_caca_pred"].size for r in per_repeat_metrics) else np.array([])
    save_caca_pair_distributions = bool(config.get("save_caca_pair_distributions", False))
    caca_pair_pred_pool = (
        np.concatenate([r["_caca_pair_pred"] for r in per_repeat_metrics], axis=0).astype(np.float32, copy=False)
        if save_caca_pair_distributions and all(r["_caca_pair_pred"].size for r in per_repeat_metrics)
        else np.array([])
    )
    caca_pred_pool_before_shake = (
        np.concatenate([compute_caca_distances(t) for t in pre_shake_trajectories], axis=0).astype(np.float32, copy=False)
        if pre_shake_trajectories is not None
        else np.array([])
    )
    caca_pair_pred_pool_before_shake = (
        np.concatenate([compute_caca_pair_distances(t) for t in pre_shake_trajectories], axis=0).astype(np.float32, copy=False)
        if save_caca_pair_distributions and pre_shake_trajectories is not None
        else np.array([])
    )
    pair_min_pred_pool = np.concatenate([r["_pair_min_pred"] for r in per_repeat_metrics], axis=0) \
        if all(r["_pair_min_pred"].size for r in per_repeat_metrics) else np.array([])
    pair_count_pred_lt_3p5_pool = np.concatenate([r["_pair_count_pred_lt_3p5"] for r in per_repeat_metrics], axis=0) \
        if all(r["_pair_count_pred_lt_3p5"].size for r in per_repeat_metrics) else np.array([])
    pair_count_pred_lt_3p0_pool = np.concatenate([r["_pair_count_pred_lt_3p0"] for r in per_repeat_metrics], axis=0) \
        if all(r["_pair_count_pred_lt_3p0"].size for r in per_repeat_metrics) else np.array([])
    pair_count_pred_lt_2p5_pool = np.concatenate([r["_pair_count_pred_lt_2p5"] for r in per_repeat_metrics], axis=0) \
        if all(r["_pair_count_pred_lt_2p5"].size for r in per_repeat_metrics) else np.array([])
    lddt_pred_pool = np.concatenate([r["_lddt_pred"] for r in per_repeat_metrics], axis=0) \
        if all(r["_lddt_pred"].size for r in per_repeat_metrics) else np.array([])
    segment_min_pred_pool = np.concatenate([r["_segment_min_pred"] for r in per_repeat_metrics], axis=0) \
        if all(r["_segment_min_pred"].size for r in per_repeat_metrics) else np.array([])
    segment_count_pred_lt_0p5_pool = np.concatenate([r["_segment_count_pred_lt_0p5"] for r in per_repeat_metrics], axis=0) \
        if all(r["_segment_count_pred_lt_0p5"].size for r in per_repeat_metrics) else np.array([])
    segment_count_pred_lt_1p0_pool = np.concatenate([r["_segment_count_pred_lt_1p0"] for r in per_repeat_metrics], axis=0) \
        if all(r["_segment_count_pred_lt_1p0"].size for r in per_repeat_metrics) else np.array([])

    oracle = compute_oracle_baseline(sample, anchor, config, hist_bins, include_spectral)

    result = {
        "dataset": sample["dataset"],
        "domain_id": sample["domain_id"],
        "target_id": sample["target_id"],
        "temp": int(sample["temp"]),
        "n_repeats": n_repeats,
        "n_ref_replicates": len(reps),
        "n_pred_frames": int(pred_trajectories[0].shape[0]),
        "n_ref_frames_per_replicate": int(reps[0].shape[0]),
        "runtime_sec": float(inference_sec),
        "runtime_sec_per_repeat": float(inference_sec / max(n_repeats, 1)),
    }
    result.update(_average_metric_repeats(per_repeat_metrics, include_spectral=include_spectral))

    # Repeat-averaged signed PC cosine can be driven toward zero by arbitrary
    # PCA sign flips across repeats. Recompute the AlphaFlow-style PC metrics on
    # the concatenated prediction ensemble as a more stable one-ensemble view.
    pred_concat = np.concatenate(pred_trajectories, axis=0).astype(np.float64)
    ref_pooled = np.concatenate(reps, axis=0).astype(np.float64)
    pred_concat_aligned = kabsch_align_traj(pred_concat, anchor.astype(np.float64))
    ref_pooled_aligned = kabsch_align_traj(ref_pooled, anchor.astype(np.float64))

    # MarS-FM-style per-target RMSF profile comparison: build one RMSF profile
    # for the full generated ensemble and compare it to the pooled MD ensemble.
    pred_concat_rmsf = compute_rmsf(pred_concat_aligned)
    ref_pooled_rmsf = compute_rmsf(ref_pooled_aligned)
    per_target_rmsf_r_concat = pearson_corr(ref_pooled_rmsf, pred_concat_rmsf)
    per_target_rmsf_sp_concat = spearman_corr(ref_pooled_rmsf, pred_concat_rmsf)
    concat_seed = hash((sample["target_id"], "concat_alphaflow_pc")) & 0xFFFFFFFF
    (
        _,
        _,
        alphaflow_pc_sim_concat,
        alphaflow_ref_pc1_pc2_ratio_concat,
        alphaflow_pred_pc1_pc2_ratio_concat,
    ) = compute_alphaflow_pca_metrics(
        pred_concat_aligned,
        ref_pooled_aligned,
        np.random.default_rng(np.uint32(concat_seed)),
    )

    # Override the repeat-averaged RMSF profile statistics with the
    # repeat-concatenated ensemble values so downstream JSON consumers, tables,
    # and figure notebooks default to the MarS-FM-compatible definition.
    result["rmsf_pred_mean_mean"] = float(pred_concat_rmsf.mean())
    result["rmsf_pred_mean_std"] = float("nan")
    result["rmsf_ref_mean_mean"] = float(ref_pooled_rmsf.mean())
    result["rmsf_ref_mean_std"] = float("nan")
    result["per_target_rmsf_r_mean"] = per_target_rmsf_r_concat
    result["per_target_rmsf_r_std"] = float("nan")
    result["per_target_rmsf_spearman_mean"] = per_target_rmsf_sp_concat
    result["per_target_rmsf_spearman_std"] = float("nan")

    result["alphaflow_pc_sim_concat_mean"] = alphaflow_pc_sim_concat
    result["alphaflow_pc_sim_concat_std"] = float("nan")
    result["alphaflow_pc_sim_concat_abs_mean"] = (
        abs(alphaflow_pc_sim_concat) if math.isfinite(alphaflow_pc_sim_concat) else float("nan")
    )
    result["alphaflow_pc_sim_concat_abs_std"] = float("nan")
    result["alphaflow_ref_pc1_pc2_ratio_concat_mean"] = alphaflow_ref_pc1_pc2_ratio_concat
    result["alphaflow_ref_pc1_pc2_ratio_concat_std"] = float("nan")
    result["alphaflow_pred_pc1_pc2_ratio_concat_mean"] = alphaflow_pred_pc1_pc2_ratio_concat
    result["alphaflow_pred_pc1_pc2_ratio_concat_std"] = float("nan")

    for k, v in oracle.items():
        result[f"oracle_{k}"] = v

    # Store the ensemble RMSF profiles used for the MarS-FM-style global /
    # per-target RMSF calculations, while keeping the old repeat-average profile
    # for debugging/comparison.
    result["_pred_rmsf_mean"] = pred_concat_rmsf
    result["_pred_rmsf_mean_repeat_avg"] = pred_rmsfs.mean(axis=0)
    result["_ref_rmsf"] = ref_pooled_rmsf
    result["_pairwise_rmsd_pred_pool"] = pair_pred_pool
    result["_pairwise_rmsd_ref_pool"] = per_repeat_metrics[0]["_pair_ref"]
    result["_caca_pred_pool"] = caca_pred_pool
    result["_caca_pred_pool_before_shake"] = caca_pred_pool_before_shake
    result["_caca_ref_pool"] = per_repeat_metrics[0]["_caca_ref"]
    result["_caca_pair_pred_pool"] = caca_pair_pred_pool
    result["_caca_pair_pred_pool_before_shake"] = caca_pair_pred_pool_before_shake
    result["_caca_pair_ref_pool"] = (
        per_repeat_metrics[0]["_caca_pair_ref"].astype(np.float32, copy=False)
        if save_caca_pair_distributions else np.array([])
    )
    result["_lddt_pred_pool"] = lddt_pred_pool
    result["_lddt_ref_pool"] = per_repeat_metrics[0]["_lddt_ref"]
    result["_pair_min_pred_pool"] = pair_min_pred_pool
    result["_pair_min_ref_pool"]  = per_repeat_metrics[0]["_pair_min_ref"]
    result["_pair_count_pred_lt_3p5_pool"] = pair_count_pred_lt_3p5_pool
    result["_pair_count_ref_lt_3p5_pool"] = per_repeat_metrics[0]["_pair_count_ref_lt_3p5"]
    result["_pair_count_pred_lt_3p0_pool"] = pair_count_pred_lt_3p0_pool
    result["_pair_count_ref_lt_3p0_pool"] = per_repeat_metrics[0]["_pair_count_ref_lt_3p0"]
    result["_pair_count_pred_lt_2p5_pool"] = pair_count_pred_lt_2p5_pool
    result["_pair_count_ref_lt_2p5_pool"] = per_repeat_metrics[0]["_pair_count_ref_lt_2p5"]
    result["_segment_min_pred_pool"] = segment_min_pred_pool
    result["_segment_min_ref_pool"] = per_repeat_metrics[0]["_segment_min_ref"]
    result["_segment_count_pred_lt_0p5_pool"] = segment_count_pred_lt_0p5_pool
    result["_segment_count_ref_lt_0p5_pool"] = per_repeat_metrics[0]["_segment_count_ref_lt_0p5"]
    result["_segment_count_pred_lt_1p0_pool"] = segment_count_pred_lt_1p0_pool
    result["_segment_count_ref_lt_1p0_pool"] = per_repeat_metrics[0]["_segment_count_ref_lt_1p0"]
    if bool(config.get("save_trajectories", False)):
        result["_pred_trajectories"] = np.stack(pred_trajectories, axis=0)

    return result

# =============================================================================
# Dataset-level summaries (Optimized for vectorized operations)
# =============================================================================
def summarise_dataset(
    per_target: list[dict], dataset_name: str, include_spectral: bool,
) -> dict:
    if not per_target:
        return {}
    
    keys = list(_SCALAR_KEYS)
    if include_spectral:
        keys += _SPEC_KEYS

    summary = {f"{dataset_name}/n_targets": len(per_target)}
    
    # Pre-extract values to avoid repetitive dict lookups
    for k in keys:
        means = np.array([r.get(f"{k}_mean", np.nan) for r in per_target], dtype=np.float64)
        stds = np.array([r.get(f"{k}_std", np.nan) for r in per_target], dtype=np.float64)
        oracle_vals = np.array([r.get(f"oracle_{k}_mean", np.nan) for r in per_target], dtype=np.float64)
        
        summary.update({
            f"{dataset_name}/{k}_median": median_or_nan(means),
            f"{dataset_name}/{k}_mean": mean_or_nan(means),
            f"{dataset_name}/{k}_mean_of_stds": mean_or_nan(stds),
            f"{dataset_name}/oracle/{k}_median": median_or_nan(oracle_vals),
            f"{dataset_name}/oracle/{k}_mean": mean_or_nan(oracle_vals),
        })

    # Global (concatenated) RMSF correlation across all proteins
    if all("_pred_rmsf_mean" in r and "_ref_rmsf" in r for r in per_target):
        pred = np.concatenate([r["_pred_rmsf_mean"] for r in per_target], axis=0)
        ref = np.concatenate([r["_ref_rmsf"] for r in per_target], axis=0)
        summary[f"{dataset_name}/global_rmsf_r"] = pearson_corr(ref, pred)
        summary[f"{dataset_name}/global_rmsf_spearman"] = spearman_corr(ref, pred)

    # Per-temperature breakdown for mdCATH
    temps = sorted({int(r["temp"]) for r in per_target if "temp" in r})
    if dataset_name == "mdcath" and len(temps) > 1:
        for t in temps:
            subset = [r for r in per_target if int(r["temp"]) == t]
            if not subset:
                continue
            summary[f"{dataset_name}/temp_{t}/n_targets"] = len(subset)
            for k in keys:
                means = np.array([r.get(f"{k}_mean", np.nan) for r in subset], dtype=np.float64)
                summary[f"{dataset_name}/temp_{t}/{k}_median"] = median_or_nan(means)
                summary[f"{dataset_name}/temp_{t}/{k}_mean"] = mean_or_nan(means)

    # Pearson correlations across targets
    def _pair_exact(key_pred, key_ref):
        pred = np.asarray([r.get(key_pred, np.nan) for r in per_target], dtype=np.float64)
        ref = np.asarray([r.get(key_ref, np.nan) for r in per_target], dtype=np.float64)
        # Mask out NaNs before correlation
        mask = ~(np.isnan(pred) | np.isnan(ref))
        if not np.any(mask):
            return np.nan
        return pearson_corr(ref[mask], pred[mask])

    pc_sim_vals = np.asarray([r.get("pc_sim_mean", np.nan) for r in per_target], dtype=np.float64)
    pc_sim_mask = np.isfinite(pc_sim_vals)
    pc_sim_gt_0_5_pct = (
        float(100.0 * np.mean(pc_sim_vals[pc_sim_mask] > 0.5))
        if np.any(pc_sim_mask) else np.nan
    )

    alphaflow_pc_sim_vals = np.asarray(
        [r.get("alphaflow_pc_sim_mean", np.nan) for r in per_target],
        dtype=np.float64,
    )
    alphaflow_pc_sim_mask = np.isfinite(alphaflow_pc_sim_vals)
    alphaflow_pc_sim_gt_0_5_pct = (
        float(100.0 * np.mean(alphaflow_pc_sim_vals[alphaflow_pc_sim_mask] > 0.5))
        if np.any(alphaflow_pc_sim_mask) else np.nan
    )

    alphaflow_pc_sim_abs_vals = np.asarray(
        [r.get("alphaflow_pc_sim_abs_mean", np.nan) for r in per_target],
        dtype=np.float64,
    )
    alphaflow_pc_sim_abs_mask = np.isfinite(alphaflow_pc_sim_abs_vals)
    alphaflow_pc_sim_abs_gt_0_5_pct = (
        float(100.0 * np.mean(alphaflow_pc_sim_abs_vals[alphaflow_pc_sim_abs_mask] > 0.5))
        if np.any(alphaflow_pc_sim_abs_mask) else np.nan
    )

    alphaflow_pc_sim_concat_vals = np.asarray(
        [r.get("alphaflow_pc_sim_concat_mean", np.nan) for r in per_target],
        dtype=np.float64,
    )
    alphaflow_pc_sim_concat_mask = np.isfinite(alphaflow_pc_sim_concat_vals)
    alphaflow_pc_sim_concat_gt_0_5_pct = (
        float(100.0 * np.mean(alphaflow_pc_sim_concat_vals[alphaflow_pc_sim_concat_mask] > 0.5))
        if np.any(alphaflow_pc_sim_concat_mask) else np.nan
    )

    alphaflow_pc_sim_concat_abs_vals = np.asarray(
        [r.get("alphaflow_pc_sim_concat_abs_mean", np.nan) for r in per_target],
        dtype=np.float64,
    )
    alphaflow_pc_sim_concat_abs_mask = np.isfinite(alphaflow_pc_sim_concat_abs_vals)
    alphaflow_pc_sim_concat_abs_gt_0_5_pct = (
        float(
            100.0
            * np.mean(alphaflow_pc_sim_concat_abs_vals[alphaflow_pc_sim_concat_abs_mask] > 0.5)
        )
        if np.any(alphaflow_pc_sim_concat_abs_mask) else np.nan
    )

    summary.update({
        f"{dataset_name}/pairwise_rmsd_r_across_targets": _pair_exact(
            "pairwise_rmsd_pred_mean_mean", "pairwise_rmsd_ref_mean_mean"
        ),
        f"{dataset_name}/dG_fold_r_across_targets": _pair_exact(
            "dG_fold_pred_mean", "dG_fold_ref_mean"
        ),
        f"{dataset_name}/rmsf_r_across_targets": _pair_exact(
            "rmsf_pred_mean_mean", "rmsf_ref_mean_mean"
        ),
        f"{dataset_name}/pc_sim_gt_0_5_pct": pc_sim_gt_0_5_pct,
        f"{dataset_name}/alphaflow_pc_sim_gt_0_5_pct": alphaflow_pc_sim_gt_0_5_pct,
        f"{dataset_name}/alphaflow_pc_sim_abs_gt_0_5_pct": alphaflow_pc_sim_abs_gt_0_5_pct,
        f"{dataset_name}/alphaflow_pc_sim_concat_gt_0_5_pct": alphaflow_pc_sim_concat_gt_0_5_pct,
        f"{dataset_name}/alphaflow_pc_sim_concat_abs_gt_0_5_pct": alphaflow_pc_sim_concat_abs_gt_0_5_pct,
    })

    return summary


def _strip_private(d: dict) -> dict:
    return {k: v for k, v in d.items() if not str(k).startswith("_")}


# Main
# =============================================================================

def main(config: dict):
    rank, local_rank, world_size, is_distributed, device = init_process()
    seed_everything(int(config.get("seed", 42)) + rank)

    if rank == 0:
        print("Run Name:", config["run_name"])
        print("Final Evaluation Config:")
        pprint(config, sort_dicts=True, indent=4)

    runtime = build_model_stack(config, device)
    dtype_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else contextlib.nullcontext()
    )

    # Extraction settings
    coords_type = config.get("coords_type", "ca")
    include_angles = bool(config.get("include_angles", False))
    use_zarr = bool(config.get("use_zarr", True))
    mdcath_total = int(config.get("mdcath_total_frames", MDCATH_TOTAL_FRAMES))
    atlas_total = int(config.get("atlas_total_frames", ATLAS_TOTAL_FRAMES))
    atlas_stride = int(config.get("atlas_stride", 100))
    num_workers = int(config.get("num_workers", min(8, os.cpu_count() or 4)))

    exclude_ids = read_id_set(config.get("exclude_ids_path"))
    atlas_extra_exclude = read_id_set(config.get("atlas_exclude_ids_path"))
    atlas_exclude_ids = (exclude_ids or set()) | (atlas_extra_exclude or set()) or None

    # Target Discovery (mdCATH & ATLAS) 
    # ------------------------------------
    mdcath_targets, atlas_targets = [], []
    mdcath_handle, atlas_handle = None, None

    if config.get("eval_mdcath", True):
        default_test_ids_path = os.path.join(config["checkpoint_dir"], "test_ids.txt")
        test_ids_path = config.get("test_ids_path") or (
            default_test_ids_path if os.path.exists(default_test_ids_path) else None
        )
        mdcath_test_ids = merge_id_sets(config.get("combined_test_ids_path"), test_ids_path)
        if not use_zarr or not config.get("mdcath_zarr_path"):
            raise ValueError("Evaluation script currently supports zarr-backed mdCATH only.")
        mdcath_handle, discovered = list_mdcath_targets_zarr(
            config["mdcath_zarr_path"], mdcath_test_ids, exclude_ids=exclude_ids
        )
        mdcath_targets = [(d, t) for d, t, _ in discovered]

    if config.get("eval_atlas", True):
        if not use_zarr or not config.get("atlas_zarr_path"):
            raise ValueError("Evaluation script currently supports zarr-backed ATLAS only.")
        atlas_test_ids = merge_id_sets(config.get("combined_test_ids_path"), config.get("atlas_test_ids_path"))
        atlas_handle, discovered = list_atlas_domains_zarr(
            config["atlas_zarr_path"], exclude_ids=atlas_exclude_ids
        )
        atlas_targets = [d for d, _ in discovered if atlas_test_ids is None or d in atlas_test_ids]

    # Apply limits and combine
    cap = config.get("max_domains")
    if cap is not None:
        mdcath_targets, atlas_targets = mdcath_targets[:int(cap)], atlas_targets[:int(cap)]

    all_targets = [("mdcath", d, t) for d, t in mdcath_targets] + [("atlas", d, None) for d in atlas_targets]
    if not all_targets:
        raise RuntimeError("No targets discovered from mdCATH or ATLAS.")

    local_targets = shard_items(all_targets, rank, world_size)
    log(f"Rank {rank} processing {len(local_targets)} targets.")

    # 1. Parallel load all samples into memory
    # ------------------------------------------------------------------
    def load_target(tup):
        source, domain_id, temp = tup
        try:
            if source == "mdcath":
                return load_mdcath_target_zarr(mdcath_handle, domain_id, temp, coords_type, include_angles, total_frames=mdcath_total)
            return load_atlas_target_zarr(atlas_handle, domain_id, coords_type, include_angles, total_frames=atlas_total, atlas_stride=atlas_stride)
        except Exception as e:
            print(f"[rank {rank}] Skipping {source}:{domain_id}: {e}", flush=True)
            return None

    all_samples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # map preserves order which guarantees sync with local_targets
        for sample in executor.map(load_target, local_targets):
            if sample is not None:
                all_samples.append(sample)

    print(f"[rank {rank}] Loaded {len(all_samples)} samples.", flush=True)

    # 2. Batched inference over repeats
    # ------------------------------------------------------------------
    batch_size = int(config.get("batch_size", 1))
    n_repeats = int(config.get("n_repeats", 5))
    mdcath_pred = int(config.get("mdcath_pred_frames", MDCATH_PRED_FRAMES))
    atlas_pred = int(config.get("atlas_pred_frames", ATLAS_PRED_FRAMES))
    win_pos_b = float(MDCATH_WINDOW_START_FRAMES[1]) / max(mdcath_total - 1, 1)

    accumulated_trajs: list[list[np.ndarray]] = [[] for _ in all_samples]
    spectral_volume_rows: list[dict] = []
    mdcath_idxs = [i for i, s in enumerate(all_samples) if s["dataset"] == "mdcath"]
    all_idxs = list(range(len(all_samples)))

    def run_batched_pass(indices: list[int], win_pos_scalar: float) -> dict[int, dict]:
        out = {}
        wp = torch.full((batch_size,), win_pos_scalar, dtype=torch.float32, device=device)
        for start in range(0, len(indices), batch_size):
            chunk_idxs = indices[start:start + batch_size]
            chunk_samples = [all_samples[i] for i in chunk_idxs]
            
            # Slice wp if chunk is smaller than batch_size
            curr_wp = wp[:len(chunk_idxs)] if len(chunk_idxs) < batch_size else wp
            pred_list = infer_batch_eval(chunk_samples, curr_wp, runtime, config, device, dtype_ctx)
            
            for i, pred_item in zip(chunk_idxs, pred_list):
                out[i] = pred_item
        return out

    t_infer_start = time.perf_counter()
    for rep_idx in range(n_repeats):
        seed_everything(int(config.get("seed", 42)) + rep_idx * 10007 + rank)
        
        pass_a = run_batched_pass(all_idxs, 0.0)
        pass_b = run_batched_pass(mdcath_idxs, win_pos_b)

        for i, sample in enumerate(all_samples):
            if sample["dataset"] == "mdcath":
                full = np.concatenate([pass_a[i]["ca"], pass_b[i]["ca"]], axis=0)[:mdcath_pred]
            else:
                full = pass_a[i]["ca"][:atlas_pred]
            accumulated_trajs[i].append(full)
            for row in pass_a[i].get("spectral_volume_rows", []):
                row["repeat"] = int(rep_idx)
                spectral_volume_rows.append(row)
            if sample["dataset"] == "mdcath":
                for row in pass_b[i].get("spectral_volume_rows", []):
                    row["repeat"] = int(rep_idx)
                    spectral_volume_rows.append(row)

    inference_sec = time.perf_counter() - t_infer_start
    avg_inf_sec = inference_sec / max(len(all_samples), 1)
    avg_inf_sec_per_repeat = avg_inf_sec / max(n_repeats, 1)
    print(
        f"[rank {rank}] Inference done in {inference_sec:.1f}s "
        f"({avg_inf_sec:.3f}s/target, {avg_inf_sec_per_repeat:.3f}s/target/repeat).",
        flush=True,
    )

    # 3. Parallel compute metrics and I/O per protein
    # ------------------------------------------------------------------
    per_target_results: list[dict] = []
    save_trajs = bool(config.get("save_trajectories", False))
    traj_out_dir = os.path.join(config["checkpoint_dir"], "eval_trajectories")
    if save_trajs:
        os.makedirs(traj_out_dir, exist_ok=True)

    def compute_metrics(i):
        sample = all_samples[i]
        tag = sample["target_id"]
        try:
            result = evaluate_from_trajectories(sample, accumulated_trajs[i], config, inference_sec=avg_inf_sec)
            
            # Thread-safe write out
            if save_trajs and "_pred_trajectories" in result:
                file_path = os.path.join(traj_out_dir, f"{sample['dataset']}_{tag.replace('@', '_')}.npz")
                np.savez_compressed(
                    file_path,
                    pred=result["_pred_trajectories"],
                    ref=np.stack([r.float().cpu().numpy() for r in sample["replicate_ca"]], axis=0),
                    native_ca=sample["native_bb"][:, 1, :].float().cpu().numpy(),
                )
            return result
        except Exception as e:
            print(f"[rank {rank}] Metrics failed {tag}: {e}", flush=True)
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for res in executor.map(compute_metrics, range(len(all_samples))):
            if res is not None:
                per_target_results.append(res)

    # Gather across ranks
    gathered = [None for _ in range(world_size)] if is_distributed else [per_target_results]
    gathered_spectral_rows = [None for _ in range(world_size)] if is_distributed else [spectral_volume_rows]
    if is_distributed:
        dist.all_gather_object(gathered, per_target_results)
        dist.all_gather_object(gathered_spectral_rows, spectral_volume_rows)
    merged = [item for chunk in gathered for item in chunk]
    merged_spectral_rows = [item for chunk in gathered_spectral_rows for item in chunk]

    if rank != 0:
        maybe_barrier()
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    # Summaries 
    include_spec = bool(config.get("compute_spectral_metrics", True))
    summary = {
        "evaluation/run_name": config["run_name"],
        "evaluation/n_repeats": n_repeats,
        "evaluation/total_targets": len(merged),
        "evaluation/inference_sec_total_rank0": float(inference_sec),
        "evaluation/inference_sec_per_loaded_target_rank0": float(avg_inf_sec),
        "evaluation/inference_sec_per_loaded_target_repeat_rank0": float(avg_inf_sec_per_repeat),
    }
    runtime_vals = np.asarray([r.get("runtime_sec", np.nan) for r in merged], dtype=np.float64)
    runtime_repeat_vals = np.asarray(
        [r.get("runtime_sec_per_repeat", np.nan) for r in merged],
        dtype=np.float64,
    )
    runtime_mask = np.isfinite(runtime_vals)
    runtime_repeat_mask = np.isfinite(runtime_repeat_vals)
    summary["evaluation/inference_sec_per_target_mean"] = (
        float(runtime_vals[runtime_mask].mean()) if np.any(runtime_mask) else float("nan")
    )
    summary["evaluation/inference_sec_per_target_repeat_mean"] = (
        float(runtime_repeat_vals[runtime_repeat_mask].mean())
        if np.any(runtime_repeat_mask) else float("nan")
    )
    summary.update(summarise_dataset([r for r in merged if r["dataset"] == "mdcath"], "mdcath", include_spec))
    summary.update(summarise_dataset([r for r in merged if r["dataset"] == "atlas"], "atlas", include_spec))
    summary.update(summarise_spectral_volume_rows(merged_spectral_rows))

    # Out
    ckpt_dir = config["checkpoint_dir"]
    _suffix = "_bond_normalised" if bool(config.get("normalize_caca_bonds", False)) else ""
    summary_path = os.path.join(ckpt_dir, f"evaluation_summary{_suffix}.json")

    summary_json = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in summary.items()}
    with open(summary_path, "w") as f:
        json.dump(summary_json, f, indent=4)

    with open(os.path.join(ckpt_dir, f"evaluation_per_target{_suffix}.json"), "w") as f:
        json.dump([_strip_private(r) for r in merged], f, indent=2, default=str)

    spectral_summary = summarise_spectral_volume_rows(merged_spectral_rows)
    with open(os.path.join(ckpt_dir, f"evaluation_spectral_volume_summary{_suffix}.json"), "w") as f:
        json.dump(
            {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in spectral_summary.items()},
            f,
            indent=2,
        )
    spectral_rows_path = os.path.join(ckpt_dir, f"evaluation_spectral_volume_metrics{_suffix}.csv")
    if merged_spectral_rows:
        fieldnames = [
            "target_id", "dataset", "domain_id", "temp", "repeat", "window_start",
            "window_size", "band", "n_ref_replicates", "n_residues", "n_freqs", "n_channels",
        ]
        metric_fields = sorted(k for k in merged_spectral_rows[0].keys() if k not in fieldnames)
        with open(spectral_rows_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames + metric_fields)
            writer.writeheader()
            writer.writerows(merged_spectral_rows)
    else:
        with open(spectral_rows_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["note"])
            writer.writeheader()
            writer.writerow({"note": "no matching full-length DCT spectral windows were available"})

    # Dump raw distributions
    torch.save({"per_target": [{
        "target_id": r["target_id"], "dataset": r["dataset"], "temp": r.get("temp"),
        "pred_rmsf_mean": r.get("_pred_rmsf_mean"), "ref_rmsf": r.get("_ref_rmsf"),
        "pairwise_rmsd_pred_pool": r.get("_pairwise_rmsd_pred_pool"),
        "pairwise_rmsd_ref_pool": r.get("_pairwise_rmsd_ref_pool"),
        "caca_pred_pool": r.get("_caca_pred_pool"),
        "caca_pred_pool_before_shake": r.get("_caca_pred_pool_before_shake"),
        "caca_pred_pool_after_shake": r.get("_caca_pred_pool") if _suffix else None,
        "caca_ref_pool": r.get("_caca_ref_pool"),
        "caca_pair_pred_pool": r.get("_caca_pair_pred_pool"),
        "caca_pair_pred_pool_before_shake": r.get("_caca_pair_pred_pool_before_shake"),
        "caca_pair_pred_pool_after_shake": r.get("_caca_pair_pred_pool") if _suffix else None,
        "caca_pair_ref_pool": r.get("_caca_pair_ref_pool"),
        "lddt_pred_pool": r.get("_lddt_pred_pool"),
        "lddt_ref_pool": r.get("_lddt_ref_pool"),
        "pair_min_pred_pool": r.get("_pair_min_pred_pool"),
        "pair_min_ref_pool":  r.get("_pair_min_ref_pool"),
        "pair_count_pred_lt_3p5_pool": r.get("_pair_count_pred_lt_3p5_pool"),
        "pair_count_ref_lt_3p5_pool": r.get("_pair_count_ref_lt_3p5_pool"),
        "pair_count_pred_lt_3p0_pool": r.get("_pair_count_pred_lt_3p0_pool"),
        "pair_count_ref_lt_3p0_pool": r.get("_pair_count_ref_lt_3p0_pool"),
        "pair_count_pred_lt_2p5_pool": r.get("_pair_count_pred_lt_2p5_pool"),
        "pair_count_ref_lt_2p5_pool": r.get("_pair_count_ref_lt_2p5_pool"),
        "segment_min_pred_pool": r.get("_segment_min_pred_pool"),
        "segment_min_ref_pool": r.get("_segment_min_ref_pool"),
        "segment_count_pred_lt_0p5_pool": r.get("_segment_count_pred_lt_0p5_pool"),
        "segment_count_ref_lt_0p5_pool": r.get("_segment_count_ref_lt_0p5_pool"),
        "segment_count_pred_lt_1p0_pool": r.get("_segment_count_pred_lt_1p0_pool"),
        "segment_count_ref_lt_1p0_pool": r.get("_segment_count_ref_lt_1p0_pool"),
    } for r in merged]}, os.path.join(ckpt_dir, f"evaluation_raw_distributions{_suffix}.pt"))

    maybe_barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
    print("FINISHED EVALUATION")


# CLI
# =============================================================================

def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Unified test-set evaluation for DynaMode checkpoints.")
    
    # Core
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Protocol
    parser.add_argument("--n_repeats", type=int, default=None)
    parser.add_argument("--eval_mdcath", action="store_true", default=None)
    parser.add_argument("--eval_atlas", action="store_true", default=None)
    parser.add_argument("--no_mdcath", action="store_true", default=None)
    parser.add_argument("--no_atlas", action="store_true", default=None)
    parser.add_argument("--mdcath_total_frames", type=int, default=None)
    parser.add_argument("--mdcath_pred_frames", type=int, default=None)
    parser.add_argument("--atlas_total_frames", type=int, default=None)
    parser.add_argument("--atlas_pred_frames", type=int, default=None)
    parser.add_argument("--atlas_stride", type=int, default=None)
    parser.add_argument("--pairwise_rmsd_samples", type=int, default=None)
    parser.add_argument("--hist_bins", type=int, default=None)
    parser.add_argument("--compute_spectral_metrics", action="store_true", default=None)
    parser.add_argument("--no_spectral_metrics", action="store_true", default=None)
    parser.add_argument("--compute_spectral_volume_metrics", action="store_true", default=None)
    parser.add_argument("--no_spectral_volume_metrics", action="store_true", default=None)
    parser.add_argument("--spectral_volume_hist_bins", type=int, default=None)
    parser.add_argument("--save_trajectories", action="store_true", default=None)
    parser.add_argument(
        "--save_caca_pair_distributions",
        action="store_true",
        default=None,
        help="Store every adjacent CA-CA bond distance in evaluation_raw_distributions*.pt. "
             "This is useful for bond histograms but can add several GB on full SpecConv runs.",
    )
    parser.add_argument("--normalize_caca_bonds", action="store_true", default=None)
    parser.add_argument("--caca_bond_target", type=float, default=None)
    parser.add_argument("--caca_bond_tolerance", type=float, default=None,
                        help="Half-width of the 'don't touch' band around the target bond length. "
                             "~0.05 A reproduces MD thermal spread.")
    parser.add_argument("--caca_bond_n_iter", type=int, default=None)
    parser.add_argument("--caca_bond_step", type=float, default=None,
                        help="Per-iteration SHAKE correction fraction (0 < step <= 0.5).")

    # Data sources
    parser.add_argument("--use_zarr", action="store_true", default=None)
    parser.add_argument("--mdcath_zarr_path", type=str, default=None)
    parser.add_argument("--atlas_zarr_path", type=str, default=None)
    parser.add_argument("--test_ids_path", type=str, default=None)
    parser.add_argument("--atlas_test_ids_path", type=str, default=None)
    parser.add_argument("--combined_test_ids_path", type=str, default=None)
    parser.add_argument("--exclude_ids_path", type=str, default=None)
    parser.add_argument("--atlas_exclude_ids_path", type=str, default=None)
    parser.add_argument("--max_domains", type=int, default=None)

    # Model / inference
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--coords_type", type=str, default=None)
    parser.add_argument("--include_angles", action="store_true", default=None)
    parser.add_argument("--displacement", action="store_true", default=None)
    parser.add_argument(
        "--representation",
        type=str,
        default=None,
        choices=[
            "raw_coords",
            "displacement",
            "unit_chain_mean_lengths",
            "unit_chain_native_lengths",
            "unit_chain_pred_lengths",
        ],
    )
    parser.add_argument("--representation_length_min", type=float, default=None)
    parser.add_argument("--representation_length_max", type=float, default=None)
    parser.add_argument("--representation_length_residual_max", type=float, default=None)
    parser.add_argument(
        "--freq_normalization",
        type=str,
        default=None,
        choices=["auto", "none", "global", "conditioned"],
    )
    parser.add_argument(
        "--dc_residualization",
        type=str,
        default=None,
        choices=["auto", "none", "bucket", "per_residue"],
    )
    parser.add_argument(
        "--aniso_source",
        type=str,
        default=None,
        choices=["auto", "none", "freq_scales", "artifact"],
    )
    parser.add_argument("--use_DCT", action="store_true", default=None)
    parser.add_argument("--model_type", type=str, default=None, choices=SUPPORTED_MODEL_TYPES)
    parser.add_argument("--top_k_freqs", type=int, default=None)
    parser.add_argument("--freq_hidden_size", type=int, default=None)
    parser.add_argument("--spectral_modes", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--prediction_target", type=str, default=None)
    parser.add_argument("--cond_dim", type=int, default=None)
    parser.add_argument("--use_seq_conditioning", action="store_true", default=None)
    parser.add_argument("--seq_embed_dim", type=int, default=None)
    parser.add_argument("--use_ss_conditioning", action="store_true", default=None)
    parser.add_argument("--ss_embed_dim", type=int, default=None)
    parser.add_argument("--use_hilbert_spatial", action="store_true", default=None)
    parser.add_argument("--use_hilbert_spatial_dct", action="store_true", default=None)
    parser.add_argument("--hilbert_mode", type=str, default=None)
    parser.add_argument("--use_rmsf_prior_gain", action="store_true", default=None)
    parser.add_argument("--use_low_k_correction_head", action="store_true", default=None)
    parser.add_argument("--low_k_correction_modes", type=str, default=None)
    parser.add_argument("--band_edges", type=str, default=None)
    parser.add_argument("--amp_head_context_modes", type=int, default=None)
    parser.add_argument("--amp_head_target_modes", type=int, default=None)
    parser.add_argument("--amp_head_d_model", type=int, default=None)
    parser.add_argument("--amp_head_depth", type=int, default=None)
    parser.add_argument("--amp_head_num_heads", type=int, default=None)
    parser.add_argument("--amp_head_mlp_ratio", type=float, default=None)
    parser.add_argument("--amp_head_attn_dropout", type=float, default=None)
    parser.add_argument("--amp_head_use_rmsf_prior", action="store_true", default=None)
    parser.add_argument("--use_shake", action="store_true", default=None)
    parser.add_argument("--shake_n_iter", type=int, default=None)
    parser.add_argument("--shake_target", type=float, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--num_ode_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--schedule", type=str, default=None)
    parser.add_argument("--shift_value", type=str, default=None)
    parser.add_argument("--min_snr_gamma", type=float, default=None)
    parser.add_argument("--aniso_gamma", type=float, default=None)
    parser.add_argument("--noise_schedule", type=str, default=None)
    parser.add_argument("--noise_space", type=str, default=None, choices=["raw_gamma", "model_normalized"])
    parser.add_argument("--noise_band_edges", type=str, default=None)
    parser.add_argument("--noise_group_model_multipliers", type=str, default=None)
    parser.add_argument("--noise_target_crossings", type=str, default=None)
    parser.add_argument("--noise_anchor_band", type=str, default=None)
    parser.add_argument("--noise_power_normalization", type=str, default=None, choices=["raw_mean_square"])
    parser.add_argument("--noise_auto_shift", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--freq_scales_path", type=str, default=None)
    parser.add_argument("--aniso_scales_path", type=str, default=None)

    args = parser.parse_args()

    # YAML base + CLI overrides
    config = {}
    base_config_path = resolve_base_config_path(args)
    if base_config_path is not None and os.path.exists(base_config_path):
        config.update(flatten_yaml_config(base_config_path))
        
    config = coerce_config_types(config)
    config.update({k: v for k, v in vars(args).items() if v is not None and k != "config"})
    config = coerce_config_types(config)

    # Essential Defaults
    if config.get("checkpoint_path") is None:
        raise ValueError("--checkpoint_path is required.")
        
    config.setdefault("coords_type", "ca")
    if config.get("representation") is None:
        config["representation"] = "displacement" if bool(config.get("displacement", True)) else "raw_coords"
    config["representation"] = canonical_representation(config["representation"])
    config["displacement"] = config["representation"] == "displacement"
    config["freq_normalization"] = canonical_freq_normalization(config.get("freq_normalization", "auto"))
    config["dc_residualization"] = canonical_dc_residualization(config.get("dc_residualization", "auto"))
    config["aniso_source"] = canonical_aniso_source(config.get("aniso_source", "auto"))
    config.setdefault("model_type", "spectral_dit_low_k")
    if config["model_type"] not in SUPPORTED_MODEL_TYPES:
        supported = ", ".join(SUPPORTED_MODEL_TYPES)
        raise ValueError(f"model_type must be one of: {supported}")
    config.setdefault("use_zarr", True)
    config.setdefault("n_repeats", 5)
    config.setdefault("pairwise_rmsd_samples", 10000)
    config.setdefault("hist_bins", 100)
    config.setdefault("atlas_stride", 100)
    config.setdefault("mdcath_total_frames", MDCATH_TOTAL_FRAMES)
    config.setdefault("mdcath_pred_frames", MDCATH_PRED_FRAMES)
    config.setdefault("atlas_total_frames", ATLAS_TOTAL_FRAMES)
    config.setdefault("atlas_pred_frames", ATLAS_PRED_FRAMES)

    if config.pop("no_mdcath", False): config["eval_mdcath"] = False
    if config.pop("no_atlas", False): config["eval_atlas"] = False
    if config.pop("no_spectral_metrics", False): config["compute_spectral_metrics"] = False
    if config.pop("no_spectral_volume_metrics", False): config["compute_spectral_volume_metrics"] = False
    
    config.setdefault("eval_mdcath", True)
    config.setdefault("eval_atlas", True)
    config.setdefault("compute_spectral_metrics", True)
    config.setdefault("compute_spectral_volume_metrics", True)
    config.setdefault("spectral_volume_hist_bins", config.get("hist_bins", 100))
    config.setdefault("save_caca_pair_distributions", False)

    if config.get("checkpoint_dir") is None:
        config["checkpoint_dir"] = resolve_checkpoint_dir(config)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["run_name"] = f"UNIFIED_EVAL_{date_str}_{config.get('run_name', '')}".strip("_")

    return config

if __name__ == "__main__":
    main(parse_args())
