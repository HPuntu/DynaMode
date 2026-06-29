"""Streamlined training loop for CNO spectral volume diffusion.

Supports x_0, v, and noise prediction targets. Validation uses full
inference and reports only: LDDT, CA-CA distance, Spearman RMSF,
spectral volume MSE, and JSD Ramachandran distributions.
"""

import os
import math
import re
import inspect
import contextlib
import shutil
from datetime import datetime, timedelta
from pprint import pprint
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy.stats import spearmanr
from dotenv import load_dotenv
import yaml
import wandb

load_dotenv()

from src.features.zarr_loader import ZarrTrajectoriesDataset, FeaturizerWindowZarr
from src.features.safe_data_loader import TrajectoriesDataset
from src.features.features import FeaturizerWindow, Aligner
from src.spectral.adapters import DCT, compute_frequency_stats
from src.spectral.representation import (
    CoordinateRepresentation,
    SpectralRepresentationPipeline,
    canonical_aniso_source,
    canonical_dc_residualization,
    canonical_freq_normalization,
    canonical_representation,
)
from src.models.diffusion import SpectralDiffusion
from src.models.noise_schedule import format_noise_diagnostics, resolve_noise_schedule
from src.metrics import compute_batch_caca_dist, compute_batch_lddt, RamaValidator
from src.utils import maintain_cache_size, no_spill_stratified_split
from src.models.model_wrapper import UnifiedWrapper
from src.models.registry import (
    SpectralDiTConfig, SpectralConvDiTConfig,
    SpectralConvSlowBranchConfig, SpectralConvBlockMixConfig,
    SpectralConvBlockMixAmplitudeConfig, SpectralConvBlockMixAmplitudeRefinedConfig,
    SpectralConvBlockMixAmplitudeBondGraphRefinedConfig,
    SpectralConvBlockMixAmplitudeSpectralGraphRefinedConfig,
    SpectralConvBlockMixAmplitudeSpecGraphConfig,
    SpectralConvBlockMixAmplitudeEGNNConfig,
    SpectralConvBlockMixSlowHybridConfig, SpectralConvBlockMixSlowHybridEGNNConfig,
    CascadeSpectralConfig,
    DualBranchConfig, FNOConfig, FNOManifoldConfig, HNOConfig, FNO2Config, FNO2BishopConfig,
    V14AConfig, V14BConfig, V14CConfig, V15Config, V16Config,
)
from src.models.conditioned_freq_scale import load_freq_scale_artifact
from src.models.models.v17.losses import v17_auxiliary_losses

# Key env variables for stable distributed training
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
BACKEND = "nccl"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MDTRAJ_NUM_THREADS"] = "1"
import numcodecs
numcodecs.blosc.use_threads = False

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')

date_str = datetime.now().strftime("%d.%m.%y")



# HELPERS
# -------
def init_wandb(project_name="pancakes_spectral_diffusion", run_name=None, config=None, id=None, resume=None):
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    resume_strategy = resume if id is not None else None
    wandb.init(project=project_name, name=run_name, config=config or {}, reinit=True, id=id, resume=resume_strategy)

def init_process():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=BACKEND,
        timeout=timedelta(seconds=3600),
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    if rank == 0:
        print(f"Distributed training initialized with {world_size} processes.")


def maybe_residualise_dc(spectral_adapter, x: torch.Tensor, temp: torch.Tensor, mask: torch.Tensor | None, coord_channels: int, per_residue_baseline: torch.Tensor | None = None):
    if spectral_adapter is None or x is None or not hasattr(spectral_adapter, "residualise_dc"):
        return x, None
    return spectral_adapter.residualise_dc(
        x, temp, mask,
        coord_channels=coord_channels,
        per_residue_baseline=per_residue_baseline,
    )


def maybe_restore_dc(spectral_adapter, x: torch.Tensor, dc_baseline: torch.Tensor | None, coord_channels: int):
    if (
        spectral_adapter is None
        or x is None
        or dc_baseline is None
        or not hasattr(spectral_adapter, "restore_dc")
    ):
        return x
    return spectral_adapter.restore_dc(x, dc_baseline, coord_channels=coord_channels)


def safe_vector_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-4) -> torch.Tensor:
    """Vector norm with a finite gradient at exactly-zero vectors."""
    return x.square().sum(dim=dim).clamp_min(float(eps) ** 2).sqrt()


class NonFiniteMonitor:
    """Forward/backward hook monitor for first non-finite activations.

    The hooks are intentionally opt-in: scanning every leaf-module tensor is
    useful for diagnosing NaN cascades, but too expensive for normal training.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        check_forward: bool = True,
        check_backward: bool = True,
        name_filter: str | None = None,
        max_modules: int = 0,
    ) -> None:
        self.enabled = bool(enabled)
        self.check_forward = bool(check_forward)
        self.check_backward = bool(check_backward)
        self.filters = [
            part.strip()
            for part in str(name_filter or "").split(",")
            if part.strip()
        ]
        self.max_modules = int(max_modules or 0)
        self.handles = []
        self.step = None
        self.epoch = None
        self.batch_idx = None
        self.forward_record = None
        self.backward_record = None

    def _matches(self, name: str) -> bool:
        return not self.filters or any(part in name for part in self.filters)

    def register(self, model: torch.nn.Module) -> int:
        if not self.enabled:
            return 0
        count = 0
        for name, module in model.named_modules():
            if not name or list(module.children()) or not self._matches(name):
                continue
            if self.max_modules > 0 and count >= self.max_modules:
                break
            if self.check_forward:
                self.handles.append(module.register_forward_hook(self._forward_hook(name)))
            if self.check_backward:
                self.handles.append(module.register_full_backward_hook(self._backward_hook(name)))
            count += 1
        return count

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset(self, *, step: int, epoch: int, batch_idx: int) -> None:
        self.step = int(step)
        self.epoch = int(epoch)
        self.batch_idx = int(batch_idx)
        self.forward_record = None
        self.backward_record = None

    def _forward_hook(self, name: str):
        def hook(module, inputs, output):
            if self.forward_record is None:
                self.forward_record = self._inspect_any(output, kind="forward", module_name=name)
        return hook

    def _backward_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            if self.backward_record is None:
                record = self._inspect_any(grad_output, kind="backward_grad_output", module_name=name)
                if record is None:
                    record = self._inspect_any(grad_input, kind="backward_grad_input", module_name=name)
                self.backward_record = record
        return hook

    def _inspect_any(self, value, *, kind: str, module_name: str, path: str = "output"):
        if torch.is_tensor(value):
            return self._inspect_tensor(value, kind=kind, module_name=module_name, path=path)
        if isinstance(value, dict):
            for key, item in value.items():
                record = self._inspect_any(item, kind=kind, module_name=module_name, path=f"{path}.{key}")
                if record is not None:
                    return record
        elif isinstance(value, (tuple, list)):
            for idx, item in enumerate(value):
                record = self._inspect_any(item, kind=kind, module_name=module_name, path=f"{path}[{idx}]")
                if record is not None:
                    return record
        return None

    @torch.no_grad()
    def _inspect_tensor(self, tensor: torch.Tensor, *, kind: str, module_name: str, path: str):
        if not tensor.is_floating_point() or tensor.numel() == 0:
            return None
        data = tensor.detach()
        finite = torch.isfinite(data)
        if bool(finite.all().item()):
            return None
        bad = ~finite
        finite_vals = data[finite].float()
        return {
            "kind": kind,
            "module": module_name,
            "path": path,
            "shape": tuple(data.shape),
            "dtype": str(data.dtype),
            "device": str(data.device),
            "numel": int(data.numel()),
            "bad": int(bad.sum().item()),
            "nan": int(torch.isnan(data).sum().item()),
            "posinf": int(torch.isposinf(data).sum().item()),
            "neginf": int(torch.isneginf(data).sum().item()),
            "finite_max_abs": (
                float(finite_vals.abs().max().item()) if finite_vals.numel() else float("nan")
            ),
            "finite_mean": (
                float(finite_vals.mean().item()) if finite_vals.numel() else float("nan")
            ),
            "step": self.step,
            "epoch": self.epoch,
            "batch_idx": self.batch_idx,
        }

    def has_record(self) -> bool:
        return self.forward_record is not None or self.backward_record is not None

    @staticmethod
    def _format_record(record: dict) -> str:
        return (
            f"{record['kind']} module={record['module']} path={record['path']} "
            f"shape={record['shape']} dtype={record['dtype']} "
            f"bad={record['bad']}/{record['numel']} "
            f"nan={record['nan']} +inf={record['posinf']} -inf={record['neginf']} "
            f"finite_max_abs={record['finite_max_abs']:.3e} "
            f"finite_mean={record['finite_mean']:.3e}"
        )

    def emit(self, *, rank: int, prefix: str = "[NonFiniteMonitor]") -> dict[str, float | str]:
        if not self.has_record():
            return {}
        payload = {}
        for label, record in (("forward", self.forward_record), ("backward", self.backward_record)):
            if record is None:
                continue
            print(f"{prefix} rank={rank} {self._format_record(record)}", flush=True)
            payload[f"debug/nonfinite_{label}_bad"] = float(record["bad"])
            payload[f"debug/nonfinite_{label}_finite_max_abs"] = float(record["finite_max_abs"])
            payload[f"debug/nonfinite_{label}_module"] = str(record["module"])
            payload[f"debug/nonfinite_{label}_kind"] = str(record["kind"])
        return payload


def compute_caca_distance_stats(ca_coords: torch.Tensor, mask: torch.Tensor):
    """Summarise consecutive CA-CA distances for a batch of trajectories."""
    if ca_coords is None or ca_coords.ndim != 4 or ca_coords.shape[2] < 2:
        return None

    pair_mask = (mask[:, 1:] * mask[:, :-1]).bool()
    if not pair_mask.any():
        return None

    caca = torch.linalg.vector_norm(ca_coords[:, :, 1:, :] - ca_coords[:, :, :-1, :], dim=-1)
    valid = pair_mask.unsqueeze(1).expand(-1, caca.shape[1], -1)
    values = caca[valid]
    if values.numel() == 0:
        return None

    return {
        "mean_A": values.mean().item(),
        "std_A": values.std(unbiased=False).item(),
        "min_A": values.min().item(),
        "max_A": values.max().item(),
        "count": int(values.numel()),
    }


@torch.no_grad()
def compute_nonbonded_clash_counts(
    ca_coords: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 3.5,
    min_sep: int = 2,
    pair_chunk: int = 2048,
    max_pairs: int | None = None,
    estimate_total_from_sample: bool = False,
):
    """Count non-bonded CA-CA clashes per trajectory.

    ``count_per_traj`` is the number of non-bonded pair-frame distances below
    ``threshold`` summed across all frames in each trajectory.
    """
    if ca_coords is None or mask is None or ca_coords.ndim != 4:
        return None
    B, T, L, _ = ca_coords.shape
    if L <= min_sep:
        zeros = torch.zeros(B, device=ca_coords.device, dtype=torch.float32)
        return {
            "count_per_traj": zeros,
            "pair_frame_frac": zeros,
            "valid_pair_frames": zeros,
            "sampled": False,
        }

    all_i, all_j = torch.triu_indices(L, L, offset=int(min_sep), device=ca_coords.device)
    n_pairs = all_i.numel()
    if n_pairs == 0:
        zeros = torch.zeros(B, device=ca_coords.device, dtype=torch.float32)
        return {
            "count_per_traj": zeros,
            "pair_frame_frac": zeros,
            "valid_pair_frames": zeros,
            "sampled": False,
        }

    sampled = max_pairs is not None and int(max_pairs) > 0 and int(max_pairs) < int(n_pairs)
    if sampled:
        sel = torch.randint(n_pairs, (int(max_pairs),), device=ca_coords.device)
        pair_i = all_i[sel]
        pair_j = all_j[sel]
    else:
        pair_i = all_i
        pair_j = all_j

    counts = torch.zeros(B, device=ca_coords.device, dtype=torch.float32)
    valid_pair_frames = torch.zeros(B, device=ca_coords.device, dtype=torch.float32)
    chunk = max(int(pair_chunk), 1)
    for start in range(0, pair_i.numel(), chunk):
        pi = pair_i[start:start + chunk]
        pj = pair_j[start:start + chunk]
        pair_valid = (mask[:, pi] * mask[:, pj]).bool()
        if not pair_valid.any():
            continue
        d = torch.linalg.vector_norm(ca_coords[:, :, pi, :] - ca_coords[:, :, pj, :], dim=-1)
        clash = (d < float(threshold)) & pair_valid.unsqueeze(1)
        counts += clash.sum(dim=(1, 2)).float()
        valid_pair_frames += pair_valid.sum(dim=1).float() * T

    if sampled and estimate_total_from_sample:
        total_valid_pairs = (mask[:, all_i] * mask[:, all_j]).sum(dim=1).float()
        sampled_valid_pairs = (mask[:, pair_i] * mask[:, pair_j]).sum(dim=1).float()
        scale = total_valid_pairs / sampled_valid_pairs.clamp_min(1.0)
        counts = counts * scale
        valid_pair_frames = total_valid_pairs * T

    return {
        "count_per_traj": counts,
        "pair_frame_frac": counts / valid_pair_frames.clamp_min(1.0),
        "valid_pair_frames": valid_pair_frames,
        "sampled": bool(sampled),
    }


def collect_temp_caca_debug_metrics(
    raw_temps: torch.Tensor,
    pred_ca: torch.Tensor | None,
    target_ca: torch.Tensor | None,
    mask: torch.Tensor,
    tracked_temps=(413.0, 450.0),
    clash_threshold: float = 3.5,
    clash_max_pairs: int = 4096,
    clash_pair_chunk: int = 512,
):
    """Collect per-temperature CA-CA and cheap sampled clash debug stats."""
    metrics = {}
    if pred_ca is None or target_ca is None or raw_temps.numel() == 0:
        return metrics

    raw_temps = raw_temps.float()
    for tracked_temp in tracked_temps:
        select = torch.isclose(
            raw_temps,
            torch.full_like(raw_temps, float(tracked_temp)),
            atol=0.5,
            rtol=0.0,
        )
        if not select.any():
            continue

        pred_stats = compute_caca_distance_stats(pred_ca[select], mask[select])
        gt_stats = compute_caca_distance_stats(target_ca[select], mask[select])
        if pred_stats is None or gt_stats is None:
            continue

        temp_key = f"{int(round(tracked_temp))}k"
        metrics[f"debug/caca_{temp_key}_n_samples"] = int(select.sum().item())
        for stat_name, stat_value in pred_stats.items():
            metrics[f"debug/caca_{temp_key}_pred_{stat_name}"] = stat_value
        for stat_name, stat_value in gt_stats.items():
            metrics[f"debug/caca_{temp_key}_gt_{stat_name}"] = stat_value

        pred_clash = compute_nonbonded_clash_counts(
            pred_ca[select],
            mask[select],
            threshold=clash_threshold,
            pair_chunk=clash_pair_chunk,
            max_pairs=clash_max_pairs,
            estimate_total_from_sample=True,
        )
        gt_clash = compute_nonbonded_clash_counts(
            target_ca[select],
            mask[select],
            threshold=clash_threshold,
            pair_chunk=clash_pair_chunk,
            max_pairs=clash_max_pairs,
            estimate_total_from_sample=True,
        )
        if pred_clash is not None and gt_clash is not None:
            metrics[f"debug/clash_{temp_key}_pred_est_count_per_traj_3p5"] = (
                pred_clash["count_per_traj"].mean().item()
            )
            metrics[f"debug/clash_{temp_key}_pred_est_ratio_3p5"] = (
                pred_clash["count_per_traj"].mean().item()
            )
            metrics[f"debug/clash_{temp_key}_gt_est_count_per_traj_3p5"] = (
                gt_clash["count_per_traj"].mean().item()
            )
            metrics[f"debug/clash_{temp_key}_gt_est_ratio_3p5"] = (
                gt_clash["count_per_traj"].mean().item()
            )
            metrics[f"debug/clash_{temp_key}_pred_pair_frame_frac_3p5"] = (
                pred_clash["pair_frame_frac"].mean().item()
            )
            metrics[f"debug/clash_{temp_key}_gt_pair_frame_frac_3p5"] = (
                gt_clash["pair_frame_frac"].mean().item()
            )

    return metrics


def init_temp_caca_accumulator(tracked_temps=(413.0, 450.0), device="cpu"):
    """Create an accumulator for per-temperature CA-CA validation stats."""
    n_temps = len(tracked_temps)
    count = torch.zeros((n_temps, 2), device=device)
    sum_ = torch.zeros((n_temps, 2), device=device)
    sumsq = torch.zeros((n_temps, 2), device=device)
    min_ = torch.full((n_temps, 2), float("inf"), device=device)
    max_ = torch.full((n_temps, 2), float("-inf"), device=device)
    return {
        "tracked_temps": tuple(float(t) for t in tracked_temps),
        "count": count,
        "sum": sum_,
        "sumsq": sumsq,
        "min": min_,
        "max": max_,
    }


def init_temp_clash_accumulator(tracked_temps=(413.0, 450.0), device="cpu"):
    """Create an accumulator for per-temperature validation clash counts."""
    n_temps = len(tracked_temps)
    return {
        "tracked_temps": tuple(float(t) for t in tracked_temps),
        "count": torch.zeros((n_temps, 2), device=device),
        "traj_count": torch.zeros((n_temps, 2), device=device),
        "pair_frame_count": torch.zeros((n_temps, 2), device=device),
    }


def update_temp_caca_accumulator(accumulator, raw_temps, pred_ca, target_ca, mask):
    """Accumulate per-temperature consecutive CA-CA stats."""
    if pred_ca is None or target_ca is None:
        return

    raw_temps = raw_temps.float()
    for temp_idx, tracked_temp in enumerate(accumulator["tracked_temps"]):
        select = torch.isclose(
            raw_temps,
            torch.full_like(raw_temps, tracked_temp),
            atol=0.5,
            rtol=0.0,
        )
        if not select.any():
            continue

        pair_mask = (mask[select, 1:] * mask[select, :-1]).bool()
        if not pair_mask.any():
            continue

        for role_idx, ca_tensor in enumerate((pred_ca[select], target_ca[select])):
            caca = torch.linalg.vector_norm(ca_tensor[:, :, 1:, :] - ca_tensor[:, :, :-1, :], dim=-1)
            valid = pair_mask.unsqueeze(1).expand(-1, caca.shape[1], -1)
            values = caca[valid]
            if values.numel() == 0:
                continue

            values = values.float()
            accumulator["count"][temp_idx, role_idx] += values.numel()
            accumulator["sum"][temp_idx, role_idx] += values.sum()
            accumulator["sumsq"][temp_idx, role_idx] += values.square().sum()
            accumulator["min"][temp_idx, role_idx] = torch.minimum(
                accumulator["min"][temp_idx, role_idx], values.min()
            )
            accumulator["max"][temp_idx, role_idx] = torch.maximum(
                accumulator["max"][temp_idx, role_idx], values.max()
            )


def update_temp_clash_accumulator(
    accumulator,
    raw_temps,
    pred_ca,
    target_ca,
    mask,
    threshold: float = 3.5,
    pair_chunk: int = 2048,
):
    """Accumulate exact non-bonded clash counts per validation trajectory."""
    if pred_ca is None or target_ca is None:
        return

    raw_temps = raw_temps.float()
    for temp_idx, tracked_temp in enumerate(accumulator["tracked_temps"]):
        select = torch.isclose(
            raw_temps,
            torch.full_like(raw_temps, tracked_temp),
            atol=0.5,
            rtol=0.0,
        )
        if not select.any():
            continue

        for role_idx, ca_tensor in enumerate((pred_ca[select], target_ca[select])):
            clash = compute_nonbonded_clash_counts(
                ca_tensor,
                mask[select],
                threshold=threshold,
                pair_chunk=pair_chunk,
                max_pairs=None,
            )
            if clash is None:
                continue
            accumulator["count"][temp_idx, role_idx] += clash["count_per_traj"].sum()
            accumulator["traj_count"][temp_idx, role_idx] += select.sum().float()
            accumulator["pair_frame_count"][temp_idx, role_idx] += clash["valid_pair_frames"].sum()


def finalize_temp_caca_accumulator(accumulator, is_distributed=False):
    """Convert accumulated CA-CA stats into W&B-friendly validation metrics."""
    if is_distributed:
        dist.all_reduce(accumulator["count"], op=dist.ReduceOp.SUM)
        dist.all_reduce(accumulator["sum"], op=dist.ReduceOp.SUM)
        dist.all_reduce(accumulator["sumsq"], op=dist.ReduceOp.SUM)
        dist.all_reduce(accumulator["min"], op=dist.ReduceOp.MIN)
        dist.all_reduce(accumulator["max"], op=dist.ReduceOp.MAX)

    results = {}
    role_names = ("pred", "gt")
    for temp_idx, tracked_temp in enumerate(accumulator["tracked_temps"]):
        temp_key = f"{int(round(tracked_temp))}k"
        for role_idx, role_name in enumerate(role_names):
            count = accumulator["count"][temp_idx, role_idx].item()
            if count <= 0:
                continue

            sum_val = accumulator["sum"][temp_idx, role_idx].item()
            sumsq_val = accumulator["sumsq"][temp_idx, role_idx].item()
            mean_val = sum_val / count
            var_val = max(sumsq_val / count - mean_val ** 2, 0.0)
            results[f"val/caca_{temp_key}_{role_name}_mean_A"] = mean_val
            results[f"val/caca_{temp_key}_{role_name}_std_A"] = var_val ** 0.5
            results[f"val/caca_{temp_key}_{role_name}_min_A"] = accumulator["min"][temp_idx, role_idx].item()
            results[f"val/caca_{temp_key}_{role_name}_max_A"] = accumulator["max"][temp_idx, role_idx].item()
            results[f"val/caca_{temp_key}_{role_name}_count"] = int(count)

    return results


def finalize_temp_clash_accumulator(accumulator, is_distributed=False):
    """Convert validation clash accumulators into W&B-friendly metrics."""
    if is_distributed:
        dist.all_reduce(accumulator["count"], op=dist.ReduceOp.SUM)
        dist.all_reduce(accumulator["traj_count"], op=dist.ReduceOp.SUM)
        dist.all_reduce(accumulator["pair_frame_count"], op=dist.ReduceOp.SUM)

    results = {}
    role_names = ("pred", "gt")
    for temp_idx, tracked_temp in enumerate(accumulator["tracked_temps"]):
        temp_key = f"{int(round(tracked_temp))}k"
        for role_idx, role_name in enumerate(role_names):
            n_traj = accumulator["traj_count"][temp_idx, role_idx].item()
            if n_traj <= 0:
                continue
            count = accumulator["count"][temp_idx, role_idx].item()
            pair_frames = accumulator["pair_frame_count"][temp_idx, role_idx].item()
            results[f"val/clash_{temp_key}_{role_name}_total_3p5"] = count
            results[f"val/clash_{temp_key}_{role_name}_n_traj"] = int(n_traj)
            results[f"val/clash_{temp_key}_{role_name}_count_per_traj_3p5"] = count / n_traj
            results[f"val/clash_{temp_key}_{role_name}_ratio_3p5"] = count / n_traj
            results[f"val/clash_{temp_key}_{role_name}_pair_frame_frac_3p5"] = (
                count / max(pair_frames, 1.0)
            )
    return results


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    seed = worker_info.seed

    np.random.seed(seed % 2**32)
    random.seed(seed)
    torch.manual_seed(seed)


def set_randomize_windows(dataset, enabled: bool):
    """Toggle temporal-window resampling on a dataset or Subset wrapper."""
    target = dataset.dataset if isinstance(dataset, Subset) else dataset
    if hasattr(target, "randomize_windows"):
        target.randomize_windows = bool(enabled)


def get_frequency_weights(weighting_type, n_steps, device, tau=32.0, min_weight=0.1):
    """Generate 1-D frequency-band loss weights."""
    indices = torch.arange(n_steps, device=device, dtype=torch.float32)
    if weighting_type == "exponential_decay":
        return torch.exp(-indices / tau)
    elif weighting_type == "linear_decay":
        return torch.linspace(1.0, min_weight, steps=n_steps, device=device)
    elif weighting_type == "quadratic_decay":
        progress = torch.linspace(1.0, math.sqrt(min_weight), steps=n_steps, device=device)
        return progress ** 2
    return torch.ones(n_steps, device=device)


def _prepare_frequency_weights(freq_weights, n_steps, device, dtype):
    """Return a length-n_steps frequency-weight vector on the requested device/dtype."""
    if freq_weights is None:
        return torch.ones(n_steps, device=device, dtype=dtype)
    if freq_weights.shape[0] < n_steps:
        raise ValueError(
            f"freq_weights too short: got {freq_weights.shape[0]}, expected at least {n_steps}"
        )
    return freq_weights[:n_steps].to(device=device, dtype=dtype)

def build_spectral_mask(mask, torsion_mask, top_k, is_dct, coord_channels=3, representation=None):
    '''Build per-element mask for the flattened spectral volume.'''
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
    else:
        full = feature_mask.unsqueeze(2).unsqueeze(-1).expand(-1, -1, top_k, -1, 2)
        return full.reshape(feature_mask.shape[0], feature_mask.shape[1], -1)


def compute_rmsf(coords, mask):
    '''
    Compute per-residue RMSF from a trajectory tensor directly on GPU.

    coords: (B, T, L, 3) predicted or ground-truth coordinates.
    mask: (B, L) residue validity mask.

    Returns 1-D PyTorch tensor of valid RMSF values across the entire batch.
    '''
    # Mean position per residue: (B, 1, L, 3)
    mask_expanded = mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, L, 1)
    coords_masked = coords * mask_expanded

    # Count valid frames per residue for mean
    T = coords.shape[1]
    mean_pos = coords_masked.sum(dim=1, keepdim=True) / T  # (B, 1, L, 3)

    # Squared deviation from mean
    sq_dev = ((coords_masked - mean_pos) ** 2).sum(dim=-1)  # (B, T, L)

    # Mean over time -> sqrt -> RMSF per residue: (B, L)
    rmsf = torch.sqrt(sq_dev.mean(dim=1) + 1e-8)

    # Return a single 1-D tensor of only the valid residues across the whole batch!
    return rmsf[mask.bool()]

def spearman_corr_pytorch(x, y):
    if x.numel() < 2:
        return float("nan")
    
    # argsort twice brilliantly gives the rank of each element
    x_rank = x.argsort().argsort().float()
    y_rank = y.argsort().argsort().float()
    
    # Pearson correlation of the ranks
    x_rank = x_rank - x_rank.mean()
    y_rank = y_rank - y_rank.mean()
    
    cov = (x_rank * y_rank).sum()
    var_x = (x_rank ** 2).sum()
    var_y = (y_rank ** 2).sum()
    
    corr = cov / (torch.sqrt(var_x * var_y) + 1e-8)
    return corr.item()


def pearson_corr_pytorch(x, y):
    if x.numel() < 2:
        return float("nan")
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    cov = (x * y).sum()
    denom = torch.sqrt((x ** 2).sum() * (y ** 2).sum()) + 1e-8
    return (cov / denom).item()


def _band_metric_label(start: int, end: int, top_k_freqs: int) -> str:
    if start == 0 and end == 1:
        return "dc"
    if end >= top_k_freqs:
        return f"k{start}_plus"
    return f"k{start}_{end - 1}"


def _low_k_group_index(band_edges) -> int:
    edges = tuple(int(edge) for edge in band_edges)
    for idx, (start, _end) in enumerate(zip(edges[:-1], edges[1:])):
        if start == 1:
            return idx
    return 0


def _safe_flat_pearson(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    finite = torch.isfinite(pred) & torch.isfinite(target)
    pred = pred[finite]
    target = target[finite]
    if pred.numel() < 2:
        return float("nan")
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = torch.sqrt(pred.square().sum() * target.square().sum()).clamp_min(1e-8)
    return float((pred * target).sum().div(denom).item())


def spectral_band_diagnostics(
    pred_spectral: torch.Tensor,
    target_spectral: torch.Tensor,
    spectral_mask: torch.Tensor,
    *,
    top_k_freqs: int,
    channels_per_mode: int,
    band_edges,
    amplitude_channels: int,
    include_all_groups: bool = True,
) -> dict[str, float]:
    """Band-wise signed coefficient and amplitude diagnostics."""
    metrics: dict[str, float] = {}
    edges = tuple(int(edge) for edge in band_edges)
    if pred_spectral.shape != target_spectral.shape:
        return metrics
    B, L, D = pred_spectral.shape
    K = int(top_k_freqs)
    C = int(channels_per_mode)
    if K <= 0 or C <= 0 or D != K * C:
        return metrics

    pred = pred_spectral.reshape(B, L, K, C).float()
    target = target_spectral.reshape(B, L, K, C).float()
    mask4 = spectral_mask.reshape(B, L, K, C).to(dtype=torch.bool, device=pred_spectral.device)
    amp_c = max(1, min(int(amplitude_channels), C))
    low_idx = _low_k_group_index(edges)

    for group_idx, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        if not include_all_groups and group_idx != low_idx:
            continue
        label = _band_metric_label(start, end, K)
        coeff_mask = mask4[:, :, start:end, :]
        denom = coeff_mask.sum().clamp_min(1)
        diff = pred[:, :, start:end, :] - target[:, :, start:end, :]
        signed_mse = (diff.square() * coeff_mask).sum() / denom

        amp_pred = torch.linalg.vector_norm(pred[:, :, start:end, :amp_c].float(), dim=-1)
        amp_target = torch.linalg.vector_norm(target[:, :, start:end, :amp_c].float(), dim=-1)
        amp_mask = mask4[:, :, start:end, :amp_c].any(dim=-1)
        amp_denom = amp_mask.sum().clamp_min(1)
        amp_mae = ((amp_pred - amp_target).abs() * amp_mask).sum() / amp_denom

        valid_pred = pred[:, :, start:end, :][coeff_mask]
        valid_target = target[:, :, start:end, :][coeff_mask]
        signed_pearson = _safe_flat_pearson(valid_pred, valid_target)

        metrics[f"spectral/{label}_signed_mse"] = float(signed_mse.detach().float().item())
        metrics[f"spectral/{label}_amp_mae"] = float(amp_mae.detach().float().item())
        metrics[f"spectral/{label}_signed_pearson"] = signed_pearson
        if group_idx == low_idx:
            metrics["spectral/low_k_signed_mse"] = metrics[f"spectral/{label}_signed_mse"]
            metrics["spectral/low_k_amp_mae"] = metrics[f"spectral/{label}_amp_mae"]
            metrics["spectral/low_k_signed_pearson"] = signed_pearson

    return metrics


def _model_band_edges(inner_model, top_k_freqs):
    if hasattr(inner_model, "band_edges"):
        return tuple(int(edge) for edge in inner_model.band_edges)
    trunk = getattr(inner_model, "trunk", None)
    if trunk is not None and hasattr(trunk, "band_edges"):
        return tuple(int(edge) for edge in trunk.band_edges)
    edges = []
    for edge in (0, 1, 9, 33, 129, int(top_k_freqs)):
        edge = min(max(int(edge), 0), int(top_k_freqs))
        if not edges or edge > edges[-1]:
            edges.append(edge)
    if edges[-1] != int(top_k_freqs):
        edges.append(int(top_k_freqs))
    return tuple(edges)



# INFERENCE
# ----------
class CFGModelWrapper(torch.nn.Module):
    '''CFG wrapper that exposes prediction_target for denoise_ode.'''

    def __init__(self, base_model, guidance_scale=1.0):
        super().__init__()
        self.base_model = base_model
        self.guidance_scale = guidance_scale
        # Expose prediction_target so denoise_ode can read it
        self.prediction_target = getattr(base_model, "prediction_target", "noise")

    def forward(self, x_t, t, norm_temps, native_coords, native_angles, mask=None, **kwargs):
        if self.guidance_scale <= 1.0:
            return self.base_model({
                "x": x_t, "t": t, "temp": norm_temps,
                "native_coords": native_coords, "native_angles": native_angles,
                "mask": mask, "cond_drop_mask": None, **kwargs,
            })

        B = x_t.shape[0]
        device = x_t.device

        cond_mask = torch.cat([
            torch.zeros(B, dtype=torch.bool, device=device),
            torch.ones(B, dtype=torch.bool, device=device),
        ])

        # Any tensor kwargs (e.g. win_pos, rmsf_prior) must be batch-duplicated
        # to match the doubled x_t; scalar / None kwargs pass through unchanged.
        kwargs_dup = {
            k: (torch.cat([v, v], dim=0) if torch.is_tensor(v) else v)
            for k, v in kwargs.items()
        }
        out_d = self.base_model({
            "x": torch.cat([x_t, x_t], dim=0),
            "t": torch.cat([t, t], dim=0),
            "temp": torch.cat([norm_temps, norm_temps], dim=0),
            "native_coords": torch.cat([native_coords, native_coords], dim=0),
            "native_angles": torch.cat([native_angles, native_angles], dim=0) if native_angles is not None else None,
            "mask": torch.cat([mask, mask], dim=0) if mask is not None else None,
            "cond_drop_mask": cond_mask,
            **kwargs_dup,
        })
        cond, uncond = torch.chunk(out_d, 2, dim=0)
        return uncond + self.guidance_scale * (cond - uncond)


def run_inference(
    model, diffusion, transform_engine, shape, native_coords, native_angles,
    temps, window_size, mask=None, torsion_mask=None,
    device="cpu", guidance_scale=1.0, num_ode_steps=20,
    displacement=True, representation=None, win_pos=None, rmsf_prior=None, res_type=None, dssp=None,
    dc_baseline_per_res=None,
):
    '''Full inference using the universal ODE sampler. Works with DCT and any prediction target.'''
    model.eval()
    real_model = model.module if hasattr(model, "module") else model
    is_dct = getattr(real_model, "is_dct", True)

    # If the underlying model consumes an NMA RMSF prior and the caller
    # did not provide one (real-world inference path: no sidecar), compute
    # it on-the-fly from the native structure. NMA is a closed-form function
    # of native_coords alone, so nothing extra is needed from the user —
    # the same PDB they already pass as structural conditioning is enough.
    inner = getattr(real_model, "model", real_model)
    needs_prior = getattr(inner, "use_rmsf_prior_gain", False)
    if needs_prior and rmsf_prior is None and native_coords.shape[-1] == 3:
        from src.features.nma import compute_rmsf_prior_for_batch
        rmsf_prior = compute_rmsf_prior_for_batch(
            native_coords, mask=mask, cutoff=12.0, gamma=1.0, n_modes=None,
        )

    # Derive channels from native tensors — coord channels + angle channels
    coord_channels = native_coords.shape[-1]
    representation = representation or CoordinateRepresentation(
        displacement=displacement,
        coord_channels=coord_channels,
    )
    repr_coord_channels = representation.model_coord_channels
    angle_channels = native_angles.shape[-1] if native_angles is not None else 0
    channels = repr_coord_channels + angle_channels

    B, L, D = shape
    complex_mult = 1 if is_dct else 2
    is_manifold_domain = getattr(real_model, "is_manifold_domain", False)
    latent_dim = int(getattr(inner, "latent_dim", channels))
    K = D // ((latent_dim if is_manifold_domain else channels) * complex_mult)

    # Torsion mask (only when angles are present)
    if native_angles is not None:
        if torsion_mask is None and mask is not None:
            torsion_mask = mask.unsqueeze(-1).expand(-1, -1, angle_channels)
        elif torsion_mask is not None and torsion_mask.dim() == 2:
            torsion_mask = torsion_mask.unsqueeze(-1).expand(-1, -1, angle_channels)
    else:
        torsion_mask = None

    # Use aniso-weighted initial noise to match the training forward process.
    # diffusion.sample_initial_noise applies aniso_weights if set; falls back to torch.randn.
    input_noise = diffusion.sample_initial_noise(shape, device=device)

    # Apply representation mask to noise
    spec_mask = None
    if mask is not None and is_manifold_domain:
        if hasattr(inner, "latent_loss_mask"):
            input_noise = input_noise * inner.latent_loss_mask(mask, window_size=window_size).to(
                device=input_noise.device, dtype=input_noise.dtype
            )
        else:
            input_noise = input_noise * mask.unsqueeze(-1)
    elif mask is not None:
        spec_mask = build_spectral_mask(
            mask, torsion_mask, K, is_dct,
            coord_channels=repr_coord_channels,
            representation=representation,
        )
        input_noise = input_noise * spec_mask

    # Normalise temperatures
    norm_temps = torch.clamp((temps - 250.0) / 200.0, 0.0, 1.0)

    # Wrap for CFG
    cfg_model = CFGModelWrapper(real_model, guidance_scale=guidance_scale)

    # ODE sampling (universal for x_0 / v / noise prediction)
    x_t = diffusion.denoise_ode(
        cfg_model, input_noise, native_coords, native_angles, norm_temps,
        mask, torsion_mask=torsion_mask,
        is_dct=is_dct, num_steps=num_ode_steps,
        win_pos=win_pos, rmsf_prior=rmsf_prior, res_type=res_type, dssp=dssp,
        feature_dim=latent_dim if is_manifold_domain else None,
        spectral_mask=spec_mask,
    )
    dc_baseline = None
    if dc_baseline_per_res is not None:
        dc_baseline = dc_baseline_per_res.to(device=x_t.device, dtype=x_t.dtype)[..., :repr_coord_channels]
    if hasattr(transform_engine, "lookup_dc_baselines"):
        if dc_baseline is None:
            dc_baseline = transform_engine.lookup_dc_baselines(
                temps, mask, coord_channels=repr_coord_channels, device=x_t.device
            )
    x_t = maybe_restore_dc(transform_engine, x_t, dc_baseline, repr_coord_channels)

    is_time_domain = getattr(real_model, "is_time_domain", False)

    if is_manifold_domain:
        x_time = inner.decode_latents(x_t, native_coords, mask=mask, window_size=window_size)
        return {"coords": x_time, "spectral": None, "manifold": x_t}

    if is_time_domain:
        # x_t is (B, L, T*C) normalised coordinate deviations — inverse-normalise and reshape
        coord_scale = real_model.model.coord_scale
        x_dev = x_t * coord_scale                                                        # (B, L, T*C)
        x_time = x_dev.reshape(B, L, window_size, coord_channels).permute(0, 2, 1, 3)  # (B, T, L, C)
        x_time = x_time * mask.unsqueeze(1).unsqueeze(-1)
        if representation.is_displacement:
            x_time = x_time + native_coords.unsqueeze(1)
        return {"coords": x_time, "spectral": None}

    # Spectral -> time domain
    x_time = transform_engine.spectral_to_time(x_t, n_time_steps=window_size, n_channels=channels)
    x_time = x_time * mask.unsqueeze(1).unsqueeze(-1)
    coord_repr_time = x_time[..., :repr_coord_channels]
    coords_abs = representation.inverse(coord_repr_time, native_coords, mask=mask)
    if hasattr(inner, "refine_ca"):
        if coords_abs.shape[-1] == 3:
            _, coords_abs = inner.refine_ca(coords_abs, mask)
        elif coords_abs.shape[-1] == 12:
            ca_refined, ca_shaken = inner.refine_ca(coords_abs[..., 3:6], mask)
            del ca_refined
            coords_abs = coords_abs.clone()
            coords_abs[..., 3:6] = ca_shaken
    elif getattr(inner, "use_shake", False):
        from src.models.shake import shake_caca as _shake_caca

        if coords_abs.shape[-1] == 3:
            coords_abs = _shake_caca(
                coords_abs,
                mask=mask,
                target=getattr(inner, "shake_target", 3.8),
                n_iter=getattr(inner, "shake_n_iter", 20),
            )
        elif coords_abs.shape[-1] == 12:
            ca_shaken = _shake_caca(
                coords_abs[..., 3:6],
                mask=mask,
                target=getattr(inner, "shake_target", 3.8),
                n_iter=getattr(inner, "shake_n_iter", 20),
            )
            coords_abs = coords_abs.clone()
            coords_abs[..., 3:6] = ca_shaken
    pred_dict = {"coords": coords_abs}
    if angle_channels > 0:
        pred_dict["angles"] = x_time[..., repr_coord_channels:]
        if torsion_mask is not None:
            pred_dict["angles"] = pred_dict["angles"] * torsion_mask.unsqueeze(1)

    pred_dict["spectral"] = x_t
    return pred_dict



# PIPELINE DIAGNOSTICS
@torch.no_grad()
def debug_print_pipeline(batch, transform_engine, real_model, device, top_k_freqs, displacement, label="DEBUG"):
    """One-shot sanity check. Call at step 0 and first validation batch.

    Assumes coordinates are in collapsed channel form:
      CA: (B, T, L, 3)
      BB: (B, T, L, 12)

    Prints five checks:
      1. Displacement active — raw vs displaced coordinate range.
      2. Stable core centering — native_coords mean should be ≈ [0,0,0].
      3. Spectral k=0 amplitude — displaced DCT DC should be << raw DCT DC.
      4. Internal normalisation — spectral volume std should move towards ~1.
      5. Masking — invalid (padded) residues should have zero spectral energy.
    """
    sep = "=" * 62
    print(f"\n{sep}\n  PIPELINE DEBUG [{label}]\n{sep}")

    coords     = batch["coords"].to(device)         # (B, T, L, C)
    native     = batch["native_coords"].to(device)  # (B, L, C)
    mask       = batch["mask"].to(device)           # (B, L)
    mask_bool  = mask.bool()

    # Defensive collapse in case an upstream path still emits BB as (..., 4, 3)
    if coords.ndim == 5:
        B, T, L, A, xyz = coords.shape
        coords = coords.reshape(B, T, L, A * xyz)
    if native.ndim == 4:
        B, L, A, xyz = native.shape
        native = native.reshape(B, L, A * xyz)

    displaced = coords - native.unsqueeze(1)        # (B, T, L, C)

    # ── 1. Displacement ────────────────────────────────────────────
    raw_abs  = coords.abs()
    disp_abs = displaced.abs()
    print(f"\n[1] DISPLACEMENT  (displacement={displacement})")
    print(f"    Raw  coords  — mean|x|: {raw_abs.mean():.2f} Å   max|x|: {raw_abs.max():.2f} Å")
    print(f"    Disp coords  — mean|x|: {disp_abs.mean():.2f} Å   max|x|: {disp_abs.max():.2f} Å")
    ratio = raw_abs.mean() / disp_abs.mean().clamp(min=1e-8)
    print(f"    Ratio raw/disp: {ratio:.1f}×  (expect >> 1 with stable-core centering)")

    # ── 2. Stable core centering ───────────────────────────────────
    valid_native = native[mask_bool]                # (N_valid, C)

    print(f"\n[2] STABLE CORE CENTERING")
    if valid_native.shape[-1] == 3:
        core_mean = valid_native.mean(dim=0)        # (3,)
        print(
            f"    native_coords mean (valid): "
            f"[{core_mean[0].item():.3f}, {core_mean[1].item():.3f}, {core_mean[2].item():.3f}] Å"
        )
    elif valid_native.shape[-1] == 12:
        core_mean = valid_native.view(valid_native.shape[0], 4, 3).mean(dim=0)  # (4, 3)
        cm = core_mean.mean(dim=0)  # overall xyz mean across BB atoms, shape (3,)
        print(
            f"    native_coords mean (valid): "
            f"[{cm[0].item():.3f}, {cm[1].item():.3f}, {cm[2].item():.3f}] Å"
        )
        print(f"    native_coords mean per BB atom:")
        for i, atom_name in enumerate(["N", "CA", "C", "O"]):
            v = core_mean[i]
            print(f"      {atom_name}: [{v[0].item():.3f}, {v[1].item():.3f}, {v[2].item():.3f}] Å")
    else:
        core_mean = valid_native.mean(dim=0)
        print(f"    native_coords mean shape: {tuple(core_mean.shape)}")
        print(f"    native_coords mean (valid): {core_mean}")

    print(f"    (expect ≈ [0, 0, 0] — stable core residues centered at origin)")
    print(f"    native_coords std  (valid): {valid_native.std():.2f} Å  (typical 10–30 Å for a folded protein)")

    # ── 3. Spectral k=0 amplitude ──────────────────────────────────
    # time_to_spectral expects (B, T, L, C)
    raw_masked  = coords * mask_bool[:, None, :, None]
    disp_masked = displaced * mask_bool[:, None, :, None]

    raw_spec  = transform_engine.time_to_spectral(raw_masked,  top_k=top_k_freqs)  # (B, L, K*C)
    disp_spec = transform_engine.time_to_spectral(disp_masked, top_k=top_k_freqs)  # (B, L, K*C)

    # First frequency block = k=0, next block = k=1
    coord_channels = native.shape[-1]
    raw_k0  = raw_spec[..., 0:coord_channels][mask_bool].abs().mean().item()
    disp_k0 = disp_spec[..., 0:coord_channels][mask_bool].abs().mean().item()
    raw_k1  = raw_spec[..., coord_channels:2 * coord_channels][mask_bool].abs().mean().item()
    disp_k1 = disp_spec[..., coord_channels:2 * coord_channels][mask_bool].abs().mean().item()

    print(f"\n[3] SPECTRAL k=0 DC AMPLITUDE (coord channels only)")
    print(f"    Raw  DCT k=0 mean: {raw_k0:.3f}   k=1 mean: {raw_k1:.3f}")
    print(f"    Disp DCT k=0 mean: {disp_k0:.3f}   k=1 mean: {disp_k1:.3f}")
    print(f"    Ratio raw/disp k=0: {raw_k0 / max(disp_k0, 1e-8):.1f}×  (expect >> 1)")

    # ── 4. Internal normalisation ──────────────────────────────────
    input_spec = disp_spec if displacement else raw_spec
    valid_spec = input_spec[mask_bool]

    print(f"\n[4] INTERNAL NORMALISATION")
    print(f"    Pre-normalise  std: {valid_spec.std():.4f}   mean: {valid_spec.mean():.4f}")

    if hasattr(real_model, "freq_scale") and real_model.freq_scale is not None:
        fs       = real_model.freq_scale.to(device)
        fs_slice = fs[: input_spec.shape[-1]].clamp(min=1e-8)
        normed   = input_spec / fs_slice
        valid_normed = normed[mask_bool]
        print(f"    Post-normalise std: {valid_normed.std():.4f}   mean: {valid_normed.mean():.4f}  (expect ≈ 1.0 / 0.0)")
        print(f"    freq_scale range: [{fs.min():.4f}, {fs.max():.4f}]  (all > 0 confirms loaded)")
        recon_err = (normed * fs_slice - input_spec).abs().max().item()
        print(f"    Normalise → denormalise max round-trip error: {recon_err:.2e}  (expect < 1e-5)")
    else:
        print(f"    freq_scale not registered on model — normalisation check skipped")

    # ── 5. Masking ────────────────────────────────────────────────
    print(f"\n[5] MASKING")
    invalid_mask = ~mask_bool
    n_invalid = invalid_mask.sum().item()
    if n_invalid > 0:
        invalid_energy = input_spec[invalid_mask].abs().max().item()
        print(f"    Max spectral energy at {int(n_invalid)} INVALID residues: {invalid_energy:.2e}  (expect 0.0)")
    else:
        print(f"    All {mask_bool.sum().item()} residues valid — no padding residues to check")
    print(f"\n{sep}\n")


# TRAINING
# --------
def compute_bending_weight(coords, native_coords, mask, bending_lambda, displacement):
    """Per-residue bending weight from the spatial gradient of deviations.

    For spectral models: returns (B, L) weight tensor to reweight the L dimension
    of the spectral loss.  For time-domain models the caller uses it differently.

    Args:
        coords:        (B, T, L, 3) — aligned trajectory coords in Å
        native_coords: (B, L, 3)    — native/reference coords in Å
        mask:          (B, L)       — bool/float residue mask
        bending_lambda: float       — scale factor; 0.0 disables
        displacement:  bool         — whether coords are already displacements

    Returns:
        (B, L) float tensor — values ≥ 1, unmasked residues = 1
    """
    with torch.no_grad():
        if displacement:
            dev = coords                                      # (B, T, L, 3) — already displaced
        else:
            dev = coords - native_coords.unsqueeze(1)        # (B, T, L, 3)

        # Spatial gradient of deviations along residue index, mean over time
        diff        = dev[:, :, 1:, :] - dev[:, :, :-1, :]  # (B, T, L-1, 3)
        bend        = (diff ** 2).sum(-1).mean(1)             # (B, L-1)
        bend        = F.pad(bend, (0, 1))                     # (B, L)  pad last residue

        # Zero out padded (invalid) positions
        bend        = bend * mask.float()

        # Normalise per-sample to mean=1 over valid residues
        valid_count = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        bend_mean   = (bend * mask.float()).sum(dim=1, keepdim=True) / valid_count
        bend_norm   = bend / (bend_mean + 1e-8)               # (B, L)

        weight      = 1.0 + bending_lambda * bend_norm
        weight      = weight * mask.float() + (1.0 - mask.float())  # invalid → 1

    return weight


def local_geometry_loss(
    pred_ca, target_ca, mask,
    target_caca=3.8, tol=0.05,
    clash_lambda=0.0, clash_threshold=3.5, min_sep=2,
    clash_max_pairs=4096, clash_pair_chunk=512,
):
    """CA-CA tolerance-band hinge loss + optional non-bonded clash hinge.

    Bonded term: penalises adjacent-CA distances only outside ``[3.8 ± tol]``,
    so the gradient targets outliers directly rather than driving the mean
    (MSE = Var + bias², so MSE-on-bonds can decrease while Var grows).

    Clash term (optional): penalises non-bonded CA-CA distances below
    ``clash_threshold`` for residue separations ≥ ``min_sep``.

    Returns per-sample loss ``(B,)`` so callers can apply SNR weighting.

    Args:
        pred_ca: (B, T, L, 3) predicted absolute CA coords.
        target_ca: unused; kept for backward-compatible call sites.
        mask: (B, L) residue validity mask.
        target_caca: ideal CA-CA bond length in Å.
        tol: half-width of the bonded tolerance band in Å.
        clash_lambda: weight on the non-bonded clash hinge (0 disables).
        clash_threshold: distance below which non-bonded pairs are penalised.
        min_sep: minimum |i-j| for the clash term.
        clash_max_pairs: maximum non-bonded residue pairs sampled per step.
        clash_pair_chunk: pair chunk size used to cap activation memory.
    """
    del target_ca
    B, T, L, _ = pred_ca.shape

    # Bonded tolerance hinge
    d_pred = safe_vector_norm(pred_ca[:, :, 1:] - pred_ca[:, :, :-1], dim=-1)  # (B, T, L-1)
    bond_mask = (mask[:, 1:] * mask[:, :-1]).unsqueeze(1).float()               # (B, 1, L-1)
    violation = (d_pred - target_caca).abs() - tol
    bond_loss = (violation.clamp(min=0) ** 2) * bond_mask                       # (B, T, L-1)
    denom_bond = bond_mask.sum(dim=(1, 2)) * T + 1e-8                           # (B,)
    bond_per_sample = bond_loss.sum(dim=(1, 2)) / denom_bond                    # (B,)

    if clash_lambda <= 0.0:
        return bond_per_sample

    # Non-bonded clash hinge (sep ≥ min_sep), sampled and chunked to avoid
    # materialising the full (B,T,L,L,3) tensor for long crops.
    pair_i, pair_j = torch.triu_indices(L, L, offset=int(min_sep), device=pred_ca.device)
    n_pairs = pair_i.numel()
    max_pairs = min(int(clash_max_pairs), int(n_pairs)) if clash_max_pairs is not None else int(n_pairs)
    if max_pairs <= 0:
        return bond_per_sample
    if max_pairs < n_pairs:
        sel = torch.randint(int(n_pairs), (int(max_pairs),), device=pred_ca.device)
        pair_i = pair_i[sel]
        pair_j = pair_j[sel]

    clash_loss = pred_ca.new_zeros(B)
    denom_clash = pred_ca.new_zeros(B)
    chunk = max(int(clash_pair_chunk), 1)
    for start in range(0, max_pairs, chunk):
        pi = pair_i[start:start + chunk]
        pj = pair_j[start:start + chunk]
        diff = pred_ca[:, :, pi, :] - pred_ca[:, :, pj, :]                      # (B, T, P, 3)
        d = safe_vector_norm(diff, dim=-1)                                      # (B, T, P)
        pair_valid = (mask[:, pi] * mask[:, pj]).float().unsqueeze(1)            # (B, 1, P)
        clash = (clash_threshold - d).clamp(min=0) ** 2
        clash_loss = clash_loss + (clash * pair_valid).sum(dim=(1, 2))
        denom_clash = denom_clash + pair_valid.sum(dim=(1, 2)) * T

    clash_per_sample = clash_loss / (denom_clash + 1e-8)

    return bond_per_sample + clash_lambda * clash_per_sample


GEO_LOSS_ALIASES = {
    "idct_ca-ca": "idct_ca-ca",
    "idct-ca-ca": "idct_ca-ca",
    "idct_caca": "idct_ca-ca",
    "idct-caca": "idct_ca-ca",
    "caca": "idct_ca-ca",
    "ca-ca": "idct_ca-ca",
    "old": "idct_ca-ca",
    "legacy": "idct_ca-ca",
    "spec_geo": "spec_geo",
    "spectral_geo": "spec_geo",
    "spectral_geometry": "spec_geo",
    "risk_band": "risk_band",
    "risk_bond": "risk_band",
    "band_risk": "risk_band",
}


def parse_geo_loss_modes(value) -> tuple[str, ...]:
    """Parse comma-delimited geometry auxiliary losses."""
    if value is None or value is False:
        return ("idct_ca-ca",)
    if isinstance(value, (list, tuple, set)):
        raw = []
        for item in value:
            raw.extend(str(item).split(","))
    else:
        text = str(value).strip()
        if text.lower() in {"", "none", "off", "false", "0"}:
            return tuple()
        raw = text.split(",")
    modes = []
    for item in raw:
        key = item.strip().lower().replace("_", "-")
        key = key.replace("spectral-geo", "spectral_geo").replace("spec-geo", "spec_geo")
        key = key.replace("risk-band", "risk_band").replace("risk-bond", "risk_bond")
        if not key:
            continue
        if key not in GEO_LOSS_ALIASES:
            valid = ", ".join(sorted(set(GEO_LOSS_ALIASES.values())))
            raise ValueError(f"Unknown geo_loss mode {item!r}; expected one of {valid}")
        mode = GEO_LOSS_ALIASES[key]
        if mode not in modes:
            modes.append(mode)
    return tuple(modes)


def _torch_segment_segment_distances(p1, q1, p2, q2, eps=1e-8):
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

    before_s = s_num < 0.0
    s_num = torch.where(before_s, torch.zeros_like(s_num), s_num)
    t_num = torch.where(before_s, e, t_num)
    t_den = torch.where(before_s, c, t_den)

    after_s = s_num > s_den
    s_num = torch.where(after_s, s_den, s_num)
    t_num = torch.where(after_s, e + b, t_num)
    t_den = torch.where(after_s, c, t_den)

    before_t = t_num < 0.0
    t_num = torch.where(before_t, torch.zeros_like(t_num), t_num)
    s_before_t = torch.minimum(torch.clamp(-d, min=0.0), a)
    s_num = torch.where(before_t, s_before_t, s_num)
    s_den = torch.where(before_t, a, s_den)

    after_t = t_num > t_den
    t_num = torch.where(after_t, t_den, t_num)
    s_after_t = torch.minimum(torch.clamp(-d + b, min=0.0), a)
    s_num = torch.where(after_t, s_after_t, s_num)
    s_den = torch.where(after_t, a, s_den)

    sc = torch.where(torch.abs(s_num) < small, torch.zeros_like(s_num), s_num / s_den.clamp_min(small))
    tc = torch.where(torch.abs(t_num) < small, torch.zeros_like(t_num), t_num / t_den.clamp_min(small))
    delta = w + sc.unsqueeze(-1) * u - tc.unsqueeze(-1) * v
    return safe_vector_norm(delta, dim=-1, eps=eps)


def _sample_upper_tri_pairs(length, *, offset, max_pairs, device):
    i, j = torch.triu_indices(int(length), int(length), offset=int(offset), device=device)
    n_pairs = int(i.numel())
    if max_pairs is not None and int(max_pairs) > 0 and int(max_pairs) < n_pairs:
        sel = torch.randint(n_pairs, (int(max_pairs),), device=device)
        i = i[sel]
        j = j[sel]
    return i, j


def _dct_time_from_coeffs(coeff, n_time_steps):
    """Convert selected DCT coeffs (..., K, C) to (..., T, C)."""
    K = coeff.shape[-2]
    W_inv = DCT.get_idct_matrix(int(n_time_steps), coeff.device).to(dtype=coeff.dtype)[:, :K]
    return torch.einsum("...kc,tk->...tc", coeff, W_inv)


def _ca_coeff_view(x_spec, *, top_k_freqs, channels, coord_channels, representation):
    if not getattr(representation, "name", representation) in ("raw_coords", "displacement"):
        return None, None
    B, L, D = x_spec.shape
    K = int(top_k_freqs)
    C = int(channels)
    if D < K * C:
        return None, None
    x = x_spec[..., :K * C].view(B, L, K, C)
    ca_start = 3 if int(coord_channels) == 12 else 0
    if ca_start + 3 > C:
        return None, None
    return x[:, :, :, ca_start:ca_start + 3], ca_start


def _bucket_label_for_sample(temp, length, size_bins):
    if size_bins is None:
        size_bins = [100, 200, 300, 400, 500, 600]
    size_label = None
    for cutoff in size_bins:
        if int(length) <= int(cutoff):
            size_label = f"le_{int(cutoff)}"
            break
    if size_label is None:
        size_label = f"gt_{int(size_bins[-1])}"
    return f"temp_size:{int(round(float(temp)))}|{size_label}", f"size:{size_label}", f"temp:{int(round(float(temp)))}", "overall"


def load_topology_margin_artifact(path, map_location="cpu"):
    if path is None:
        return None
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or "buckets" not in payload:
        raise ValueError(f"{path} is not a spectral topology margin artifact")
    return payload


def resolve_topology_thresholds(
    raw_temps,
    mask,
    margin_artifact=None,
    *,
    pair_default=3.5,
    segment_default=1.0,
    quantile_key="q01",
):
    """Per-sample lower-bound distance thresholds from the margin artifact."""
    B = mask.shape[0]
    pair = torch.full((B,), float(pair_default), device=mask.device, dtype=torch.float32)
    segment = torch.full((B,), float(segment_default), device=mask.device, dtype=torch.float32)
    if not margin_artifact:
        return pair, segment

    buckets = margin_artifact.get("buckets", {})
    meta = margin_artifact.get("metadata", {})
    size_bins = meta.get("size_bins", [100, 200, 300, 400, 500, 600])
    temps = raw_temps.detach().float().cpu().tolist()
    lengths = mask.detach().bool().sum(dim=1).cpu().tolist()
    for b, (temp, length) in enumerate(zip(temps, lengths)):
        for name in _bucket_label_for_sample(temp, length, size_bins):
            stats = buckets.get(name)
            if not stats:
                continue
            nb = stats.get("nonbonded_ca", {})
            seg = stats.get("chain_segment", {})
            nb_val = nb.get(quantile_key, nb.get("q05", nb.get("min")))
            seg_val = seg.get(quantile_key, seg.get("q05", seg.get("min")))
            if nb_val is not None and math.isfinite(float(nb_val)):
                pair[b] = min(float(pair_default), float(nb_val))
            if seg_val is not None and math.isfinite(float(seg_val)):
                segment[b] = min(float(segment_default), float(seg_val))
            break
    return pair, segment


def spectral_geometry_losses(
    pred_spec,
    target_spec,
    native_coords,
    mask,
    raw_temps,
    *,
    top_k_freqs,
    channels,
    coord_channels,
    representation,
    n_time_steps,
    margin_artifact=None,
    pair_lambda=1.0,
    segment_lambda=1.0,
    clash_threshold=3.5,
    segment_threshold=1.0,
    max_pairs=4096,
    max_segment_pairs=1024,
    pair_chunk=512,
):
    """Direct spectral-coordinate topology loss on nonlocal pairs and segments."""
    pred_coeff, _ = _ca_coeff_view(
        pred_spec, top_k_freqs=top_k_freqs, channels=channels,
        coord_channels=coord_channels, representation=representation,
    )
    if pred_coeff is None:
        z = pred_spec.new_tensor(0.0)
        return z, {"spec_geo_pair": 0.0, "spec_geo_segment": 0.0}

    B, L, K, _ = pred_coeff.shape
    pair_thresh, seg_thresh = resolve_topology_thresholds(
        raw_temps, mask, margin_artifact,
        pair_default=clash_threshold, segment_default=segment_threshold,
    )
    total = pred_spec.new_tensor(0.0)
    metrics = {}

    if pair_lambda > 0.0 and L > 2:
        pi, pj = _sample_upper_tri_pairs(L, offset=2, max_pairs=max_pairs, device=pred_spec.device)
        pair_loss = pred_spec.new_zeros(B)
        pair_denom = pred_spec.new_zeros(B)
        for start in range(0, int(pi.numel()), max(int(pair_chunk), 1)):
            ci = pi[start:start + int(pair_chunk)]
            cj = pj[start:start + int(pair_chunk)]
            coeff_diff = pred_coeff[:, ci, :, :] - pred_coeff[:, cj, :, :]  # (B,P,K,3)
            pair_time = _dct_time_from_coeffs(coeff_diff, n_time_steps).permute(0, 2, 1, 3)  # (B,T,P,3)
            if getattr(representation, "is_displacement", False):
                pair_time = pair_time + (native_coords[:, ci, :] - native_coords[:, cj, :]).unsqueeze(1)
            d2 = torch.sum(pair_time * pair_time, dim=-1)
            valid = (mask[:, ci] & mask[:, cj]).float().unsqueeze(1)
            hinge = (pair_thresh.view(B, 1, 1).square() - d2).clamp_min(0.0).square()
            pair_loss = pair_loss + (hinge * valid).sum(dim=(1, 2))
            pair_denom = pair_denom + valid.sum(dim=(1, 2)) * int(n_time_steps)
        pair_per = pair_loss / pair_denom.clamp_min(1.0)
        pair_scalar = pair_per.mean()
        total = total + float(pair_lambda) * pair_scalar
        metrics["spec_geo_pair"] = float(pair_scalar.detach().item())

    if segment_lambda > 0.0 and L > 3 and max_segment_pairs is not None and int(max_segment_pairs) > 0:
        si, sj = _sample_upper_tri_pairs(L - 1, offset=2, max_pairs=max_segment_pairs, device=pred_spec.device)
        endpoints = torch.stack([si, si + 1, sj, sj + 1], dim=1).reshape(-1)
        unique, inverse = torch.unique(endpoints, sorted=True, return_inverse=True)
        ep_coeff = pred_coeff[:, unique, :, :]
        ep_time = _dct_time_from_coeffs(ep_coeff, n_time_steps).permute(0, 2, 1, 3)
        if getattr(representation, "is_displacement", False):
            ep_time = ep_time + native_coords[:, unique, :].unsqueeze(1)
        ep_time = ep_time[:, :, inverse, :].view(B, int(n_time_steps), int(si.numel()), 4, 3)
        dist = _torch_segment_segment_distances(
            ep_time[:, :, :, 0, :], ep_time[:, :, :, 1, :],
            ep_time[:, :, :, 2, :], ep_time[:, :, :, 3, :],
        )
        valid = (mask[:, si] & mask[:, si + 1] & mask[:, sj] & mask[:, sj + 1]).float().unsqueeze(1)
        hinge = (seg_thresh.view(B, 1, 1).square() - dist.square()).clamp_min(0.0).square()
        seg_per = (hinge * valid).sum(dim=(1, 2)) / (valid.sum(dim=(1, 2)) * int(n_time_steps)).clamp_min(1.0)
        seg_scalar = seg_per.mean()
        total = total + float(segment_lambda) * seg_scalar
        metrics["spec_geo_segment"] = float(seg_scalar.detach().item())

    return total, metrics


def risk_band_geometry_loss(
    pred_spec,
    target_spec,
    native_coords,
    mask,
    raw_temps,
    *,
    top_k_freqs,
    channels,
    coord_channels,
    representation,
    n_time_steps,
    margin_artifact=None,
    pair_lambda=1.0,
    segment_lambda=1.0,
    clash_threshold=3.5,
    segment_threshold=1.0,
    max_pairs=2048,
    max_segment_pairs=512,
    pair_chunk=512,
    band_edges=(0, 1, 9, 33, 129),
):
    """Band-wise triangle-inequality risk loss on coefficient error."""
    pred_coeff, _ = _ca_coeff_view(
        pred_spec, top_k_freqs=top_k_freqs, channels=channels,
        coord_channels=coord_channels, representation=representation,
    )
    target_coeff, _ = _ca_coeff_view(
        target_spec, top_k_freqs=top_k_freqs, channels=channels,
        coord_channels=coord_channels, representation=representation,
    )
    if pred_coeff is None or target_coeff is None:
        z = pred_spec.new_tensor(0.0)
        return z, {"risk_band_pair": 0.0, "risk_band_segment": 0.0}

    B, L, K, _ = pred_coeff.shape
    err_coeff = pred_coeff - target_coeff
    pair_thresh, seg_thresh = resolve_topology_thresholds(
        raw_temps, mask, margin_artifact,
        pair_default=clash_threshold, segment_default=segment_threshold,
    )
    edges = [int(e) for e in band_edges if int(e) < K]
    if not edges or edges[0] != 0:
        edges = [0] + edges
    if edges[-1] != K:
        edges.append(K)

    total = pred_spec.new_tensor(0.0)
    metrics = {}
    band_count = max(len(edges) - 1, 1)

    if pair_lambda > 0.0 and L > 2:
        pi, pj = _sample_upper_tri_pairs(L, offset=2, max_pairs=max_pairs, device=pred_spec.device)
        pair_valid = (mask[:, pi] & mask[:, pj]).float()
        target_pair_coeff = target_coeff[:, pi, :, :] - target_coeff[:, pj, :, :]
        target_pair_time = _dct_time_from_coeffs(target_pair_coeff, n_time_steps).permute(0, 2, 1, 3)
        if getattr(representation, "is_displacement", False):
            target_pair_time = target_pair_time + (native_coords[:, pi, :] - native_coords[:, pj, :]).unsqueeze(1)
        target_min = safe_vector_norm(target_pair_time, dim=-1).amin(dim=1)
        margin = (target_min - pair_thresh.view(B, 1)).clamp_min(0.0)
        pair_loss_total = pred_spec.new_tensor(0.0)
        for lo, hi in zip(edges[:-1], edges[1:]):
            band_loss = pred_spec.new_zeros(B)
            band_denom = pred_spec.new_zeros(B)
            for start in range(0, int(pi.numel()), max(int(pair_chunk), 1)):
                ci = pi[start:start + int(pair_chunk)]
                cj = pj[start:start + int(pair_chunk)]
                valid = (mask[:, ci] & mask[:, cj]).float()
                err_pair = err_coeff[:, ci, lo:hi, :] - err_coeff[:, cj, lo:hi, :]
                err_time = _dct_time_from_coeffs(err_pair, n_time_steps)
                risk = safe_vector_norm(err_time, dim=-1).amax(dim=-1)
                local_margin = margin[:, start:start + ci.numel()]
                hinge = (risk - local_margin).clamp_min(0.0).square()
                band_loss = band_loss + (hinge * valid).sum(dim=1)
                band_denom = band_denom + valid.sum(dim=1)
            pair_loss_total = pair_loss_total + (band_loss / band_denom.clamp_min(1.0)).mean()
        pair_scalar = pair_loss_total / band_count
        total = total + float(pair_lambda) * pair_scalar
        metrics["risk_band_pair"] = float(pair_scalar.detach().item())

    if segment_lambda > 0.0 and L > 3 and max_segment_pairs is not None and int(max_segment_pairs) > 0:
        si, sj = _sample_upper_tri_pairs(L - 1, offset=2, max_pairs=max_segment_pairs, device=pred_spec.device)
        endpoints = torch.stack([si, si + 1, sj, sj + 1], dim=1).reshape(-1)
        unique, inverse = torch.unique(endpoints, sorted=True, return_inverse=True)
        target_ep = _dct_time_from_coeffs(target_coeff[:, unique, :, :], n_time_steps).permute(0, 2, 1, 3)
        if getattr(representation, "is_displacement", False):
            target_ep = target_ep + native_coords[:, unique, :].unsqueeze(1)
        target_ep = target_ep[:, :, inverse, :].view(B, int(n_time_steps), int(si.numel()), 4, 3)
        target_dist = _torch_segment_segment_distances(
            target_ep[:, :, :, 0, :], target_ep[:, :, :, 1, :],
            target_ep[:, :, :, 2, :], target_ep[:, :, :, 3, :],
        )
        margin = (target_dist.amin(dim=1) - seg_thresh.view(B, 1)).clamp_min(0.0)
        seg_valid = (mask[:, si] & mask[:, si + 1] & mask[:, sj] & mask[:, sj + 1]).float()
        seg_loss_total = pred_spec.new_tensor(0.0)
        for lo, hi in zip(edges[:-1], edges[1:]):
            err_ep = _dct_time_from_coeffs(err_coeff[:, unique, lo:hi, :], n_time_steps)
            risk_ep = safe_vector_norm(err_ep, dim=-1).amax(dim=-1)
            risk_ep = risk_ep[:, inverse].view(B, int(si.numel()), 4).amax(dim=-1)
            hinge = (risk_ep - margin).clamp_min(0.0).square()
            seg_loss_total = seg_loss_total + ((hinge * seg_valid).sum(dim=1) / seg_valid.sum(dim=1).clamp_min(1.0)).mean()
        seg_scalar = seg_loss_total / band_count
        total = total + float(segment_lambda) * seg_scalar
        metrics["risk_band_segment"] = float(seg_scalar.detach().item())

    return total, metrics


def backbone_bond_loss(pred_bb_abs, mask):
    """Constrain backbone covalent bond lengths to ideal values.

    Penalises predicted (absolute) backbone atom positions whose intra-residue
    and peptide-bond lengths deviate from their quasi-rigid ideal values.
    Bond length variance in folded proteins is < 0.05 Å, so this is a hard
    constraint. Essential for BB (12-channel) models to maintain bonded geometry.

    Ideal bonds (Å):
        N–CA : 1.46   CA–C  : 1.52
        C=O  : 1.23   C(i)–N(i+1) : 1.33  (peptide bond)

    Args:
        pred_bb_abs: (B, T, L, 4, 3) absolute predicted backbone coords,
                     atom order N / CA / C / O.
        mask: (B, L) residue validity mask.

    Returns:
        Scalar loss averaged over all four bond types.
    """
    B, T, L, _, _ = pred_bb_abs.shape
    pred_N  = pred_bb_abs[..., 0, :]   # (B, T, L, 3)
    pred_CA = pred_bb_abs[..., 1, :]
    pred_C  = pred_bb_abs[..., 2, :]
    pred_O  = pred_bb_abs[..., 3, :]

    DELTA = 0.1  # Å — bonds are quasi-rigid; penalise anything beyond ±0.1 Å
    res_mask = mask.unsqueeze(1).float()  # (B, 1, L)

    def _bond(a, b, ideal, bmask):
        d = safe_vector_norm(a - b, dim=-1)  # (B, T, L) or (B, T, L-1)
        err = F.huber_loss(d, torch.full_like(d, ideal), reduction='none', delta=DELTA)
        return (err * bmask).sum() / (bmask.sum() * T + 1e-8)

    pep_mask = (mask[:, :-1] * mask[:, 1:]).unsqueeze(1).float()  # (B, 1, L-1)
    losses = torch.stack([
        _bond(pred_N,          pred_CA,          1.46, res_mask),
        _bond(pred_CA,         pred_C,           1.52, res_mask),
        _bond(pred_C,          pred_O,           1.23, res_mask),
        _bond(pred_C[:, :, :-1], pred_N[:, :, 1:], 1.33, pep_mask),  # peptide
    ])
    return losses.mean()


def geometry_schedule_factor(epoch, warmup_start=50, warmup_epochs=10,
                              decay_start=200, decay_epochs=200, min_factor=0.1):
    """Warmup then slow decay: 0 → 1.0 (over warmup) → min_factor (over decay)."""
    if epoch < warmup_start:
        return 0.0
    if epoch < warmup_start + warmup_epochs:
        return (epoch - warmup_start) / warmup_epochs
    if epoch < decay_start:
        return 1.0
    if epoch < decay_start + decay_epochs:
        progress = (epoch - decay_start) / decay_epochs
        return 1.0 - (1.0 - min_factor) * progress
    return min_factor


def spectral_amplitude_loss(x_0_pred, x_0_gt, mask, coord_channels=3, freq_weights=None):
    """Per-residue signed spectral coefficient matching loss.

    Signed L1 loss over per-residue, per-frequency, per-channel spectral
    coefficients. Uncapped (no Huber) so high-T residues with large amplitudes
    contribute proportional gradient rather than a clipped constant pull.
    Direction-aware (acts on signed residuals, not norms) so the low-k / DC
    slow-mode sign is supervised, not just magnitude.

    Args:
        x_0_pred: (B, L, K*C) predicted clean spectral volume.
        x_0_gt: (B, L, K*C) ground-truth spectral volume.
        mask: (B, L) residue validity mask.
        coord_channels: C, the number of coordinate channels (3 for CA, 12 for BB).
        freq_weights: Optional (K,) weights to upweight specific frequencies.

    Returns:
        Scalar signed-L1 loss averaged across (B, L, K, C).
    """
    B, L, D = x_0_pred.shape
    K = D // coord_channels

    pred = x_0_pred.view(B, L, K, coord_channels)                           # (B, L, K, C)
    gt   = x_0_gt.view(B, L, K, coord_channels).detach()                    # (B, L, K, C)

    loss = (pred - gt).abs()                                                 # (B, L, K, C)
    mask_exp = mask.float().unsqueeze(-1).unsqueeze(-1)                      # (B, L, 1, 1)
    weights = _prepare_frequency_weights(freq_weights, K, x_0_pred.device, loss.dtype)
    weights = weights.view(1, 1, K, 1)
    return (loss * mask_exp * weights).sum() / (
        mask.float().sum() * coord_channels * weights.sum() + 1e-8
    )


def spectral_mode_vector_norm_loss(
    x_pred,
    x_gt,
    mask,
    coord_channels=3,
    n_modes=1,
):
    """MSE on per-residue vector amplitudes for the first ``n_modes`` modes."""
    B, L, D = x_pred.shape
    K = D // coord_channels
    n_modes = min(int(n_modes), K)
    pred = x_pred.view(B, L, K, coord_channels)[:, :, :n_modes, :]
    gt = x_gt.view(B, L, K, coord_channels)[:, :, :n_modes, :].detach()
    pred_amp = safe_vector_norm(pred, dim=-1)
    gt_amp = safe_vector_norm(gt, dim=-1)
    sq = (pred_amp - gt_amp) ** 2
    if mask is None:
        return sq.mean()
    m = mask.float().unsqueeze(-1)
    return (sq * m).sum() / (m.sum() * n_modes + 1e-8)


def masked_feature_mse(pred, target, mask):
    """Mean squared error over valid residues for tensors shaped (B, L, C)."""
    mask_exp = mask.float().unsqueeze(-1)
    loss = F.mse_loss(pred, target.detach(), reduction="none")
    return (loss * mask_exp).sum() / (mask.float().sum() * pred.shape[-1] + 1e-8)


def spectral_dc_mse_loss(x_0_pred, x_0_gt, mask, coord_channels=3):
    """Signed DC-only clean-spectrum MSE on k=0 across coordinate channels."""
    return masked_feature_mse(
        x_0_pred[:, :, :coord_channels],
        x_0_gt[:, :, :coord_channels],
        mask,
    )


def spectral_low_k_loss(
    x_0_pred,
    x_0_gt,
    mask,
    coord_channels=3,
    n_modes=8,
    freq_weights=None,
):
    """Signed low-frequency clean-spectrum MSE on the first n_modes.

    Uncapped signed MSE (no Huber) so high-T low-k coefficients with large
    magnitudes get proportional gradient rather than clipped constant pull.
    Directly supervises the DC / slow collective modes where unfolding drift
    lives, preserving coefficient sign.
    """
    B, L, D = x_0_pred.shape
    K = D // coord_channels
    n = min(max(int(n_modes), 1), K)

    pred = x_0_pred.view(B, L, K, coord_channels)[:, :, :n, :]
    gt = x_0_gt.view(B, L, K, coord_channels)[:, :, :n, :].detach()

    loss = (pred - gt).pow(2)                                               # (B, L, n, C)
    mask_exp = mask.float().unsqueeze(-1).unsqueeze(-1)                     # (B, L, 1, 1)
    weights = _prepare_frequency_weights(freq_weights, n, x_0_pred.device, loss.dtype)
    weights = weights.view(1, 1, n, 1)
    return (loss * mask_exp * weights).sum() / (
        mask.float().sum() * coord_channels * weights.sum() + 1e-8
    )


def time_domain_low_k_loss(
    x_0_pred: torch.Tensor,
    x_0_gt: torch.Tensor,
    mask: torch.Tensor,
    *,
    window_size: int,
    coord_channels: int,
    n_modes: int = 8,
) -> torch.Tensor:
    """Signed low-k DCT loss for time-domain coordinate residual models."""
    b, l, _ = x_0_gt.shape
    t = int(window_size)
    c = int(coord_channels)
    n = min(max(int(n_modes), 1), t)
    pred = x_0_pred.view(b, l, t, c)
    target = x_0_gt.view(b, l, t, c).detach()
    w_dct = DCT.get_dct_matrix(t, x_0_gt.device)[:n].to(dtype=x_0_gt.dtype)
    pred_k = torch.einsum("bltc,kt->blkc", pred, w_dct)
    target_k = torch.einsum("bltc,kt->blkc", target, w_dct)
    loss = (pred_k - target_k).pow(2)
    mask_exp = mask.float().view(b, l, 1, 1)
    return (loss * mask_exp).sum() / (mask.float().sum() * n * c + 1e-8)


def time_domain_dct_coeff_l1_loss(
    x_0_pred: torch.Tensor,
    x_0_gt: torch.Tensor,
    mask: torch.Tensor,
    *,
    window_size: int,
    coord_channels: int,
) -> torch.Tensor:
    """Signed full-DCT coefficient L1 loss for decoded time-domain residuals."""
    b, l, _ = x_0_gt.shape
    t = int(window_size)
    c = int(coord_channels)
    pred = x_0_pred.view(b, l, t, c)
    target = x_0_gt.view(b, l, t, c).detach()
    w_dct = DCT.get_dct_matrix(t, x_0_gt.device).to(dtype=x_0_gt.dtype)
    pred_k = torch.einsum("bltc,kt->blkc", pred, w_dct)
    target_k = torch.einsum("bltc,kt->blkc", target, w_dct)
    loss = (pred_k - target_k).abs()
    mask_exp = mask.float().view(b, l, 1, 1)
    return (loss * mask_exp).sum() / (mask.float().sum() * t * c + 1e-8)


def train_step(
    model, diffusion, transform_engine, batch, device, dtype_ctx,
    top_k_freqs=64, cond_drop_prob=0.15, displacement=True, representation=None,
    freq_weighting=None, bending_lambda=0.0,
    geo_loss="idct_ca-ca",
    geometry_lambda=0.0, geometry_warmup_start=50, geometry_warmup_epochs=10,
    geometry_decay_start=200, geometry_decay_epochs=200,
    geometry_tol=0.05, clash_lambda=0.0, clash_threshold=3.5,
    clash_max_pairs=4096, clash_pair_chunk=512,
    topology_margin_artifact=None, spectral_geo_segment_threshold=1.0,
    spectral_geo_max_segment_pairs=1024, risk_band_max_pairs=2048,
    risk_band_max_segment_pairs=512,
    representation_barrier_lambda=0.0,
    rmsf_lambda=0.0, rmsf_warmup_start=100, rmsf_warmup_epochs=10,
    low_freq_lambda=0.0, low_freq_modes=8,
    dc_lambda=0.0, dc_start_epoch=10,
    v17_aux_modes=17,
    v17_low_mode_lambda=0.0,
    v17_adjacent_lambda=0.0,
    v17_idct_bond_lambda=0.0,
    v17_caca_tolerance_lambda=0.0,
    v17_caca_target=3.84,
    v17_caca_tolerance=0.05,
    v12e_spec_graph_residual_lambda=0.0,
    loss_slow_weight=1.0, loss_fast_weight=1.0, loss_total_weight=0.1,
    epoch=0,
    is_validation=False, is_main_process=True, global_step=0, log_every=1000
):
    '''Single training step supporting x_0, v, and noise prediction targets.

    Returns:
        total_loss: Scalar loss tensor.
        metrics: Dict of loggable metrics.
    '''
    coords_abs = batch["coords"].to(device)
    coords = coords_abs
    mask = batch["mask"].to(device)
    raw_temps = batch["temp"].to(device)
    temps = raw_temps.clone()

    native_coords = batch["native_coords"].to(device)
    representation = representation or CoordinateRepresentation(
        displacement=displacement,
        coord_channels=native_coords.shape[-1],
    )
    native_angles = batch.get("native_angles", None)
    if native_angles is not None:
        native_angles = native_angles.to(device)
    res_type = batch.get("res_type", None)
    if res_type is not None:
        res_type = res_type.to(device)
    dssp = batch.get("dssp", None)
    if dssp is not None:
        dssp = dssp.to(device)

    real_model = model.module if hasattr(model, "module") else model
    inner_model = getattr(real_model, "model", real_model)
    prediction_target = getattr(real_model, "prediction_target", "noise")
    is_dct = getattr(real_model, "is_dct", True)
    is_time_domain = getattr(real_model, "is_time_domain", False)
    is_manifold_domain = getattr(real_model, "is_manifold_domain", False)

    # Pipeline sanity check on the very first training step
    if (
        is_main_process
        and global_step == 0
        and not is_validation
        and not getattr(real_model, "is_time_domain", False)
        and not is_manifold_domain
        and representation.name in ("raw_coords", "displacement")
    ):
        debug_print_pipeline(
            batch, transform_engine, real_model, device,
            top_k_freqs=top_k_freqs, displacement=displacement,
            label=f"TRAIN step=0",
        )

    if representation.is_unit_chain and (is_time_domain or is_manifold_domain):
        raise ValueError(f"{representation.name} is currently supported for spectral models only")

    # Build INPUT and derive channels from actual data shapes
    angles = batch.get("angles", None)
    if angles is not None:
        angles = angles.to(device)
        torsion_mask = batch.get("torsion_mask", None)
        if torsion_mask is not None:
            torsion_mask = torsion_mask.to(device)
            if torsion_mask.dim() == 2:
                torsion_mask = torsion_mask.unsqueeze(-1).expand(-1, -1, angles.shape[-1])
        coord_repr, repr_context = representation.forward(
            coords_abs, native_coords, mask=mask, return_context=True
        )
        INPUT = torch.cat([coord_repr, angles], dim=-1)
    else:
        torsion_mask = None
        coord_repr, repr_context = representation.forward(
            coords_abs, native_coords, mask=mask, return_context=True
        )
        INPUT = coord_repr

    channels = INPUT.shape[-1]  # 3 (CA), 7 (CA+torsions), 12 (BB), 16 (BB+torsions), etc.
    coord_channels = native_coords.shape[-1]
    repr_coord_channels = representation.model_coord_channels

    # Temperature jitter (training only)
    if not is_validation:
        jitter = (torch.rand_like(temps, dtype=torch.float32) * 2 - 1) * 10.0
        temps = temps + jitter

    norm_temps = torch.clamp((temps - 250.0) / 200.0, 0.0, 1.0)

    # Window position conditioning (normalised start index ∈ [0,1])
    win_pos = batch.get("win_pos", None)
    if win_pos is not None:
        win_pos = win_pos.float().to(device)

    # CFG dropout
    cond_drop_mask = None
    if not is_validation and cond_drop_prob > 0.0:
        B = native_coords.shape[0]
        cond_drop_mask = torch.rand(B, device=device) < cond_drop_prob

    with dtype_ctx:
        dc_baseline = None

        # 1. Representation: spectral volume, raw deviations, or frame/torsion manifold latents.
        if is_manifold_domain:
            B_td, T_frames, L_td, C_td = coords_abs.shape
            expected_coord_channels = int(getattr(inner_model, "coord_channels", C_td))
            if C_td != expected_coord_channels:
                raise ValueError(
                    f"{inner_model.__class__.__name__} expects coords with {expected_coord_channels} channels; "
                    f"got {C_td}. Check coords_type and model_type."
                )
            x_0 = inner_model.encode_coords(coords_abs, native_coords, mask=mask)
        elif is_time_domain: # FNO/HNO/CNO path: diffuse directly on coordinate deviations (B, L, T*C)
            C_td = real_model.config.in_channels
            B_td, T_frames, L_td, _ = coords.shape
            if representation.name not in ("raw_coords", "displacement"):
                raise ValueError(f"{representation.name} is not supported for time-domain models yet")
            coords = coord_repr
            x_dev = coords[..., :C_td]  # (B, T, L, C)
            x_0 = x_dev.permute(0, 2, 1, 3).reshape(B_td, L_td, T_frames * C_td)
            x_0 = (x_0 / real_model.model.coord_scale) * mask.unsqueeze(-1)
        else:
            #x_0 = adapter.forward_transform(INPUT, mask, top_k=top_k_freqs) # REPLACED ADAPTER AS NORMALISATION NOW DONE INTERNALLY IN MODE 06.04.26
            #INPUT_masked = INPUT * mask.unsqueeze(1).unsqueeze(-1)
            INPUT_masked = INPUT * mask[:, None, :, None]
            x_0 = transform_engine.time_to_spectral(INPUT_masked, top_k=top_k_freqs)
            dc_baseline_per_res = batch.get("dc_baseline_per_res", None)
            if dc_baseline_per_res is not None:
                dc_baseline_per_res = dc_baseline_per_res.to(device=device, dtype=x_0.dtype)
            x_0, dc_baseline = maybe_residualise_dc(
                transform_engine, x_0, temps, mask, repr_coord_channels,
                per_residue_baseline=dc_baseline_per_res,
            )

        # 2. Diffusion forward (same for both paths)
        t = diffusion.sample_timesteps(x_0.shape[0]).to(device)
        x_t, noise = diffusion.q_sample(x_0, t)

        # 3. Residue/feature mask protects invalid residues/features in the
        # noisy input, clean target, noise target, and loss.
        if is_manifold_domain:
            if hasattr(inner_model, "latent_loss_mask"):
                base_loss_mask = inner_model.latent_loss_mask(mask, window_size=T_frames).to(
                    device=x_0.device, dtype=x_0.dtype
                )
            else:
                base_loss_mask = mask.unsqueeze(-1).expand_as(x_0).float()
        elif is_time_domain:
            base_loss_mask = mask.unsqueeze(-1).expand_as(x_0).float()
        else:
            base_loss_mask = build_spectral_mask(
                mask, torsion_mask, top_k_freqs, is_dct,
                coord_channels=repr_coord_channels,
                representation=representation,
            ).to(device=x_0.device, dtype=x_0.dtype)

        x_t = x_t * base_loss_mask
        noise = noise * base_loss_mask
        x_0 = x_0 * base_loss_mask

        full_loss_mask = base_loss_mask

        # 4. Model prediction
        # if is_time_domain:
        #     model_out = model(
        #         x_t, t, norm_temps, native_coords,
        #         mask=mask, win_pos=win_pos, cond_drop_mask=cond_drop_mask,
        #     )
        # else:
        #     model_out = model(
        #         x_t, t, norm_temps, native_coords, native_angles,
        #         mask=mask, win_pos=win_pos, cond_drop_mask=cond_drop_mask,
        #     )
        # Optional NMA RMSF prior (dataloader surfaces it when a sidecar is loaded).
        rmsf_prior_b = batch.get("rmsf_prior", None)
        if rmsf_prior_b is not None:
            rmsf_prior_b = rmsf_prior_b.to(device)

        dc_aux_active = dc_lambda > 0.0 and epoch >= dc_start_epoch
        is_dual_branch = getattr(inner_model, "is_dual_branch", False)
        is_v17_model = (
            hasattr(inner_model, "spectral_graph_refiner_modes")
            or hasattr(inner_model, "bond_spectral_graph_refiner_modes")
        )
        v17_aux_active = is_v17_model and any(
            float(v) > 0.0
            for v in (
                v17_low_mode_lambda,
                v17_adjacent_lambda,
                v17_idct_bond_lambda,
                v17_caca_tolerance_lambda,
            )
        )
        amp_log_gain_mean = float("nan")
        amp_log_gain_min = float("nan")
        amp_log_gain_max = float("nan")
        amp_gain_max = float("nan")
        need_aux_outputs = (
            not is_time_domain
            and (
                (dc_aux_active and getattr(inner_model, "use_low_k_correction_head", False))
                or is_dual_branch
                or v17_aux_active
                or hasattr(inner_model, "amp_head")
                or hasattr(inner_model, "v12e_spec_graph")
            )
        )
        model_batch = {
            "x": x_t,
            "t": t,
            "temp": norm_temps,
            "native_coords": native_coords,
            "native_angles": native_angles,
            "res_type": res_type,
            "dssp": dssp,
            "mask": mask,
            "win_pos": win_pos,
            "cond_drop_mask": cond_drop_mask,
            "rmsf_prior": rmsf_prior_b,
            "return_aux": need_aux_outputs,
        }
        model_out = model(model_batch)
        aux_model_out = model_out if isinstance(model_out, dict) else {}
        if isinstance(model_out, dict):
            model_out = model_out["pred"]
        amp_log_gain = aux_model_out.get("amp_log_gain")
        amp_gain = aux_model_out.get("amp_gain")
        if amp_log_gain is not None:
            with torch.no_grad():
                valid_amp = amp_log_gain[mask.bool()] if mask is not None else amp_log_gain.reshape(-1, amp_log_gain.shape[-1])
                amp_log_gain_mean = valid_amp.mean().item()
                amp_log_gain_min = valid_amp.min().item()
                amp_log_gain_max = valid_amp.max().item()
        if amp_gain is not None:
            with torch.no_grad():
                valid_gain = amp_gain[mask.bool()] if mask is not None else amp_gain.reshape(-1, amp_gain.shape[-1])
                amp_gain_max = valid_gain.max().item()
        # # CNO5/6 return (pred, amp_list) tuple — unpack
        # if isinstance(model_out, tuple):
        #     model_out = model_out[0]
        #model_out = torch.clamp(model_out, -10.0, 10.0)

        # 5. Determine loss target based on prediction type
        sqrt_ab = torch.sqrt(diffusion.alpha_bar[t]).view(-1, 1, 1)
        sqrt_one_minus_ab = torch.sqrt(1.0 - diffusion.alpha_bar[t]).view(-1, 1, 1)
        x_0_pred = None

        if prediction_target == "noise":
            target = noise
        elif prediction_target == "x_0":
            target = x_0
        elif prediction_target == "v":
            target = sqrt_ab * noise - sqrt_one_minus_ab * x_0
        else:
            raise ValueError(f"Unknown prediction_target: {prediction_target}")

        # The RMSF prior is a clean-sample amplitude prior. When the model is
        # parameterised in v/noise space, apply the gain after recovering x_0,
        # then convert back to the model's target space for the main loss.
        if (
            prediction_target != "x_0"
            and hasattr(inner_model, "apply_rmsf_prior_gain")
            and rmsf_prior_b is not None
        ):
            x_0_pred, _ = diffusion.extract_x0_eps_from_prediction(
                model_out, x_t, t, prediction_target
            )
            x_0_pred = inner_model.apply_rmsf_prior_gain(
                x_0_pred, rmsf_prior_b, mask=mask
            )
            model_out = diffusion.prediction_from_x0(
                x_0_pred, x_t, t, prediction_target
            )

        # 6. Min-SNR-gamma weighting
        if diffusion.min_snr_gamma is not None:
            snr = diffusion.get_snr(t)
            clamped_snr = torch.clamp(snr, max=diffusion.min_snr_gamma)

            if prediction_target == "noise":
                loss_weights_t = clamped_snr / snr
            elif prediction_target == "x_0":
                loss_weights_t = clamped_snr
            elif prediction_target == "v":
                loss_weights_t = clamped_snr / (snr + 1.0)

            loss_weights_t = loss_weights_t.view(-1, 1, 1)
        else:
            loss_weights_t = torch.ones((x_t.shape[0], 1, 1), device=device)

        # 7. Frequency band weighting (spectral path only; uniform for CNO4)
        weights_k = None
        if is_time_domain or is_manifold_domain:
            weights_expanded = torch.ones(1, 1, x_0.shape[-1], device=device)
        else:
            complex_mult = 1 if is_dct else 2
            C_freq = channels * complex_mult
            weights_k = get_frequency_weights(freq_weighting, top_k_freqs, device)
            weights_expanded = weights_k.view(-1, 1).repeat(1, C_freq).view(1, 1, -1)

        # 8. Bending-based spatial weighting / auxiliary loss
        bending_loss = torch.tensor(0.0, device=device)
        if bending_lambda > 0.0:
            if is_time_domain:
                # Auxiliary bending loss: spatial gradient of predicted vs true coords
                # model operates in (B, L, T*C) → reshape to (B, T, L, C)
                C_td = real_model.config.in_channels
                pred_coords = model_out.view(B_td, L_td, T_frames, C_td).permute(0, 2, 1, 3)
                true_coords = x_0.view(B_td, L_td, T_frames, C_td).permute(0, 2, 1, 3)
                pred_bend   = pred_coords[:, :, 1:, :] - pred_coords[:, :, :-1, :]  # (B, T, L-1, C)
                true_bend   = true_coords[:, :, 1:, :] - true_coords[:, :, :-1, :]
                bend_mask   = (mask[:, 1:] * mask[:, :-1]).unsqueeze(1).unsqueeze(-1)  # (B, 1, L-1, 1)
                bending_loss = ((pred_bend - true_bend) ** 2 * bend_mask).sum() / (bend_mask.sum() * C_td + 1e-8)
                bending_loss = bending_lambda * bending_loss
            elif not is_manifold_domain:
                # Per-residue loss weighting: upweight hinge residues in spectral loss
                bend_w = compute_bending_weight(
                    coords_abs, native_coords, mask.bool(), bending_lambda, False
                )  # (B, L)
                # Broadcast (B, L) → (B, L, D) to reweight the spectral L dimension
                weights_expanded = weights_expanded * bend_w.unsqueeze(-1)

        total_weights = full_loss_mask * weights_expanded * loss_weights_t
        sq_diff = (model_out - target) ** 2

        # DIAGNOSTICS
        diagnostic_loss_mask = full_loss_mask.to(dtype=torch.bool)
        valid_pred = model_out[diagnostic_loss_mask].reshape(-1)
        valid_target = target[diagnostic_loss_mask].reshape(-1)

        # Initialize defaults to avoid UnboundLocalError
        pred_std = target_std = pred_mean = target_mean = pred_norm = target_norm = cos_sim = 0.0

        if valid_pred.numel() > 0:
            with torch.no_grad():
                pred_std = valid_pred.std().item()
                target_std = valid_target.std().item()
                pred_mean = valid_pred.mean().item()
                target_mean = valid_target.mean().item()
                
                pred_norm = torch.linalg.vector_norm(valid_pred).item() / (valid_pred.numel() + 1e-8)
                target_norm = torch.linalg.vector_norm(valid_target).item() / (valid_target.numel() + 1e-8)
                
                cos_sim = torch.nn.functional.cosine_similarity(
                    valid_pred.unsqueeze(0), 
                    valid_target.unsqueeze(0)
                ).item()

        # Print to console occasionally
        if is_main_process and global_step % log_every == 0:
            print(f"--- METRICS @ Step {global_step} ---")
            print(f"Shapes   | Masked: {valid_pred.shape}") # Should be (1792 * Channels,)
            print(f"Std Dev  | Pred: {pred_std:.4f}  Target: {target_std:.4f}")
            print(f"Norm     | Pred: {pred_norm:.4f}  Target: {target_norm:.4f}")
            print(f"Mean     | Pred: {pred_mean:.4f}  Target: {target_mean:.4f}")
            print(f"Cos Sim  | {cos_sim:.4f}")
            print("-----------------------------")

        loss_mse = (sq_diff * total_weights).sum() / (total_weights.sum() + 1e-8)

        # 8.5 Dual-branch three-term loss (v11 only).
        # Slow/fast branch predictions are returned as slices of the full
        # spectrum in raw coefficient space. Split the target along the mode
        # axis and compute per-branch weighted MSE in the same form as
        # loss_mse above, so the scales remain directly comparable.
        loss_slow = torch.tensor(0.0, device=device)
        loss_fast = torch.tensor(0.0, device=device)
        if is_dual_branch and not is_time_domain:
            slow_pred = aux_model_out["slow_pred"]
            fast_pred = aux_model_out["fast_pred"]
            slow_pred_type = aux_model_out.get("slow_pred_type", "xyz")
            K_slow = int(getattr(inner_model, "K_slow"))
            K_total = int(getattr(inner_model, "top_k_freqs"))
            C_per_mode = target.shape[-1] // K_total
            slow_end = K_slow * C_per_mode

            target_slow = target[..., :slow_end]
            target_fast = target[..., slow_end:]
            mask_slow = full_loss_mask[..., :slow_end]
            mask_fast = full_loss_mask[..., slow_end:]
            w_slow = weights_expanded[..., :slow_end]
            w_fast = weights_expanded[..., slow_end:]
            tw_slow = mask_slow * w_slow * loss_weights_t
            tw_fast = mask_fast * w_fast * loss_weights_t

            if slow_pred_type == "amplitude":
                slow_amp_pred = aux_model_out["slow_amp_pred"]
                target_slow_kc = target_slow.view(target.shape[0], target.shape[1], K_slow, C_per_mode)
                slow_amp_target = torch.linalg.norm(target_slow_kc, dim=-1)
                amp_sq = (slow_amp_pred - slow_amp_target) ** 2
                amp_mask = mask.unsqueeze(-1).to(amp_sq.dtype)
                amp_weights = weights_k[:K_slow].view(1, 1, K_slow) if weights_k is not None else 1.0
                tw_amp = amp_mask * amp_weights * loss_weights_t
                loss_slow = (amp_sq * tw_amp).sum() / (tw_amp.sum() + 1e-8)
            else:
                sd_slow = (slow_pred - target_slow) ** 2
                loss_slow = (sd_slow * tw_slow).sum() / (tw_slow.sum() + 1e-8)
            sd_fast = (fast_pred - target_fast) ** 2
            loss_fast = (sd_fast * tw_fast).sum() / (tw_fast.sum() + 1e-8)

        # 9. Auxiliary losses (spectral models only)
        eff_geometry_lambda = geometry_lambda * geometry_schedule_factor(
            epoch, geometry_warmup_start, geometry_warmup_epochs,
            geometry_decay_start, geometry_decay_epochs,
        )
        eff_rmsf_lambda = rmsf_lambda * geometry_schedule_factor(
            epoch, rmsf_warmup_start, rmsf_warmup_epochs,
            decay_start=99999, decay_epochs=1,  # no decay for rmsf
        )
        eff_low_freq_lambda = low_freq_lambda * geometry_schedule_factor(
            epoch, rmsf_warmup_start, rmsf_warmup_epochs,
            decay_start=99999, decay_epochs=1,
        )
        eff_dc_lambda = dc_lambda if epoch >= dc_start_epoch else 0.0
        geo_modes = parse_geo_loss_modes(geo_loss)
        use_idct_geo = "idct_ca-ca" in geo_modes
        use_spec_geo = "spec_geo" in geo_modes
        use_risk_band_geo = "risk_band" in geo_modes

        geometry_loss = torch.tensor(0.0, device=device)
        geometry_loss_raw = torch.tensor(0.0, device=device)
        spectral_amp_loss = torch.tensor(0.0, device=device)
        low_freq_loss = torch.tensor(0.0, device=device)
        dc_loss = torch.tensor(0.0, device=device)
        v17_loss = torch.tensor(0.0, device=device)
        v17_low_mode_loss = torch.tensor(0.0, device=device)
        v17_adjacent_loss = torch.tensor(0.0, device=device)
        v17_idct_bond_loss = torch.tensor(0.0, device=device)
        v17_caca_tol_loss = torch.tensor(0.0, device=device)
        v12e_residual_loss = torch.tensor(0.0, device=device)
        v12e_delta_rms = float("nan")
        v12e_delta_max = float("nan")
        dc_pred = float("nan")
        dc_gt   = float("nan")
        dc_head_mean = float("nan")
        dc_final_mse = float("nan")
        dc_final_abs_error = float("nan")
        refiner_delta_rms = float("nan")
        shake_delta_rms = float("nan")
        caca_debug_metrics = {}
        track_caca_debug = bool(
            torch.isclose(raw_temps.float(), torch.full_like(raw_temps.float(), 413.0), atol=0.5, rtol=0.0).any()
            or torch.isclose(raw_temps.float(), torch.full_like(raw_temps.float(), 450.0), atol=0.5, rtol=0.0).any()
        )

        need_x0_pred = (
            eff_geometry_lambda > 0.0
            or eff_rmsf_lambda > 0.0
            or eff_low_freq_lambda > 0.0
            or eff_dc_lambda > 0.0
            or representation_barrier_lambda > 0.0
            or v17_aux_active
            or track_caca_debug
        ) and not is_time_domain and not is_manifold_domain
        if need_x0_pred:
            # Recover predicted x_0 in spectral domain from the possibly
            # gain-adjusted target-space prediction.
            if x_0_pred is None:
                x_0_pred, _ = diffusion.extract_x0_eps_from_prediction(
                    model_out, x_t, t, prediction_target
                )

            x_0_pred_raw = maybe_restore_dc(transform_engine, x_0_pred, dc_baseline, repr_coord_channels)
            x_0_target_raw = maybe_restore_dc(transform_engine, x_0, dc_baseline, repr_coord_channels)

            if eff_geometry_lambda > 0.0 and (use_spec_geo or use_risk_band_geo) and is_dct:
                if use_spec_geo:
                    spec_geo_loss, spec_geo_metrics = spectral_geometry_losses(
                        x_0_pred_raw,
                        x_0_target_raw,
                        native_coords,
                        mask,
                        raw_temps,
                        top_k_freqs=top_k_freqs,
                        channels=channels,
                        coord_channels=coord_channels,
                        representation=representation,
                        n_time_steps=coords_abs.shape[1],
                        margin_artifact=topology_margin_artifact,
                        clash_threshold=clash_threshold,
                        segment_threshold=spectral_geo_segment_threshold,
                        max_pairs=clash_max_pairs,
                        max_segment_pairs=spectral_geo_max_segment_pairs,
                        pair_chunk=clash_pair_chunk,
                    )
                    geometry_loss = geometry_loss + eff_geometry_lambda * spec_geo_loss
                    for key, value in spec_geo_metrics.items():
                        caca_debug_metrics[f"debug/{key}"] = value
                if use_risk_band_geo:
                    risk_loss, risk_metrics = risk_band_geometry_loss(
                        x_0_pred_raw,
                        x_0_target_raw,
                        native_coords,
                        mask,
                        raw_temps,
                        top_k_freqs=top_k_freqs,
                        channels=channels,
                        coord_channels=coord_channels,
                        representation=representation,
                        n_time_steps=coords_abs.shape[1],
                        margin_artifact=topology_margin_artifact,
                        clash_threshold=clash_threshold,
                        segment_threshold=spectral_geo_segment_threshold,
                        max_pairs=risk_band_max_pairs,
                        max_segment_pairs=risk_band_max_segment_pairs,
                        pair_chunk=clash_pair_chunk,
                    )
                    geometry_loss = geometry_loss + eff_geometry_lambda * risk_loss
                    for key, value in risk_metrics.items():
                        caca_debug_metrics[f"debug/{key}"] = value

            # DC component diagnostic: x_0 layout is (B, L, K*C), DC slot is indices 0:repr_coord_channels
            with torch.no_grad():
                dc_pred_slice = x_0_pred_raw[:, :, :repr_coord_channels]
                dc_gt_slice = x_0_target_raw[:, :, :repr_coord_channels]
                dc_pred = dc_pred_slice[mask.bool()].mean().item()
                dc_gt   = dc_gt_slice[mask.bool()].mean().item()
                dc_diff = dc_pred_slice - dc_gt_slice
                valid_dc = mask.unsqueeze(-1).expand_as(dc_diff)
                dc_final_mse = ((dc_diff.square() * valid_dc).sum() / valid_dc.sum().clamp_min(1.0)).item()
                dc_final_abs_error = ((dc_diff.abs() * valid_dc).sum() / valid_dc.sum().clamp_min(1.0)).item()

            # Spectral amplitude loss: match per-residue per-frequency amplitude (power spectral density)
            if eff_rmsf_lambda > 0.0:
                spectral_amp_loss = eff_rmsf_lambda * spectral_amplitude_loss(
                    x_0_pred,
                    x_0,
                    mask,
                    coord_channels=repr_coord_channels,
                    freq_weights=weights_k,
                )

            # Explicit signed supervision on DC / low-k clean coefficients.
            if eff_low_freq_lambda > 0.0:
                low_freq_loss = eff_low_freq_lambda * spectral_low_k_loss(
                    x_0_pred,
                    x_0,
                    mask,
                    coord_channels=repr_coord_channels,
                    n_modes=low_freq_modes,
                    freq_weights=weights_k,
                )

            if eff_dc_lambda > 0.0:
                # Supervise the final corrected DC coefficient directly. This is
                # more stable than targeting the isolated correction branch
                # against a moving residual formed from the base prediction.
                dc_loss = eff_dc_lambda * spectral_dc_mse_loss(
                    x_0_pred_raw,
                    x_0_target_raw,
                    mask,
                    coord_channels=repr_coord_channels,
                )

                # Keep the head output as a diagnostic so we can still inspect
                # whether the correction branch is active and well-behaved.
                dc_head_pred = aux_model_out.get("low_k_correction_dc")
                if dc_head_pred is not None:
                    with torch.no_grad():
                        dc_head_mean = dc_head_pred[mask.bool()].mean().item()

            if v17_aux_active:
                caca_tol_weight = float(v17_caca_tolerance_lambda)
                if caca_tol_weight > 0.0 and not representation.is_displacement:
                    raise ValueError(
                        "v17_caca_tolerance_lambda currently expects displacement spectra; "
                        f"got representation={representation.name!r}."
                    )
                ca_start_native = 3 if coord_channels == 12 else 0
                native_ca_for_v17 = native_coords[..., ca_start_native:ca_start_native + 3]
                v17_losses = v17_auxiliary_losses(
                    x_0_pred_raw,
                    x_0_target_raw,
                    mask,
                    coord_channels=repr_coord_channels,
                    n_modes=v17_aux_modes,
                    n_time_steps=coords_abs.shape[1],
                    native_ca=native_ca_for_v17,
                    low_mode_weight=v17_low_mode_lambda,
                    adjacent_weight=v17_adjacent_lambda,
                    idct_bond_weight=v17_idct_bond_lambda,
                    caca_tolerance_weight=caca_tol_weight,
                    caca_target=v17_caca_target,
                    caca_tolerance=v17_caca_tolerance,
                )
                v17_loss = v17_losses["loss_total"]
                v17_low_mode_loss = v17_losses.get(
                    "v17_low_mode_spectral_mse", v17_low_mode_loss
                )
                v17_adjacent_loss = v17_losses.get(
                    "v17_adjacent_spectral_mse", v17_adjacent_loss
                )
                v17_idct_bond_loss = v17_losses.get(
                    "v17_idct_bond_vector_mse", v17_idct_bond_loss
                )
                v17_caca_tol_loss = v17_losses.get(
                    "v17_caca_tolerance_loss", v17_caca_tol_loss
                )

            if representation_barrier_lambda > 0.0:
                pred_repr_time_for_barrier = transform_engine.spectral_to_time(
                    x_0_pred_raw, n_time_steps=coords_abs.shape[1], n_channels=channels
                )[..., :repr_coord_channels]
                geometry_loss = geometry_loss + representation_barrier_lambda * representation.length_barrier_loss(
                    pred_repr_time_for_barrier,
                    mask=mask,
                )

            pred_ca = None
            target_ca = None
            # Local geometry loss: CA inter-residue distances at sep=1..5
            if (eff_geometry_lambda > 0.0 and use_idct_geo) or track_caca_debug:
                T_frames = coords.shape[1]
                pred_repr_time = transform_engine.spectral_to_time(
                    x_0_pred_raw, n_time_steps=T_frames, n_channels=channels
                )

                ca_start = 3 if coord_channels == 12 else 0
                pred_time_abs = representation.inverse(
                    pred_repr_time[..., :repr_coord_channels],
                    native_coords,
                    mask=mask,
                    context=repr_context,
                )
                target_time_abs = coords_abs
                pred_ca = pred_time_abs[..., ca_start:ca_start + 3]    # (B, T, L, 3)
                target_ca = target_time_abs[..., ca_start:ca_start + 3]     # (B, T, L, 3)

                # v12c: post-IDCT CA refinement + differentiable SHAKE.
                # v12a: optional SHAKE-only pass-through when use_shake=True.
                shake_residual_loss = None
                if hasattr(inner_model, "refine_ca"):
                    base_ca = pred_ca
                    refined_ca, pred_ca = inner_model.refine_ca(base_ca, mask)
                    shake_residual_loss = ((refined_ca - pred_ca) ** 2).mean()
                    with torch.no_grad():
                        refiner_delta_rms = torch.sqrt((refined_ca - base_ca).square().mean()).item()
                        shake_delta_rms = torch.sqrt((pred_ca - refined_ca).square().mean()).item()
                elif getattr(inner_model, "use_shake", False):
                    from src.models.shake import shake_caca as _shake_caca
                    refined_ca = pred_ca
                    pred_ca = _shake_caca(
                        pred_ca, mask=mask,
                        target=getattr(real_model, "shake_target", 3.8),
                        n_iter=getattr(real_model, "shake_n_iter", 20),
                    )
                    with torch.no_grad():
                        shake_delta_rms = torch.sqrt((pred_ca - refined_ca).square().mean()).item()

                if track_caca_debug:
                    caca_debug_metrics = collect_temp_caca_debug_metrics(
                        raw_temps,
                        pred_ca,
                        target_ca,
                        mask,
                        clash_threshold=clash_threshold,
                        clash_max_pairs=clash_max_pairs,
                        clash_pair_chunk=clash_pair_chunk,
                    )

                if eff_geometry_lambda > 0.0 and use_idct_geo:
                    ca_loss_per = local_geometry_loss(
                        pred_ca, target_ca, mask,
                        tol=geometry_tol,
                        clash_lambda=clash_lambda,
                        clash_threshold=clash_threshold,
                        clash_max_pairs=clash_max_pairs,
                        clash_pair_chunk=clash_pair_chunk,
                    )
                    snr_w = loss_weights_t.view(-1).detach()
                    ca_loss = (ca_loss_per * snr_w).sum() / (snr_w.sum() + 1e-8)

                    bond_loss_val = torch.tensor(0.0, device=device)
                    if coord_channels == 12:
                        pB, pT, pL, _ = pred_time_abs.shape
                        pred_bb_abs = pred_time_abs[..., :12].view(pB, pT, pL, 4, 3)
                        bond_loss_val = backbone_bond_loss(pred_bb_abs, mask)

                    geometry_loss = geometry_loss + eff_geometry_lambda * (ca_loss + bond_loss_val)
                    if shake_residual_loss is not None:
                        geometry_loss = geometry_loss + eff_geometry_lambda * shake_residual_loss

        if is_manifold_domain and (
            (eff_geometry_lambda > 0.0 and use_idct_geo)
            or track_caca_debug
            or eff_rmsf_lambda > 0.0
            or eff_low_freq_lambda > 0.0
        ):
            if x_0_pred is None:
                x_0_pred, _ = diffusion.extract_x0_eps_from_prediction(
                    model_out, x_t, t, prediction_target
                )

            pred_time_abs = inner_model.decode_latents(
                x_0_pred, native_coords, mask=mask, window_size=T_frames
            )
            target_time_abs = inner_model.decode_latents(
                x_0, native_coords, mask=mask, window_size=T_frames
            )

            pred_channels = pred_time_abs.shape[-1]
            ca_start = int(getattr(inner_model, "ca_start", 3 if pred_channels == 12 else 0))
            pred_ca = pred_time_abs[..., ca_start:ca_start + 3]
            target_ca = target_time_abs[..., ca_start:ca_start + 3]

            if track_caca_debug:
                caca_debug_metrics = collect_temp_caca_debug_metrics(
                    raw_temps,
                    pred_ca,
                    target_ca,
                    mask,
                    clash_threshold=clash_threshold,
                    clash_max_pairs=clash_max_pairs,
                    clash_pair_chunk=clash_pair_chunk,
                )

            if eff_geometry_lambda > 0.0 and use_idct_geo:
                ca_loss_per = local_geometry_loss(
                    pred_ca, target_ca, mask,
                    tol=geometry_tol,
                    clash_lambda=clash_lambda,
                    clash_threshold=clash_threshold,
                    clash_max_pairs=clash_max_pairs,
                    clash_pair_chunk=clash_pair_chunk,
                )
                snr_w = loss_weights_t.view(-1).detach()
                ca_loss = (ca_loss_per * snr_w).sum() / (snr_w.sum() + 1e-8)
                bond_loss_val = torch.tensor(0.0, device=device)
                if pred_channels == 12:
                    pred_bb_abs = pred_time_abs[..., :12].view(B_td, T_frames, L_td, 4, 3)
                    bond_loss_val = backbone_bond_loss(pred_bb_abs, mask)
                geometry_loss = geometry_loss + eff_geometry_lambda * (ca_loss + bond_loss_val)

            if eff_low_freq_lambda > 0.0:
                pred_res = pred_time_abs - native_coords.unsqueeze(1)
                target_res = target_time_abs - native_coords.unsqueeze(1)
                pred_res_flat = pred_res.permute(0, 2, 1, 3).reshape(B_td, L_td, T_frames * C_td)
                target_res_flat = target_res.permute(0, 2, 1, 3).reshape(B_td, L_td, T_frames * C_td)
                low_freq_loss = eff_low_freq_lambda * time_domain_low_k_loss(
                    pred_res_flat,
                    target_res_flat,
                    mask,
                    window_size=T_frames,
                    coord_channels=C_td,
                    n_modes=low_freq_modes,
                )

            if eff_rmsf_lambda > 0.0:
                pred_res = pred_time_abs - native_coords.unsqueeze(1)
                target_res = target_time_abs - native_coords.unsqueeze(1)
                pred_res_flat = pred_res.permute(0, 2, 1, 3).reshape(B_td, L_td, T_frames * C_td)
                target_res_flat = target_res.permute(0, 2, 1, 3).reshape(B_td, L_td, T_frames * C_td)
                spectral_amp_loss = eff_rmsf_lambda * time_domain_dct_coeff_l1_loss(
                    pred_res_flat,
                    target_res_flat,
                    mask,
                    window_size=T_frames,
                    coord_channels=C_td,
                )

        if is_time_domain and ((eff_geometry_lambda > 0.0 and use_idct_geo) or track_caca_debug):
            if x_0_pred is None:
                x_0_pred, _ = diffusion.extract_x0_eps_from_prediction(
                    model_out, x_t, t, prediction_target
                )

            pred_time = x_0_pred.view(B_td, L_td, T_frames, C_td).permute(0, 2, 1, 3)
            target_time = x_0.view(B_td, L_td, T_frames, C_td).permute(0, 2, 1, 3)
            pred_time_abs = pred_time * real_model.model.coord_scale
            target_time_abs = target_time * real_model.model.coord_scale
            if displacement:
                pred_time_abs = pred_time_abs + native_coords.unsqueeze(1)
                target_time_abs = target_time_abs + native_coords.unsqueeze(1)

            ca_start = 3 if coord_channels == 12 else 0
            pred_ca = pred_time_abs[..., ca_start:ca_start + 3]
            target_ca = target_time_abs[..., ca_start:ca_start + 3]

            if track_caca_debug:
                caca_debug_metrics = collect_temp_caca_debug_metrics(
                    raw_temps,
                    pred_ca,
                    target_ca,
                    mask,
                    clash_threshold=clash_threshold,
                    clash_max_pairs=clash_max_pairs,
                    clash_pair_chunk=clash_pair_chunk,
                )

            if eff_geometry_lambda > 0.0 and use_idct_geo:
                ca_loss_per = local_geometry_loss(
                    pred_ca, target_ca, mask,
                    tol=geometry_tol,
                    clash_lambda=clash_lambda,
                    clash_threshold=clash_threshold,
                    clash_max_pairs=clash_max_pairs,
                    clash_pair_chunk=clash_pair_chunk,
                )
                snr_w = loss_weights_t.view(-1).detach()
                ca_loss = (ca_loss_per * snr_w).sum() / (snr_w.sum() + 1e-8)
                bond_loss_val = torch.tensor(0.0, device=device)
                if coord_channels == 12:
                    pred_bb_abs = pred_time_abs[..., :12].view(B_td, T_frames, L_td, 4, 3)
                    bond_loss_val = backbone_bond_loss(pred_bb_abs, mask)
                geometry_loss = geometry_loss + eff_geometry_lambda * (ca_loss + bond_loss_val)

        if is_time_domain and eff_low_freq_lambda > 0.0:
            if x_0_pred is None:
                x_0_pred, _ = diffusion.extract_x0_eps_from_prediction(
                    model_out, x_t, t, prediction_target
                )
            low_freq_loss = eff_low_freq_lambda * time_domain_low_k_loss(
                x_0_pred,
                x_0,
                mask,
                window_size=T_frames,
                coord_channels=C_td,
                n_modes=low_freq_modes,
            )

        if is_dual_branch and not is_time_domain:
            # Three-term v11 objective: per-branch MSE dominates; small
            # concat-consistency term on the full spectrum.
            main_loss = (
                loss_slow_weight * loss_slow
                + loss_fast_weight * loss_fast
                + loss_total_weight * loss_mse
            )
        else:
            main_loss = loss_mse

        total_loss = (
            main_loss
            + bending_loss
            + geometry_loss
            + spectral_amp_loss
            + low_freq_loss
            + dc_loss
            + v17_loss
            + v12e_residual_loss
        )
        v12e_delta = aux_model_out.get("v12e_spec_graph_delta")
        if v12e_delta is not None:
            if float(v12e_spec_graph_residual_lambda) > 0.0:
                v12e_residual_loss = (
                    float(v12e_spec_graph_residual_lambda) * v12e_delta.square().mean()
                )
                total_loss = total_loss + v12e_residual_loss
            with torch.no_grad():
                v12e_delta_rms = torch.sqrt(v12e_delta.detach().square().mean()).item()
                v12e_delta_max = v12e_delta.detach().abs().max().item()
        train_band_metrics = {}
        if not is_time_domain and not is_manifold_domain and prediction_target == "x_0":
            with torch.no_grad():
                band_edges = _model_band_edges(inner_model, top_k_freqs)
                band_diag = spectral_band_diagnostics(
                    model_out.detach(),
                    target.detach(),
                    base_loss_mask.detach(),
                    top_k_freqs=top_k_freqs,
                    channels_per_mode=target.shape[-1] // top_k_freqs,
                    band_edges=band_edges,
                    amplitude_channels=repr_coord_channels,
                    include_all_groups=False,
                )
                train_band_metrics = {
                    f"train/{key}": value for key, value in band_diag.items()
                }
        if hasattr(inner_model, "refine_ca"):
            # v12c calls the CA refiner after inverse-DCT inside train_step,
            # outside the DDP-wrapped forward. Keep those parameters present in
            # the autograd graph from step 0 so DDP static_graph does not see a
            # changing used-parameter set when geometry warmup activates.
            refiner_module = getattr(inner_model, "refiner", None)
            if refiner_module is not None:
                total_loss = total_loss + sum((p.sum() * 0.0) for p in refiner_module.parameters())

    # log for wandb
    metrics = {
        "train/mse": loss_mse.item(),
        "train/loss_slow": loss_slow.item(),
        "train/loss_fast": loss_fast.item(),
        "train/bending_loss": bending_loss.item(),
        "train/geometry_loss": geometry_loss.item(),
        "train/spectral_amp_loss": spectral_amp_loss.item(),
        "train/low_freq_loss": low_freq_loss.item(),
        "train/dc_loss": dc_loss.item(),
        "train/v17_loss": v17_loss.item(),
        "train/v17_low_mode_spectral_mse": v17_low_mode_loss.item(),
        "train/v17_adjacent_spectral_mse": v17_adjacent_loss.item(),
        "train/v17_idct_bond_vector_mse": v17_idct_bond_loss.item(),
        "train/v17_caca_tolerance_loss": v17_caca_tol_loss.item(),
        "train/v12e_spec_graph_residual_loss": v12e_residual_loss.item(),
        "train/geometry_frac": geometry_loss.item() / (loss_mse.item() + 1e-8),
        "debug/dc_pred": dc_pred,
        "debug/dc_gt": dc_gt,
        "debug/dc_error": (dc_pred - dc_gt) if not (math.isnan(dc_pred) or math.isnan(dc_gt)) else float("nan"),
        "debug/dc_final_mse": dc_final_mse,
        "debug/dc_final_abs_error": dc_final_abs_error,
        "debug/dc_head_mean": dc_head_mean,
        "debug/amp_log_gain_mean": amp_log_gain_mean,
        "debug/amp_log_gain_min": amp_log_gain_min,
        "debug/amp_log_gain_max": amp_log_gain_max,
        "debug/amp_gain_max": amp_gain_max,
        "debug/v12e_spec_graph_delta_rms": v12e_delta_rms,
        "debug/v12e_spec_graph_delta_max": v12e_delta_max,
        "debug/refiner_delta_rms_A": refiner_delta_rms,
        "debug/shake_delta_rms_A": shake_delta_rms,
        "debug/pred_std": pred_std,
        "debug/target_std": target_std,
        "debug/pred_norm": pred_norm,
        "debug/target_norm": target_norm,
        "debug/pred_mean": pred_mean,
        "debug/target_mean": target_mean,
        "debug/cos_sim": cos_sim,
    }
    metrics.update(train_band_metrics)
    if hasattr(inner_model, "gate_diagnostics"):
        metrics.update(inner_model.gate_diagnostics())
    metrics.update(caca_debug_metrics)

    return total_loss, metrics


@torch.no_grad()
def validate(
    model, diffusion, loader, transform_engine, device, is_distributed, dtype_ctx,
    top_k_freqs=64, window_size=256, guidance_scale=1.0,
    max_val_batches=10, num_ode_steps=20, freq_weighting=None, displacement=True, representation=None,
    bending_lambda=0.0,
    geo_loss="idct_ca-ca",
    geometry_lambda=0.0, geometry_warmup_start=50, geometry_warmup_epochs=10,
    geometry_decay_start=200, geometry_decay_epochs=200,
    geometry_tol=0.05, clash_lambda=0.0, clash_threshold=3.5,
    clash_max_pairs=4096, clash_pair_chunk=512,
    topology_margin_artifact=None, spectral_geo_segment_threshold=1.0,
    spectral_geo_max_segment_pairs=1024, risk_band_max_pairs=2048,
    risk_band_max_segment_pairs=512,
    representation_barrier_lambda=0.0,
    rmsf_lambda=0.0, rmsf_warmup_start=100, rmsf_warmup_epochs=10,
    low_freq_lambda=0.0, low_freq_modes=8,
    dc_lambda=0.0, dc_start_epoch=10,
    v17_aux_modes=17,
    v17_low_mode_lambda=0.0,
    v17_adjacent_lambda=0.0,
    v17_idct_bond_lambda=0.0,
    v17_caca_tolerance_lambda=0.0,
    v17_caca_target=3.84,
    v17_caca_tolerance=0.05,
    v12e_spec_graph_residual_lambda=0.0,
    loss_slow_weight=1.0, loss_fast_weight=1.0, loss_total_weight=0.1,
    epoch=0,
):
    '''
    Run full inference on small subset of the validation set.

    Reports LDDT, CA-CA distance, Spearman RMSF, spectral volume MSE,
    and JSD Ramachandran distributions.
    '''
    model.eval()
    real_model = model.module if hasattr(model, "module") else model
    is_dct = getattr(real_model, "is_dct", True)
    is_time_domain = getattr(real_model, "is_time_domain", False)
    is_manifold_domain = getattr(real_model, "is_manifold_domain", False)

    rama_val = RamaValidator(bins=64, device=device)

    # Accumulators
    # [lddt_sum, lddt_frames, caca_sum, caca_count, spec_mse_sum, spec_mse_count, 1step_mse_sum, 1step_mse_count]
    stats = torch.zeros(8, device=device)
    all_rmsf_pred = []
    all_rmsf_gt = []
    temp_caca_acc = init_temp_caca_accumulator(device=device)
    temp_clash_acc = init_temp_clash_accumulator(device=device)
    dc_pred_acc = 0.0
    dc_gt_acc   = 0.0
    dc_amp_pred_acc = 0.0
    dc_amp_gt_acc   = 0.0
    dc_n        = 0
    band_metric_sums: dict[str, float] = {}
    band_metric_counts: dict[str, int] = {}
    step_band_metric_sums: dict[str, float] = {}
    step_band_metric_counts: dict[str, int] = {}
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_val_batches:
            break
        if batch is None or (isinstance(batch, dict) and batch["coords"].shape[0] == 0):
            continue

        # 1-Step True Validation Loss - best_model for cpkt saving is determined from this
        # ---------------------------
        # We run this BEFORE full inference, using your exact train_step logic
        with dtype_ctx:
            _, step_metrics = train_step(
                model, diffusion, transform_engine, batch, device, dtype_ctx,
                top_k_freqs=top_k_freqs,
                cond_drop_prob=0.0, # Force 0 dropout for clean validation
                displacement=displacement,
                representation=representation,
                freq_weighting=freq_weighting,
                bending_lambda=bending_lambda,
                geo_loss=geo_loss,
                geometry_lambda=geometry_lambda,
                representation_barrier_lambda=representation_barrier_lambda,
                geometry_warmup_start=geometry_warmup_start,
                geometry_warmup_epochs=geometry_warmup_epochs,
                geometry_decay_start=geometry_decay_start,
                geometry_decay_epochs=geometry_decay_epochs,
                geometry_tol=geometry_tol,
                clash_lambda=clash_lambda,
                clash_threshold=clash_threshold,
                clash_max_pairs=clash_max_pairs,
                clash_pair_chunk=clash_pair_chunk,
                topology_margin_artifact=topology_margin_artifact,
                spectral_geo_segment_threshold=spectral_geo_segment_threshold,
                spectral_geo_max_segment_pairs=spectral_geo_max_segment_pairs,
                risk_band_max_pairs=risk_band_max_pairs,
                risk_band_max_segment_pairs=risk_band_max_segment_pairs,
                rmsf_lambda=rmsf_lambda,
                rmsf_warmup_start=rmsf_warmup_start,
                rmsf_warmup_epochs=rmsf_warmup_epochs,
                low_freq_lambda=low_freq_lambda,
                low_freq_modes=low_freq_modes,
                dc_lambda=dc_lambda,
                dc_start_epoch=dc_start_epoch,
                v17_aux_modes=v17_aux_modes,
                v17_low_mode_lambda=v17_low_mode_lambda,
                v17_adjacent_lambda=v17_adjacent_lambda,
                v17_idct_bond_lambda=v17_idct_bond_lambda,
                v17_caca_tolerance_lambda=v17_caca_tolerance_lambda,
                v17_caca_target=v17_caca_target,
                v17_caca_tolerance=v17_caca_tolerance,
                v12e_spec_graph_residual_lambda=v12e_spec_graph_residual_lambda,
                loss_slow_weight=loss_slow_weight,
                loss_fast_weight=loss_fast_weight,
                loss_total_weight=loss_total_weight,
                epoch=epoch,
                is_validation=True,
                is_main_process=False,
                global_step=0
            )
        stats[6] += step_metrics["train/mse"]  # pure spectral MSE, no auxiliary losses
        stats[7] += 1
        for key, value in step_metrics.items():
            if key.startswith("train/spectral/") and isinstance(value, (int, float)) and math.isfinite(float(value)):
                out_key = key.replace("train/", "1step_", 1)
                step_band_metric_sums[out_key] = step_band_metric_sums.get(out_key, 0.0) + float(value)
                step_band_metric_counts[out_key] = step_band_metric_counts.get(out_key, 0) + 1
        # ----------------------------

        gt_coords = batch["coords"].to(device)  # (B, T, L, 3)
        native_coords = batch["native_coords"].to(device)
        mask = batch["mask"].to(device)
        temp = batch["temp"].to(device)
        win_pos = batch.get("win_pos", None)
        if win_pos is not None:
            win_pos = win_pos.float().to(device)

        gt_angles = batch.get("angles", None)
        native_angles = batch.get("native_angles", None)
        torsion_mask = batch.get("torsion_mask", None)

        if gt_angles is not None:
            gt_angles = gt_angles.to(device)
        if native_angles is not None:
            native_angles = native_angles.to(device)
        if torsion_mask is not None:
            torsion_mask = torsion_mask.to(device)
            if torsion_mask.dim() == 2:
                angle_channels = gt_angles.shape[-1]
                torsion_mask = torsion_mask.unsqueeze(-1).expand(-1, -1, angle_channels)

        coord_channels = gt_coords.shape[-1]
        representation = representation or CoordinateRepresentation(
            displacement=displacement,
            coord_channels=coord_channels,
        )
        repr_coord_channels = representation.model_coord_channels
        angle_channels = gt_angles.shape[-1] if gt_angles is not None else 0
        channels = repr_coord_channels + angle_channels
        B, T_frames, L, _ = gt_coords.shape
        complex_mult = 1 if is_dct else 2
        if is_manifold_domain:
            D = window_size * int(getattr(getattr(real_model, "model", real_model), "latent_dim", coord_channels))
        elif is_time_domain:
            D = window_size * coord_channels  # FNO diffuses on (B, L, T*C)
        else:
            D = channels * top_k_freqs * complex_mult
        shape = (B, L, D)

        # Pipeline sanity check on the very first validation batch
        is_main_process = (not is_distributed) or (dist.get_rank() == 0)
        # if is_main_process and batch_idx == 0:
        #     debug_print_pipeline(
        #         batch, transform_engine, real_model, device,
        #         top_k_freqs=top_k_freqs, displacement=displacement,
        #         label="VALIDATE batch=0",
        #     )

        # Optional NMA RMSF prior (present when a sidecar is loaded).
        rmsf_prior_val = batch.get("rmsf_prior", None)
        if rmsf_prior_val is not None:
            rmsf_prior_val = rmsf_prior_val.to(device)
        res_type = batch.get("res_type", None)
        if res_type is not None:
            res_type = res_type.to(device)
        dssp = batch.get("dssp", None)
        if dssp is not None:
            dssp = dssp.to(device)
        dc_baseline_per_res = batch.get("dc_baseline_per_res", None)
        if dc_baseline_per_res is not None:
            dc_baseline_per_res = dc_baseline_per_res.to(device)

        # Inference
        with dtype_ctx:
            pred = run_inference(
                model, diffusion, transform_engine, shape,
                native_coords, native_angles, temp, T_frames,
                mask=mask, torsion_mask=torsion_mask,
                device=device,
                guidance_scale=guidance_scale,
                num_ode_steps=num_ode_steps,
                displacement=displacement,
                representation=representation,
                win_pos=win_pos,
                rmsf_prior=rmsf_prior_val,
                res_type=res_type,
                dssp=dssp,
                dc_baseline_per_res=dc_baseline_per_res,
            )
        pred_coords = pred["coords"]  # (B, T, L, C)
        pred_ca = pred_coords[..., :3] if coord_channels != 12 else pred_coords[..., 3:6]
        gt_ca = gt_coords[..., :3] if coord_channels != 12 else gt_coords[..., 3:6]
        update_temp_caca_accumulator(temp_caca_acc, temp, pred_ca, gt_ca, mask)
        update_temp_clash_accumulator(
            temp_clash_acc,
            temp,
            pred_ca,
            gt_ca,
            mask,
            threshold=clash_threshold,
            pair_chunk=max(clash_pair_chunk, 2048),
        )

        # 1. Spectral Volume MSE (spectral models only)
        if not is_time_domain and not is_manifold_domain:
            pred_spectral = pred["spectral"]  # (B, L, D)
            if gt_angles is not None:
                gt_repr = representation.forward(gt_coords, native_coords, mask=mask)
                gt_input = torch.cat([gt_repr, gt_angles], dim=-1)
            else:
                gt_input = representation.forward(gt_coords, native_coords, mask=mask)
            gt_input_masked = gt_input * mask.unsqueeze(1).unsqueeze(-1)
            gt_spectral = transform_engine.time_to_spectral(gt_input_masked, top_k=top_k_freqs)
            spec_mask = build_spectral_mask(
                mask, torsion_mask, top_k_freqs, is_dct,
                coord_channels=repr_coord_channels,
                representation=representation,
            )
            spec_diff = (pred_spectral - gt_spectral) ** 2 * spec_mask
            spec_mse = spec_diff.sum() / (spec_mask.sum() + 1e-8)
            stats[4] += spec_mse.item()
            stats[5] += 1

            band_edges = _model_band_edges(getattr(real_model, "model", real_model), top_k_freqs)
            band_metrics = spectral_band_diagnostics(
                pred_spectral,
                gt_spectral,
                spec_mask,
                top_k_freqs=top_k_freqs,
                channels_per_mode=channels * complex_mult,
                band_edges=band_edges,
                amplitude_channels=repr_coord_channels,
                include_all_groups=True,
            )
            for key, value in band_metrics.items():
                if math.isfinite(float(value)):
                    band_metric_sums[key] = band_metric_sums.get(key, 0.0) + float(value)
                    band_metric_counts[key] = band_metric_counts.get(key, 0) + 1

            # DC component diagnostic: k=0 slot, first coord_channels elements
            valid = mask.bool()
            if valid.any():
                dc_pred_vals = pred_spectral[:, :, :repr_coord_channels][valid]
                dc_gt_vals   = gt_spectral[:, :, :repr_coord_channels][valid]
                dc_pred_mean = dc_pred_vals.mean().item()
                dc_gt_mean   = dc_gt_vals.mean().item()
                dc_amp_pred  = dc_pred_vals.abs().mean().item()
                dc_amp_gt    = dc_gt_vals.abs().mean().item()
                if math.isfinite(dc_pred_mean) and math.isfinite(dc_gt_mean):
                    dc_pred_acc     += dc_pred_mean
                    dc_gt_acc       += dc_gt_mean
                    dc_amp_pred_acc += dc_amp_pred
                    dc_amp_gt_acc   += dc_amp_gt
                    dc_n            += 1

        # 2. CA-CA Distance
        caca = compute_batch_caca_dist(pred_coords, mask)
        stats[2] += caca.item()
        stats[3] += 1

        # 3. LDDT
        max_eval_frames = 20
        stride = max(1, T_frames // max_eval_frames)
        p_eval = pred_coords[:, ::stride]
        t_eval = gt_coords[:, ::stride]
        lddt = compute_batch_lddt(p_eval, t_eval, mask)
        frames_evaluated = p_eval.size(1) * B
        stats[0] += lddt.item() * frames_evaluated
        stats[1] += frames_evaluated

        # 4. RMSF Spearman 
        rmsf_p = compute_rmsf(pred_coords, mask)
        rmsf_g = compute_rmsf(gt_coords, mask)
        all_rmsf_pred.append(rmsf_p.cpu())
        all_rmsf_gt.append(rmsf_g.cpu())

        # 5. Ramachandran JSD
        if gt_angles is not None and torsion_mask is not None:
            pred_angles = pred["angles"]
            phi_pred = torch.rad2deg(torch.atan2(pred_angles[..., 0], pred_angles[..., 1]))
            psi_pred = torch.rad2deg(torch.atan2(pred_angles[..., 2], pred_angles[..., 3]))
            phi_gt = torch.rad2deg(torch.atan2(gt_angles[..., 0], gt_angles[..., 1]))
            psi_gt = torch.rad2deg(torch.atan2(gt_angles[..., 2], gt_angles[..., 3]))

            valid_mask = torsion_mask[..., 0].bool().unsqueeze(1).expand(-1, T_frames, -1)
            rama_val.update(
                phi_pred[valid_mask].flatten(),
                psi_pred[valid_mask].flatten(),
                phi_gt[valid_mask].flatten(),
                psi_gt[valid_mask].flatten(),
            )

        # clear up inference intermediates between batches
        to_del = [pred, pred_coords, gt_coords]
        if not is_time_domain and not is_manifold_domain:
            to_del += [pred_spectral, gt_spectral]
        del to_del
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    # Aggregation
    if is_distributed:
        dist.barrier()
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    rama_metrics = rama_val.compute(is_distributed=is_distributed) if rama_val.hist_gt.sum() > 0 else {}

    # Global Spearman
    if all_rmsf_pred:
        # Concat local batch lists into single local 1-D tensors
        local_pred = torch.cat(all_rmsf_pred).to(device)
        local_gt = torch.cat(all_rmsf_gt).to(device)
    else:
        local_pred = torch.tensor([], device=device)
        local_gt = torch.tensor([], device=device)

    if is_distributed:
        # Gather the lengths of the arrays from all GPUs
        world_size = dist.get_world_size()
        local_size = torch.tensor([local_pred.size(0)], dtype=torch.long, device=device)
        size_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        
        dist.all_gather(size_list, local_size)
        max_size = max([s.item() for s in size_list])

        # Pad local tensors to max_size so all_gather doesn't crash
        pad_len = max_size - local_size.item()
        padded_pred = torch.nn.functional.pad(local_pred, (0, pad_len))
        padded_gt = torch.nn.functional.pad(local_gt, (0, pad_len))

        # Allocate buffers and execute global gather
        pred_gather = [torch.zeros(max_size, dtype=padded_pred.dtype, device=device) for _ in range(world_size)]
        gt_gather = [torch.zeros(max_size, dtype=padded_gt.dtype, device=device) for _ in range(world_size)]

        dist.all_gather(pred_gather, padded_pred)
        dist.all_gather(gt_gather, padded_gt)

        # Un-pad and concatenate into the final global 1-D tensors
        global_rmsf_pred = torch.cat([p[:s.item()] for p, s in zip(pred_gather, size_list)])
        global_rmsf_gt = torch.cat([g[:s.item()] for g, s in zip(gt_gather, size_list)])
    else:
        global_rmsf_pred = local_pred
        global_rmsf_gt = local_gt

    spearman_rmsf = spearman_corr_pytorch(global_rmsf_gt, global_rmsf_pred)
    pearson_rmsf  = pearson_corr_pytorch(global_rmsf_gt, global_rmsf_pred)

    results = {
        "val/lddt": (stats[0] / (stats[1] + 1e-8)).item(),
        "val/caca_dist_A": (stats[2] / (stats[3] + 1e-8)).item(),
        "val/1_step_mse": (stats[6] / (stats[7] + 1e-8)).item(),
        "val/rmsf_spearman": spearman_rmsf,
        "val/rmsf_pearson": pearson_rmsf,
    }
    results.update(finalize_temp_caca_accumulator(temp_caca_acc, is_distributed=is_distributed))
    results.update(finalize_temp_clash_accumulator(temp_clash_acc, is_distributed=is_distributed))
    if not is_time_domain and not is_manifold_domain:
        results["val/spectral_mse"] = (stats[4] / (stats[5] + 1e-8)).item()
        for key, value_sum in sorted(step_band_metric_sums.items()):
            count = max(1, step_band_metric_counts.get(key, 0))
            results[f"val/{key}"] = value_sum / count
        for key, value_sum in sorted(band_metric_sums.items()):
            count = max(1, band_metric_counts.get(key, 0))
            results[f"val/{key}"] = value_sum / count
        if dc_n > 0:
            results["val/dc_pred_mean"]      = dc_pred_acc / dc_n
            results["val/dc_gt_mean"]        = dc_gt_acc   / dc_n
            results["val/dc_error"]          = (dc_pred_acc - dc_gt_acc) / dc_n
            results["val/dc_amp_pred_mean"]  = dc_amp_pred_acc / dc_n
            results["val/dc_amp_gt_mean"]    = dc_amp_gt_acc   / dc_n
            results["val/dc_amp_ratio"]      = (dc_amp_pred_acc / dc_n) / max(dc_amp_gt_acc / dc_n, 1e-8)

    # Rama JSD (global only, skip baseline keys)
    for k, v in rama_metrics.items():
        if "baseline" not in k and "beta" not in k and "alpha" not in k:
            results[f"val/{k}"] = v

    return results


def TRAIN(
    model, train_loader, val_loader, optimizer, diffusion, transform_engine,
    device="cuda", epochs=100, start_epoch=0, start_step=0,
    best_model_score=float("-inf"), coord_channels=3, angle_channels=0, checkpoint_dir="",
    mdcath_path="", sampler=None, scheduler=None, displacement=True, representation=None,
    window_size=256, top_k_freqs=64, freq_weighting=None,
    cond_drop_prob=0.15, trim_cache=False, crop_size=384,
    guidance_scale=1.0, num_ode_steps=20, max_val_batches=10,
    bending_lambda=0.0,
    geo_loss="idct_ca-ca",
    geometry_lambda=0.0, geometry_warmup_start=50, geometry_warmup_epochs=10,
    geometry_decay_start=200, geometry_decay_epochs=200,
    geometry_tol=0.05, clash_lambda=0.0, clash_threshold=3.5,
    clash_max_pairs=4096, clash_pair_chunk=512,
    topology_margin_artifact=None, spectral_geo_segment_threshold=1.0,
    spectral_geo_max_segment_pairs=1024, risk_band_max_pairs=2048,
    risk_band_max_segment_pairs=512,
    representation_barrier_lambda=0.0,
    rmsf_lambda=0.0, rmsf_warmup_start=100, rmsf_warmup_epochs=10,
    low_freq_lambda=0.0, low_freq_modes=8,
    dc_lambda=0.0, dc_start_epoch=10,
    v17_aux_modes=17,
    v17_low_mode_lambda=0.0,
    v17_adjacent_lambda=0.0,
    v17_idct_bond_lambda=0.0,
    v17_caca_tolerance_lambda=0.0,
    v17_caca_target=3.84,
    v17_caca_tolerance=0.05,
    v12e_spec_graph_residual_lambda=0.0,
    loss_slow_weight=1.0, loss_fast_weight=1.0, loss_total_weight=0.1,
    randomize_train_windows=True,
    max_bad_update_streak=25,
    max_bad_update_total=1000,
    debug_nonfinite_hooks=False,
    debug_nonfinite_forward=True,
    debug_nonfinite_backward=True,
    debug_nonfinite_filter="",
    debug_nonfinite_max_modules=0,
):
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    is_main_process = (not is_distributed) or (dist.get_rank() == 0)

    if is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Starting training on {device}. Checkpoints -> '{checkpoint_dir}'")

    if str(device).startswith("cuda") and torch.cuda.is_bf16_supported():
        if is_main_process:
            print(f"BFloat16 enabled on {torch.cuda.get_device_name(0)}.")
        dtype_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        dtype_ctx = contextlib.nullcontext()

    model.train()
    best_score = best_model_score
    global_step = start_step
    bad_update_streak = 0
    bad_update_total = 0
    nonfinite_monitor = NonFiniteMonitor(
        enabled=debug_nonfinite_hooks,
        check_forward=debug_nonfinite_forward,
        check_backward=debug_nonfinite_backward,
        name_filter=debug_nonfinite_filter,
        max_modules=debug_nonfinite_max_modules,
    )
    n_monitored_modules = nonfinite_monitor.register(model)
    if is_main_process and debug_nonfinite_hooks:
        filt_msg = debug_nonfinite_filter or "<all leaf modules>"
        print(
            f"Non-finite activation monitor enabled: modules={n_monitored_modules}, "
            f"forward={debug_nonfinite_forward}, backward={debug_nonfinite_backward}, "
            f"filter={filt_msg!r}",
            flush=True,
        )

    def emit_nonfinite_monitor(reason: str):
        payload = nonfinite_monitor.emit(rank=rank, prefix=f"[NonFiniteMonitor:{reason}]")
        if is_main_process and debug_nonfinite_hooks and not payload:
            print(
                f"[NonFiniteMonitor:{reason}] no non-finite forward activations or "
                "activation gradients captured on rank 0; source may be on another "
                "rank, inside an unhooked functional op, or in parameter-gradient/optimizer math.",
                flush=True,
            )
        if is_main_process and payload and wandb.run is not None:
            wandb.log(payload, step=global_step)
        return payload

    def record_skipped_update(kind, epoch, batch_idx, metrics=None, extra=None):
        nonlocal bad_update_streak, bad_update_total
        bad_update_streak += 1
        bad_update_total += 1
        extra = extra or {}
        metrics = metrics or {}

        if is_main_process:
            print(
                f"Skipped optimizer update at optimizer_step={global_step} "
                f"(epoch={epoch}, batch={batch_idx}, kind={kind}, "
                f"streak={bad_update_streak}, total={bad_update_total}).",
                flush=True,
            )
            log_payload = {
                "train/skipped_update": 1.0,
                f"train/skipped_{kind}": 1.0,
                "debug/bad_update_streak": float(bad_update_streak),
                "debug/bad_update_total": float(bad_update_total),
                "debug/optimizer_step": float(global_step),
                "epoch": epoch,
            }
            for key in (
                "train/mse",
                "train/bending_loss",
                "train/geometry_loss",
                "train/spectral_amp_loss",
                "train/low_freq_loss",
                "train/dc_loss",
                "debug/dc_pred",
                "debug/dc_gt",
                "debug/dc_error",
                "debug/pred_std",
                "debug/target_std",
                "debug/amp_log_gain_max",
                "debug/amp_gain_max",
            ):
                value = metrics.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    log_payload[f"skipped_context/{key.replace('/', '_')}"] = float(value)
            for key, value in extra.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    log_payload[f"skipped_context/{key}"] = float(value)
            if wandb.run is not None:
                wandb.log(log_payload, step=global_step)

        if (
            max_bad_update_streak is not None
            and max_bad_update_streak > 0
            and bad_update_streak >= max_bad_update_streak
        ):
            raise RuntimeError(
                f"Aborting after {bad_update_streak} consecutive skipped optimizer "
                f"updates at optimizer_step={global_step}. Last skip kind={kind}."
            )
        if (
            max_bad_update_total is not None
            and max_bad_update_total > 0
            and bad_update_total >= max_bad_update_total
        ):
            raise RuntimeError(
                f"Aborting after {bad_update_total} total skipped optimizer updates. "
                f"Last skip kind={kind}, optimizer_step={global_step}."
            )

    for epoch in range(start_epoch, epochs):
        set_randomize_windows(train_loader.dataset, randomize_train_windows)
        if is_distributed and sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        if is_main_process:
            print(f"\nEpoch {epoch} started. Batches: {len(train_loader)}")

        epoch_loss = 0.0
        num_batches = 0

        for i, batch in enumerate(train_loader):
            if is_main_process and global_step == 0:
                print("[step 0] fetched first training batch", flush=True)
            # Bad batch handling
            # 1. Local Check
            local_bad = torch.tensor([1 if (batch is None or batch["coords"].shape[0] == 0) else 0], device=device)
            # 2. Global Sync (DDP requirement: all ranks must do this or none)
            if is_distributed:
                dist.all_reduce(local_bad, op=dist.ReduceOp.MAX)

            is_bad_batch = local_bad.item() > 0
            if is_bad_batch:
                optimizer.zero_grad() # If ANY rank is bad, ALL ranks run a "dummy" zero-grad step to stay in sync
                safe_n = crop_size - 1
                batch = {
                    "coords": torch.zeros(1, window_size, safe_n, coord_channels, device=device),
                    "mask": torch.zeros(1, safe_n, device=device),
                    "temp": torch.tensor([300.0], device=device),
                    "native_coords": torch.zeros(1, safe_n, coord_channels, device=device),
                }
                if angle_channels > 0:
                    batch["angles"] = torch.zeros(1, window_size, safe_n, angle_channels, device=device)
                    batch["torsion_mask"] = torch.zeros(1, safe_n, angle_channels, device=device)
                    batch["native_angles"] = torch.zeros(1, safe_n, angle_channels, device=device)
                model.eval()

            optimizer.zero_grad()
            if debug_nonfinite_hooks:
                nonfinite_monitor.reset(step=global_step, epoch=epoch, batch_idx=i)

            if is_main_process and global_step == 0:
                print("[step 0] entering train_step forward/loss", flush=True)
            loss, metrics = train_step(
                model, diffusion, transform_engine, batch, device, dtype_ctx,
                top_k_freqs=top_k_freqs,
                cond_drop_prob=cond_drop_prob, displacement=displacement, representation=representation,
                freq_weighting=freq_weighting, bending_lambda=bending_lambda,
                geo_loss=geo_loss,
                geometry_lambda=geometry_lambda,
                geometry_warmup_start=geometry_warmup_start,
                geometry_warmup_epochs=geometry_warmup_epochs,
                geometry_decay_start=geometry_decay_start,
                geometry_decay_epochs=geometry_decay_epochs,
                geometry_tol=geometry_tol,
                clash_lambda=clash_lambda,
                clash_threshold=clash_threshold,
                clash_max_pairs=clash_max_pairs,
                clash_pair_chunk=clash_pair_chunk,
                topology_margin_artifact=topology_margin_artifact,
                spectral_geo_segment_threshold=spectral_geo_segment_threshold,
                spectral_geo_max_segment_pairs=spectral_geo_max_segment_pairs,
                risk_band_max_pairs=risk_band_max_pairs,
                risk_band_max_segment_pairs=risk_band_max_segment_pairs,
                representation_barrier_lambda=representation_barrier_lambda,
                rmsf_lambda=rmsf_lambda,
                rmsf_warmup_start=rmsf_warmup_start,
                rmsf_warmup_epochs=rmsf_warmup_epochs,
                low_freq_lambda=low_freq_lambda,
                low_freq_modes=low_freq_modes,
                dc_lambda=dc_lambda,
                dc_start_epoch=dc_start_epoch,
                v17_aux_modes=v17_aux_modes,
                v17_low_mode_lambda=v17_low_mode_lambda,
                v17_adjacent_lambda=v17_adjacent_lambda,
                v17_idct_bond_lambda=v17_idct_bond_lambda,
                v17_caca_tolerance_lambda=v17_caca_tolerance_lambda,
                v17_caca_target=v17_caca_target,
                v17_caca_tolerance=v17_caca_tolerance,
                v12e_spec_graph_residual_lambda=v12e_spec_graph_residual_lambda,
                loss_slow_weight=loss_slow_weight,
                loss_fast_weight=loss_fast_weight,
                loss_total_weight=loss_total_weight,
                epoch=epoch,
                is_validation=False, is_main_process=is_main_process, global_step=global_step
            )
            if is_main_process and global_step == 0:
                print("[step 0] train_step returned; checking loss", flush=True)
            
            # DDP-SAFE LOSS CHECK
            # --------------------------
            is_nan = torch.tensor(1 if not bool(torch.isfinite(loss).item()) else 0, device=device)
            if is_distributed:
                dist.all_reduce(is_nan, op=dist.ReduceOp.MAX)

            if is_nan.item() > 0:
                if is_main_process:
                    print(f"WARNING: NaN loss detected at step {global_step}. Skipping batch across ALL ranks.")
                if debug_nonfinite_hooks:
                    emit_nonfinite_monitor("bad_loss")
                optimizer.zero_grad() 
                record_skipped_update("bad_loss", epoch, i, metrics=metrics)
                continue 
            # --------------------------- 

            if is_bad_batch:
                loss = loss * 0.0

            if is_main_process and global_step == 0:
                print("[step 0] starting backward", flush=True)
            loss.backward()
            if is_main_process and global_step == 0:
                print("[step 0] backward finished; clipping gradients", flush=True)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if is_main_process and global_step == 0:
                print("[step 0] grad clip finished; checking gradients", flush=True)

            # GRADIENT SAFETY CHECK
            # --------------------------------
            is_grad_nan = torch.tensor(1 if (torch.isnan(grad_norm) or torch.isinf(grad_norm)) else 0, device=device)
            if is_distributed:
                dist.all_reduce(is_grad_nan, op=dist.ReduceOp.MAX)

            if is_grad_nan.item() > 0:
                if is_main_process:
                    print(f"WARNING: NaN/Inf gradients detected at step {global_step}. Skipping optimizer step.")
                    bad_grad_names = []
                    real_model_for_grad = model.module if hasattr(model, "module") else model
                    for name, param in real_model_for_grad.named_parameters():
                        if param.grad is None:
                            continue
                        finite = torch.isfinite(param.grad)
                        if not bool(finite.all()):
                            n_bad = int((~finite).sum().item())
                            total = int(param.grad.numel())
                            finite_abs = param.grad.detach().abs()[finite]
                            max_abs = float(finite_abs.max().item()) if finite_abs.numel() else float("nan")
                            bad_grad_names.append(f"{name} bad={n_bad}/{total} finite_max={max_abs:.3e}")
                            if len(bad_grad_names) >= 8:
                                break
                    if bad_grad_names:
                        print("Non-finite gradient tensors: " + " | ".join(bad_grad_names), flush=True)
                if debug_nonfinite_hooks:
                    emit_nonfinite_monitor("bad_grad")
                optimizer.zero_grad()
                record_skipped_update(
                    "bad_grad",
                    epoch,
                    i,
                    metrics=metrics,
                    extra={"grad_norm": float(grad_norm.detach().float().item()) if torch.isfinite(grad_norm.detach()).item() else float("nan")},
                )
                continue
            # ---------------------------------

            if is_main_process and global_step == 0:
                print("[step 0] optimizer step", flush=True)
            optimizer.step()
            if scheduler:
                scheduler.step()
            bad_update_streak = 0
            if is_main_process and global_step == 0:
                print("[step 0] optimizer/scheduler finished; reducing loss", flush=True)

            current_loss = loss.item()
            if is_distributed:
                loss_tensor = torch.tensor(current_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                current_loss = loss_tensor.item()

            epoch_loss += current_loss
            num_batches += 1

            if is_main_process:
                # Reach through DDP and UnifiedWrapper to expose the raw model
                _real = model.module if hasattr(model, "module") else model
                _inner = getattr(_real, "model", _real)
                extra_log = {}
                if hasattr(_inner, "rmsf_gate"):
                    extra_log["debug/rmsf_gate"] = _inner.rmsf_gate.detach().float().item()
                if global_step == 0:
                    print("[step 0] calling wandb.log", flush=True)
                wandb.log(
                    {
                        "train/total": current_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "debug/grad_norm": grad_norm.item(),
                        **metrics,
                        **extra_log,
                    },
                    step=global_step,
                )
                if global_step == 0:
                    print("[step 0] wandb.log returned", flush=True)

            if is_main_process and trim_cache and i % 100 == 0:
                maintain_cache_size(mdcath_path, max_size_gb=20)

            if is_bad_batch:
                model.train()

            global_step += 1

        # Validation
        # ----------
        set_randomize_windows(val_loader.dataset, False)
        val_results = validate(
            model, diffusion, val_loader, transform_engine, device,
            is_distributed, dtype_ctx, displacement=displacement,
            representation=representation,
            top_k_freqs=top_k_freqs,
            window_size=window_size, guidance_scale=guidance_scale,
            max_val_batches=max_val_batches, num_ode_steps=num_ode_steps,
            freq_weighting=freq_weighting, bending_lambda=bending_lambda,
            geo_loss=geo_loss,
            geometry_lambda=geometry_lambda,
            representation_barrier_lambda=representation_barrier_lambda,
            geometry_warmup_start=geometry_warmup_start,
            geometry_warmup_epochs=geometry_warmup_epochs,
            geometry_decay_start=geometry_decay_start,
            geometry_decay_epochs=geometry_decay_epochs,
            geometry_tol=geometry_tol,
            clash_lambda=clash_lambda,
            clash_threshold=clash_threshold,
            clash_max_pairs=clash_max_pairs,
            clash_pair_chunk=clash_pair_chunk,
            topology_margin_artifact=topology_margin_artifact,
            spectral_geo_segment_threshold=spectral_geo_segment_threshold,
            spectral_geo_max_segment_pairs=spectral_geo_max_segment_pairs,
            risk_band_max_pairs=risk_band_max_pairs,
            risk_band_max_segment_pairs=risk_band_max_segment_pairs,
            rmsf_lambda=rmsf_lambda,
            rmsf_warmup_start=rmsf_warmup_start,
            rmsf_warmup_epochs=rmsf_warmup_epochs,
            low_freq_lambda=low_freq_lambda,
            low_freq_modes=low_freq_modes,
            dc_lambda=dc_lambda,
            dc_start_epoch=dc_start_epoch,
            v17_aux_modes=v17_aux_modes,
            v17_low_mode_lambda=v17_low_mode_lambda,
            v17_adjacent_lambda=v17_adjacent_lambda,
            v17_idct_bond_lambda=v17_idct_bond_lambda,
            v17_caca_tolerance_lambda=v17_caca_tolerance_lambda,
            v17_caca_target=v17_caca_target,
            v17_caca_tolerance=v17_caca_tolerance,
            v12e_spec_graph_residual_lambda=v12e_spec_graph_residual_lambda,
            loss_slow_weight=loss_slow_weight,
            loss_fast_weight=loss_fast_weight,
            loss_total_weight=loss_total_weight,
            epoch=epoch,
        )
        set_randomize_windows(train_loader.dataset, randomize_train_windows)
        #global_val_loss = val_results["val/spectral_mse"]
        global_val_loss = val_results["val/1_step_mse"]
        balanced_val_score = (
            float(val_results.get("val/rmsf_spearman", float("nan")))
            + float(val_results.get("val/lddt", float("nan")))
        )
        val_results["val/balanced_score"] = balanced_val_score

        if is_main_process:
            print(f"\nEpoch {epoch} Validation:")
            print(f"  1-Step MSE:   {val_results['val/1_step_mse']:.4f}")
            if "val/spectral_mse" in val_results:
                print(f"  Spectral MSE: {val_results['val/spectral_mse']:.4f}")
            if "val/spectral/low_k_signed_mse" in val_results:
                print(
                    "  Low-k signed/amp/Pearson: "
                    f"{val_results['val/spectral/low_k_signed_mse']:.4f} / "
                    f"{val_results.get('val/spectral/low_k_amp_mae', float('nan')):.4f} / "
                    f"{val_results.get('val/spectral/low_k_signed_pearson', float('nan')):.4f}"
                )
            if "val/1step_spectral/low_k_signed_mse" in val_results:
                print(
                    "  1-step low-k signed/amp/Pearson: "
                    f"{val_results['val/1step_spectral/low_k_signed_mse']:.4f} / "
                    f"{val_results.get('val/1step_spectral/low_k_amp_mae', float('nan')):.4f} / "
                    f"{val_results.get('val/1step_spectral/low_k_signed_pearson', float('nan')):.4f}"
                )
            if "val/lddt" in val_results:
                print(f"  LDDT:         {val_results['val/lddt']:.4f}")
            if "val/caca_dist_A" in val_results:
                print(f"  CA-CA (Å):    {val_results['val/caca_dist_A']:.2f}")
            if "val/clash_450k_pred_count_per_traj_3p5" in val_results:
                print(
                    "  450K clashes: "
                    f"{val_results['val/clash_450k_pred_count_per_traj_3p5']:.2f}/traj "
                    f"(n={val_results.get('val/clash_450k_pred_n_traj', 0)})"
                )
            if "val/rmsf_spearman" in val_results:
                print(f"  RMSF Spearman:{val_results['val/rmsf_spearman']:.4f}")
            if math.isfinite(balanced_val_score):
                print(f"  Balanced Score (RMSF+LDDT): {balanced_val_score:.4f}")
            if "val/dc_gt_mean" in val_results:
                print(f"  DC gt/pred/err: {val_results['val/dc_gt_mean']:.4f} / {val_results['val/dc_pred_mean']:.4f} / {val_results['val/dc_error']:.4f}")
            jsd_key = "val/rama_global_JSD"
            if jsd_key in val_results:
                print(f"  Rama JSD:     {val_results[jsd_key]:.4f}")

            wandb.log({"epoch": epoch, **val_results})

            model_state = (
                model.module.state_dict()
                if isinstance(model, DDP)
                else model.state_dict()
            )

            resume_state = {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "global_val_loss": global_val_loss,
                "best_model_score": best_score,
                "wandb_id": wandb.run.id if wandb.run else None,
                "global_step": global_step,
            }
            torch.save(resume_state, os.path.join(checkpoint_dir, "checkpoint_latest.pt"))

            inference_state = {
                "epoch": epoch,
                "model_state_dict": model_state,
                "global_val_loss": global_val_loss,
                "balanced_val_score": balanced_val_score,
            }

            if epoch % 5 == 0:
                torch.save(inference_state, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))

            if math.isfinite(balanced_val_score) and balanced_val_score > best_score:
                print(f"New best balanced model: ({best_score:.4f} -> {balanced_val_score:.4f})")
                best_score = balanced_val_score
                torch.save(inference_state, os.path.join(checkpoint_dir, "best_model.pt"))

        model.train()

    return os.path.join(checkpoint_dir, "best_model.pt"), global_step, best_score


def TEST(
    model, diffusion, test_loader, transform_engine, device,
    checkpoint_path, is_distributed=True, displacement=True, representation=None,
    top_k_freqs=64, window_size=256, guidance_scale=1.0,
    num_ode_steps=20,
    geometry_tol=0.05, clash_lambda=0.0, clash_threshold=3.5,
    clash_max_pairs=4096, clash_pair_chunk=512,
):
    '''Run full inference on the test set. Same metrics as validation.'''
    log = lambda msg: print(msg) if (not is_distributed or dist.get_rank() == 0) else None

    log(f"\n--- Starting TEST on {device} ---")
    log(f"Loading weights from: {checkpoint_path}")

    load_model_weights(checkpoint_path, model, device=device, log_fn=log)

    if str(device).startswith("cuda") and torch.cuda.is_bf16_supported():
        dtype_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        dtype_ctx = contextlib.nullcontext()

    results = validate(
        model, diffusion, test_loader, transform_engine, device,
        is_distributed, dtype_ctx, displacement=displacement, representation=representation,
        top_k_freqs=top_k_freqs,
        window_size=window_size, guidance_scale=guidance_scale,
        num_ode_steps=num_ode_steps,
        geometry_tol=geometry_tol,
        clash_lambda=clash_lambda,
        clash_threshold=clash_threshold,
        clash_max_pairs=clash_max_pairs,
        clash_pair_chunk=clash_pair_chunk,
    )

    if (not is_distributed) or (dist.get_rank() == 0):
        log(f"\n[FINAL TEST METRICS]")
        for k, v in sorted(results.items()):
            log(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        wandb.log({k.replace("val/", "test/"): v for k, v in results.items()})

    return results


_LEGACY_LOW_K_RE = re.compile(r"\blow_k_correction_head\.(\d+)\.")


def _remap_legacy_state_dict_keys(
    state_dict: dict[str, "torch.Tensor"],
    target_state_dict: dict[str, "torch.Tensor"] | None = None,
) -> dict[str, "torch.Tensor"]:
    """Apply known checkpoint-schema renames, guarded by shape compatibility.

    - ``low_k_correction_head.{layer}`` -> ``low_k_correction_heads.0.{layer}``.
      The single-head ``nn.Sequential`` was refactored into a
      ``nn.ModuleList`` of heads; the first entry is the original single
      head.

    The rename is only applied when the target model actually has a key
    with the renamed path AND the shapes match. If the current model's
    low-k head has a different width (e.g. the user configured
    ``low_k_correction_modes: 4`` while the checkpoint was trained with
    width 1), we leave the source key untouched so ``strict=False``
    reports it as ``unexpected`` — the old, shape-safe behaviour.
    """
    remapped: dict[str, "torch.Tensor"] = {}
    for k, v in state_dict.items():
        match = _LEGACY_LOW_K_RE.search(k)
        if match is None:
            remapped[k] = v
            continue
        new_k = _LEGACY_LOW_K_RE.sub(r"low_k_correction_heads.0.\1.", k)
        shape_ok = (
            target_state_dict is None
            or (new_k in target_state_dict
                and tuple(target_state_dict[new_k].shape) == tuple(v.shape))
        )
        if shape_ok:
            remapped[new_k] = v
        else:
            # Preserve the legacy key name: strict=False will then drop it
            # into ``unexpected_keys`` without corrupting the target head.
            remapped[k] = v
    return remapped


def _drop_shape_incompatible_state_dict_keys(
    state_dict: dict[str, "torch.Tensor"],
    target_state_dict: dict[str, "torch.Tensor"],
    log_fn=print,
    prefix: str = "load",
) -> dict[str, "torch.Tensor"]:
    """Remove same-name checkpoint tensors whose shapes do not match target.

    ``strict=False`` ignores missing and unexpected keys, but PyTorch still
    raises for keys that exist in both state dicts with incompatible shapes.
    Warm-starts across small architecture/config changes should keep loading
    compatible trunk weights while leaving changed optional heads at init.
    """
    filtered: dict[str, "torch.Tensor"] = {}
    skipped: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for k, v in state_dict.items():
        target_v = target_state_dict.get(k)
        if (
            target_v is not None
            and torch.is_tensor(v)
            and torch.is_tensor(target_v)
            and tuple(v.shape) != tuple(target_v.shape)
        ):
            skipped.append((k, tuple(v.shape), tuple(target_v.shape)))
            continue
        filtered[k] = v

    if skipped:
        preview = ", ".join(
            f"{k}: ckpt{src_shape}->model{dst_shape}"
            for k, src_shape, dst_shape in skipped[:12]
        )
        more = "" if len(skipped) <= 12 else f", ... +{len(skipped) - 12} more"
        log_fn(f"  {prefix}: shape-mismatched keys skipped = {preview}{more}")
    return filtered


def load_model_weights(checkpoint_path, model, device="cpu", log_fn=print):
    """Load model weights only, without optimizer/scheduler state.

    Accepts either the training checkpoints written by this repo
    (``model_state_dict``) or a raw state-dict checkpoint (``state_dict`` or
    a plain tensor dict). Uses ``strict=False`` so target-space warm-starts
    (for example ``v -> x_0``) and additive optional parameters do not fail.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if isinstance(model, DDP):
        target = model.module
    else:
        target = model

    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    target_state_dict = target.state_dict()
    state_dict = _remap_legacy_state_dict_keys(state_dict, target_state_dict)
    state_dict = _drop_shape_incompatible_state_dict_keys(
        state_dict, target_state_dict, log_fn=log_fn, prefix="load"
    )

    result = target.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        log_fn(f"  load: missing keys (kept at init) = {result.missing_keys}")
    if result.unexpected_keys:
        log_fn(f"  load: unexpected keys (ignored)   = {result.unexpected_keys}")
    return result


class RobustCollate:
    '''
    A collate function that delegates to the featurizer's collate_fn.

    Wraps in None-filtering for bad sample handling.
    '''
    def __init__(self, featurizer):
        # Store the featurizer as a class attribute so the workers can access it
        self.featurizer = featurizer

    def __call__(self, batch):
        # Filter out bad samples
        batch = [b for b in batch if b is not None]
        
        # Handle empty batches
        if len(batch) == 0:
            return {"coords": torch.empty(0)}
            
        # Delegate to the featurizer's collate function
        return self.featurizer.collate_fn(batch)



def TRAIN_DISTRIBUTED(
    # Data
    epochs=50,
    batch_size=200,
    crop_size=384,
    max_domains=5320,
    window_size=256,
    samples_per_traj=1,
    randomize_train_windows=True,
    dataloader_num_workers=None,
    dataloader_timeout=None,
    dataloader_prefetch_factor=None,
    dataloader_persistent_workers=True,
    # Model
    model_type="dit",  # "dit", "spec_conv", "fno", "hno", "cno"
    top_k_freqs=64,
    include_angles=False,
    hidden_dim=1024,
    freq_hidden_size=4,
    hidden_size=None,
    spectral_modes=None,
    num_layers=12,
    num_heads=12,
    prediction_target="x_0",
    bridge_scaling="none",  # "none" | "unit_max" | "full" — cno3 only
    coord_scale=5.0,         # normalisation scale in Å — cno4 only
    # Optimizer
    optimizer="adamw",
    max_lr=1e-4,
    lr_schedule_steps=None,
    # Diffusion
    schedule="cosine",
    num_steps=200,
    shift_value=2.0,
    min_snr_gamma=None,
    # Data pipeline
    use_DCT=True,
    smoothing=None,
    smoothing_sigma=1.0,
    smooth_start_freq=64,
    shelf_value=0.1,
    freq_weighting=None,
    conditioning_dropout=False,
    cond_drop_prob=0.15,
    # Paths
    checkpoint_dir="./checkpoints",
    mdcath_path="./mdCATH/data",
    atlas_path=None,
    use_atlas=False,
    freq_scales_path=None,
    offline_mode=True,
    crop=True,
    trim_cache=False,
    log_dir="logs",
    # Workflow
    test_only=False,
    run_name="train_run",
    resume_from_latest=False,
    checkpoint_path=None,
    guidance_scale=1.0,
    num_ode_steps=20,
    max_val_batches=10,
    use_zarr=False,
    mdcath_zarr_path="./mdcath_zarr",
    atlas_zarr_path="./atlas_zarr",
    split_ids_dir=None,
    displacement=True,
    representation=None,
    freq_normalization="auto",
    dc_residualization="auto",
    aniso_source="auto",
    aniso_scales_path=None,
    noise_schedule=None,
    noise_space="raw_gamma",
    noise_band_edges=None,
    noise_group_model_multipliers=None,
    noise_target_crossings=None,
    noise_anchor_band=None,
    noise_power_normalization="raw_mean_square",
    noise_auto_shift=False,
    representation_length_min=3.5,
    representation_length_max=4.1,
    representation_length_residual_max=0.30,
    representation_barrier_lambda=0.0,
    bending_lambda=0.0,
    geo_loss="idct_ca-ca",
    geometry_lambda=0.0,
    geometry_warmup_start=50,
    geometry_warmup_epochs=10,
    geometry_decay_start=200,
    geometry_decay_epochs=200,
    geometry_tol=0.05,
    clash_lambda=0.0,
    clash_threshold=3.5,
    clash_max_pairs=4096,
    clash_pair_chunk=512,
    topology_margin_path=None,
    spectral_geo_segment_threshold=1.0,
    spectral_geo_max_segment_pairs=1024,
    risk_band_max_pairs=2048,
    risk_band_max_segment_pairs=512,
    rmsf_lambda=0.0,
    rmsf_warmup_start=100,
    rmsf_warmup_epochs=10,
    low_freq_lambda=0.0,
    low_freq_modes=8,
    dc_lambda=0.0,
    dc_start_epoch=10,
    coords_type="ca", # "ca" or "bb"
    atlas_stride=100, # atlas base is 10ps, 100 stride -> 1ns
    aniso_gamma=None, # None = isotropic; 0.3 = recommended gentle anisotropy
    use_hilbert_spatial=False, # spectral_conv_dit only: Hilbert spatial envelope in Phase 2 (FFT-based, legacy)
    use_hilbert_spatial_dct=False, # spectral_conv_dit only: DCT-based boundary-safe Hilbert envelope (preferred)
    hilbert_mode="every_block", # spectral_conv_dit only: every_block | every_3_blocks | input_only | off
    rmsf_prior_path=None, # optional sidecar .pt of per-domain NMA RMSF prior (from scripts/precompute_nma_tica.py)
    use_rmsf_prior_gain=False, # spectral_conv_dit only: apply per-residue NMA RMSF prior gain on the model output
    use_low_k_correction_head=False, # spectral_conv_dit only: additive low-k correction branch, zero-init for warm-start safety
    low_k_correction_modes=1, # number of lowest modes handled by the additive correction branch
    use_seq_conditioning=False, # spectral_dit / spectral_conv_dit only: use learned residue-type embeddings and pooled global sequence conditioning
    seq_embed_dim=16, # width of the learned residue-type embedding
    use_ss_conditioning=False, # spectral_dit / spectral_conv_dit only: use DSSP-based secondary-structure conditioning
    ss_embed_dim=8, # latent width for DSSP conditioning
    temporal_ablation_mode="normal", # fno2: normal | off | freq_noise
    block_ablation_mode="normal", # fno2: normal | no_mlp | temporal_only
    temporal_gate_init=0.0, # fno2: nonzero initial gate bias for physical DCT-FNO path
    # dual_branch (v11) specific hyperparameters
    K_slow=16, # dual_branch/v12b: number of modes handled by the slow branch
    slow_mode_start=0, # v12b: first spectral mode receiving the slow residual; 1 skips DC
    slow_d_model=192, # dual_branch: slow-branch hidden width
    slow_depth=6, # dual_branch: slow-branch transformer depth
    slow_num_heads=4, # dual_branch: slow-branch attention heads
    slow_mlp_ratio=4.0, # dual_branch: slow-branch SwiGLU expansion ratio
    slow_attn_dropout=0.0, # dual_branch: slow-branch attention dropout
    slow_use_rmsf_prior=False, # dual_branch: feed per-residue NMA RMSF into the slow branch
    slow_predicts_amplitude_only=False, # dual_branch: if True, slow branch predicts amplitudes and reconstructs vectors along noisy low-k directions
    fast_band_edges=None, # block_mix/v12: spectral band edges/scheme; pass "legacy" for pre-DC-split checkpoints
    fast_cond_dim=512, # dual_branch: fast-branch AdaLN conditioning width
    cascade_band_edges=None, # cascade: DC, low-k, then wider spectral groups; default (0,1,9,33,129,K)
    cascade_context_mode="idct_summary", # cascade: idct_summary | none
    cascade_detach_context=True, # cascade: stop later-stage losses backpropagating through previous stage context
    cascade_dc_depth=None, # cascade: optional DC transformer depth
    cascade_low_depth=None, # cascade: optional low-k spectral transformer depth
    cascade_high_depth=None, # cascade: optional spectral-conv stage depth
    loss_slow_weight=1.0, # dual_branch: weight on per-branch slow MSE in the three-term loss
    loss_fast_weight=1.0, # dual_branch: weight on per-branch fast MSE in the three-term loss
    loss_total_weight=0.1, # dual_branch: small concat-consistency weight on the full-spectrum MSE
    max_bad_update_streak=25, # fail loudly after consecutive NaN-loss/NaN-grad skipped optimizer updates
    max_bad_update_total=1000, # fail loudly after total skipped optimizer updates in one run
    amp_head_context_modes=4, # v12a: number of low-k modes supplied as context to the amplitude head
    amp_head_target_modes=1, # v12a: number of lowest modes amplitude-calibrated
    amp_head_d_model=128, # v12a: amplitude-head hidden width
    amp_head_depth=3, # v12a: amplitude-head transformer depth
    amp_head_num_heads=4, # v12a: amplitude-head attention heads
    amp_head_mlp_ratio=4.0, # v12a: amplitude-head SwiGLU ratio
    amp_head_attn_dropout=0.0, # v12a: amplitude-head attention dropout
    amp_head_use_rmsf_prior=False, # v12a: feed RMSF prior into the amplitude head
    use_shake=False, # v12a/v12c: apply differentiable SHAKE to reconstructed CA coords
    shake_n_iter=20, # v12a/v12c: SHAKE iterations (>=20 converges residual <1e-5 A)
    shake_target=3.8, # v12a/v12c: ideal CA-CA bond length in Angstroms
    refiner_hidden=32, # v12c: CA-coord refiner hidden width
    refiner_depth=2, # v12c: CA-coord refiner depth
    refiner_kernel_size=5, # v12c: CA-coord refiner conv kernel size
    refiner_max_delta=0.5, # v12c: cap per-coordinate refiner residual in Angstroms
    use_spectral_graph_refiner=True, # v17a: native-graph absolute low-mode spectral correction
    spectral_graph_refiner_modes=17,
    spectral_graph_refiner_hidden=128,
    spectral_graph_refiner_depth=3,
    spectral_graph_refiner_msg_hidden=None,
    spectral_graph_refiner_sequence_window=2,
    spectral_graph_refiner_knn=16,
    spectral_graph_refiner_use_sequence_edges=True,
    spectral_graph_refiner_use_native_knn=True,
    spectral_graph_refiner_max_delta=0.25,
    use_v12e_spec_graph=True, # v12e: enable sparse spectral-volume graph residual
    v12e_spec_graph_k_min=1, # v12e: first DCT mode refined, inclusive
    v12e_spec_graph_k_max=128, # v12e: last DCT mode refined, exclusive
    v12e_spec_graph_hidden=128,
    v12e_spec_graph_depth=3,
    v12e_spec_graph_msg_hidden=None,
    v12e_spec_graph_sequence_window=4,
    v12e_spec_graph_knn=8,
    v12e_spec_graph_use_sequence_edges=True,
    v12e_spec_graph_use_native_knn=True,
    v12e_spec_graph_max_delta=0.10,
    v12e_spec_graph_residual_lambda=0.0,
    use_bond_spectral_graph_refiner=True, # v17b: native-graph adjacent bond spectral correction
    bond_spectral_graph_refiner_modes=17,
    bond_spectral_graph_refiner_hidden=128,
    bond_spectral_graph_refiner_depth=3,
    bond_spectral_graph_refiner_msg_hidden=None,
    bond_spectral_graph_refiner_sequence_window=2,
    bond_spectral_graph_refiner_knn=16,
    bond_spectral_graph_refiner_use_sequence_edges=True,
    bond_spectral_graph_refiner_use_native_knn=True,
    bond_spectral_graph_refiner_max_delta=0.25,
    bond_spectral_graph_refiner_blend=1.0,
    v17_aux_modes=17, # v17 auxiliary losses: active only for model_type in {"v17a", "v17b"}
    v17_low_mode_lambda=0.0,
    v17_adjacent_lambda=0.0,
    v17_idct_bond_lambda=0.0,
    v17_caca_tolerance_lambda=0.0,
    v17_caca_target=3.84,
    v17_caca_tolerance=0.05,
    egnn_h_dim=32, # v12d: EGNN refiner node feature width
    egnn_hidden=64, # v12d: EGNN refiner edge MLP hidden width
    egnn_depth=3, # v12d: EGNN refiner number of layers
    egnn_seq_window=12, # v12d: EGNN edge sequence-window radius (|i-j| <= W)
    egnn_max_len=1024, # v12d: max residue index for the positional embedding
    egnn_t_chunk=64, # v12d: frames processed per refiner chunk (caps activation memory)
    jepa_latent_dim=128, # v14c: future-latent auxiliary width
    ddp_find_unused_parameters=None,
    ddp_static_graph=None,
    debug_nonfinite_hooks=False,
    debug_nonfinite_forward=True,
    debug_nonfinite_backward=True,
    debug_nonfinite_filter="",
    debug_nonfinite_max_modules=0,
):
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    per_gpu_batch_size = batch_size // world_size
    device = f"cuda:{local_rank}"

    if rank == 0:
        print(f"World size: {world_size}, batch: {batch_size}, per-GPU: {per_gpu_batch_size}")

    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    base_seed = 42
    seed_everything(base_seed + rank)

    # Featurizer
    coord_channels = 12 if coords_type == 'bb' else 3
    representation_name = canonical_representation(representation, displacement=displacement)
    representation_obj = CoordinateRepresentation(
        representation_name,
        coord_channels=coord_channels,
        length_min=representation_length_min,
        length_max=representation_length_max,
        length_residual_max=representation_length_residual_max,
    )
    representation_channels = representation_obj.model_coord_channels
    displacement = representation_obj.is_displacement
    if representation_obj.is_unit_chain and coords_type != "ca":
        raise ValueError(f"{representation_obj.name} requires coords_type='ca'")
    if representation_obj.is_unit_chain and use_rmsf_prior_gain:
        if rank == 0:
            print(
                "WARNING: Disabling use_rmsf_prior_gain for unit-chain representation. "
                "The RMSF prior gain is defined for Cartesian displacement spectra, "
                "not bond-direction/length spectral channels."
            )
        use_rmsf_prior_gain = False
    freq_normalization = canonical_freq_normalization(freq_normalization)
    dc_residualization = canonical_dc_residualization(dc_residualization)
    aniso_source = canonical_aniso_source(aniso_source)
    angle_channels = 4 if include_angles else 0
    if rank == 0:
        print(
            f"Coordinate representation: {representation_obj.name} "
            f"(raw_coord_channels={coord_channels}, model_coord_channels={representation_channels})"
        )
        print(f"Geometry auxiliary losses: {','.join(parse_geo_loss_modes(geo_loss)) or 'none'}")
        print(
            "Spectral policies: "
            f"freq_normalization={freq_normalization}, "
            f"dc_residualization={dc_residualization}, "
            f"aniso_source={aniso_source}"
        )
    topology_margin_artifact = None
    if topology_margin_path:
        topology_margin_artifact = load_topology_margin_artifact(topology_margin_path, map_location="cpu")
        if rank == 0:
            n_buckets = len(topology_margin_artifact.get("buckets", {}))
            print(f"Loaded topology margin artifact from {topology_margin_path} ({n_buckets} buckets)")
    aligner = Aligner()
    if use_zarr and os.path.exists(mdcath_zarr_path):
        featurizer = FeaturizerWindowZarr(window_size=window_size, include_angles=include_angles, coords_type=coords_type)
    else:
        featurizer = FeaturizerWindow(aligner, window_size=window_size, device="cpu", max_alignment_iters=2, include_angles=include_angles, coords_type=coords_type)
    collate_fn = RobustCollate(featurizer)

    # DATA LOADER
    # ------------
    if use_zarr and os.path.exists(mdcath_zarr_path):
        full_dataset = ZarrTrajectoriesDataset(
            featuriser=featurizer,
            mdcath_zarr_path=mdcath_zarr_path,
            max_domains=max_domains,
            window_size=window_size,
            samples_per_traj=samples_per_traj,
            crop=crop,
            crop_size=crop_size,
            use_atlas=use_atlas,
            atlas_zarr_path=atlas_zarr_path,
            native_aligned=True,
            coords_type=coords_type,
            atlas_stride=atlas_stride,
            rmsf_prior_path=rmsf_prior_path,
            randomize_windows=randomize_train_windows,
        )
        heavy_cascade = str(model_type).lower() == "cascade"
        default_num_workers = 4 if heavy_cascade else 16
        num_workers = default_num_workers if dataloader_num_workers is None else int(dataloader_num_workers)
        if rank == 0:
            print(
                "Using ZarrTrajectoriesDataset with:\n"
                f" paths:{mdcath_zarr_path},{atlas_zarr_path if use_atlas else None}\n"
                f" num_workers: {num_workers} (default {default_num_workers})"
            )
    else:
        full_dataset = TrajectoriesDataset(
            featurizer, aligner,
            mdcath_path=mdcath_path,
            use_atlas=use_atlas,
            atlas_path=atlas_path,
            max_domains=max_domains,
            samples_per_traj=samples_per_traj,
            window_size=window_size,
            crop=crop,
            crop_size=crop_size,
            offline_mode=offline_mode,
            max_alignment_iters=2,
            native_aligned=True,
            coords_type=coords_type,
            use_zarr=use_zarr,
            zarr_path=mdcath_zarr_path,
            atlas_stride=atlas_stride,
            randomize_windows=randomize_train_windows,
        )
        heavy_cascade = str(model_type).lower() == "cascade"
        default_num_workers = 4 if heavy_cascade else 6
        num_workers = default_num_workers if dataloader_num_workers is None else int(dataloader_num_workers)
        if rank == 0:
            print(
                "Using TrajectoriesDataset with: \n"
                f" path: {mdcath_path}\n"
                f" num_workers: {num_workers} (default {default_num_workers})"
            )

    # Train/Val/Test split
    train_id_file = os.path.join(checkpoint_dir, "train_ids.txt")
    val_id_file = os.path.join(checkpoint_dir, "val_ids.txt")
    test_id_file = os.path.join(checkpoint_dir, "test_ids.txt")
    split_source_dir = split_ids_dir if split_ids_dir is not None else checkpoint_dir
    source_train_id_file = os.path.join(split_source_dir, "train_ids.txt")
    source_val_id_file = os.path.join(split_source_dir, "val_ids.txt")
    source_test_id_file = os.path.join(split_source_dir, "test_ids.txt")
    has_split_files = os.path.exists(source_train_id_file) and os.path.exists(source_val_id_file)

    if has_split_files:
        if rank == 0:
            print(f"Loading split IDs from {split_source_dir}...")

        def load_id_set(path):
            with open(path, "r") as f:
                return set(line.strip() for line in f if line.strip())

        train_domains = load_id_set(source_train_id_file)
        val_domains = load_id_set(source_val_id_file)
        test_domains = load_id_set(source_test_id_file) if os.path.exists(source_test_id_file) else set()

        train_idx, val_idx, test_idx = [], [], []
        for i, (domain_id, _, _, _) in enumerate(full_dataset.index_map):
            if domain_id in train_domains:
                train_idx.append(i)
            elif domain_id in val_domains:
                val_idx.append(i)
            elif domain_id in test_domains:
                test_idx.append(i)

        if rank == 0:
            print(f"Restored -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
            if len(train_idx) == 0:
                raise RuntimeError("Resume failed: no matching train domains!")
            for src_path, dst_path in [
                (source_train_id_file, train_id_file),
                (source_val_id_file, val_id_file),
                (source_test_id_file, test_id_file),
            ]:
                if os.path.exists(src_path) and os.path.abspath(src_path) != os.path.abspath(dst_path):
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied split file -> {dst_path}")
        if dist.is_initialized():
            dist.barrier()
    else:
        if rank == 0:
            print(f"Dataset Size: {len(full_dataset)}")

        train_idx, val_idx, test_idx = no_spill_stratified_split(
            full_dataset, test_size=0.05, val_size=0.05, seed=42,
        )

        if rank == 0:
            print(f"Split -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

            def save_ids(indices, filename):
                unique_ids = sorted(set(full_dataset.index_map[i][0] for i in indices))
                with open(filename, "w") as f:
                    for did in unique_ids:
                        f.write(f"{did}\n")

            save_ids(train_idx, train_id_file)
            save_ids(val_idx, val_id_file)
            save_ids(test_idx, test_id_file)

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)
    set_randomize_windows(train_dataset, randomize_train_windows)
    if rank == 0:
        mode = "fresh per __getitem__" if randomize_train_windows else "deterministic seeded"
        print(f"Training temporal windows: {mode}; validation/test windows: deterministic seeded.")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    g = torch.Generator()
    g.manual_seed(base_seed)

    train_num_workers = max(0, int(num_workers))
    train_persistent_workers = bool(dataloader_persistent_workers) and train_num_workers > 0
    heavy_cascade = str(model_type).lower() == "cascade"
    default_timeout = 300 if heavy_cascade else 60
    default_prefetch = 2 if heavy_cascade else 4
    resolved_timeout = default_timeout if dataloader_timeout is None else int(dataloader_timeout)
    resolved_prefetch = default_prefetch if dataloader_prefetch_factor is None else int(dataloader_prefetch_factor)
    train_timeout = max(0, resolved_timeout) if train_num_workers > 0 else 0
    train_loader_kwargs = {
        "batch_size": per_gpu_batch_size,
        "sampler": train_sampler,
        "persistent_workers": train_persistent_workers,
        "num_workers": train_num_workers,
        "pin_memory": True,
        "generator": g,
        "collate_fn": collate_fn,
        "drop_last": True,
        "timeout": train_timeout,
        "worker_init_fn": worker_init_fn,
    }
    if train_num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = max(1, resolved_prefetch)
    if rank == 0:
        print(
            "Train DataLoader: "
            f"workers={train_num_workers}, timeout={train_timeout}s, "
            f"prefetch={train_loader_kwargs.get('prefetch_factor', 'n/a')}, "
            f"persistent={train_persistent_workers}"
        )

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(
        val_dataset, batch_size=int(per_gpu_batch_size/2), sampler=val_sampler,
        persistent_workers=False, num_workers=0, pin_memory=True,
        collate_fn=collate_fn, drop_last=False, timeout=0, worker_init_fn=worker_init_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=int(per_gpu_batch_size/2), sampler=test_sampler,
        persistent_workers=False, num_workers=0, pin_memory=True, 
        collate_fn=collate_fn, drop_last=False, timeout=0, worker_init_fn=worker_init_fn
    )

    total_channels = representation_channels + angle_channels
    effective_spectral_modes = top_k_freqs if spectral_modes is None else int(spectral_modes)
    
    # CENTRAL SPECTRAL REPRESENTATION PIPELINE
    # ----------------
    freq_scales = None
    conditioned_freq_scale = None
    aniso_freq_scales = None
    spectral_model = model_type not in ("fno", "fno_manifold", "hno", "fno2", "fno2_bishop", "v14a", "v14b", "v14c", "v15", "v16")
    is_v17_model_type = model_type in {"v17a", "v17b"}
    is_clean_spectral_graph_model_type = model_type in {"v12e_spec_graph", "v17a", "v17b"}
    if is_clean_spectral_graph_model_type and prediction_target != "x_0":
        raise ValueError(f"{model_type} currently requires prediction_target='x_0'")
    if not is_v17_model_type:
        if rank == 0 and any(
            float(v) > 0.0
            for v in (
                v17_low_mode_lambda,
                v17_adjacent_lambda,
                v17_idct_bond_lambda,
                v17_caca_tolerance_lambda,
            )
        ):
            print("WARNING: v17 auxiliary loss weights were set for a non-v17 model; disabling them.")
        v17_low_mode_lambda = 0.0
        v17_adjacent_lambda = 0.0
        v17_idct_bond_lambda = 0.0
        v17_caca_tolerance_lambda = 0.0
    expected_freq_dim = top_k_freqs * total_channels

    needs_main_scale_artifact = False
    needs_aniso_artifact = aniso_scales_path is not None
    if spectral_model:
        needs_main_scale_artifact = (
            freq_normalization != "none"
            or dc_residualization in {"bucket", "per_residue"}
            or (dc_residualization == "auto" and freq_scales_path is not None)
            or aniso_source == "freq_scales"
            or (aniso_source == "auto" and freq_normalization != "none")
        )

    if spectral_model and (needs_main_scale_artifact or needs_aniso_artifact):
        expected_freq_dim = top_k_freqs * total_channels  # or in_channels, whichever variable is your true model input channels

        if needs_main_scale_artifact and freq_scales_path is None:
            stats_pipeline = SpectralRepresentationPipeline(
                coordinate=representation_obj,
                raw_coord_channels=coord_channels,
                total_channels=total_channels,
                use_dct=use_DCT,
                freq_normalization="none",
                dc_residualization="none",
                aniso_source="none",
                device=device,
            )
            time_to_spectral_fn = stats_pipeline.time_to_spectral

            freq_scale_path = os.path.join(checkpoint_dir, "freq_scales.pt")
            if rank == 0:
                print("Computing frequency scaling statistics...")
                if os.path.exists(freq_scale_path):
                    os.remove(freq_scale_path)

                freq_scales = compute_frequency_stats(
                    train_loader,
                    time_to_spectral_fn,
                    samples=1000,
                    top_k=top_k_freqs,
                    device=device,
                    is_dct=use_DCT,
                    use_angles=include_angles,
                    coords_type=coords_type,
                    representation=representation_obj,
                )

                # Defensive trim in case the stats function returns more than needed
                if freq_scales.shape[0] < expected_freq_dim:
                    raise ValueError(
                        f"Computed freq_scales too short: got {freq_scales.shape[0]}, "
                        f"expected at least {expected_freq_dim}"
                    )
                freq_scales = freq_scales[:expected_freq_dim].contiguous()

                torch.save(freq_scales, freq_scale_path)

            dist.barrier()
            freq_scales = torch.load(freq_scale_path, map_location=device)

        elif needs_main_scale_artifact:
            freq_scales, conditioned_freq_scale = load_freq_scale_artifact(
                freq_scales_path, expected_freq_dim, map_location=device
            )

        if aniso_scales_path is not None:
            aniso_freq_scales, _ = load_freq_scale_artifact(
                aniso_scales_path, expected_freq_dim, map_location=device
            )

        if rank == 0:
            print("top_k_freqs =", top_k_freqs)
            print("channels =", total_channels)
            print("expected_freq_dim =", expected_freq_dim)
            if freq_scales is not None:
                print("trimmed freq_scales.shape =", tuple(freq_scales.shape))
            if aniso_freq_scales is not None:
                print("aniso_freq_scales.shape =", tuple(aniso_freq_scales.shape))
            if conditioned_freq_scale is not None:
                meta = conditioned_freq_scale.get("metadata", {})
                print(
                    "Using conditioned freq scales:",
                    {
                        "scheme": meta.get("scheme"),
                        "alpha": meta.get("alpha"),
                        "scale_condition_modes": meta.get("scale_condition_modes"),
                    },
                )

    transform_engine = SpectralRepresentationPipeline(
        coordinate=representation_obj,
        raw_coord_channels=coord_channels,
        total_channels=total_channels,
        use_dct=use_DCT,
        scale_factors=freq_scales,
        conditioned_freq_scale=conditioned_freq_scale,
        freq_normalization=freq_normalization if spectral_model else "none",
        dc_residualization=dc_residualization if spectral_model else "none",
        aniso_source=aniso_source if spectral_model else "none",
        aniso_scale_factors=aniso_freq_scales,
        device=device,
    )
    representation_obj = transform_engine.coordinate
    freq_scales_for_model = transform_engine.model_freq_scale
    conditioned_freq_scale_for_model = transform_engine.model_conditioned_freq_scale

    # Wire per-residue DC baselines into the dataset(s) if that policy is active.
    # Dataset.__getitem__ will then emit `dc_baseline_per_res` per sample and
    # train_step can subtract per-residue DC instead of bucket DC.
    prdc_payload = transform_engine.per_residue_dc_baselines
    if prdc_payload:
        prdc = {
            str(k): (v if torch.is_tensor(v) else torch.as_tensor(v)).float().contiguous()
            for k, v in prdc_payload.items()
        }
        for ds_holder in (full_dataset, getattr(full_dataset, "dataset", None)):
            if ds_holder is not None and hasattr(ds_holder, "per_residue_dc_baselines"):
                ds_holder.per_residue_dc_baselines = prdc
        if rank == 0:
            print(f"Attached per-residue DC baselines to dataset: {len(prdc)} (key,temp) pairs")

    if rank == 0:
        model_scale_msg = "conditioned" if conditioned_freq_scale_for_model is not None else (
            "global" if freq_scales_for_model is not None else "none"
        )
        print(
            "Effective spectral policies: "
            f"model_normalization={model_scale_msg}, "
            f"dc_residualization={transform_engine.effective_dc_residualization}, "
            f"aniso_source={transform_engine.effective_aniso_source}"
        )


    # MODEL INIT
    # ----------
    _configs = {
        "spectral_dit": SpectralDiTConfig(
            in_channels=total_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            top_k_freqs=top_k_freqs, freq_hidden_size=freq_hidden_size,
            prediction_target=prediction_target,
            freq_scale=freq_scales_for_model, cfg_dropout=conditioning_dropout, is_dct=use_DCT,
            conditioned_freq_scale=conditioned_freq_scale_for_model,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            use_shake=use_shake,
            shake_n_iter=shake_n_iter,
            shake_target=shake_target,
        ),
        "spectral_dit_low_k": SpectralDiTConfig(
            in_channels=total_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            top_k_freqs=top_k_freqs, freq_hidden_size=freq_hidden_size,
            prediction_target=prediction_target,
            freq_scale=freq_scales_for_model, cfg_dropout=conditioning_dropout, is_dct=use_DCT,
            conditioned_freq_scale=conditioned_freq_scale_for_model,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            use_low_k_correction_head=True,
            low_k_correction_modes=low_k_correction_modes,
            use_shake=use_shake,
            shake_n_iter=shake_n_iter,
            shake_target=shake_target,
        ),
        "spectral_conv_dit": SpectralConvDiTConfig(
            in_channels=total_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            top_k_freqs=top_k_freqs, freq_hidden_size=freq_hidden_size,
            spectral_modes=effective_spectral_modes,
            prediction_target=prediction_target,
            freq_scale=freq_scales_for_model, cfg_dropout=conditioning_dropout, is_dct=use_DCT,
            conditioned_freq_scale=conditioned_freq_scale_for_model,
            use_hilbert=use_hilbert_spatial,
            use_hilbert_dct=use_hilbert_spatial_dct,
            hilbert_mode=hilbert_mode,
            use_rmsf_prior_gain=use_rmsf_prior_gain,
            use_low_k_correction_head=use_low_k_correction_head,
            low_k_correction_modes=low_k_correction_modes,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
        ),
        # v9: SpectralConvDiT trunk + additive slow-branch transformer head.
        "spectral_conv_slow_branch": SpectralConvSlowBranchConfig(
            in_channels=total_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            top_k_freqs=top_k_freqs, freq_hidden_size=freq_hidden_size,
            spectral_modes=effective_spectral_modes,
            prediction_target=prediction_target,
            freq_scale=freq_scales_for_model, cfg_dropout=conditioning_dropout, is_dct=use_DCT,
            conditioned_freq_scale=conditioned_freq_scale_for_model,
            use_hilbert=use_hilbert_spatial,
            use_hilbert_dct=use_hilbert_spatial_dct,
            hilbert_mode=hilbert_mode,
            use_rmsf_prior_gain=use_rmsf_prior_gain,
            use_low_k_correction_head=use_low_k_correction_head,
            low_k_correction_modes=low_k_correction_modes,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            K_slow=K_slow,
            slow_d_model=slow_d_model,
            slow_depth=slow_depth,
            slow_num_heads=slow_num_heads,
            slow_mlp_ratio=slow_mlp_ratio,
            slow_attn_dropout=slow_attn_dropout,
            slow_use_rmsf_prior=slow_use_rmsf_prior,
        ),
        # v10: SpectralConvDiT trunk with block-diagonal frequency mixers.
        "spectral_conv_block_mix": SpectralConvBlockMixConfig(
            in_channels=total_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            top_k_freqs=top_k_freqs, freq_hidden_size=freq_hidden_size,
            spectral_modes=effective_spectral_modes,
            prediction_target=prediction_target,
            freq_scale=freq_scales_for_model, cfg_dropout=conditioning_dropout, is_dct=use_DCT,
            conditioned_freq_scale=conditioned_freq_scale_for_model,
            use_hilbert=use_hilbert_spatial,
            use_hilbert_dct=use_hilbert_spatial_dct,
            hilbert_mode=hilbert_mode,
            use_rmsf_prior_gain=use_rmsf_prior_gain,
            use_low_k_correction_head=use_low_k_correction_head,
            low_k_correction_modes=low_k_correction_modes,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            band_edges=fast_band_edges,
        ),
        "cascade": CascadeSpectralConfig(
            in_channels=total_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            top_k_freqs=top_k_freqs, freq_hidden_size=freq_hidden_size,
            spectral_modes=effective_spectral_modes,
            prediction_target=prediction_target,
            freq_scale=freq_scales_for_model, cfg_dropout=conditioning_dropout, is_dct=use_DCT,
            conditioned_freq_scale=conditioned_freq_scale_for_model,
            use_hilbert=use_hilbert_spatial,
            use_hilbert_dct=use_hilbert_spatial_dct,
            hilbert_mode=hilbert_mode,
            use_rmsf_prior_gain=use_rmsf_prior_gain,
            use_low_k_correction_head=False,
            low_k_correction_modes=low_k_correction_modes,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            window_size=window_size,
            cascade_band_edges=cascade_band_edges,
            cascade_context_mode=cascade_context_mode,
            cascade_detach_context=cascade_detach_context,
            cascade_dc_depth=cascade_dc_depth,
            cascade_low_depth=cascade_low_depth,
            cascade_high_depth=cascade_high_depth,
        ),
        "spectral_conv_block_mix_amplitude": SpectralConvBlockMixAmplitudeConfig(
            in_channels=total_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            top_k_freqs=top_k_freqs, freq_hidden_size=freq_hidden_size,
            spectral_modes=effective_spectral_modes,
            prediction_target=prediction_target,
            freq_scale=freq_scales_for_model, cfg_dropout=conditioning_dropout, is_dct=use_DCT,
            conditioned_freq_scale=conditioned_freq_scale_for_model,
            use_hilbert=use_hilbert_spatial,
            use_hilbert_dct=use_hilbert_spatial_dct,
            hilbert_mode=hilbert_mode,
            use_rmsf_prior_gain=use_rmsf_prior_gain,
            use_low_k_correction_head=use_low_k_correction_head,
            low_k_correction_modes=low_k_correction_modes,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            band_edges=fast_band_edges,
            amp_head_context_modes=amp_head_context_modes,
            amp_head_target_modes=amp_head_target_modes,
            amp_head_d_model=amp_head_d_model,
            amp_head_depth=amp_head_depth,
            amp_head_num_heads=amp_head_num_heads,
            amp_head_mlp_ratio=amp_head_mlp_ratio,
            amp_head_attn_dropout=amp_head_attn_dropout,
            amp_head_use_rmsf_prior=amp_head_use_rmsf_prior,
            use_shake=use_shake,
            shake_n_iter=shake_n_iter,
            shake_target=shake_target,
        ),
        "v12c": SpectralConvBlockMixAmplitudeRefinedConfig(
            in_channels=total_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            top_k_freqs=top_k_freqs, freq_hidden_size=freq_hidden_size,
            spectral_modes=effective_spectral_modes,
            prediction_target=prediction_target,
            freq_scale=freq_scales_for_model, cfg_dropout=conditioning_dropout, is_dct=use_DCT,
            conditioned_freq_scale=conditioned_freq_scale_for_model,
            use_hilbert=use_hilbert_spatial,
            use_hilbert_dct=use_hilbert_spatial_dct,
            hilbert_mode=hilbert_mode,
            use_rmsf_prior_gain=use_rmsf_prior_gain,
            use_low_k_correction_head=use_low_k_correction_head,
            low_k_correction_modes=low_k_correction_modes,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            band_edges=fast_band_edges,
            amp_head_context_modes=amp_head_context_modes,
            amp_head_target_modes=amp_head_target_modes,
            amp_head_d_model=amp_head_d_model,
            amp_head_depth=amp_head_depth,
            amp_head_num_heads=amp_head_num_heads,
            amp_head_mlp_ratio=amp_head_mlp_ratio,
            amp_head_attn_dropout=amp_head_attn_dropout,
            amp_head_use_rmsf_prior=amp_head_use_rmsf_prior,
            use_shake=use_shake,
            shake_n_iter=shake_n_iter,
            shake_target=shake_target,
            refiner_hidden=refiner_hidden,
            refiner_depth=refiner_depth,
            refiner_kernel_size=refiner_kernel_size,
            refiner_max_delta=refiner_max_delta,
        ),
        "v12e_spec_graph": SpectralConvBlockMixAmplitudeSpecGraphConfig(
            in_channels=total_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            top_k_freqs=top_k_freqs, freq_hidden_size=freq_hidden_size,
            spectral_modes=effective_spectral_modes,
            prediction_target=prediction_target,
            freq_scale=freq_scales_for_model, cfg_dropout=conditioning_dropout, is_dct=use_DCT,
            conditioned_freq_scale=conditioned_freq_scale_for_model,
            use_hilbert=use_hilbert_spatial,
            use_hilbert_dct=use_hilbert_spatial_dct,
            hilbert_mode=hilbert_mode,
            use_rmsf_prior_gain=use_rmsf_prior_gain,
            use_low_k_correction_head=use_low_k_correction_head,
            low_k_correction_modes=low_k_correction_modes,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            band_edges=fast_band_edges,
            amp_head_context_modes=amp_head_context_modes,
            amp_head_target_modes=amp_head_target_modes,
            amp_head_d_model=amp_head_d_model,
            amp_head_depth=amp_head_depth,
            amp_head_num_heads=amp_head_num_heads,
            amp_head_mlp_ratio=amp_head_mlp_ratio,
            amp_head_attn_dropout=amp_head_attn_dropout,
            amp_head_use_rmsf_prior=amp_head_use_rmsf_prior,
            use_shake=use_shake,
            shake_n_iter=shake_n_iter,
            shake_target=shake_target,
            use_v12e_spec_graph=use_v12e_spec_graph,
            v12e_spec_graph_k_min=v12e_spec_graph_k_min,
            v12e_spec_graph_k_max=v12e_spec_graph_k_max,
            v12e_spec_graph_hidden=v12e_spec_graph_hidden,
            v12e_spec_graph_depth=v12e_spec_graph_depth,
            v12e_spec_graph_msg_hidden=v12e_spec_graph_msg_hidden,
            v12e_spec_graph_sequence_window=v12e_spec_graph_sequence_window,
            v12e_spec_graph_knn=v12e_spec_graph_knn,
            v12e_spec_graph_use_sequence_edges=v12e_spec_graph_use_sequence_edges,
            v12e_spec_graph_use_native_knn=v12e_spec_graph_use_native_knn,
            v12e_spec_graph_max_delta=v12e_spec_graph_max_delta,
        ),
        "v17a": SpectralConvBlockMixAmplitudeSpectralGraphRefinedConfig(
            in_channels=total_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            top_k_freqs=top_k_freqs, freq_hidden_size=freq_hidden_size,
            spectral_modes=effective_spectral_modes,
            prediction_target=prediction_target,
            freq_scale=freq_scales_for_model, cfg_dropout=conditioning_dropout, is_dct=use_DCT,
            conditioned_freq_scale=conditioned_freq_scale_for_model,
            use_hilbert=use_hilbert_spatial,
            use_hilbert_dct=use_hilbert_spatial_dct,
            hilbert_mode=hilbert_mode,
            use_rmsf_prior_gain=use_rmsf_prior_gain,
            use_low_k_correction_head=use_low_k_correction_head,
            low_k_correction_modes=low_k_correction_modes,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            band_edges=fast_band_edges,
            amp_head_context_modes=amp_head_context_modes,
            amp_head_target_modes=amp_head_target_modes,
            amp_head_d_model=amp_head_d_model,
            amp_head_depth=amp_head_depth,
            amp_head_num_heads=amp_head_num_heads,
            amp_head_mlp_ratio=amp_head_mlp_ratio,
            amp_head_attn_dropout=amp_head_attn_dropout,
            amp_head_use_rmsf_prior=amp_head_use_rmsf_prior,
            use_shake=use_shake,
            shake_n_iter=shake_n_iter,
            shake_target=shake_target,
            refiner_hidden=refiner_hidden,
            refiner_depth=refiner_depth,
            refiner_kernel_size=refiner_kernel_size,
            refiner_max_delta=refiner_max_delta,
            use_spectral_graph_refiner=use_spectral_graph_refiner,
            spectral_graph_refiner_modes=spectral_graph_refiner_modes,
            spectral_graph_refiner_hidden=spectral_graph_refiner_hidden,
            spectral_graph_refiner_depth=spectral_graph_refiner_depth,
            spectral_graph_refiner_msg_hidden=spectral_graph_refiner_msg_hidden,
            spectral_graph_refiner_sequence_window=spectral_graph_refiner_sequence_window,
            spectral_graph_refiner_knn=spectral_graph_refiner_knn,
            spectral_graph_refiner_use_sequence_edges=spectral_graph_refiner_use_sequence_edges,
            spectral_graph_refiner_use_native_knn=spectral_graph_refiner_use_native_knn,
            spectral_graph_refiner_max_delta=spectral_graph_refiner_max_delta,
        ),
        "v17b": SpectralConvBlockMixAmplitudeBondGraphRefinedConfig(
            in_channels=total_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            top_k_freqs=top_k_freqs, freq_hidden_size=freq_hidden_size,
            spectral_modes=effective_spectral_modes,
            prediction_target=prediction_target,
            freq_scale=freq_scales_for_model, cfg_dropout=conditioning_dropout, is_dct=use_DCT,
            conditioned_freq_scale=conditioned_freq_scale_for_model,
            use_hilbert=use_hilbert_spatial,
            use_hilbert_dct=use_hilbert_spatial_dct,
            hilbert_mode=hilbert_mode,
            use_rmsf_prior_gain=use_rmsf_prior_gain,
            use_low_k_correction_head=use_low_k_correction_head,
            low_k_correction_modes=low_k_correction_modes,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            band_edges=fast_band_edges,
            amp_head_context_modes=amp_head_context_modes,
            amp_head_target_modes=amp_head_target_modes,
            amp_head_d_model=amp_head_d_model,
            amp_head_depth=amp_head_depth,
            amp_head_num_heads=amp_head_num_heads,
            amp_head_mlp_ratio=amp_head_mlp_ratio,
            amp_head_attn_dropout=amp_head_attn_dropout,
            amp_head_use_rmsf_prior=amp_head_use_rmsf_prior,
            use_shake=use_shake,
            shake_n_iter=shake_n_iter,
            shake_target=shake_target,
            refiner_hidden=refiner_hidden,
            refiner_depth=refiner_depth,
            refiner_kernel_size=refiner_kernel_size,
            refiner_max_delta=refiner_max_delta,
            use_bond_spectral_graph_refiner=use_bond_spectral_graph_refiner,
            bond_spectral_graph_refiner_modes=bond_spectral_graph_refiner_modes,
            bond_spectral_graph_refiner_hidden=bond_spectral_graph_refiner_hidden,
            bond_spectral_graph_refiner_depth=bond_spectral_graph_refiner_depth,
            bond_spectral_graph_refiner_msg_hidden=bond_spectral_graph_refiner_msg_hidden,
            bond_spectral_graph_refiner_sequence_window=bond_spectral_graph_refiner_sequence_window,
            bond_spectral_graph_refiner_knn=bond_spectral_graph_refiner_knn,
            bond_spectral_graph_refiner_use_sequence_edges=bond_spectral_graph_refiner_use_sequence_edges,
            bond_spectral_graph_refiner_use_native_knn=bond_spectral_graph_refiner_use_native_knn,
            bond_spectral_graph_refiner_max_delta=bond_spectral_graph_refiner_max_delta,
            bond_spectral_graph_refiner_blend=bond_spectral_graph_refiner_blend,
        ),
        "v12a_egnn": SpectralConvBlockMixAmplitudeEGNNConfig(
            in_channels=total_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            top_k_freqs=top_k_freqs, freq_hidden_size=freq_hidden_size,
            spectral_modes=effective_spectral_modes,
            prediction_target=prediction_target,
            freq_scale=freq_scales_for_model, cfg_dropout=conditioning_dropout, is_dct=use_DCT,
            conditioned_freq_scale=conditioned_freq_scale_for_model,
            use_hilbert=use_hilbert_spatial,
            use_hilbert_dct=use_hilbert_spatial_dct,
            hilbert_mode=hilbert_mode,
            use_rmsf_prior_gain=use_rmsf_prior_gain,
            use_low_k_correction_head=use_low_k_correction_head,
            low_k_correction_modes=low_k_correction_modes,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            band_edges=fast_band_edges,
            amp_head_context_modes=amp_head_context_modes,
            amp_head_target_modes=amp_head_target_modes,
            amp_head_d_model=amp_head_d_model,
            amp_head_depth=amp_head_depth,
            amp_head_num_heads=amp_head_num_heads,
            amp_head_mlp_ratio=amp_head_mlp_ratio,
            amp_head_attn_dropout=amp_head_attn_dropout,
            amp_head_use_rmsf_prior=amp_head_use_rmsf_prior,
            use_shake=use_shake,
            shake_n_iter=shake_n_iter,
            shake_target=shake_target,
            egnn_h_dim=egnn_h_dim,
            egnn_hidden=egnn_hidden,
            egnn_depth=egnn_depth,
            egnn_seq_window=egnn_seq_window,
            egnn_max_len=egnn_max_len,
            egnn_t_chunk=egnn_t_chunk,
        ),
        "v12b_egnn": SpectralConvBlockMixSlowHybridEGNNConfig(
            in_channels=total_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            top_k_freqs=top_k_freqs, freq_hidden_size=freq_hidden_size,
            spectral_modes=effective_spectral_modes,
            prediction_target=prediction_target,
            freq_scale=freq_scales_for_model, cfg_dropout=conditioning_dropout, is_dct=use_DCT,
            conditioned_freq_scale=conditioned_freq_scale_for_model,
            use_hilbert=use_hilbert_spatial,
            use_hilbert_dct=use_hilbert_spatial_dct,
            hilbert_mode=hilbert_mode,
            use_rmsf_prior_gain=use_rmsf_prior_gain,
            use_low_k_correction_head=use_low_k_correction_head,
            low_k_correction_modes=low_k_correction_modes,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            band_edges=fast_band_edges,
            K_slow=K_slow,
            slow_mode_start=slow_mode_start,
            slow_d_model=slow_d_model,
            slow_depth=slow_depth,
            slow_num_heads=slow_num_heads,
            slow_mlp_ratio=slow_mlp_ratio,
            slow_attn_dropout=slow_attn_dropout,
            slow_use_rmsf_prior=slow_use_rmsf_prior,
            use_shake=use_shake,
            shake_n_iter=shake_n_iter,
            shake_target=shake_target,
            egnn_h_dim=egnn_h_dim,
            egnn_hidden=egnn_hidden,
            egnn_depth=egnn_depth,
            egnn_seq_window=egnn_seq_window,
            egnn_max_len=egnn_max_len,
            egnn_t_chunk=egnn_t_chunk,
        ),
        "spectral_conv_block_mix_slow_hybrid": SpectralConvBlockMixSlowHybridConfig(
            in_channels=total_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            top_k_freqs=top_k_freqs, freq_hidden_size=freq_hidden_size,
            spectral_modes=effective_spectral_modes,
            prediction_target=prediction_target,
            freq_scale=freq_scales_for_model, cfg_dropout=conditioning_dropout, is_dct=use_DCT,
            conditioned_freq_scale=conditioned_freq_scale_for_model,
            use_hilbert=use_hilbert_spatial,
            use_hilbert_dct=use_hilbert_spatial_dct,
            hilbert_mode=hilbert_mode,
            use_rmsf_prior_gain=use_rmsf_prior_gain,
            use_low_k_correction_head=use_low_k_correction_head,
            low_k_correction_modes=low_k_correction_modes,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            band_edges=fast_band_edges,
            K_slow=K_slow,
            slow_mode_start=slow_mode_start,
            slow_d_model=slow_d_model,
            slow_depth=slow_depth,
            slow_num_heads=slow_num_heads,
            slow_mlp_ratio=slow_mlp_ratio,
            slow_attn_dropout=slow_attn_dropout,
            slow_use_rmsf_prior=slow_use_rmsf_prior,
        ),
        "dual_branch": DualBranchConfig(
            top_k_freqs=top_k_freqs,
            K_slow=K_slow,
            in_channels=total_channels,
            cond_channels=coord_channels,
            prediction_target=prediction_target,
            is_dct=use_DCT,
            freq_scale=freq_scales_for_model,
            conditioned_freq_scale=conditioned_freq_scale_for_model,
            cfg_dropout=conditioning_dropout,
            slow_d_model=slow_d_model,
            slow_depth=slow_depth,
            slow_num_heads=slow_num_heads,
            slow_mlp_ratio=slow_mlp_ratio,
            slow_attn_dropout=slow_attn_dropout,
            slow_use_rmsf_prior=slow_use_rmsf_prior,
            slow_predicts_amplitude_only=slow_predicts_amplitude_only,
            fast_freq_hidden_size=freq_hidden_size,
            fast_depth=num_layers,
            fast_num_heads=num_heads,
            fast_spectral_modes=effective_spectral_modes,
            fast_band_edges=fast_band_edges,
            fast_cond_dim=fast_cond_dim,
            fast_use_hilbert=use_hilbert_spatial,
            fast_use_hilbert_dct=use_hilbert_spatial_dct,
            fast_hilbert_mode=hilbert_mode,
            fast_use_rmsf_prior_gain=use_rmsf_prior_gain,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
        ),
        "fno": FNOConfig(
            in_channels=coord_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            window_size=window_size, hidden_per_time=freq_hidden_size,
            spectral_modes=effective_spectral_modes, coord_scale=coord_scale,
            use_dropout=conditioning_dropout,
            prediction_target=prediction_target,
        ),
        "fno_manifold": FNOManifoldConfig(
            in_channels=7, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            window_size=window_size, hidden_per_time=freq_hidden_size,
            spectral_modes=effective_spectral_modes,
            use_dropout=conditioning_dropout,
            prediction_target=prediction_target,
        ),
        "hno": HNOConfig(
            in_channels=coord_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            window_size=window_size, hidden_per_time=freq_hidden_size,
            spectral_modes=effective_spectral_modes, coord_scale=coord_scale,
            use_dropout=conditioning_dropout,
            prediction_target=prediction_target,
        ),
        "fno2": FNO2Config(
            in_channels=coord_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            window_size=window_size, hidden_per_time=max(freq_hidden_size, 8),
            spectral_modes=effective_spectral_modes, coord_scale=coord_scale,
            use_dropout=conditioning_dropout,
            prediction_target=prediction_target,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            temporal_ablation_mode=temporal_ablation_mode,
            block_ablation_mode=block_ablation_mode,
            temporal_gate_init=temporal_gate_init,
        ),
        "fno2_bishop": FNO2BishopConfig(
            in_channels=coord_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            window_size=window_size, hidden_per_time=max(freq_hidden_size, 8),
            spectral_modes=effective_spectral_modes, coord_scale=coord_scale,
            use_dropout=conditioning_dropout,
            prediction_target=prediction_target,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            temporal_ablation_mode=temporal_ablation_mode,
            block_ablation_mode=block_ablation_mode,
            temporal_gate_init=temporal_gate_init,
        ),
        "v15": V15Config(
            in_channels=coord_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            window_size=window_size, hidden_per_time=max(freq_hidden_size, 8),
            spectral_modes=effective_spectral_modes, coord_scale=coord_scale,
            use_dropout=conditioning_dropout,
            prediction_target=prediction_target,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            temporal_ablation_mode=temporal_ablation_mode,
            block_ablation_mode=block_ablation_mode,
            temporal_gate_init=temporal_gate_init,
        ),
        "v16": V16Config(
            in_channels=coord_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            window_size=window_size, hidden_per_time=max(freq_hidden_size, 8),
            spectral_modes=effective_spectral_modes, coord_scale=coord_scale,
            use_dropout=conditioning_dropout,
            prediction_target=prediction_target,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            temporal_ablation_mode=temporal_ablation_mode,
            block_ablation_mode=block_ablation_mode,
            temporal_gate_init=temporal_gate_init,
        ),
        "v14a": V14AConfig(
            in_channels=coord_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            window_size=window_size, hidden_size=hidden_size or max(freq_hidden_size * 16, 128),
            spectral_modes=effective_spectral_modes, coord_scale=coord_scale,
            use_dropout=conditioning_dropout,
            prediction_target=prediction_target,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
        ),
        "v14b": V14BConfig(
            in_channels=coord_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            window_size=window_size, hidden_size=hidden_size or max(freq_hidden_size * 16, 128),
            spectral_modes=effective_spectral_modes, coord_scale=coord_scale,
            use_dropout=conditioning_dropout,
            prediction_target=prediction_target,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
        ),
        "v14c": V14CConfig(
            in_channels=coord_channels, cond_channels=coord_channels,
            depth=num_layers, num_heads=num_heads,
            window_size=window_size, hidden_size=hidden_size or max(freq_hidden_size * 16, 128),
            spectral_modes=effective_spectral_modes, coord_scale=coord_scale,
            use_dropout=conditioning_dropout,
            prediction_target=prediction_target,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            jepa_latent_dim=jepa_latent_dim,
        ),
    }

    if model_type not in _configs:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = UnifiedWrapper(model_name=model_type, config=_configs[model_type]).to(device)

    latest_ckpt_for_resume = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    if checkpoint_path is not None and resume_from_latest:
        if os.path.exists(latest_ckpt_for_resume):
            if rank == 0:
                print(
                    "resume_from_latest=True and checkpoint_latest.pt exists; "
                    f"ignoring warm-start checkpoint_path={checkpoint_path}"
                )
            checkpoint_path = None
        else:
            raise ValueError(
                "Use either checkpoint_path (weights-only warm start) or resume_from_latest. "
                f"No checkpoint_latest.pt found at {latest_ckpt_for_resume!r}."
            )

    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        if rank == 0:
            print(f"Warm-starting model weights from: {checkpoint_path}")
        load_model_weights(checkpoint_path, model, device=device, log_fn=print if rank == 0 else (lambda *_args, **_kwargs: None))

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {model_type} | Params: {n_params:,}")

    if per_gpu_batch_size < 16:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    _inner = getattr(model, "model", model)
    has_external_refiner = hasattr(_inner, "refine_ca")
    if ddp_find_unused_parameters is None:
        # Refiner parameters are intentionally used after IDCT in train_step,
        # i.e. outside the DDP-wrapped forward. DDP unused-parameter traversal
        # can pre-mark them unused and then mark them ready again during
        # backward, producing "marked as ready twice" crashes.
        ddp_find_unused_parameters = False
    if ddp_static_graph is None:
        ddp_static_graph = has_external_refiner

    ddp_kwargs = {
        "device_ids": [local_rank],
        "find_unused_parameters": bool(ddp_find_unused_parameters),
    }
    ddp_supports_static_graph = "static_graph" in inspect.signature(DDP).parameters
    if bool(ddp_static_graph) and ddp_supports_static_graph:
        ddp_kwargs["static_graph"] = True
        ddp_kwargs["find_unused_parameters"] = False
    if rank == 0:
        print(
            "DDP settings: "
            f"find_unused_parameters={ddp_kwargs['find_unused_parameters']}, "
            f"static_graph={ddp_kwargs.get('static_graph', False)}, "
            f"external_refiner={has_external_refiner}"
        )
    model = DDP(model, **ddp_kwargs)

    # Diffusion init
    noise_config = {
        "schedule": schedule,
        "shift_value": shift_value,
        "aniso_gamma": aniso_gamma,
        "noise_schedule": noise_schedule,
        "noise_space": noise_space,
        "noise_band_edges": noise_band_edges,
        "noise_group_model_multipliers": noise_group_model_multipliers,
        "noise_target_crossings": noise_target_crossings,
        "noise_anchor_band": noise_anchor_band,
        "noise_power_normalization": noise_power_normalization,
        "noise_auto_shift": noise_auto_shift,
    }
    resolved_noise = resolve_noise_schedule(
        config=noise_config,
        freq_scales=transform_engine.aniso_freq_scale,
        top_k_freqs=top_k_freqs,
        channels=total_channels,
        num_steps=num_steps,
        device=device,
    )
    if rank == 0:
        print(format_noise_diagnostics(resolved_noise.diagnostics))
        try:
            wandb.config.update(
                {
                    "resolved_shift_value": resolved_noise.shift_value,
                    "resolved_noise_schedule": resolved_noise.diagnostics,
                },
                allow_val_change=True,
            )
        except Exception:
            pass

    diffusion = SpectralDiffusion(
        T=num_steps, device=device, schedule=resolved_noise.schedule,
        min_snr_gamma=min_snr_gamma, shift_value=resolved_noise.shift_value,
        aniso_weights=resolved_noise.aniso_weights,
    )

    # TRAINING 
    # ---------
    if not test_only:
        start_epoch = 0
        global_step = 0
        best_model_score = float("-inf")
        latest_ckpt = os.path.join(checkpoint_dir, "checkpoint_latest.pt")

        # optimizer with specified weight decay
        param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_params = [
            {"params": decay_params, "weight_decay": 0.05},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        if optimizer.lower() == "adamw":
            opt = optim.AdamW(optim_params, lr=max_lr)
        elif optimizer.lower() == "sgd":
            opt = optim.SGD(optim_params, lr=max_lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # init scheduler
        num_total_steps = epochs * len(train_loader) # <-- DO NOT subtract start_epoch here!
        warmup_pct = min(0.1, 5 * len(train_loader) / max(num_total_steps, 1))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=max_lr, total_steps=num_total_steps,
            pct_start=warmup_pct, anneal_strategy="cos",
        )

        # Now load checkpoint if we are resuming from one
        if resume_from_latest and os.path.exists(latest_ckpt):
            model, opt, sched, start_epoch, wandb_id, global_step, best_model_score = load_checkpoint(
                latest_ckpt, model, optimizer=opt, scheduler=sched, device=device,
            )
            sched = maybe_rebuild_onecycle_scheduler(
                sched,
                opt,
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                max_lr=max_lr,
                global_step=global_step,
                rank=rank,
            )
            if rank == 0:
                print(f"Resumed flawlessly from epoch {start_epoch}")

        # DDP sync
        state_tensor = torch.tensor([start_epoch, global_step, best_model_score], device=device)
        dist.broadcast(state_tensor, src=0)
        start_epoch = int(state_tensor[0].item())
        global_step = int(state_tensor[1].item())
        best_model_score = float(state_tensor[2].item())

        # RUN
        best_model_ckpt, global_step, best_model_score = TRAIN(
            model, train_loader, val_loader, opt, diffusion, transform_engine,
            device=device, epochs=epochs, start_epoch=start_epoch,
            start_step=global_step, best_model_score=best_model_score,
            coord_channels=coord_channels, angle_channels=angle_channels,
            checkpoint_dir=checkpoint_dir, mdcath_path=mdcath_path, sampler=train_sampler,
            scheduler=sched, window_size=window_size, displacement=displacement,
            representation=representation_obj,
            top_k_freqs=top_k_freqs, freq_weighting=freq_weighting,
            cond_drop_prob=cond_drop_prob if conditioning_dropout else 0.0,
            trim_cache=trim_cache, crop_size=crop_size,
            guidance_scale=guidance_scale, num_ode_steps=num_ode_steps, 
            max_val_batches=max_val_batches,
            bending_lambda=bending_lambda,
            geo_loss=geo_loss,
            geometry_lambda=geometry_lambda,
            representation_barrier_lambda=representation_barrier_lambda,
            geometry_warmup_start=geometry_warmup_start,
            geometry_warmup_epochs=geometry_warmup_epochs,
            geometry_decay_start=geometry_decay_start,
            geometry_decay_epochs=geometry_decay_epochs,
            geometry_tol=geometry_tol,
            clash_lambda=clash_lambda,
            clash_threshold=clash_threshold,
            clash_max_pairs=clash_max_pairs,
            clash_pair_chunk=clash_pair_chunk,
            topology_margin_artifact=topology_margin_artifact,
            spectral_geo_segment_threshold=spectral_geo_segment_threshold,
            spectral_geo_max_segment_pairs=spectral_geo_max_segment_pairs,
            risk_band_max_pairs=risk_band_max_pairs,
            risk_band_max_segment_pairs=risk_band_max_segment_pairs,
            rmsf_lambda=rmsf_lambda,
            rmsf_warmup_start=rmsf_warmup_start,
            rmsf_warmup_epochs=rmsf_warmup_epochs,
            low_freq_lambda=low_freq_lambda,
            low_freq_modes=low_freq_modes,
            dc_lambda=dc_lambda,
            dc_start_epoch=dc_start_epoch,
            v17_aux_modes=v17_aux_modes,
            v17_low_mode_lambda=v17_low_mode_lambda,
            v17_adjacent_lambda=v17_adjacent_lambda,
            v17_idct_bond_lambda=v17_idct_bond_lambda,
            v17_caca_tolerance_lambda=v17_caca_tolerance_lambda,
            v17_caca_target=v17_caca_target,
            v17_caca_tolerance=v17_caca_tolerance,
            loss_slow_weight=loss_slow_weight,
            loss_fast_weight=loss_fast_weight,
            loss_total_weight=loss_total_weight,
            v12e_spec_graph_residual_lambda=v12e_spec_graph_residual_lambda,
            randomize_train_windows=randomize_train_windows,
            max_bad_update_streak=max_bad_update_streak,
            max_bad_update_total=max_bad_update_total,
            debug_nonfinite_hooks=debug_nonfinite_hooks,
            debug_nonfinite_forward=debug_nonfinite_forward,
            debug_nonfinite_backward=debug_nonfinite_backward,
            debug_nonfinite_filter=debug_nonfinite_filter,
            debug_nonfinite_max_modules=debug_nonfinite_max_modules,
        )
    else:
        best_model_ckpt = os.path.join(checkpoint_dir, "best_model.pt")

    # Test
    if conditioning_dropout:
        guidance_scale = max(guidance_scale, 2.0)

    set_randomize_windows(test_loader.dataset, False)
    test_metrics = TEST(
        model, diffusion, test_loader, transform_engine, device,
        checkpoint_path=best_model_ckpt, is_distributed=True,
        top_k_freqs=top_k_freqs, displacement=displacement,
        representation=representation_obj,
        window_size=window_size, guidance_scale=guidance_scale,
        num_ode_steps=num_ode_steps,
        geometry_tol=geometry_tol,
        clash_lambda=clash_lambda,
        clash_threshold=clash_threshold,
        clash_max_pairs=clash_max_pairs,
        clash_pair_chunk=clash_pair_chunk,
    )

    if dist.is_initialized():
        dist.barrier()

    return test_metrics


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    ckpt_has_module = any(k.startswith("module.") for k in state_dict)
    is_ddp = isinstance(model, DDP)
    target = model.module if is_ddp else model

    # strict=False so that additive model changes (e.g. new optional
    # parameters like `rmsf_gate`) do not break checkpoint resume. Missing
    # and unexpected keys are logged for auditability.
    if is_ddp and not ckpt_has_module:
        target_state_dict = target.state_dict()
        state_dict = _remap_legacy_state_dict_keys(state_dict, target_state_dict)
        state_dict = _drop_shape_incompatible_state_dict_keys(
            state_dict, target_state_dict, log_fn=print, prefix="load_checkpoint"
        )
        result = target.load_state_dict(state_dict, strict=False)
    elif not is_ddp and ckpt_has_module:
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        target_state_dict = target.state_dict()
        state_dict = _remap_legacy_state_dict_keys(state_dict, target_state_dict)
        state_dict = _drop_shape_incompatible_state_dict_keys(
            state_dict, target_state_dict, log_fn=print, prefix="load_checkpoint"
        )
        result = target.load_state_dict(state_dict, strict=False)
    else:
        if is_ddp and ckpt_has_module:
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        target_state_dict = target.state_dict()
        state_dict = _remap_legacy_state_dict_keys(state_dict, target_state_dict)
        state_dict = _drop_shape_incompatible_state_dict_keys(
            state_dict, target_state_dict, log_fn=print, prefix="load_checkpoint"
        )
        result = target.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"load_checkpoint: missing keys (kept at init) = {result.missing_keys}")
    if result.unexpected_keys:
        print(f"load_checkpoint: unexpected keys (ignored)   = {result.unexpected_keys}")

    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError as e:
            # Catch both "parameter groups" (group count mismatch) and
            # "size of optimizer's group" (group length mismatch from new params)
            if "parameter groups" in str(e) or "size of optimizer" in str(e):
                print(f"Optimizer state mismatch — reinitializing optimizer. ({e})")
            else:
                raise

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception:
            print("Scheduler mismatch — resetting.")

    start_epoch = checkpoint["epoch"] + 1
    wandb_id = checkpoint.get("wandb_id", None)
    global_step = checkpoint.get("global_step", 0)
    best_model_score = checkpoint.get("best_model_score", float("-inf"))

    return model, optimizer, scheduler, start_epoch, wandb_id, global_step, best_model_score


def maybe_rebuild_onecycle_scheduler(
    scheduler,
    optimizer,
    *,
    epochs,
    steps_per_epoch,
    max_lr,
    global_step,
    rank=0,
):
    """Rebuild OneCycleLR when the resumed run changes the total step budget."""
    if scheduler is None or not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        return scheduler

    expected_total_steps = epochs * steps_per_epoch
    loaded_total_steps = getattr(scheduler, "total_steps", None)

    if loaded_total_steps == expected_total_steps:
        return scheduler

    if global_step >= expected_total_steps:
        raise ValueError(
            f"Cannot resume scheduler at global_step={global_step} with total_steps={expected_total_steps}. "
            "Increase epochs or disable scheduler resume."
        )

    warmup_pct = min(0.1, 5 * steps_per_epoch / max(expected_total_steps, 1))
    rebuilt = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=expected_total_steps,
        pct_start=warmup_pct,
        anneal_strategy="cos",
        last_epoch=max(global_step - 1, -1),
    )

    if rank == 0:
        print(
            "Scheduler total_steps changed on resume; rebuilding OneCycleLR "
            f"from {loaded_total_steps} to {expected_total_steps} at global_step={global_step}."
        )

    return rebuilt



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pancakes CNO spectral diffusion training")

    # Core
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--include_angles", action="store_true", default=None)
    parser.add_argument('--coords_type', type=str, default=None)
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
        help="Coordinate representation to DCT/diffuse. Replaces legacy --displacement.",
    )
    parser.add_argument("--representation_length_min", type=float, default=None)
    parser.add_argument("--representation_length_max", type=float, default=None)
    parser.add_argument("--representation_length_residual_max", type=float, default=None)
    parser.add_argument("--representation_barrier_lambda", type=float, default=None)
    parser.add_argument(
        "--freq_normalization",
        type=str,
        default=None,
        choices=["auto", "none", "global", "conditioned"],
        help="Amplitude normalisation policy for spectral model internals.",
    )
    parser.add_argument(
        "--dc_residualization",
        type=str,
        default=None,
        choices=["auto", "none", "bucket", "per_residue"],
        help="DC residualisation policy, decoupled from amplitude normalisation.",
    )
    parser.add_argument(
        "--aniso_source",
        type=str,
        default=None,
        choices=["auto", "none", "freq_scales", "artifact"],
        help="Scale source for anisotropic diffusion noise.",
    )
    parser.add_argument("--use_DCT", action="store_true", default=None)
    parser.add_argument("--use_zarr", action="store_true", default=None)
    parser.add_argument("--use_atlas", action="store_true", default=None)

    # Model
    parser.add_argument("--model_type", type=str, default=None, choices=["spectral_dit", "spectral_dit_low_k", "spectral_conv_dit", "spectral_conv_slow_branch", "spectral_conv_block_mix", "cascade", "spectral_conv_block_mix_amplitude", "spectral_conv_block_mix_slow_hybrid", "dual_branch", "fno", "fno_manifold", "hno", "fno2", "fno2_bishop", "v14a", "v14b", "v14c", "v15", "v16", "v12c", "v12e_spec_graph", "v17a", "v17b", "v12a_egnn", "v12b_egnn"])
    parser.add_argument("--top_k_freqs", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--freq_hidden_size", type=int, default=None)
    parser.add_argument("--spectral_modes", type=int, default=None,
                        help="Number of modes used by SpectralConv/FNO branches; default preserves the full top_k_freqs setting.")
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--prediction_target", type=str, default=None, choices=["x_0", "v", "noise"])
    parser.add_argument("--bridge_scaling", type=str, default=None, choices=["none", "unit_max", "full"])
    parser.add_argument("--coord_scale", type=float, default=None)
    parser.add_argument("--use_seq_conditioning", action="store_true", default=None,
                        help="spectral_dit / spectral_conv_dit only: add learned residue-type embeddings per residue plus a pooled global sequence summary.")
    parser.add_argument("--seq_embed_dim", type=int, default=None,
                        help="spectral_dit / spectral_conv_dit only: residue embedding width for sequence conditioning.")
    parser.add_argument("--use_ss_conditioning", action="store_true", default=None,
                        help="spectral_dit / spectral_conv_dit only: add DSSP-based secondary-structure conditioning.")
    parser.add_argument("--ss_embed_dim", type=int, default=None,
                        help="spectral_dit / spectral_conv_dit only: latent width for DSSP conditioning.")
    parser.add_argument("--temporal_ablation_mode", type=str, default=None,
                        choices=["normal", "off", "freq_noise"],
                        help="fno2 only: ablate the physical coordinate DCT-FNO path.")
    parser.add_argument("--block_ablation_mode", type=str, default=None,
                        choices=["normal", "no_mlp", "temporal_only"],
                        help="fno2 only: remove shortcut branches inside FNO2 blocks.")
    parser.add_argument("--temporal_gate_init", type=float, default=None,
                        help="fno2 only: initial AdaLN gate bias for physical coordinate DCT-FNO branch.")

    # dual_branch (v11)
    parser.add_argument("--K_slow", type=int, default=None,
                        help="dual_branch/v12b: number of modes handled by the slow branch.")
    parser.add_argument("--slow_mode_start", type=int, default=None,
                        help="v12b: first spectral mode receiving the slow residual; use 1 to skip DC.")
    parser.add_argument("--slow_d_model", type=int, default=None,
                        help="dual_branch: slow-branch hidden width.")
    parser.add_argument("--slow_depth", type=int, default=None,
                        help="dual_branch: slow-branch transformer depth.")
    parser.add_argument("--slow_num_heads", type=int, default=None,
                        help="dual_branch: slow-branch attention heads.")
    parser.add_argument("--slow_mlp_ratio", type=float, default=None,
                        help="dual_branch: slow-branch SwiGLU expansion ratio.")
    parser.add_argument("--slow_attn_dropout", type=float, default=None,
                        help="dual_branch: slow-branch attention dropout.")
    parser.add_argument("--slow_use_rmsf_prior", action="store_true", default=None,
                        help="dual_branch: feed NMA RMSF prior into the slow branch.")
    parser.add_argument("--slow_predicts_amplitude_only", action="store_true", default=None,
                        help="dual_branch: make the slow branch predict amplitudes and reconstruct vectors along noisy low-k directions.")
    parser.add_argument("--fast_cond_dim", type=int, default=None,
                        help="dual_branch: fast-branch AdaLN conditioning width.")
    parser.add_argument("--fast_band_edges", type=str, default=None,
                        help="block_mix/v12/dual_branch: frequency mixer bands. Use 'legacy' for old 0-8-32-128-K checkpoints or e.g. 'DC,1-8,9-32,33-128,129+'.")
    parser.add_argument("--cascade_band_edges", type=str, default=None,
                        help="cascade: frequency edges/ranges, e.g. '0,1,9,33,129,256' or 'DC,1-8,9-32,33+'.")
    parser.add_argument("--cascade_context_mode", type=str, default=None,
                        choices=["idct_summary", "none"],
                        help="cascade: append IDCT low-pass mean/RMS context to native conditioning.")
    parser.add_argument("--cascade_detach_context", action=argparse.BooleanOptionalAction, default=None,
                        help="cascade: detach previous-stage predictions before feeding later-stage context.")
    parser.add_argument("--cascade_dc_depth", type=int, default=None,
                        help="cascade: override DC transformer depth.")
    parser.add_argument("--cascade_low_depth", type=int, default=None,
                        help="cascade: override low-k spectral transformer depth.")
    parser.add_argument("--cascade_high_depth", type=int, default=None,
                        help="cascade: override spectral-conv stage depth.")
    parser.add_argument("--loss_slow_weight", type=float, default=None,
                        help="dual_branch: weight on per-branch slow MSE in the three-term loss.")
    parser.add_argument("--loss_fast_weight", type=float, default=None,
                        help="dual_branch: weight on per-branch fast MSE in the three-term loss.")
    parser.add_argument("--loss_total_weight", type=float, default=None,
                        help="dual_branch: small concat-consistency weight on the full-spectrum MSE.")
    parser.add_argument("--amp_head_context_modes", type=int, default=None,
                        help="v12a: number of low-k modes supplied to the amplitude head as context.")
    parser.add_argument("--amp_head_target_modes", type=int, default=None,
                        help="v12a: number of lowest modes amplitude-calibrated by the head.")
    parser.add_argument("--amp_head_d_model", type=int, default=None,
                        help="v12a: amplitude-head hidden width.")
    parser.add_argument("--amp_head_depth", type=int, default=None,
                        help="v12a: amplitude-head transformer depth.")
    parser.add_argument("--amp_head_num_heads", type=int, default=None,
                        help="v12a: amplitude-head attention heads.")
    parser.add_argument("--amp_head_mlp_ratio", type=float, default=None,
                        help="v12a: amplitude-head SwiGLU expansion ratio.")
    parser.add_argument("--amp_head_attn_dropout", type=float, default=None,
                        help="v12a: amplitude-head attention dropout.")
    parser.add_argument("--amp_head_use_rmsf_prior", action="store_true", default=None,
                        help="v12a: feed the RMSF prior into the amplitude head.")
    parser.add_argument("--use_shake", action="store_true", default=None,
                        help="v12a/v12c: apply differentiable SHAKE to reconstructed CA coords.")
    parser.add_argument("--shake_n_iter", type=int, default=None,
                        help="v12a/v12c: SHAKE iterations.")
    parser.add_argument("--shake_target", type=float, default=None,
                        help="v12a/v12c: ideal CA-CA bond length in Angstroms.")
    parser.add_argument("--refiner_hidden", type=int, default=None,
                        help="v12c: CA-coord refiner hidden width.")
    parser.add_argument("--refiner_depth", type=int, default=None,
                        help="v12c: CA-coord refiner depth.")
    parser.add_argument("--refiner_kernel_size", type=int, default=None,
                        help="v12c: CA-coord refiner conv kernel size.")
    parser.add_argument("--refiner_max_delta", type=float, default=None,
                        help="v12c: cap per-coordinate refiner residual in Angstroms; <=0 disables the cap.")
    parser.add_argument("--use_spectral_graph_refiner", action=argparse.BooleanOptionalAction, default=None,
                        help="v17a: enable native-graph absolute low-mode spectral correction.")
    parser.add_argument("--spectral_graph_refiner_modes", type=int, default=None,
                        help="v17a: number of low DCT modes refined by the graph module.")
    parser.add_argument("--spectral_graph_refiner_hidden", type=int, default=None,
                        help="v17a: graph refiner hidden width.")
    parser.add_argument("--spectral_graph_refiner_depth", type=int, default=None,
                        help="v17a: graph message-passing depth.")
    parser.add_argument("--spectral_graph_refiner_msg_hidden", type=int, default=None,
                        help="v17a: optional graph edge-message hidden width.")
    parser.add_argument("--spectral_graph_refiner_sequence_window", type=int, default=None,
                        help="v17a: sequence-neighbour window radius.")
    parser.add_argument("--spectral_graph_refiner_knn", type=int, default=None,
                        help="v17a: native-structure kNN degree.")
    parser.add_argument("--spectral_graph_refiner_use_sequence_edges", action=argparse.BooleanOptionalAction, default=None,
                        help="v17a: include sequence-neighbour graph edges.")
    parser.add_argument("--spectral_graph_refiner_use_native_knn", action=argparse.BooleanOptionalAction, default=None,
                        help="v17a: include native-structure kNN graph edges.")
    parser.add_argument("--spectral_graph_refiner_max_delta", type=float, default=None,
                        help="v17a: tanh cap on spectral correction; <=0 disables the cap.")
    parser.add_argument("--use_v12e_spec_graph", action=argparse.BooleanOptionalAction, default=None,
                        help="v12e: enable sparse residue graph residual on spectral coefficients.")
    parser.add_argument("--v12e_spec_graph_k_min", type=int, default=None,
                        help="v12e: first DCT mode refined by the graph, inclusive.")
    parser.add_argument("--v12e_spec_graph_k_max", type=int, default=None,
                        help="v12e: final DCT mode refined by the graph, exclusive.")
    parser.add_argument("--v12e_spec_graph_hidden", type=int, default=None,
                        help="v12e: graph refiner hidden width.")
    parser.add_argument("--v12e_spec_graph_depth", type=int, default=None,
                        help="v12e: graph message-passing depth.")
    parser.add_argument("--v12e_spec_graph_msg_hidden", type=int, default=None,
                        help="v12e: optional graph edge-message hidden width.")
    parser.add_argument("--v12e_spec_graph_sequence_window", type=int, default=None,
                        help="v12e: sequence-neighbour window radius.")
    parser.add_argument("--v12e_spec_graph_knn", type=int, default=None,
                        help="v12e: native-structure kNN degree.")
    parser.add_argument("--v12e_spec_graph_use_sequence_edges", action=argparse.BooleanOptionalAction, default=None,
                        help="v12e: include sequence-neighbour graph edges.")
    parser.add_argument("--v12e_spec_graph_use_native_knn", action=argparse.BooleanOptionalAction, default=None,
                        help="v12e: include native-structure kNN graph edges.")
    parser.add_argument("--v12e_spec_graph_max_delta", type=float, default=None,
                        help="v12e: tanh cap on spectral residual; <=0 disables the cap.")
    parser.add_argument("--v12e_spec_graph_residual_lambda", type=float, default=None,
                        help="v12e: optional L2 penalty on graph spectral residual.")
    parser.add_argument("--use_bond_spectral_graph_refiner", action=argparse.BooleanOptionalAction, default=None,
                        help="v17b: enable adjacent-bond spectral graph correction.")
    parser.add_argument("--bond_spectral_graph_refiner_modes", type=int, default=None,
                        help="v17b: number of low DCT modes refined by the bond graph module.")
    parser.add_argument("--bond_spectral_graph_refiner_hidden", type=int, default=None,
                        help="v17b: bond graph refiner hidden width.")
    parser.add_argument("--bond_spectral_graph_refiner_depth", type=int, default=None,
                        help="v17b: bond graph message-passing depth.")
    parser.add_argument("--bond_spectral_graph_refiner_msg_hidden", type=int, default=None,
                        help="v17b: optional bond graph edge-message hidden width.")
    parser.add_argument("--bond_spectral_graph_refiner_sequence_window", type=int, default=None,
                        help="v17b: bond-sequence graph window radius.")
    parser.add_argument("--bond_spectral_graph_refiner_knn", type=int, default=None,
                        help="v17b: native bond-midpoint kNN degree.")
    parser.add_argument("--bond_spectral_graph_refiner_use_sequence_edges", action=argparse.BooleanOptionalAction, default=None,
                        help="v17b: include sequence-neighbour bond graph edges.")
    parser.add_argument("--bond_spectral_graph_refiner_use_native_knn", action=argparse.BooleanOptionalAction, default=None,
                        help="v17b: include native bond-midpoint kNN graph edges.")
    parser.add_argument("--bond_spectral_graph_refiner_max_delta", type=float, default=None,
                        help="v17b: tanh cap on bond spectral correction; <=0 disables the cap.")
    parser.add_argument("--bond_spectral_graph_refiner_blend", type=float, default=None,
                        help="v17b: blend factor for integrating corrected bond spectra back to residues.")
    parser.add_argument("--egnn_h_dim", type=int, default=None,
                        help="v12d: EGNN refiner node feature width.")
    parser.add_argument("--egnn_hidden", type=int, default=None,
                        help="v12d: EGNN refiner edge MLP hidden width.")
    parser.add_argument("--egnn_depth", type=int, default=None,
                        help="v12d: EGNN refiner number of layers.")
    parser.add_argument("--egnn_seq_window", type=int, default=None,
                        help="v12d: EGNN edge sequence-window radius (|i-j| <= W).")
    parser.add_argument("--egnn_max_len", type=int, default=None,
                        help="v12d: max residue index for the EGNN positional embedding.")
    parser.add_argument("--egnn_t_chunk", type=int, default=None,
                        help="v12d: frames processed per refiner chunk (caps activation memory).")

    # Optimizer
    parser.add_argument("--optimizer", type=str, default=None, choices=["adamw", "sgd"])
    parser.add_argument("--max_lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)

    # Diffusion
    parser.add_argument("--schedule", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--shift_value", type=str, default=None,
                        help="Log-SNR shift. Use a float or 'auto' for targeted noise schedules.")
    parser.add_argument("--min_snr_gamma", type=float, default=None)
    parser.add_argument("--aniso_gamma", type=float, default=None,
                        help="Anisotropic noise exponent (0.0=off, 0.3=recommended). "
                             "Default None=isotropic.")
    parser.add_argument("--noise_schedule", type=str, default=None,
                        help="High-level noise preset, e.g. freq_scale_gamma, cosine_low_first_targeted.")
    parser.add_argument("--noise_space", type=str, default=None,
                        choices=["raw_gamma", "model_normalized"],
                        help="Interpretation space for custom noise schedules.")
    parser.add_argument("--noise_band_edges", type=str, default=None,
                        help="Frequency grouping for custom schedules, e.g. 'DC,1-8,9-32,33-128,129+'.")
    parser.add_argument("--noise_group_model_multipliers", type=str, default=None,
                        help="Comma-separated per-band model-space noise multipliers.")
    parser.add_argument("--noise_target_crossings", type=str, default=None,
                        help="Comma-separated per-band forward SNR=1 crossing timesteps.")
    parser.add_argument("--noise_anchor_band", type=str, default=None,
                        help="Band name/index used to tune shift_value='auto', e.g. 'k9-32'.")
    parser.add_argument("--noise_power_normalization", type=str, default=None,
                        choices=["raw_mean_square"],
                        help="How to normalise custom raw aniso weights.")
    parser.add_argument("--noise_auto_shift", action=argparse.BooleanOptionalAction, default=None,
                        help="Tune global shift to the anchor band even when shift_value is numeric.")
    parser.add_argument("--use_hilbert_spatial", action="store_true", default=None,
                        help="spectral_conv_dit only: inject FFT-based Hilbert spatial envelope in Phase 2 (legacy).")
    parser.add_argument("--use_hilbert_spatial_dct", action="store_true", default=None,
                        help="spectral_conv_dit only: inject DCT-based boundary-safe Hilbert spatial envelope in Phase 2.")
    parser.add_argument("--hilbert_mode", type=str, default=None,
                        choices=["every_block", "every_3_blocks", "input_only", "off"],
                        help="spectral_conv_dit only: which blocks receive Hilbert injection.")
    parser.add_argument("--rmsf_prior_path", type=str, default=None,
                        help="Path to the NMA RMSF prior sidecar .pt produced by scripts/precompute_nma_tica.py.")
    parser.add_argument("--use_rmsf_prior_gain", action="store_true", default=None,
                        help="spectral_conv_dit only: apply per-residue NMA RMSF prior gain on the model output (Mechanism A).")
    parser.add_argument("--use_low_k_correction_head", action="store_true", default=None,
                        help="spectral_conv_dit / spectral_dit_low_k: add a zero-init additive correction head for the first low_k_correction_modes modes.")
    parser.add_argument("--low_k_correction_modes", type=str, default=None,
                        help="Low-k correction spec. Legacy ints still mean the first n modes; string forms support 'DC', '0-4', or 'DC,1-4'.")
    parser.add_argument("--num_ode_steps", type=int, default=None)

    # Data
    parser.add_argument("--max_domains", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--samples_per_traj", type=int, default=None)
    parser.add_argument("--randomize_train_windows", dest="randomize_train_windows",
                        action="store_true", default=None,
                        help="Resample temporal windows freshly for training dataset fetches.")
    parser.add_argument("--deterministic_train_windows", dest="randomize_train_windows",
                        action="store_false",
                        help="Use deterministic seeded temporal windows for training.")
    parser.add_argument("--dataloader_num_workers", type=int, default=None,
                        help="Training DataLoader workers per DDP rank. Default: 16 for Zarr, 6 for HDF5.")
    parser.add_argument("--dataloader_timeout", type=int, default=None,
                        help="Training DataLoader timeout in seconds. Increase on shared filesystems.")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=None,
                        help="Training DataLoader prefetch factor per worker.")
    parser.add_argument("--dataloader_persistent_workers", action=argparse.BooleanOptionalAction, default=None,
                        help="Keep training DataLoader workers alive across epochs.")
    parser.add_argument("--smoothing", type=str, default=None)
    parser.add_argument("--smoothing_sigma", type=float, default=None)
    parser.add_argument("--smooth_start_freq", type=float, default=None)
    parser.add_argument("--shelf_value", type=float, default=None)
    parser.add_argument("--freq_weighting", type=str, default=None)
    parser.add_argument("--bending_lambda", type=float, default=None)
    parser.add_argument("--geo_loss", type=str, default=None,
                        help=("Comma-delimited geometry auxiliary losses. "
                              "Options: idct_ca-ca (legacy), spec_geo, risk_band. "
                              "Example: spec_geo,risk_band. Alias risk_bond is accepted."))
    parser.add_argument("--geometry_lambda", type=float, default=None,
                        help="Weight for local sequential geometry loss (spectral models). "
                             "Penalises predicted CA inter-residue distances at separations 1..5 "
                             "vs ground truth. sep=1 implicitly covers CA-CA bonds.")
    parser.add_argument("--geometry_warmup_start", type=int, default=None,
                        help="Epoch at which geometry loss begins to ramp up (default: 50).")
    parser.add_argument("--geometry_warmup_epochs", type=int, default=None,
                        help="Number of epochs over which geometry loss ramps from 0 to full weight (default: 10).")
    parser.add_argument("--geometry_decay_start", type=int, default=None,
                        help="Epoch at which geometry loss begins to decay from 1.0 toward min (default: 200).")
    parser.add_argument("--geometry_decay_epochs", type=int, default=None,
                        help="Number of epochs over which geometry loss decays to min_factor=0.1 (default: 200).")
    parser.add_argument("--geometry_tol", type=float, default=None,
                        help="Tolerance in Angstroms for local sequential geometry loss.")
    parser.add_argument("--clash_lambda", type=float, default=None,
                        help="Weight on non-bonded CA clash hinge inside local geometry loss.")
    parser.add_argument("--clash_threshold", type=float, default=None,
                        help="Non-bonded CA distance threshold in Angstroms below which clash loss activates.")
    parser.add_argument("--clash_max_pairs", type=int, default=None,
                        help="Maximum non-bonded residue pairs sampled per batch for clash loss.")
    parser.add_argument("--clash_pair_chunk", type=int, default=None,
                        help="Chunk size for sampled non-bonded clash pairs.")
    parser.add_argument("--topology_margin_path", type=str, default=None,
                        help="Optional topology lower-bound artifact from compute_freq_stats_conditioned_dct.py.")
    parser.add_argument("--spectral_geo_segment_threshold", type=float, default=None,
                        help="Fallback segment-segment lower-bound threshold for spec_geo/risk_band.")
    parser.add_argument("--spectral_geo_max_segment_pairs", type=int, default=None,
                        help="Maximum sampled non-adjacent segment pairs for spec_geo.")
    parser.add_argument("--risk_band_max_pairs", type=int, default=None,
                        help="Maximum sampled nonlocal CA pairs for risk_band.")
    parser.add_argument("--risk_band_max_segment_pairs", type=int, default=None,
                        help="Maximum sampled non-adjacent segment pairs for risk_band.")
    parser.add_argument("--rmsf_lambda", type=float, default=None,
                        help="Weight for per-residue RMSF spectral power loss (default: 0.0=disabled).")
    parser.add_argument("--rmsf_warmup_start", type=int, default=None,
                        help="Epoch at which RMSF loss begins to ramp up (default: 100).")
    parser.add_argument("--rmsf_warmup_epochs", type=int, default=None,
                        help="Number of epochs over which RMSF loss ramps from 0 to full weight (default: 10).")
    parser.add_argument("--low_freq_lambda", type=float, default=None,
                        help="Weight for explicit signed low-frequency x_0 loss on the first low_freq_modes modes.")
    parser.add_argument("--low_freq_modes", type=int, default=None,
                        help="Number of lowest spectral modes supervised by low_freq_lambda (default: 8).")
    parser.add_argument("--dc_lambda", type=float, default=None,
                        help="Weight for DC-only clean-spectrum MSE on k=0. Targets the additive low-k head when enabled.")
    parser.add_argument("--dc_start_epoch", type=int, default=None,
                        help="Epoch at which the DC-only auxiliary loss activates (default: 10).")
    parser.add_argument("--v17_aux_modes", type=int, default=None,
                        help="v17 only: number of low DCT modes used by v17 auxiliary losses.")
    parser.add_argument("--v17_low_mode_lambda", type=float, default=None,
                        help="v17 only: weight for signed low-mode spectral MSE.")
    parser.add_argument("--v17_adjacent_lambda", type=float, default=None,
                        help="v17 only: weight for adjacent-residue spectral-difference MSE.")
    parser.add_argument("--v17_idct_bond_lambda", type=float, default=None,
                        help="v17 only: weight for IDCT low-mode bond-vector MSE.")
    parser.add_argument("--v17_caca_tolerance_lambda", type=float, default=None,
                        help="v17 only: weight for low-mode CA-CA tolerance penalty.")
    parser.add_argument("--v17_caca_target", type=float, default=None,
                        help="v17 only: target CA-CA distance for tolerance penalty.")
    parser.add_argument("--v17_caca_tolerance", type=float, default=None,
                        help="v17 only: tolerance around v17_caca_target.")
    parser.add_argument("--max_bad_update_streak", type=int, default=None,
                        help="Abort after this many consecutive skipped optimizer updates from NaN loss/grad.")
    parser.add_argument("--max_bad_update_total", type=int, default=None,
                        help="Abort after this many total skipped optimizer updates from NaN loss/grad.")
    parser.add_argument("--debug_nonfinite_hooks", action=argparse.BooleanOptionalAction, default=None,
                        help="Install forward/backward hooks to log the first non-finite activation or activation gradient.")
    parser.add_argument("--debug_nonfinite_forward", action=argparse.BooleanOptionalAction, default=None,
                        help="When debug_nonfinite_hooks is enabled, check leaf-module forward outputs.")
    parser.add_argument("--debug_nonfinite_backward", action=argparse.BooleanOptionalAction, default=None,
                        help="When debug_nonfinite_hooks is enabled, check leaf-module backward grad inputs/outputs.")
    parser.add_argument("--debug_nonfinite_filter", type=str, default=None,
                        help="Optional comma-delimited substrings limiting monitored module names.")
    parser.add_argument("--debug_nonfinite_max_modules", type=int, default=None,
                        help="Optional cap on monitored leaf modules; 0 means no cap.")
    parser.add_argument("--conditioning_dropout", action="store_true", default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--atlas_stride", type=int, default=None)

    # Paths
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--mdcath_path", type=str, default=None)
    parser.add_argument("--atlas_path", type=str, default=None)
    parser.add_argument("--freq_scales_path", type=str, default=None)
    parser.add_argument("--aniso_scales_path", type=str, default=None)
    parser.add_argument("--mdcath_zarr_path", type=str, default=None)
    parser.add_argument("--atlas_zarr_path", type=str, default=None)
    parser.add_argument("--split_ids_dir", type=str, default=None,
                        help="Optional directory containing train_ids.txt / val_ids.txt / test_ids.txt to use for splitting, even without resume_from_latest.")

    # Workflow
    parser.add_argument("--offline_mode", action="store_true", default=None)
    parser.add_argument("--crop", action="store_true", default=None)
    parser.add_argument("--trim_cache", action="store_true", default=None)
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--resume_from_latest", action="store_true", default=False)
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override DDP find_unused_parameters. Defaults to False; external refiners use static_graph instead.",
    )
    parser.add_argument(
        "--ddp_static_graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override DDP static_graph. Defaults to True for models with post-IDCT refiners.",
    )
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()

    init_process()
    rank = dist.get_rank()

    # Config: YAML base + CLI overrides
    config = {}
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(args.config)
        raw = yaml.safe_load(open(args.config)) or {}
        for sec_k, sec_v in raw.items():
            if isinstance(sec_v, dict):
                config.update(sec_v)
            else:
                config[sec_k] = sec_v

    # Type coercions for YAML
    for float_key in ["max_lr", "min_snr_gamma", "smoothing_sigma", "shift_value", "guidance_scale", "shelf_value", "coord_scale", "bending_lambda", "geometry_lambda", "geometry_tol", "clash_lambda", "clash_threshold", "spectral_geo_segment_threshold", "aniso_gamma", "rmsf_lambda", "low_freq_lambda", "dc_lambda", "temporal_gate_init",
                      "representation_length_min", "representation_length_max", "representation_length_residual_max", "representation_barrier_lambda",
                      "slow_mlp_ratio", "slow_attn_dropout",
                      "amp_head_mlp_ratio", "amp_head_attn_dropout",
                      "loss_slow_weight", "loss_fast_weight", "loss_total_weight",
                      "refiner_max_delta",
                      "spectral_graph_refiner_max_delta",
                      "v12e_spec_graph_max_delta", "v12e_spec_graph_residual_lambda",
                      "bond_spectral_graph_refiner_max_delta",
                      "bond_spectral_graph_refiner_blend",
                      "v17_low_mode_lambda", "v17_adjacent_lambda",
                      "v17_idct_bond_lambda", "v17_caca_tolerance_lambda",
                      "v17_caca_target", "v17_caca_tolerance"]:
        if float_key in config and config[float_key] is not None:
            if float_key == "shift_value" and isinstance(config[float_key], str) and config[float_key].strip().lower() == "auto":
                continue
            config[float_key] = float(config[float_key])
    for int_key in ["epochs", "batch_size", "top_k_freqs", "hidden_dim", "freq_hidden_size",
                     "spectral_modes", "hidden_size", "jepa_latent_dim",
                     "seq_embed_dim",
                     "ss_embed_dim",
                     "num_layers", "num_heads", "num_steps",
                     "num_ode_steps", "max_val_batches",
                     "dataloader_num_workers", "dataloader_timeout",
                     "dataloader_prefetch_factor",
                     "geometry_warmup_start", "geometry_warmup_epochs",
                     "geometry_decay_start", "geometry_decay_epochs",
                     "clash_max_pairs", "clash_pair_chunk",
                     "spectral_geo_max_segment_pairs", "risk_band_max_pairs", "risk_band_max_segment_pairs",
                     "rmsf_warmup_start", "rmsf_warmup_epochs", "low_freq_modes",
                     "dc_start_epoch",
                     "K_slow", "slow_mode_start", "slow_d_model", "slow_depth", "slow_num_heads",
                     "cascade_dc_depth", "cascade_low_depth", "cascade_high_depth",
                     "fast_cond_dim", "amp_head_context_modes", "amp_head_target_modes",
                     "amp_head_d_model", "amp_head_depth", "amp_head_num_heads",
                     "spectral_graph_refiner_modes", "spectral_graph_refiner_hidden",
                     "spectral_graph_refiner_depth", "spectral_graph_refiner_msg_hidden",
                     "spectral_graph_refiner_sequence_window", "spectral_graph_refiner_knn",
                     "v12e_spec_graph_k_min", "v12e_spec_graph_k_max",
                     "v12e_spec_graph_hidden", "v12e_spec_graph_depth",
                     "v12e_spec_graph_msg_hidden", "v12e_spec_graph_sequence_window",
                     "v12e_spec_graph_knn",
                     "bond_spectral_graph_refiner_modes", "bond_spectral_graph_refiner_hidden",
                     "bond_spectral_graph_refiner_depth", "bond_spectral_graph_refiner_msg_hidden",
                     "bond_spectral_graph_refiner_sequence_window", "bond_spectral_graph_refiner_knn",
                     "v17_aux_modes",
                     "max_bad_update_streak", "max_bad_update_total",
                     "debug_nonfinite_max_modules",
                     "egnn_h_dim", "egnn_hidden", "egnn_depth", "egnn_seq_window",
                     "egnn_max_len", "egnn_t_chunk"]:
        if int_key in config and config[int_key] is not None:
            config[int_key] = int(config[int_key])
    if "low_k_correction_modes" in config and config["low_k_correction_modes"] is not None:
        value = config["low_k_correction_modes"]
        if isinstance(value, str):
            value = value.strip()
            if value.isdigit():
                value = int(value)
        config["low_k_correction_modes"] = value

    # Date-prefix checkpoint dir
    if "checkpoint_dir" in config:
        base = config["checkpoint_dir"]
        parent = os.path.dirname(base)
        name = os.path.basename(base)
        config["checkpoint_dir"] = os.path.join(parent, f"{date_str}_{name}")

    # CLI overrides
    cli_updates = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    config.update(cli_updates)
    if config.get("randomize_train_windows") is None:
        config["randomize_train_windows"] = True
    if config.get("representation") is None:
        config["representation"] = "displacement" if bool(config.get("displacement", False)) else "raw_coords"
    config["representation"] = canonical_representation(config["representation"])
    config["displacement"] = config["representation"] == "displacement"
    config["geo_loss"] = ",".join(parse_geo_loss_modes(config.get("geo_loss", "idct_ca-ca")))
    config["freq_normalization"] = canonical_freq_normalization(config.get("freq_normalization", "auto"))
    config["dc_residualization"] = canonical_dc_residualization(config.get("dc_residualization", "auto"))
    config["aniso_source"] = canonical_aniso_source(config.get("aniso_source", "auto"))

    checkpoint_dir = config.get("checkpoint_dir")
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir must be provided via CLI or config.yaml")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Run name
    if config.get("run_name") is None:
        run_name = f"{date_str}_top_k{config.get('top_k_freqs', '?')}"
    else:
        run_name = f"{date_str}_{config['run_name']}"
    if args.test_only:
        run_name = f"TEST_{run_name}"
    config["run_name"] = run_name

    # Wandb
    if rank == 0:
        print("Run Name:", run_name)
        wandb.login()

        wandb_id = None
        latest_ckpt = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
        if args.resume_from_latest and os.path.exists(latest_ckpt):
            ckpt = torch.load(latest_ckpt, map_location="cpu")
            wandb_id = ckpt.get("wandb_id")

        init_wandb(
            project_name="pancakes_spectral_diffusion",
            run_name=run_name, config=config,
            id=wandb_id, resume="allow",
        )

        with open(os.path.join(checkpoint_dir, "run_config.yaml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print("Final config:")
        pprint(config, sort_dicts=True, indent=4)

    try:
        test_metrics = TRAIN_DISTRIBUTED(**config)
        if rank == 0:
            print("Final Metrics:", test_metrics)
    except Exception as e:
        print(f"Rank {rank} failed: {e}")
        raise
    finally:
        if rank == 0:
            wandb.finish()
        dist.destroy_process_group()

    print("FINISHED")
