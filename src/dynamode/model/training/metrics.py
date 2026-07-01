'''
Collection of funcs for wandb and print val, train and debug metrics logging.
'''


import numpy as np
import torch
import torch.distributed as dist



def safe_vector_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-4) -> torch.Tensor:
    '''Vector norm with a finite gradient at exactly-zero vectors.'''
    return x.square().sum(dim=dim).clamp_min(float(eps) ** 2).sqrt()


def compute_batch_caca_dist(ca_coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    '''Mean adjacent CA-CA distance over valid residues and frames.'''
    if ca_coords.ndim != 4 or ca_coords.shape[-1] < 3 or ca_coords.shape[2] < 2:
        return ca_coords.new_tensor(float("nan"))
    ca = ca_coords[..., :3]
    pair_mask = (mask[:, 1:] & mask[:, :-1]).to(dtype=ca.dtype)
    if not bool(pair_mask.any()):
        return ca.new_tensor(float("nan"))
    dist_ca = safe_vector_norm(ca[:, :, 1:] - ca[:, :, :-1], dim=-1)
    valid = pair_mask.unsqueeze(1).expand_as(dist_ca)
    return (dist_ca * valid).sum() / valid.sum().clamp_min(1.0)


def compute_batch_lddt(
    pred_coords: torch.Tensor,
    target_coords: torch.Tensor,
    mask: torch.Tensor,
    *,
    cutoff: float = 15.0,
    thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    frame_chunk: int = 8,
) -> torch.Tensor:
    '''Approximate CA lDDT over valid non-self pairs.

    The implementation chunks frames to avoid materialising the full
    (B, T, L, L) distance tensor for long validation crops.
    '''
    if pred_coords.shape != target_coords.shape:
        raise ValueError(
            f"pred/target shape mismatch: {tuple(pred_coords.shape)} vs {tuple(target_coords.shape)}"
        )
    pred_ca = pred_coords[..., :3]
    target_ca = target_coords[..., :3]
    B, T, L, _ = pred_ca.shape
    if L < 2:
        return pred_ca.new_tensor(float("nan"))

    residue_valid = mask.bool()
    pair_valid_base = residue_valid[:, :, None] & residue_valid[:, None, :]
    eye = torch.eye(L, device=mask.device, dtype=torch.bool).unsqueeze(0)
    pair_valid_base = pair_valid_base & ~eye

    score_sum = pred_ca.new_tensor(0.0)
    score_count = pred_ca.new_tensor(0.0)
    for start in range(0, T, max(int(frame_chunk), 1)):
        end = min(start + max(int(frame_chunk), 1), T)
        pred_dist = torch.cdist(pred_ca[:, start:end], pred_ca[:, start:end])
        true_dist = torch.cdist(target_ca[:, start:end], target_ca[:, start:end])
        valid = pair_valid_base[:, None] & (true_dist < float(cutoff))
        if not bool(valid.any()):
            continue
        err = (pred_dist - true_dist).abs()
        frame_score = torch.zeros_like(err)
        for threshold in thresholds:
            frame_score = frame_score + (err < float(threshold)).to(err.dtype)
        frame_score = frame_score / float(len(thresholds))
        score_sum = score_sum + frame_score[valid].sum()
        score_count = score_count + valid.sum().to(dtype=score_sum.dtype)
    return score_sum / score_count.clamp_min(1.0)


class RamaValidator:
    '''Small Ramachandran histogram accumulator for validation diagnostics.'''

    def __init__(self, bins: int = 64, device: str | torch.device = "cpu") -> None:
        self.bins = int(bins)
        self.device = torch.device(device)
        self.hist_pred = torch.zeros((self.bins, self.bins), device=self.device)
        self.hist_gt = torch.zeros((self.bins, self.bins), device=self.device)

    def _accumulate(self, hist: torch.Tensor, phi: torch.Tensor, psi: torch.Tensor) -> None:
        phi_idx = (((phi + 180.0) / 360.0) * self.bins).floor().long().clamp(0, self.bins - 1)
        psi_idx = (((psi + 180.0) / 360.0) * self.bins).floor().long().clamp(0, self.bins - 1)
        valid = torch.isfinite(phi) & torch.isfinite(psi)
        if not bool(valid.any()):
            return
        flat_idx = phi_idx[valid] * self.bins + psi_idx[valid]
        hist.view(-1).scatter_add_(
            0,
            flat_idx,
            torch.ones_like(flat_idx, dtype=hist.dtype),
        )

    def update(
        self,
        phi_pred: torch.Tensor,
        psi_pred: torch.Tensor,
        phi_gt: torch.Tensor,
        psi_gt: torch.Tensor,
    ) -> None:
        self._accumulate(self.hist_pred, phi_pred.to(self.device), psi_pred.to(self.device))
        self._accumulate(self.hist_gt, phi_gt.to(self.device), psi_gt.to(self.device))

    def compute(self, is_distributed: bool = False) -> dict[str, float]:
        if is_distributed:
            dist.all_reduce(self.hist_pred, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.hist_gt, op=dist.ReduceOp.SUM)
        if self.hist_pred.sum() <= 0 or self.hist_gt.sum() <= 0:
            return {}
        eps = 1e-8
        p = self.hist_pred / self.hist_pred.sum().clamp_min(eps)
        q = self.hist_gt / self.hist_gt.sum().clamp_min(eps)
        m = 0.5 * (p + q)
        jsd = 0.5 * (
            (p * ((p + eps).log() - (m + eps).log())).sum()
            + (q * ((q + eps).log() - (m + eps).log())).sum()
        )
        return {"rama_global_JSD": float(jsd.detach().item())}


def compute_caca_distance_stats(ca_coords: torch.Tensor, mask: torch.Tensor):
    '''Summarise consecutive CA-CA distances for a batch of trajectories.'''
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
    '''Count non-bonded CA-CA clashes per trajectory.

    count_per_traj is the number of non-bonded pair-frame distances below
    threshold summed across all frames in each trajectory.
    '''
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
    '''Collect per-temperature CA-CA and cheap sampled clash debug stats.'''
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
    '''Create an accumulator for per-temperature CA-CA validation stats.'''
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
    '''Create an accumulator for per-temperature validation clash counts.'''
    n_temps = len(tracked_temps)
    return {
        "tracked_temps": tuple(float(t) for t in tracked_temps),
        "count": torch.zeros((n_temps, 2), device=device),
        "traj_count": torch.zeros((n_temps, 2), device=device),
        "pair_frame_count": torch.zeros((n_temps, 2), device=device),
    }


def update_temp_caca_accumulator(accumulator, raw_temps, pred_ca, target_ca, mask):
    '''Accumulate per-temperature consecutive CA-CA stats.'''
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
    '''Accumulate exact non-bonded clash counts per validation trajectory.'''
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
    '''Convert accumulated CA-CA stats into W&B-friendly validation metrics.'''
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
    '''Convert validation clash accumulators into W&B-friendly metrics.'''
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
    '''Band-wise signed coefficient and amplitude diagnostics.'''
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

# PIPELINE DIAGNOSTICS
@torch.no_grad()
def debug_print_pipeline(batch, transform_engine, real_model, device, top_k_freqs, displacement, label="DEBUG"):
    '''One-shot sanity check. Call at step 0 and first validation batch.

    Assumes coordinates are in collapsed channel form:
      CA: (B, T, L, 3)
      BB: (B, T, L, 12)

    Prints five checks:
      1. Displacement active — raw vs displaced coordinate range.
      2. Stable core centering — native_coords mean should be ≈ [0,0,0].
      3. Spectral k=0 amplitude — displaced DCT DC should be << raw DCT DC.
      4. Internal normalisation — spectral volume std should move towards ~1.
      5. Masking — invalid (padded) residues should have zero spectral energy.
    '''
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