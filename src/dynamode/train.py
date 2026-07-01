'''
Public DynaMode training loop for the supported spectral diffusion models.

The public repository trains only the transformer baseline
(`spectral_dit_low_k`) and the default SpecConv model
(`spectral_conv_block_mix_amplitude`). Coordinate encoding, DCT/DFT transforms,
normalisation, anisotropic scale lookup, and DC residualisation all flow through
dynamode.spectral.representation.SpectralRepresentationPipeline.
'''

import os
import math
import inspect
import contextlib
import shutil
from datetime import datetime, timedelta
from pprint import pprint
import random

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
import wandb

load_dotenv()

from dynamode.dataloader.zarr_loader import ZarrTrajectoriesDataset, FeaturizerWindowZarr
from dynamode.dataloader.raw_loader import TrajectoriesDataset
from dynamode.dataloader.features import FeaturizerWindow, Aligner
from dynamode.spectral.representation import (
    CoordinateRepresentation,
    canonical_aniso_source,
    canonical_dc_residualization,
    canonical_freq_normalization,
    canonical_representation,
)
from dynamode.inference import build_spectral_mask, maybe_restore_dc, run_inference
from dynamode.model.training.loss import (
    backbone_bond_loss,
    compute_bending_weight,
    geometry_schedule_factor,
    get_frequency_weights,
    load_topology_margin_artifact,
    local_geometry_loss,
    parse_geo_loss_modes,
    risk_band_geometry_loss,
    spectral_amplitude_loss,
    spectral_dc_mse_loss,
    spectral_geometry_losses,
    spectral_low_k_loss,
)
from dynamode.model.stack import (
    attach_per_residue_dc_baselines,
    build_model_stack,
    load_model_weights,
)
from dynamode.model.shake import shake_caca
from dynamode.utils import maintain_cache_size, no_spill_stratified_split
from dynamode.model.training.metrics import (
    RamaValidator,
    collect_temp_caca_debug_metrics,
    compute_batch_caca_dist,
    compute_batch_lddt,
    compute_rmsf,
    debug_print_pipeline,
    finalize_temp_caca_accumulator,
    finalize_temp_clash_accumulator,
    init_temp_caca_accumulator,
    init_temp_clash_accumulator,
    pearson_corr_pytorch,
    spearman_corr_pytorch,
    spectral_band_diagnostics,
    update_temp_caca_accumulator,
    update_temp_clash_accumulator,
)

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
def init_wandb(project_name="dynamode_train", run_name=None, config=None, id=None, resume=None):
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


def worker_init_fn(_worker_id):
    worker_info = torch.utils.data.get_worker_info()
    seed = worker_info.seed

    np.random.seed(seed % 2**32)
    random.seed(seed)
    torch.manual_seed(seed)


def set_randomize_windows(dataset, enabled: bool):
    '''Toggle temporal-window resampling on a dataset or Subset wrapper.'''
    target = dataset.dataset if isinstance(dataset, Subset) else dataset
    if hasattr(target, "randomize_windows"):
        target.randomize_windows = bool(enabled)


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
    epoch=0,
    is_validation=False, is_main_process=True, global_step=0, log_every=1000
):
    '''Single training step supporting x_0, v, and noise prediction targets.

    Returns:
        total_loss: Scalar loss tensor.
        metrics: Dict of loggable metrics.
    '''
    coords_abs = batch["coords"].to(device)
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

    # Pipeline sanity check on the very first training step
    if (
        is_main_process
        and global_step == 0
        and not is_validation
        and representation.name in ("raw_coords", "displacement")
    ):
        debug_print_pipeline(
            batch, transform_engine, real_model, device,
            top_k_freqs=top_k_freqs, displacement=displacement,
            label=f"TRAIN step=0",
        )

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

        # 1. Encode the coordinate representation into the spectral volume.
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
        base_loss_mask = build_spectral_mask(
            mask, torsion_mask, top_k_freqs, is_dct,
            coord_channels=repr_coord_channels,
            representation=representation,
        ).to(device=x_0.device, dtype=x_0.dtype)

        x_t = x_t * base_loss_mask
        noise = noise * base_loss_mask
        x_0 = x_0 * base_loss_mask

        # 4. Model prediction.
        # Optional NMA RMSF prior is surfaced by the dataloader when configured.
        rmsf_prior_b = batch.get("rmsf_prior", None)
        if rmsf_prior_b is not None:
            rmsf_prior_b = rmsf_prior_b.to(device)

        dc_aux_active = dc_lambda > 0.0 and epoch >= dc_start_epoch
        amp_log_gain_mean = float("nan")
        amp_log_gain_min = float("nan")
        amp_log_gain_max = float("nan")
        amp_gain_max = float("nan")
        need_aux_outputs = (
            (dc_aux_active and getattr(inner_model, "use_low_k_correction_head", False))
            or hasattr(inner_model, "amp_head")
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

        # 5. Determine loss target based on prediction type
        x_0_pred = None

        if prediction_target == "noise":
            target = noise
        elif prediction_target == "x_0":
            target = x_0
        elif prediction_target == "v":
            sqrt_ab = torch.sqrt(diffusion.alpha_bar[t]).view(-1, 1, 1)
            sqrt_one_minus_ab = torch.sqrt(1.0 - diffusion.alpha_bar[t]).view(-1, 1, 1)
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

        # 7. Frequency band weighting.
        complex_mult = 1 if is_dct else 2
        C_freq = channels * complex_mult
        weights_k = get_frequency_weights(freq_weighting, top_k_freqs, device)
        weights_expanded = weights_k.view(-1, 1).repeat(1, C_freq).view(1, 1, -1)

        # 8. Bending-based spatial weighting / auxiliary loss
        bending_loss = torch.tensor(0.0, device=device)
        if bending_lambda > 0.0:
            bend_w = compute_bending_weight(
                coords_abs, native_coords, mask.bool(), bending_lambda, False
            )
            weights_expanded = weights_expanded * bend_w.unsqueeze(-1)

        total_weights = base_loss_mask * weights_expanded * loss_weights_t
        sq_diff = (model_out - target) ** 2

        # DIAGNOSTICS
        diagnostic_loss_mask = base_loss_mask.to(dtype=torch.bool)
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
        spectral_amp_loss = torch.tensor(0.0, device=device)
        low_freq_loss = torch.tensor(0.0, device=device)
        dc_loss = torch.tensor(0.0, device=device)
        dc_pred = float("nan")
        dc_gt   = float("nan")
        dc_head_mean = float("nan")
        dc_final_mse = float("nan")
        dc_final_abs_error = float("nan")
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
            or track_caca_debug
        )
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

            if representation_barrier_lambda > 0.0:
                pred_repr_time_for_barrier = transform_engine.spectral_to_time(
                    x_0_pred_raw, n_time_steps=coords_abs.shape[1], n_channels=channels
                )[..., :repr_coord_channels]
                geometry_loss = geometry_loss + representation_barrier_lambda * representation.length_barrier_loss(
                    pred_repr_time_for_barrier,
                    mask=mask,
                )

            # Local geometry loss: CA inter-residue distances at sep=1..5
            if (eff_geometry_lambda > 0.0 and use_idct_geo) or track_caca_debug:
                T_frames = coords_abs.shape[1]
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

                # Optional SpecConv SHAKE pass-through.
                shake_residual_loss = None
                if getattr(inner_model, "use_shake", False):
                    refined_ca = pred_ca
                    pred_ca = shake_caca(
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
                        pred_ca, mask,
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

        total_loss = (
            loss_mse
            + bending_loss
            + geometry_loss
            + spectral_amp_loss
            + low_freq_loss
            + dc_loss
        )
        train_band_metrics = {}
        if prediction_target == "x_0":
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

    # log for wandb
    metrics = {
        "train/mse": loss_mse.item(),
        "train/bending_loss": bending_loss.item(),
        "train/geometry_loss": geometry_loss.item(),
        "train/spectral_amp_loss": spectral_amp_loss.item(),
        "train/low_freq_loss": low_freq_loss.item(),
        "train/dc_loss": dc_loss.item(),
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
    top_k_freqs=64, guidance_scale=1.0,
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
        D = channels * top_k_freqs * complex_mult
        shape = (B, L, D)

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

        # 1. Spectral Volume MSE
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

        valid = mask.bool()
        if valid.any():
            dc_pred_vals = pred_spectral[:, :, :repr_coord_channels][valid]
            dc_gt_vals = gt_spectral[:, :, :repr_coord_channels][valid]
            dc_pred_mean = dc_pred_vals.mean().item()
            dc_gt_mean = dc_gt_vals.mean().item()
            dc_amp_pred = dc_pred_vals.abs().mean().item()
            dc_amp_gt = dc_gt_vals.abs().mean().item()
            if math.isfinite(dc_pred_mean) and math.isfinite(dc_gt_mean):
                dc_pred_acc += dc_pred_mean
                dc_gt_acc += dc_gt_mean
                dc_amp_pred_acc += dc_amp_pred
                dc_amp_gt_acc += dc_amp_gt
                dc_n += 1

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
        del pred, pred_coords, gt_coords, pred_spectral, gt_spectral
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
    results["val/spectral_mse"] = (stats[4] / (stats[5] + 1e-8)).item()
    for key, value_sum in sorted(step_band_metric_sums.items()):
        count = max(1, step_band_metric_counts.get(key, 0))
        results[f"val/{key}"] = value_sum / count
    for key, value_sum in sorted(band_metric_sums.items()):
        count = max(1, band_metric_counts.get(key, 0))
        results[f"val/{key}"] = value_sum / count
    if dc_n > 0:
        results["val/dc_pred_mean"] = dc_pred_acc / dc_n
        results["val/dc_gt_mean"] = dc_gt_acc / dc_n
        results["val/dc_error"] = (dc_pred_acc - dc_gt_acc) / dc_n
        results["val/dc_amp_pred_mean"] = dc_amp_pred_acc / dc_n
        results["val/dc_amp_gt_mean"] = dc_amp_gt_acc / dc_n
        results["val/dc_amp_ratio"] = (dc_amp_pred_acc / dc_n) / max(dc_amp_gt_acc / dc_n, 1e-8)

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
    randomize_train_windows=True,
    max_bad_update_streak=25,
    max_bad_update_total=1000
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

        for i, batch in enumerate(train_loader):
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
                epoch=epoch,
                is_validation=False, is_main_process=is_main_process, global_step=global_step
            )
            
            # DDP-SAFE LOSS CHECK
            # --------------------------
            is_nan = torch.tensor(1 if not bool(torch.isfinite(loss).item()) else 0, device=device)
            if is_distributed:
                dist.all_reduce(is_nan, op=dist.ReduceOp.MAX)

            if is_nan.item() > 0:
                if is_main_process:
                    print(f"WARNING: NaN loss detected at step {global_step}. Skipping batch across ALL ranks.")
                optimizer.zero_grad() 
                record_skipped_update("bad_loss", epoch, i, metrics=metrics)
                continue 
            # --------------------------- 

            if is_bad_batch:
                loss = loss * 0.0

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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

            optimizer.step()
            if scheduler:
                scheduler.step()
            bad_update_streak = 0

            current_loss = loss.item()
            if is_distributed:
                loss_tensor = torch.tensor(current_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                current_loss = loss_tensor.item()

            if is_main_process:
                # Unwrap DDP/training wrappers for model-specific diagnostics.
                _real = model.module if hasattr(model, "module") else model
                _inner = getattr(_real, "model", _real)
                extra_log = {}
                if hasattr(_inner, "rmsf_gate"):
                    extra_log["debug/rmsf_gate"] = _inner.rmsf_gate.detach().float().item()
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
            guidance_scale=guidance_scale,
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
    top_k_freqs=64, guidance_scale=1.0,
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
        guidance_scale=guidance_scale,
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
    model_type="spectral_conv_block_mix_amplitude",
    top_k_freqs=64,
    include_angles=False,
    freq_hidden_size=4,
    spectral_modes=None,
    num_layers=12,
    num_heads=12,
    prediction_target="x_0",
    # Optimizer
    optimizer="adamw",
    max_lr=1e-4,
    # Diffusion
    schedule="cosine",
    num_steps=200,
    shift_value=2.0,
    min_snr_gamma=None,
    # Data pipeline
    use_DCT=True,
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
    # Workflow
    test_only=False,
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
    use_hilbert_spatial=False, # SpecConv: FFT-based Hilbert spatial envelope.
    use_hilbert_spatial_dct=False, # SpecConv: DCT-based boundary-safe Hilbert envelope.
    hilbert_mode="every_block", # SpecConv: every_block | every_3_blocks | input_only | off.
    rmsf_prior_path=None, # optional sidecar .pt of per-domain NMA RMSF prior (from scripts/precompute_nma_tica.py)
    use_rmsf_prior_gain=False, # SpecConv: apply per-residue NMA RMSF prior gain on the model output.
    use_low_k_correction_head=False, # Additive low-k correction branch, zero-init for warm-start safety.
    low_k_correction_modes=1, # number of lowest modes handled by the additive correction branch
    use_seq_conditioning=False, # Use learned residue-type embeddings and pooled global sequence conditioning.
    seq_embed_dim=16, # width of the learned residue-type embedding
    use_ss_conditioning=False, # Use DSSP-based secondary-structure conditioning.
    ss_embed_dim=8, # latent width for DSSP conditioning
    fast_band_edges=None, # SpecConv block-diagonal frequency mixer bands.
    max_bad_update_streak=25, # fail loudly after consecutive NaN-loss/NaN-grad skipped optimizer updates
    max_bad_update_total=1000, # fail loudly after total skipped optimizer updates in one run
    amp_head_context_modes=4, # SpecConv: number of low-k modes supplied as amplitude-head context.
    amp_head_target_modes=1, # SpecConv: number of lowest modes amplitude-calibrated.
    amp_head_d_model=128, # SpecConv amplitude-head hidden width.
    amp_head_depth=3, # SpecConv amplitude-head transformer depth.
    amp_head_num_heads=4, # SpecConv amplitude-head attention heads.
    amp_head_mlp_ratio=4.0, # SpecConv amplitude-head SwiGLU ratio.
    amp_head_attn_dropout=0.0, # SpecConv amplitude-head attention dropout.
    amp_head_use_rmsf_prior=False, # Feed RMSF prior into the SpecConv amplitude head.
    use_shake=False, # SpecConv: apply differentiable SHAKE to reconstructed CA coords.
    shake_n_iter=20,
    shake_target=3.8,
    ddp_find_unused_parameters=None,
    ddp_static_graph=None,
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
        default_num_workers = 16
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
        default_num_workers = 6
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
    default_timeout = 60
    default_prefetch = 4
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

    stack_log = print if rank == 0 else (lambda *_args, **_kwargs: None)
    stack_config = {
        "coords_type": coords_type,
        "include_angles": include_angles,
        "representation": representation_obj.name,
        "displacement": displacement,
        "representation_length_min": representation_length_min,
        "representation_length_max": representation_length_max,
        "representation_length_residual_max": representation_length_residual_max,
        "window_size": window_size,
        "top_k_freqs": top_k_freqs,
        "use_DCT": use_DCT,
        "freq_normalization": freq_normalization,
        "dc_residualization": dc_residualization,
        "aniso_source": aniso_source,
        "freq_scales_path": freq_scales_path,
        "aniso_scales_path": aniso_scales_path,
        "model_type": model_type,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "freq_hidden_size": freq_hidden_size,
        "spectral_modes": spectral_modes,
        "prediction_target": prediction_target,
        "conditioning_dropout": conditioning_dropout,
        "use_hilbert_spatial": use_hilbert_spatial,
        "use_hilbert_spatial_dct": use_hilbert_spatial_dct,
        "hilbert_mode": hilbert_mode,
        "use_rmsf_prior_gain": use_rmsf_prior_gain,
        "use_low_k_correction_head": use_low_k_correction_head,
        "low_k_correction_modes": low_k_correction_modes,
        "use_seq_conditioning": use_seq_conditioning,
        "seq_embed_dim": seq_embed_dim,
        "use_ss_conditioning": use_ss_conditioning,
        "ss_embed_dim": ss_embed_dim,
        "fast_band_edges": fast_band_edges,
        "amp_head_context_modes": amp_head_context_modes,
        "amp_head_target_modes": amp_head_target_modes,
        "amp_head_d_model": amp_head_d_model,
        "amp_head_depth": amp_head_depth,
        "amp_head_num_heads": amp_head_num_heads,
        "amp_head_mlp_ratio": amp_head_mlp_ratio,
        "amp_head_attn_dropout": amp_head_attn_dropout,
        "amp_head_use_rmsf_prior": amp_head_use_rmsf_prior,
        "use_shake": use_shake,
        "shake_n_iter": shake_n_iter,
        "shake_target": shake_target,
        "schedule": schedule,
        "num_steps": num_steps,
        "shift_value": shift_value,
        "min_snr_gamma": min_snr_gamma,
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

    stack = build_model_stack(
        stack_config,
        device,
        train_loader=train_loader,
        checkpoint_dir=checkpoint_dir,
        compute_missing_freq_stats=True,
        freq_stats_samples=1000,
        load_weights_path=checkpoint_path,
        set_eval=False,
        log_fn=stack_log,
    )
    model = stack.model
    diffusion = stack.diffusion
    transform_engine = stack.transform_engine
    representation_obj = stack.representation
    coord_channels = stack.coord_channels
    angle_channels = stack.angle_channels
    top_k_freqs = stack.top_k_freqs
    displacement = representation_obj.is_displacement

    attach_per_residue_dc_baselines(
        full_dataset,
        stack.per_residue_dc_baselines,
        log_fn=stack_log,
    )

    if rank == 0:
        model_scale_msg = "conditioned" if transform_engine.model_conditioned_freq_scale is not None else (
            "global" if transform_engine.model_freq_scale is not None else "none"
        )
        print(
            "Effective spectral policies: "
            f"model_normalization={model_scale_msg}, "
            f"dc_residualization={transform_engine.effective_dc_residualization}, "
            f"aniso_source={transform_engine.effective_aniso_source}"
        )
        print(stack.noise_diagnostics_text)
        try:
            wandb.config.update(
                {
                    "resolved_shift_value": stack.resolved_noise.shift_value,
                    "resolved_noise_schedule": stack.resolved_noise.diagnostics,
                },
                allow_val_change=True,
            )
        except Exception:
            pass

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {model_type} | Params: {n_params:,}")

    if per_gpu_batch_size < 16:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if ddp_find_unused_parameters is None:
        ddp_find_unused_parameters = False
    if ddp_static_graph is None:
        ddp_static_graph = False

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
            f"static_graph={ddp_kwargs.get('static_graph', False)}"
        )
    model = DDP(model, **ddp_kwargs)

    # TRAINING 
    # ---------
    if not test_only:
        start_epoch = 0
        global_step = 0
        best_model_score = float("-inf")
        latest_ckpt = os.path.join(checkpoint_dir, "checkpoint_latest.pt")

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        decay_params = [p for p in trainable_params if p.dim() >= 2]
        nodecay_params = [p for p in trainable_params if p.dim() < 2]
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
            randomize_train_windows=randomize_train_windows,
            max_bad_update_streak=max_bad_update_streak,
            max_bad_update_total=max_bad_update_total,
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
        guidance_scale=guidance_scale,
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
    target = model.module if isinstance(model, DDP) else model
    target.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler_state = checkpoint["scheduler_state_dict"]
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

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
    '''Rebuild OneCycleLR when the resumed run changes the total step budget.'''
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



HYDRA_CONFIG_SKIP_SECTIONS = {"hydra", "inference"}
TRAIN_FLOAT_KEYS = {
    "max_lr",
    "min_snr_gamma",
    "shift_value",
    "guidance_scale",
    "bending_lambda",
    "geometry_lambda",
    "geometry_tol",
    "clash_lambda",
    "clash_threshold",
    "spectral_geo_segment_threshold",
    "aniso_gamma",
    "rmsf_lambda",
    "low_freq_lambda",
    "dc_lambda",
    "representation_length_min",
    "representation_length_max",
    "representation_length_residual_max",
    "representation_barrier_lambda",
    "amp_head_mlp_ratio",
    "amp_head_attn_dropout",
    "shake_target",
}
TRAIN_INT_KEYS = {
    "epochs",
    "batch_size",
    "top_k_freqs",
    "freq_hidden_size",
    "spectral_modes",
    "seq_embed_dim",
    "ss_embed_dim",
    "num_layers",
    "num_heads",
    "num_steps",
    "num_ode_steps",
    "max_val_batches",
    "dataloader_num_workers",
    "dataloader_timeout",
    "dataloader_prefetch_factor",
    "geometry_warmup_start",
    "geometry_warmup_epochs",
    "geometry_decay_start",
    "geometry_decay_epochs",
    "clash_max_pairs",
    "clash_pair_chunk",
    "spectral_geo_max_segment_pairs",
    "risk_band_max_pairs",
    "risk_band_max_segment_pairs",
    "rmsf_warmup_start",
    "rmsf_warmup_epochs",
    "low_freq_modes",
    "dc_start_epoch",
    "amp_head_context_modes",
    "amp_head_target_modes",
    "amp_head_d_model",
    "amp_head_depth",
    "amp_head_num_heads",
    "shake_n_iter",
    "max_bad_update_streak",
    "max_bad_update_total",
}


def flatten_hydra_config(cfg: DictConfig | dict, *, skip_sections=HYDRA_CONFIG_SKIP_SECTIONS) -> dict:
    '''Flatten the public Hydra section layout into TRAIN_DISTRIBUTED kwargs.'''
    raw = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
    config: dict = {}
    for key, value in (raw or {}).items():
        if key in skip_sections:
            continue
        if isinstance(value, dict):
            config.update(value)
        else:
            config[key] = value
    return config


def coerce_training_config_types(config: dict) -> dict:
    '''Keep YAML/Hydra overrides aligned with the TRAIN_DISTRIBUTED signature.'''
    for key in TRAIN_FLOAT_KEYS:
        if key in config and config[key] is not None:
            if key == "shift_value" and isinstance(config[key], str) and config[key].strip().lower() == "auto":
                continue
            config[key] = float(config[key])
    for key in TRAIN_INT_KEYS:
        if key in config and config[key] is not None:
            config[key] = int(config[key])
    if "low_k_correction_modes" in config and config["low_k_correction_modes"] is not None:
        value = config["low_k_correction_modes"]
        if isinstance(value, str):
            value = value.strip()
            if value.isdigit():
                value = int(value)
        config["low_k_correction_modes"] = value
    return config


def prepare_training_config(cfg: DictConfig | dict) -> dict:
    '''Resolve Hydra config into the flat public training contract.'''
    config = coerce_training_config_types(flatten_hydra_config(cfg))
    config.setdefault("randomize_train_windows", True)

    if config.get("representation") is None:
        config["representation"] = (
            "displacement" if bool(config.get("displacement", False)) else "raw_coords"
        )
    config["representation"] = canonical_representation(config["representation"])
    config["displacement"] = config["representation"] == "displacement"
    config["geo_loss"] = ",".join(parse_geo_loss_modes(config.get("geo_loss", "idct_ca-ca")))
    config["freq_normalization"] = canonical_freq_normalization(config.get("freq_normalization", "auto"))
    config["dc_residualization"] = canonical_dc_residualization(config.get("dc_residualization", "auto"))
    config["aniso_source"] = canonical_aniso_source(config.get("aniso_source", "auto"))

    if config.get("date_prefix_checkpoint_dir", True) and config.get("checkpoint_dir"):
        base = str(config["checkpoint_dir"])
        parent = os.path.dirname(base)
        name = os.path.basename(base)
        config["checkpoint_dir"] = os.path.join(parent, f"{date_str}_{name}")

    checkpoint_dir = config.get("checkpoint_dir")
    if checkpoint_dir is None:
        raise ValueError("core.checkpoint_dir must be provided in the Hydra config.")

    if config.get("run_name") is None:
        run_name = f"{date_str}_top_k{config.get('top_k_freqs', '?')}"
    else:
        run_name = f"{date_str}_{config['run_name']}"
    if bool(config.get("test_only", False)):
        run_name = f"TEST_{run_name}"
    config["run_name"] = run_name
    return config


def _filter_train_kwargs(config: dict, *, rank: int = 0) -> dict:
    accepted = set(inspect.signature(TRAIN_DISTRIBUTED).parameters)
    ignored = sorted(k for k in config if k not in accepted)
    if rank == 0 and ignored:
        print(f"Ignoring non-training config keys: {ignored}")
    return {key: value for key, value in config.items() if key in accepted}


@hydra.main(version_base=None, config_path="../../configs", config_name="spec_conv_displacement_ca_unit_var")
def main(cfg: DictConfig) -> None:
    '''
    Hydra training entrypoint.

    Example:
        torchrun --nproc_per_node=4 -m dynamode.train --config-name transformer_displacement_ca
    '''
    init_process()
    rank = dist.get_rank()
    config = prepare_training_config(cfg)
    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    if rank == 0:
        print("Run Name:", config["run_name"])
        wandb.login()

        wandb_id = None
        latest_ckpt = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
        if config.get("resume_from_latest") and os.path.exists(latest_ckpt):
            ckpt = torch.load(latest_ckpt, map_location="cpu")
            wandb_id = ckpt.get("wandb_id")

        init_wandb(
            project_name=config.get("wandb_project", "pancakes_spectral_diffusion"),
            run_name=config["run_name"],
            config=config,
            id=wandb_id,
            resume="allow",
        )

        with open(os.path.join(checkpoint_dir, "run_config.yaml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print("Final config:")
        pprint(config, sort_dicts=True, indent=4)

    try:
        train_kwargs = _filter_train_kwargs(config, rank=rank)
        test_metrics = TRAIN_DISTRIBUTED(**train_kwargs)
        if rank == 0:
            print("Final Metrics:", test_metrics)
    except Exception as exc:
        print(f"Rank {rank} failed: {exc}")
        raise
    finally:
        if rank == 0:
            wandb.finish()
        if dist.is_initialized():
            dist.destroy_process_group()

    print("FINISHED")


if __name__ == "__main__":
    main()
