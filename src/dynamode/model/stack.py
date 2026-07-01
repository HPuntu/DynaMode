'''Shared public model stack construction.

This module assembles the runtime pieces that training, inference, and
evaluation all need: coordinate representation, spectral transform pipeline,
model wrapper, optional weights-only loading, and diffusion schedule. Training
still owns dataloaders, DDP, optimizers, schedulers, and full checkpoint resume.
'''

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist

from dynamode.model.diffusion import SpectralDiffusion
from dynamode.model.noise_schedule import (
    ResolvedNoiseSchedule,
    format_noise_diagnostics,
    resolve_noise_schedule,
)
from dynamode.model.wrapper import (
    BaseDiffusionConfig,
    SUPPORTED_MODEL_TYPES,
    UnifiedWrapper,
    make_model_config,
)
from dynamode.spectral.adapters import compute_frequency_stats
from dynamode.spectral.conditioned_freq_scale import load_freq_scale_artifact
from dynamode.spectral.representation import (
    CoordinateRepresentation,
    SpectralRepresentationPipeline,
    canonical_aniso_source,
    canonical_dc_residualization,
    canonical_freq_normalization,
    canonical_representation,
)


_UNSET = object()


@dataclass
class ModelStack:
    '''Runtime bundle shared by training, inference, and evaluation.'''

    model: UnifiedWrapper
    diffusion: SpectralDiffusion
    transform_engine: SpectralRepresentationPipeline
    model_config: BaseDiffusionConfig
    representation: CoordinateRepresentation
    coord_channels: int
    repr_coord_channels: int
    angle_channels: int
    total_channels: int
    window_size: int
    top_k_freqs: int
    is_dct: bool
    is_time_domain: bool
    resolved_noise: ResolvedNoiseSchedule
    noise_diagnostics_text: str
    freq_scales: torch.Tensor | None = None
    conditioned_freq_scale: dict[str, Any] | None = None
    aniso_freq_scales: torch.Tensor | None = None
    per_residue_dc_baselines: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "diffusion": self.diffusion,
            "transform_engine": self.transform_engine,
            "model_config": self.model_config,
            "coord_channels": self.coord_channels,
            "repr_coord_channels": self.repr_coord_channels,
            "angle_channels": self.angle_channels,
            "total_channels": self.total_channels,
            "window_size": self.window_size,
            "top_k_freqs": self.top_k_freqs,
            "is_dct": self.is_dct,
            "is_time_domain": self.is_time_domain,
            "representation": self.representation,
            "resolved_noise": self.resolved_noise,
            "noise_diagnostics_text": self.noise_diagnostics_text,
            "freq_scales": self.freq_scales,
            "conditioned_freq_scale": self.conditioned_freq_scale,
            "aniso_freq_scales": self.aniso_freq_scales,
            "per_residue_dc_baselines": self.per_residue_dc_baselines,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)

    def keys(self):
        return self.as_dict().keys()

    def items(self):
        return self.as_dict().items()


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _is_rank0() -> bool:
    return _rank() == 0


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


def _log_once(log_fn: Callable[[str], None] | None, message: str) -> None:
    if log_fn is not None and _is_rank0():
        log_fn(message)


def _value(config: Mapping[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def _drop_shape_incompatible_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
    log_fn: Callable[[str], None] = print,
    prefix: str = "load",
) -> dict[str, torch.Tensor]:
    '''Remove same-name checkpoint tensors whose shapes do not match target.'''
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key, value in state_dict.items():
        target_value = target_state_dict.get(key)
        if (
            target_value is not None
            and torch.is_tensor(value)
            and torch.is_tensor(target_value)
            and tuple(value.shape) != tuple(target_value.shape)
        ):
            skipped.append((key, tuple(value.shape), tuple(target_value.shape)))
            continue
        filtered[key] = value

    if skipped:
        preview = ", ".join(
            f"{key}: ckpt{src_shape}->model{dst_shape}"
            for key, src_shape, dst_shape in skipped[:12]
        )
        more = "" if len(skipped) <= 12 else f", ... +{len(skipped) - 12} more"
        log_fn(f"  {prefix}: shape-mismatched keys skipped = {preview}{more}")
    return filtered


def load_model_weights(
    checkpoint_path: str,
    model: torch.nn.Module,
    *,
    device: torch.device | str = "cpu",
    log_fn: Callable[[str], None] = print,
):
    '''Load weights only, accepting this repo's training checkpoints or raw state dicts.'''
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    target = model.module if hasattr(model, "module") else model
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }
    state_dict = _drop_shape_incompatible_state_dict_keys(
        state_dict,
        target.state_dict(),
        log_fn=log_fn,
        prefix="load",
    )
    result = target.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        log_fn(f"  load: missing keys (kept at init) = {result.missing_keys}")
    if result.unexpected_keys:
        log_fn(f"  load: unexpected keys (ignored)   = {result.unexpected_keys}")
    return result


def attach_per_residue_dc_baselines(
    dataset: Any,
    baselines: Mapping[str, Any] | None,
    *,
    log_fn: Callable[[str], None] | None = None,
) -> int:
    '''Attach per-residue DC residualisation payload to dataset wrappers that expose it.'''
    if not baselines:
        return 0
    payload = {
        str(key): (value if torch.is_tensor(value) else torch.as_tensor(value)).float().contiguous()
        for key, value in baselines.items()
    }
    attached = 0
    for holder in (dataset, getattr(dataset, "dataset", None)):
        if holder is not None and hasattr(holder, "per_residue_dc_baselines"):
            holder.per_residue_dc_baselines = payload
            attached += 1
    if attached:
        _log_once(
            log_fn,
            f"Attached per-residue DC baselines to dataset: {len(payload)} (key,temp) pairs",
        )
    return len(payload)


def _compute_or_load_frequency_scales(
    *,
    config: Mapping[str, Any],
    train_loader: Any,
    checkpoint_dir: str | None,
    compute_missing_freq_stats: bool,
    freq_stats_samples: int,
    representation: CoordinateRepresentation,
    coord_channels: int,
    total_channels: int,
    top_k_freqs: int,
    use_dct: bool,
    include_angles: bool,
    coords_type: str,
    freq_normalization: str,
    dc_residualization: str,
    aniso_source: str,
    expected_freq_dim: int,
    device: torch.device | str,
    log_fn: Callable[[str], None] | None,
) -> tuple[torch.Tensor | None, dict[str, Any] | None, torch.Tensor | None]:
    freq_scales = None
    conditioned_freq_scale = None
    aniso_freq_scales = None

    needs_aniso_artifact = config.get("aniso_scales_path") is not None
    needs_main_scale_artifact = (
        freq_normalization != "none"
        or dc_residualization in {"bucket", "per_residue"}
        or (dc_residualization == "auto" and config.get("freq_scales_path") is not None)
        or aniso_source == "freq_scales"
        or (aniso_source == "auto" and freq_normalization != "none")
    )
    if not (needs_main_scale_artifact or needs_aniso_artifact):
        return freq_scales, conditioned_freq_scale, aniso_freq_scales

    freq_scales_path = config.get("freq_scales_path")
    if needs_main_scale_artifact and freq_scales_path is None:
        if not compute_missing_freq_stats:
            raise FileNotFoundError(
                "A valid freq_scales_path is required for the selected spectral "
                "normalization/DC/aniso policies."
            )
        if train_loader is None:
            raise ValueError("train_loader is required to compute missing frequency statistics.")
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required to save computed frequency statistics.")

        os.makedirs(checkpoint_dir, exist_ok=True)
        freq_scales_path = os.path.join(checkpoint_dir, "freq_scales.pt")
        if _is_rank0():
            _log_once(log_fn, "Computing frequency scaling statistics...")
            if os.path.exists(freq_scales_path):
                os.remove(freq_scales_path)
            stats_pipeline = SpectralRepresentationPipeline(
                coordinate=representation,
                raw_coord_channels=coord_channels,
                total_channels=total_channels,
                use_dct=use_dct,
                freq_normalization="none",
                dc_residualization="none",
                aniso_source="none",
                device=device,
            )
            computed = compute_frequency_stats(
                train_loader,
                stats_pipeline.time_to_spectral,
                samples=int(freq_stats_samples),
                top_k=top_k_freqs,
                device=device,
                is_dct=use_dct,
                use_angles=include_angles,
                coords_type=coords_type,
                representation=representation,
            )
            if computed.shape[0] < expected_freq_dim:
                raise ValueError(
                    f"Computed freq_scales too short: got {computed.shape[0]}, "
                    f"expected at least {expected_freq_dim}"
                )
            torch.save(computed[:expected_freq_dim].contiguous(), freq_scales_path)
        _barrier()

    if needs_main_scale_artifact:
        if freq_scales_path is None or not os.path.exists(freq_scales_path):
            raise FileNotFoundError(
                "A valid freq_scales_path is required for the selected spectral "
                "normalization/DC/aniso policies."
            )
        freq_scales, conditioned_freq_scale = load_freq_scale_artifact(
            freq_scales_path, expected_freq_dim, map_location=device
        )

    aniso_scales_path = config.get("aniso_scales_path")
    if aniso_scales_path is not None:
        if not os.path.exists(aniso_scales_path):
            raise FileNotFoundError(aniso_scales_path)
        aniso_freq_scales, _ = load_freq_scale_artifact(
            aniso_scales_path, expected_freq_dim, map_location=device
        )

    if _is_rank0():
        _log_once(log_fn, f"top_k_freqs = {top_k_freqs}")
        _log_once(log_fn, f"channels = {total_channels}")
        _log_once(log_fn, f"expected_freq_dim = {expected_freq_dim}")
        if freq_scales is not None:
            _log_once(log_fn, f"trimmed freq_scales.shape = {tuple(freq_scales.shape)}")
        if aniso_freq_scales is not None:
            _log_once(log_fn, f"aniso_freq_scales.shape = {tuple(aniso_freq_scales.shape)}")
        if conditioned_freq_scale is not None:
            meta = conditioned_freq_scale.get("metadata", {})
            _log_once(
                log_fn,
                "Using conditioned freq scales: "
                + str(
                    {
                        "scheme": meta.get("scheme"),
                        "alpha": meta.get("alpha"),
                        "scale_condition_modes": meta.get("scale_condition_modes"),
                    }
                ),
            )

    return freq_scales, conditioned_freq_scale, aniso_freq_scales


def build_model_stack(
    config: Mapping[str, Any],
    device: torch.device | str,
    *,
    train_loader: Any = None,
    checkpoint_dir: str | None = None,
    compute_missing_freq_stats: bool = False,
    freq_stats_samples: int = 1000,
    load_weights_path: str | None | object = _UNSET,
    set_eval: bool = False,
    log_fn: Callable[[str], None] | None = print,
) -> ModelStack:
    '''Build the shared spectral runtime stack for the two public model types.'''
    model_type = str(_value(config, "model_type", "spectral_conv_block_mix_amplitude"))
    if model_type not in SUPPORTED_MODEL_TYPES:
        supported = ", ".join(SUPPORTED_MODEL_TYPES)
        raise ValueError(f"Unknown model_type={model_type!r}. Supported public models: {supported}")

    coords_type = str(_value(config, "coords_type", "ca"))
    include_angles = bool(_value(config, "include_angles", False))
    coord_channels = 12 if coords_type == "bb" else 3
    representation_name = canonical_representation(
        config.get("representation"),
        displacement=_value(config, "displacement", True),
    )
    representation = CoordinateRepresentation(
        representation_name,
        coord_channels=coord_channels,
        length_min=float(_value(config, "representation_length_min", 3.5)),
        length_max=float(_value(config, "representation_length_max", 4.1)),
        length_residual_max=float(_value(config, "representation_length_residual_max", 0.30)),
    )
    if representation.is_unit_chain and coords_type != "ca":
        raise ValueError(f"{representation.name} requires coords_type='ca'")

    repr_coord_channels = representation.model_coord_channels
    angle_channels = 4 if include_angles else 0
    total_channels = repr_coord_channels + angle_channels
    window_size = int(_value(config, "window_size", 256))
    top_k_freqs = int(_value(config, "top_k_freqs", 64))
    use_dct = bool(_value(config, "use_DCT", True))
    freq_normalization = canonical_freq_normalization(config.get("freq_normalization", "auto"))
    dc_residualization = canonical_dc_residualization(config.get("dc_residualization", "auto"))
    aniso_source = canonical_aniso_source(config.get("aniso_source", "auto"))
    expected_freq_dim = top_k_freqs * total_channels

    freq_scales, conditioned_freq_scale, aniso_freq_scales = _compute_or_load_frequency_scales(
        config=config,
        train_loader=train_loader,
        checkpoint_dir=checkpoint_dir,
        compute_missing_freq_stats=compute_missing_freq_stats,
        freq_stats_samples=freq_stats_samples,
        representation=representation,
        coord_channels=coord_channels,
        total_channels=total_channels,
        top_k_freqs=top_k_freqs,
        use_dct=use_dct,
        include_angles=include_angles,
        coords_type=coords_type,
        freq_normalization=freq_normalization,
        dc_residualization=dc_residualization,
        aniso_source=aniso_source,
        expected_freq_dim=expected_freq_dim,
        device=device,
        log_fn=log_fn,
    )

    transform_engine = SpectralRepresentationPipeline(
        coordinate=representation,
        raw_coord_channels=coord_channels,
        total_channels=total_channels,
        use_dct=use_dct,
        scale_factors=freq_scales,
        conditioned_freq_scale=conditioned_freq_scale,
        freq_normalization=freq_normalization,
        dc_residualization=dc_residualization,
        aniso_source=aniso_source,
        aniso_scale_factors=aniso_freq_scales,
        device=device,
    )

    model_kwargs = dict(config)
    for duplicate_key in (
        "model_type",
        "in_channels",
        "cond_channels",
        "top_k_freqs",
        "freq_scale",
        "conditioned_freq_scale",
        "is_dct",
    ):
        model_kwargs.pop(duplicate_key, None)
    model_config = make_model_config(
        model_type,
        in_channels=total_channels,
        cond_channels=coord_channels,
        top_k_freqs=top_k_freqs,
        freq_scale=transform_engine.model_freq_scale,
        conditioned_freq_scale=transform_engine.model_conditioned_freq_scale,
        is_dct=use_dct,
        **model_kwargs,
    )
    model = UnifiedWrapper(model_name=model_type, config=model_config).to(device)

    if load_weights_path is _UNSET:
        load_weights_path = config.get("checkpoint_path")
    if load_weights_path is not None:
        if not os.path.exists(str(load_weights_path)):
            raise FileNotFoundError(str(load_weights_path))
        _log_once(log_fn, f"Loading model weights from: {load_weights_path}")
        load_model_weights(
            str(load_weights_path),
            model,
            device=device,
            log_fn=log_fn if log_fn is not None else _noop,
        )
    if set_eval:
        model.eval()

    resolved_noise = resolve_noise_schedule(
        config=dict(config),
        freq_scales=transform_engine.aniso_freq_scale,
        top_k_freqs=top_k_freqs,
        channels=total_channels,
        num_steps=int(_value(config, "num_steps", 1000)),
        device=device,
    )
    diffusion = SpectralDiffusion(
        T=int(_value(config, "num_steps", 1000)),
        device=device,
        schedule=resolved_noise.schedule,
        min_snr_gamma=config.get("min_snr_gamma"),
        shift_value=resolved_noise.shift_value,
        aniso_weights=resolved_noise.aniso_weights,
    )

    return ModelStack(
        model=model,
        diffusion=diffusion,
        transform_engine=transform_engine,
        model_config=model_config,
        representation=representation,
        coord_channels=coord_channels,
        repr_coord_channels=repr_coord_channels,
        angle_channels=angle_channels,
        total_channels=total_channels,
        window_size=window_size,
        top_k_freqs=top_k_freqs,
        is_dct=bool(getattr(model, "is_dct", use_dct)),
        is_time_domain=bool(getattr(model, "is_time_domain", False)),
        resolved_noise=resolved_noise,
        noise_diagnostics_text=format_noise_diagnostics(resolved_noise.diagnostics),
        freq_scales=freq_scales,
        conditioned_freq_scale=conditioned_freq_scale,
        aniso_freq_scales=aniso_freq_scales,
        per_residue_dc_baselines=transform_engine.per_residue_dc_baselines,
    )
