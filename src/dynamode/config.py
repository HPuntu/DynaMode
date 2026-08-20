"""Hydra composition and flat-runtime compatibility for DynaMode.

Hydra is the user-facing configuration layer.  The training, inference, model,
and data implementations intentionally continue to consume the established flat
runtime dictionaries so configuration migration remains independent of the
scientific code.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import yaml

DYNAMODE_SECTIONS = (
    "run",
    "core",
    "data",
    "representation",
    "model",
    "diffusion",
    "sampling",
    "training",
    "inference",
)
TRAIN_SECTIONS = tuple(section for section in DYNAMODE_SECTIONS if section != "inference")
INFERENCE_RUNTIME_KEYS = {
    "batch_size",
    "crop_size",
    "coords_type",
    "guidance_scale",
    "include_angles",
    "num_ode_steps",
}

MODEL_HANDLE_ALIASES = {
    "spectral_conv_block_mix_amplitude": "specconv-base",
    "spectral_dit_low_k": "spectral-dit",
}
RUN_IDENTITY_KEYS = (
    "experiment_handle",
    "model_handle",
    "setup_handle",
    "config_fingerprint",
    "run_descriptor",
    "run_id",
    "run_name",
    "started_at_local",
    "started_at_utc",
)
RUN_FINGERPRINT_EXCLUDE_KEYS = {
    "checkpoint_dir",
    "config_fingerprint",
    "experiment_handle",
    "model_handle",
    "output_root",
    "resume_from_latest",
    "resume_allow_unsafe_overrides",
    "resume_config_mode",
    "run_descriptor",
    "run_id",
    "run_name",
    "run_tag",
    "setup_handle",
    "setup_tags",
    "started_at_local",
    "started_at_utc",
    "timestamp_run_dir",
}

RESUME_CONTROL_KEYS = {
    "checkpoint_dir",
    "checkpoint_path",
    "resume_allow_unsafe_overrides",
    "resume_config_mode",
    "resume_from_latest",
    "test_only",
}
RESUME_MUTABLE_KEYS = {
    "atlas_path",
    "atlas_zarr_path",
    "batch_size",
    "dataloader_num_workers",
    "dataloader_persistent_workers",
    "dataloader_prefetch_factor",
    "dataloader_timeout",
    "epochs",
    "freq_scales_path",
    "max_val_batches",
    "mdcath_path",
    "mdcath_zarr_path",
    "offline_mode",
    "output_root",
    "rmsf_prior_path",
    "split_ids_dir",
    "topology_margin_path",
    "trim_cache",
}

TRAIN_FLOAT_KEYS = {
    "amp_head_attn_dropout",
    "amp_head_mlp_ratio",
    "aniso_gamma",
    "bending_lambda",
    "clash_lambda",
    "clash_threshold",
    "dc_lambda",
    "geometry_lambda",
    "geometry_tol",
    "guidance_scale",
    "low_freq_lambda",
    "max_lr",
    "min_snr_gamma",
    "representation_barrier_lambda",
    "representation_length_max",
    "representation_length_min",
    "representation_length_residual_max",
    "rmsf_lambda",
    "shake_target",
    "shift_value",
    "spectral_geo_segment_threshold",
}
TRAIN_INT_KEYS = {
    "amp_head_context_modes",
    "amp_head_d_model",
    "amp_head_depth",
    "amp_head_num_heads",
    "amp_head_target_modes",
    "batch_size",
    "clash_max_pairs",
    "clash_pair_chunk",
    "dataloader_num_workers",
    "dataloader_prefetch_factor",
    "dataloader_timeout",
    "dc_start_epoch",
    "epochs",
    "freq_hidden_size",
    "geometry_decay_epochs",
    "geometry_decay_start",
    "geometry_warmup_epochs",
    "geometry_warmup_start",
    "low_freq_modes",
    "max_bad_update_streak",
    "max_bad_update_total",
    "max_val_batches",
    "num_layers",
    "num_heads",
    "num_ode_steps",
    "num_steps",
    "risk_band_max_pairs",
    "risk_band_max_segment_pairs",
    "rmsf_warmup_epochs",
    "rmsf_warmup_start",
    "seq_embed_dim",
    "shake_n_iter",
    "spectral_geo_max_segment_pairs",
    "spectral_modes",
    "ss_embed_dim",
    "top_k_freqs",
}

REPRESENTATION_ALIASES = {
    "absolute": "raw_coords",
    "coords": "raw_coords",
    "displacement": "displacement",
    "native_displacement": "displacement",
    "raw": "raw_coords",
    "raw_coords": "raw_coords",
    "unit_chain_mean": "unit_chain_mean_lengths",
    "unit_chain_mean_lengths": "unit_chain_mean_lengths",
    "unit_chain_native": "unit_chain_native_lengths",
    "unit_chain_native_lengths": "unit_chain_native_lengths",
    "unit_chain_pred": "unit_chain_pred_lengths",
    "unit_chain_pred_lengths": "unit_chain_pred_lengths",
    "unit_chain_residual_lengths": "unit_chain_pred_lengths",
}
NORMALIZATION_ALIASES = {
    "auto": "auto",
    "conditioned": "conditioned",
    "conditioned_freq_scales": "conditioned",
    "freq": "global",
    "freq_scales": "global",
    "global": "global",
    "identity": "none",
    "none": "none",
    "off": "none",
}
DC_ALIASES = {
    "auto": "auto",
    "bucket": "bucket",
    "conditioned": "bucket",
    "none": "none",
    "off": "none",
    "per-residue": "per_residue",
    "per_residue": "per_residue",
}
ANISO_ALIASES = {
    "artifact": "artifact",
    "auto": "auto",
    "freq_scales": "freq_scales",
    "model": "freq_scales",
    "none": "none",
    "normalization": "freq_scales",
    "off": "none",
}
GEO_LOSS_ALIASES = {
    "band_risk": "risk_band",
    "ca-ca": "idct_ca-ca",
    "caca": "idct_ca-ca",
    "idct-ca-ca": "idct_ca-ca",
    "idct-caca": "idct_ca-ca",
    "idct_ca-ca": "idct_ca-ca",
    "idct_caca": "idct_ca-ca",
    "risk_band": "risk_band",
    "risk_bond": "risk_band",
    "spec_geo": "spec_geo",
    "spectral_geo": "spec_geo",
    "spectral_geometry": "spec_geo",
}


def _canonical(value: Any, aliases: Mapping[str, str], *, name: str) -> str:
    key = str(value or "auto").strip().lower()
    if key not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unknown {name}={value!r}. Expected one of: {valid}")
    return aliases[key]


def _parse_geo_loss_modes(value: Any) -> tuple[str, ...]:
    if value is None or value is False:
        return ("idct_ca-ca",)
    if isinstance(value, (list, tuple, set)):
        raw = [part for item in value for part in str(item).split(",")]
    else:
        text = str(value).strip()
        if text.lower() in {"", "0", "false", "none", "off"}:
            return tuple()
        raw = text.split(",")

    modes: list[str] = []
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


def slugify_handle(value: Any, *, fallback: str = "run") -> str:
    """Return a compact filesystem-safe descriptor."""

    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return slug or fallback


def _metadata_safe(value: Any) -> Any:
    """Convert common runtime objects into compact YAML/JSON-safe metadata."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _metadata_safe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _metadata_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_metadata_safe(item) for item in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        shape = [int(size) for size in getattr(value, "shape", ())]
        return {
            "type": type(value).__name__,
            "shape": shape,
            "dtype": str(getattr(value, "dtype", "unknown")),
        }
    return str(value)


def _fingerprint_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _metadata_safe(value)
        for key, value in config.items()
        if key not in RUN_FINGERPRINT_EXCLUDE_KEYS
    }


def config_fingerprint(config: Mapping[str, Any], *, length: int = 8) -> str:
    """Hash the resolved scientific setup while excluding volatile run fields."""

    payload = json.dumps(
        _fingerprint_payload(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[: int(length)]


def derive_run_descriptors(config: Mapping[str, Any]) -> dict[str, str]:
    """Resolve stable experiment/model names and an automatic setup handle."""

    experiment = slugify_handle(config.get("experiment_handle"), fallback="experiment")
    model_fallback = MODEL_HANDLE_ALIASES.get(
        str(config.get("model_type", "")),
        str(config.get("model_type", "model")),
    )
    model = slugify_handle(config.get("model_handle") or model_fallback, fallback="model")

    representation_aliases = {
        "displacement": "disp",
        "raw_coords": "coords",
        "unit_chain_mean_lengths": "unit-mean",
        "unit_chain_native_lengths": "unit-native",
        "unit_chain_pred_lengths": "unit-pred",
    }
    representation = representation_aliases.get(
        str(config.get("representation", "coords")),
        slugify_handle(config.get("representation"), fallback="coords"),
    )
    transform = "dct" if bool(config.get("use_DCT", True)) else "dft"
    setup_parts = [
        slugify_handle(config.get("coords_type", "ca"), fallback="ca"),
        representation,
        transform,
        f"k{int(config.get('top_k_freqs', 0))}",
    ]
    tags = config.get("setup_tags") or []
    if isinstance(tags, str):
        tags = [part for part in tags.split(",") if part.strip()]
    setup_parts.extend(slugify_handle(tag) for tag in tags)
    inferred_setup = "-".join(part for part in setup_parts if part)
    setup = slugify_handle(config.get("setup_handle") or inferred_setup, fallback="setup")

    tag = config.get("run_tag")
    descriptor_parts = [experiment, model, setup]
    if tag:
        descriptor_parts.append(slugify_handle(tag))
    return {
        "experiment_handle": experiment,
        "model_handle": model,
        "setup_handle": setup,
        "run_descriptor": "__".join(descriptor_parts),
    }


def plain_container(config: Any, *, resolve: bool = True) -> dict[str, Any]:
    """Return a detached plain dictionary from a mapping or OmegaConf config."""

    try:
        from omegaconf import DictConfig, OmegaConf
    except ImportError:
        DictConfig = ()  # type: ignore[assignment]
        OmegaConf = None

    if OmegaConf is not None and isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=resolve)
    if not isinstance(config, Mapping):
        raise TypeError(f"Expected a mapping config, got {type(config).__name__}.")
    return deepcopy(dict(config))


def flatten_sections(
    config: Any,
    *,
    sections: Iterable[str] = TRAIN_SECTIONS,
) -> dict[str, Any]:
    """Flatten named Hydra sections while rejecting ambiguous key ownership."""

    nested = plain_container(config)
    selected = tuple(sections)
    invalid_sections = sorted(set(selected) - set(DYNAMODE_SECTIONS))
    if invalid_sections:
        raise ValueError(f"Unknown DynaMode config sections: {', '.join(invalid_sections)}")

    unexpected = sorted(set(nested) - set(DYNAMODE_SECTIONS))
    if unexpected:
        raise ValueError(
            "Unexpected top-level config keys; put runtime values in a named section: "
            + ", ".join(unexpected)
        )

    flat: dict[str, Any] = {}
    owners: dict[str, str] = {}
    for section in selected:
        values = nested.get(section)
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise TypeError(f"Config section {section!r} must be a mapping.")
        for key, value in values.items():
            if key in flat:
                raise ValueError(
                    f"Config key {key!r} is defined in both {owners[key]!r} and {section!r}."
                )
            flat[key] = deepcopy(value)
            owners[key] = section
    return flat


def load_run_manifest(
    path: str | Path,
    *,
    sections: Iterable[str] = TRAIN_SECTIONS,
) -> dict[str, Any]:
    """Load the resolved runtime configuration from a DynaMode run manifest."""

    with Path(path).open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if not isinstance(manifest, Mapping):
        raise TypeError(f"Run manifest {path!s} must contain a mapping.")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping) or not isinstance(
        configuration.get("composed"), Mapping
    ):
        raise ValueError(
            f"{path!s} is not a DynaMode run_manifest.yaml: "
            "configuration.composed is required."
        )
    runtime = flatten_sections(configuration["composed"], sections=sections)
    overlay = configuration.get("runtime_overlay") or {}
    if not isinstance(overlay, Mapping):
        raise TypeError("run_manifest.yaml configuration.runtime_overlay must be a mapping.")
    runtime.update(deepcopy(dict(overlay)))
    return runtime


def split_inference_config(config: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split the composed config into model/runtime values and CLI controls."""

    nested = plain_container(config)
    inference = nested.get("inference") or {}
    if not isinstance(inference, Mapping):
        raise TypeError("Config section 'inference' must be a mapping.")
    inference_config = deepcopy(dict(inference))
    model_config = flatten_sections(nested, sections=TRAIN_SECTIONS)
    for key in INFERENCE_RUNTIME_KEYS:
        value = inference_config.get(key)
        if value is not None:
            model_config[key] = deepcopy(value)
    return model_config, inference_config


def coerce_training_config_types(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keep YAML/Hydra values aligned with the established trainer signature."""

    runtime = deepcopy(dict(config))
    for key in TRAIN_FLOAT_KEYS:
        if key not in runtime or runtime[key] is None:
            continue
        if key == "shift_value" and str(runtime[key]).strip().lower() == "auto":
            continue
        runtime[key] = float(runtime[key])
    for key in TRAIN_INT_KEYS:
        if key in runtime and runtime[key] is not None:
            runtime[key] = int(runtime[key])
    if runtime.get("low_k_correction_modes") is not None:
        value = runtime["low_k_correction_modes"]
        if isinstance(value, str):
            value = value.strip()
            if value.isdigit():
                value = int(value)
        runtime["low_k_correction_modes"] = value
    return runtime


def _task_override_keys(overrides: Iterable[str]) -> set[str]:
    keys: set[str] = set()
    for override in overrides:
        expression = str(override).split("=", 1)[0].lstrip("+~")
        if "." not in expression or expression.startswith("hydra."):
            continue
        section = expression.split(".", 1)[0]
        if section not in TRAIN_SECTIONS:
            continue
        keys.add(expression.rsplit(".", 1)[-1])
    return keys


def prepare_resume_config(
    current_config: Mapping[str, Any],
    *,
    task_overrides: Iterable[str] = (),
) -> dict[str, Any]:
    """Restore or validate the recorded run configuration for checkpoint resume."""

    current = deepcopy(dict(current_config))
    if not bool(current.get("resume_from_latest", False)):
        return current

    checkpoint_dir = current.get("checkpoint_dir")
    if not checkpoint_dir:
        raise ValueError("resume_from_latest=true requires an explicit core.checkpoint_dir.")
    latest_checkpoint = Path(str(checkpoint_dir)) / "checkpoint_latest.pt"
    if not latest_checkpoint.is_file():
        raise FileNotFoundError(
            "resume_from_latest=true requires checkpoint_latest.pt in the run directory; "
            f"none was found at {latest_checkpoint}. The manifest restores configuration, "
            "not model or optimizer state."
        )
    mode = str(current.get("resume_config_mode", "restore")).strip().lower()
    if mode not in {"restore", "strict", "current"}:
        raise ValueError("run.resume_config_mode must be restore, strict, or current.")
    if mode == "current":
        return current

    manifest_path = Path(str(checkpoint_dir)) / "run_manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No run_manifest.yaml found in {checkpoint_dir}. "
            "Set run.resume_config_mode=current only if using the active Hydra config is intentional."
        )
    recorded = load_run_manifest(manifest_path)

    if mode == "strict":
        current_hash = config_fingerprint(current)
        recorded_hash = config_fingerprint(recorded)
        if current_hash != recorded_hash:
            raise ValueError(
                "Active Hydra config does not match the recorded resume config: "
                f"current={current_hash}, recorded={recorded_hash}. "
                "Use run.resume_config_mode=restore or intentionally select current."
            )
        return current

    merged = deepcopy(recorded)
    for key in RESUME_CONTROL_KEYS:
        if key in current:
            merged[key] = deepcopy(current[key])

    explicit_keys = _task_override_keys(task_overrides)
    unsafe = bool(current.get("resume_allow_unsafe_overrides", False))
    rejected: list[str] = []
    for key in sorted(explicit_keys - RESUME_CONTROL_KEYS):
        if key not in current:
            continue
        if key not in RESUME_MUTABLE_KEYS and not unsafe:
            rejected.append(key)
            continue
        merged[key] = deepcopy(current[key])
    if rejected:
        raise ValueError(
            "Resume rejected scientific/model overrides: "
            + ", ".join(rejected)
            + ". Set run.resume_allow_unsafe_overrides=true only when checkpoint compatibility "
            "has been considered."
        )
    return merged


def validate_training_config(config: Mapping[str, Any]) -> None:
    """Validate inexpensive cross-field invariants before DDP/data setup."""

    required = ("model_type", "window_size", "top_k_freqs")
    missing = [key for key in required if config.get(key) is None]
    if missing:
        raise ValueError(f"Required training config values are missing: {', '.join(missing)}")
    if config.get("checkpoint_dir") is None and config.get("output_root") is None:
        raise ValueError("Configure run.output_root or an explicit core.checkpoint_dir.")
    if bool(config.get("resume_from_latest", False)) and not config.get("checkpoint_dir"):
        raise ValueError("resume_from_latest=true requires an explicit core.checkpoint_dir.")

    window_size = int(config["window_size"])
    top_k_freqs = int(config["top_k_freqs"])
    if window_size < 1 or not 1 <= top_k_freqs <= window_size:
        raise ValueError(
            f"Expected 1 <= top_k_freqs <= window_size, got {top_k_freqs} and {window_size}."
        )
    spectral_modes = config.get("spectral_modes")
    if spectral_modes is not None and not 1 <= int(spectral_modes) <= top_k_freqs:
        raise ValueError(
            f"Expected 1 <= spectral_modes <= top_k_freqs, got {spectral_modes} and {top_k_freqs}."
        )
    if str(config.get("coords_type", "ca")).lower() not in {"ca", "bb"}:
        raise ValueError("coords_type must be either 'ca' or 'bb'.")
    if str(config.get("prediction_target", "x_0")) not in {"x_0", "noise", "v"}:
        raise ValueError("prediction_target must be one of 'x_0', 'noise', or 'v'.")
    if int(config.get("num_steps", 1)) < 1 or int(config.get("num_ode_steps", 1)) < 1:
        raise ValueError("num_steps and num_ode_steps must be positive.")
    if int(config.get("batch_size", 1)) < 1:
        raise ValueError("batch_size must be positive.")
    if bool(config.get("use_low_k_correction_head", False)) and str(
        config.get("prediction_target", "x_0")
    ) != "x_0":
        raise ValueError("use_low_k_correction_head requires prediction_target='x_0'.")


def prepare_runtime_identity(
    config: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create a descriptive run identity while keeping explicit paths stable."""

    runtime = deepcopy(dict(config))
    if now is None:
        local_time = datetime.now().astimezone()
    elif now.tzinfo is None:
        local_time = now.astimezone()
    else:
        local_time = now
    utc_time = local_time.astimezone(timezone.utc)
    timestamp = local_time.strftime("%H-%M-%S")

    descriptors = derive_run_descriptors(runtime)
    fingerprint = config_fingerprint({**runtime, **descriptors})
    run_prefix = f"test__{timestamp}" if bool(runtime.get("test_only", False)) else timestamp
    run_suffix = f"{descriptors['run_descriptor']}__{fingerprint}"
    run_id = f"{run_prefix}__{run_suffix}"

    checkpoint_dir = runtime.get("checkpoint_dir")
    if bool(runtime.get("resume_from_latest", False)):
        if checkpoint_dir is None:
            raise ValueError("resume_from_latest=true requires an explicit core.checkpoint_dir.")
        manifest_path = Path(str(checkpoint_dir)) / "run_manifest.yaml"
        if manifest_path.exists():
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = yaml.safe_load(handle) or {}
            identity = manifest.get("identity") if isinstance(manifest, Mapping) else None
            if isinstance(identity, Mapping):
                for key in RUN_IDENTITY_KEYS:
                    if identity.get(key) is not None:
                        runtime[key] = deepcopy(identity[key])
                runtime["checkpoint_dir"] = str(checkpoint_dir)
                return runtime
    elif checkpoint_dir is None:
        output_root = runtime.get("output_root")
        if output_root is None:
            raise ValueError("Configure run.output_root or an explicit core.checkpoint_dir.")
        root = Path(str(output_root))
        if bool(runtime.get("timestamp_run_dir", True)):
            date_root = root / local_time.strftime("%Y-%m-%d")
            checkpoint_dir = date_root / run_id
            collision_index = 2
            while checkpoint_dir.exists():
                run_id = f"{run_prefix}-{collision_index:02d}__{run_suffix}"
                checkpoint_dir = date_root / run_id
                collision_index += 1
        else:
            checkpoint_dir = root / f"{descriptors['run_descriptor']}__{fingerprint}"

    runtime.update(descriptors)
    runtime.update(
        {
            "checkpoint_dir": str(checkpoint_dir),
            "config_fingerprint": fingerprint,
            "run_id": run_id,
            "run_name": run_id,
            "started_at_local": local_time.isoformat(timespec="milliseconds"),
            "started_at_utc": utc_time.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        }
    )
    return runtime


def prepare_training_config(
    config: Any,
    *,
    config_choices: Mapping[str, Any] | None = None,
    task_overrides: Iterable[str] = (),
    now: datetime | None = None,
    resolve_identity: bool = True,
) -> dict[str, Any]:
    """Resolve a composed Hydra config into the public flat training contract."""

    runtime = coerce_training_config_types(flatten_sections(config))
    runtime.setdefault("randomize_train_windows", True)
    if runtime.get("representation") is None:
        runtime["representation"] = (
            "displacement" if bool(runtime.get("displacement", False)) else "raw_coords"
        )
    runtime["representation"] = _canonical(
        runtime["representation"], REPRESENTATION_ALIASES, name="representation"
    )
    runtime["displacement"] = runtime["representation"] == "displacement"
    runtime["geo_loss"] = ",".join(
        _parse_geo_loss_modes(runtime.get("geo_loss", "idct_ca-ca"))
    )
    runtime["freq_normalization"] = _canonical(
        runtime.get("freq_normalization", "auto"),
        NORMALIZATION_ALIASES,
        name="freq_normalization",
    )
    runtime["dc_residualization"] = _canonical(
        runtime.get("dc_residualization", "auto"),
        DC_ALIASES,
        name="dc_residualization",
    )
    runtime["aniso_source"] = _canonical(
        runtime.get("aniso_source", "auto"),
        ANISO_ALIASES,
        name="aniso_source",
    )
    if config_choices:
        selected_choices = {
            str(key): str(value)
            for key, value in config_choices.items()
            if value is not None and (key == "experiment" or key in DYNAMODE_SECTIONS)
        }
        runtime["config_choices"] = selected_choices
        if not runtime.get("experiment_handle"):
            runtime["experiment_handle"] = selected_choices.get("experiment")
        if not runtime.get("model_handle"):
            runtime["model_handle"] = selected_choices.get("model")
    runtime = prepare_resume_config(runtime, task_overrides=task_overrides)
    validate_training_config(runtime)
    if not resolve_identity:
        return runtime
    return prepare_runtime_identity(runtime, now=now)


def _git_value(args: list[str], *, cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def collect_provenance(*, cwd: str | Path | None = None) -> dict[str, Any]:
    """Collect a small, non-secret execution and source provenance record."""

    workdir = Path(cwd or Path.cwd()).resolve()
    package_versions: dict[str, str] = {}
    for package in ("dynamode", "torch", "hydra-core", "omegaconf", "numpy", "zarr"):
        try:
            package_versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            continue

    git_root_text = _git_value(["rev-parse", "--show-toplevel"], cwd=workdir)
    git_root = Path(git_root_text) if git_root_text else workdir
    git_status = _git_value(["status", "--porcelain"], cwd=git_root)
    distributed_keys = (
        "LOCAL_RANK",
        "RANK",
        "SLURM_JOB_ID",
        "SLURM_PROCID",
        "TORCHELASTIC_RUN_ID",
        "WORLD_SIZE",
    )
    return {
        "command": [str(arg) for arg in sys.argv],
        "working_directory": str(workdir),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "packages": package_versions,
        "distributed": {key: os.environ[key] for key in distributed_keys if key in os.environ},
        "git": {
            "root": str(git_root) if git_root_text else None,
            "commit": _git_value(["rev-parse", "HEAD"], cwd=git_root),
            "branch": _git_value(["branch", "--show-current"], cwd=git_root),
            "dirty": bool(git_status),
        },
    }


def _identity_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(config.get(key)) for key in RUN_IDENTITY_KEYS if config.get(key) is not None}


def _static_model_spec(nested: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    core = nested.get("core") if isinstance(nested.get("core"), Mapping) else {}
    data = nested.get("data") if isinstance(nested.get("data"), Mapping) else {}
    training = nested.get("training") if isinstance(nested.get("training"), Mapping) else {}
    return {
        "schema_version": 1,
        "identity": _identity_metadata(runtime),
        "architecture": {
            "model_type": runtime.get("model_type") or core.get("model_type"),
            "configured": deepcopy(nested.get("model") or {}),
        },
        "representation": deepcopy(nested.get("representation") or {}),
        "diffusion": deepcopy(nested.get("diffusion") or {}),
        "sampling": deepcopy(nested.get("sampling") or {}),
        "artifacts": {
            "frequency_scales": data.get("freq_scales_path"),
            "rmsf_prior": data.get("rmsf_prior_path"),
            "topology_margins": training.get("topology_margin_path"),
        },
    }


def _write_run_card(target: Path, manifest: Mapping[str, Any]) -> None:
    identity = manifest.get("identity") or {}
    lines = [
        f"# {identity.get('run_id', 'DynaMode run')}",
        "",
        f"- Experiment: `{identity.get('experiment_handle', 'unknown')}`",
        f"- Model: `{identity.get('model_handle', 'unknown')}`",
        f"- Setup: `{identity.get('setup_handle', 'unknown')}`",
        f"- Config fingerprint: `{identity.get('config_fingerprint', 'unknown')}`",
        f"- Started: `{identity.get('started_at_local', 'unknown')}`",
        "",
        "## Metadata files",
        "",
        "- `run_manifest.yaml`: run identity, configuration, overrides, and provenance",
        "- `model_spec.yaml`: configured and runtime-resolved model specification",
        "",
    ]
    with (target / "RUN.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def update_model_spec(checkpoint_dir: str | Path, runtime_spec: Mapping[str, Any]) -> None:
    """Add post-construction model facts to the static model specification."""

    path = Path(checkpoint_dir) / "model_spec.yaml"
    existing: dict[str, Any] = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, Mapping):
            existing = deepcopy(dict(loaded))
    existing["runtime"] = _metadata_safe(runtime_spec)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(existing, handle, sort_keys=False)


def _runtime_overlay(
    nested_config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    configured = flatten_sections(nested_config)
    return {
        key: deepcopy(value)
        for key, value in runtime_config.items()
        if key not in configured or configured[key] != value
    }


def write_run_metadata(
    checkpoint_dir: str | Path,
    nested_config: Any,
    flat_config: Mapping[str, Any],
    overrides: Iterable[str] = (),
) -> None:
    """Write the compact manifest, model specification, and human run card."""

    target = Path(checkpoint_dir)
    target.mkdir(parents=True, exist_ok=True)
    nested = plain_container(nested_config)
    runtime = deepcopy(dict(flat_config))
    manifest_path = target / "run_manifest.yaml"
    existing_manifest: dict[str, Any] = {}
    if bool(runtime.get("resume_from_latest", False)) and manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, Mapping):
            existing_manifest = deepcopy(dict(loaded))
            previous_configuration = existing_manifest.get("configuration") or {}
            if isinstance(previous_configuration, Mapping) and isinstance(
                previous_configuration.get("composed"), Mapping
            ):
                nested = deepcopy(dict(previous_configuration["composed"]))

    identity = _identity_metadata(runtime)
    current_provenance = collect_provenance()
    previous_configuration = existing_manifest.get("configuration") or {}
    initial_overrides = (
        deepcopy(previous_configuration.get("overrides") or [])
        if isinstance(previous_configuration, Mapping)
        else []
    )
    resume_events = deepcopy(existing_manifest.get("resume_events") or [])
    if bool(runtime.get("resume_from_latest", False)):
        resume_events.append(
            {
                "resumed_at_local": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "overrides": list(overrides),
                "provenance": current_provenance,
            }
        )
    manifest = {
        "schema_version": 1,
        "identity": identity,
        "paths": {
            "checkpoint_dir": runtime.get("checkpoint_dir"),
            "output_root": runtime.get("output_root"),
        },
        "resume_from_latest": bool(runtime.get("resume_from_latest", False)),
        "configuration": {
            "choices": deepcopy(runtime.get("config_choices") or {}),
            "overrides": initial_overrides or list(overrides),
            "composed": nested,
            "runtime_overlay": _runtime_overlay(nested, runtime),
        },
        "provenance": deepcopy(existing_manifest.get("provenance") or current_provenance),
        "resume_events": resume_events,
        "artifacts": {
            "run_card": "RUN.md",
            "manifest": "run_manifest.yaml",
            "model_spec": "model_spec.yaml",
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    model_spec_path = target / "model_spec.yaml"
    model_spec = _static_model_spec(nested, runtime)
    if bool(runtime.get("resume_from_latest", False)) and model_spec_path.exists():
        with model_spec_path.open(encoding="utf-8") as handle:
            previous_model_spec = yaml.safe_load(handle) or {}
        if isinstance(previous_model_spec, Mapping) and "runtime" in previous_model_spec:
            model_spec["runtime"] = deepcopy(previous_model_spec["runtime"])
    with model_spec_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(model_spec, handle, sort_keys=False)
    _write_run_card(target, manifest)


__all__ = [
    "DYNAMODE_SECTIONS",
    "INFERENCE_RUNTIME_KEYS",
    "TRAIN_SECTIONS",
    "collect_provenance",
    "coerce_training_config_types",
    "config_fingerprint",
    "derive_run_descriptors",
    "flatten_sections",
    "load_run_manifest",
    "plain_container",
    "prepare_runtime_identity",
    "prepare_resume_config",
    "prepare_training_config",
    "slugify_handle",
    "split_inference_config",
    "update_model_spec",
    "validate_training_config",
    "write_run_metadata",
]
