'''
Configurable spectral diffusion noise schedules.

The training code diffuses raw spectral coefficients, but our spectral models
normalise those coefficients internally by freq_scales. Presets can be expressed in 
raw coefficient space or in model-normalised space and are resolved to the raw 
aniso_weights tensor for anisotropic noise.
'''

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence
import torch

from dynamode.model.diffusion import make_aniso_weights
from dynamode.model.frequency_bands import parse_band_edges



DEFAULT_5GROUP_BANDS = "DC,1-8,9-32,33-128,129+"
DEFAULT_6GROUP_BANDS = "DC,1-4,5-16,17-64,65-127,128+"


@dataclass
class ResolvedNoiseSchedule:
    schedule: str
    shift_value: float
    aniso_weights: torch.Tensor | None
    diagnostics: dict[str, Any]


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"

def _optional_float(value: Any, default: float | None = None) -> float | None:
    if value is None or _is_auto(value):
        return default
    return float(value)

def _as_list(value: Any, *, cast=float) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [cast(part.strip()) for part in text.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [cast(item) for item in value]
    return [cast(value)]

def _normalise_weights(
    raw_weights: torch.Tensor,
    *,
    mode: str | None = "raw_mean_square",
    freq_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    '''Normalise raw multipliers without changing their relative band ratios.'''
    key = str(mode or "raw_mean_square").strip().lower().replace("-", "_")
    raw_weights = raw_weights.float().flatten().clamp(min=1e-12)
    if key in {"none", "off", "false"}:
        return raw_weights
    if key in {"raw_mean_square", "raw_rms", "mean_square"}:
        return raw_weights / raw_weights.pow(2).mean().sqrt().clamp(min=1e-12)
    if key in {"model_mean_square", "model_rms"}:
        if freq_scales is None:
            raise ValueError("noise_power_normalization='model_mean_square' requires freq_scales.")
        model_noise = raw_weights / freq_scales.float().flatten().clamp(min=1e-12)
        return raw_weights / model_noise.pow(2).mean().sqrt().clamp(min=1e-12)
    raise ValueError(
        f"Unknown noise_power_normalization={mode!r}. "
        "Use 'raw_mean_square', 'model_mean_square', or 'none'."
    )

def _canonical_noise_space(value: Any) -> str:
    key = str(value or "raw_gamma").strip().lower().replace("-", "_")
    aliases = {
        "model": "model_normalized",
        "model_normalised": "model_normalized",
        "model_normalized": "model_normalized",
        "normalised": "model_normalized",
        "normalized": "model_normalized",
        "raw": "raw_gamma",
        "raw_coeff": "raw_gamma",
        "raw_coefficient": "raw_gamma",
        "raw_coefficients": "raw_gamma",
        "raw_gamma": "raw_gamma",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown noise_space={value!r}. Use 'model_normalized' or 'raw_gamma'."
        )
    return aliases[key]



def _shifted_cosine_alpha_bar(
    num_steps: int,
    shift: float = 0.0,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    s: float = 0.008,
) -> torch.Tensor:
    torch_device = torch.device(device)
    # MPS does not support float64 tensors. Use float64 as the intermediate
    # precision where available, but stay in float32 on Apple Silicon.
    work_dtype = torch.float32 if torch_device.type == "mps" else torch.float64
    steps = torch.arange(num_steps + 1, dtype=work_dtype, device=torch_device)
    f_t = torch.cos(((steps / num_steps) + s) / (1.0 + s) * (math.pi / 2.0)) ** 2
    alpha_bars = (f_t / f_t[0]).clamp(min=1e-5, max=0.99999)
    if shift != 0.0:
        log_snr = torch.log(alpha_bars / (1.0 - alpha_bars))
        alpha_bars = torch.sigmoid(log_snr + float(shift))
    return alpha_bars[1:].to(dtype=dtype)

def _base_snr(num_steps: int, shift: float, device: torch.device | str) -> torch.Tensor:
    alpha_bar = _shifted_cosine_alpha_bar(num_steps, shift=shift, device=device)
    return alpha_bar / (1.0 - alpha_bar).clamp(min=1e-12)

def _group_names(edges: Sequence[int], top_k_freqs: int) -> list[str]:
    names: list[str] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if left == 0 and right == 1:
            names.append("DC")
        elif right == top_k_freqs:
            names.append(f"k{left}+")
        else:
            names.append(f"k{left}-{right - 1}")
    return names

def _channel_slice(left: int, right: int, channels: int) -> slice:
    return slice(left * channels, right * channels)

def _default_group_multipliers(num_groups: int, *, strong: bool = False) -> list[float]:
    start, end = (0.35, 2.40) if strong else (0.55, 1.90)
    if num_groups == 1:
        return [1.0]
    return torch.exp(torch.linspace(math.log(start), math.log(end), num_groups)).tolist()

def _default_target_crossings(num_groups: int, *, pressure: bool = False) -> list[int]:
    if num_groups == 5:
        return [520, 470, 390, 300, 230] if pressure else [560, 500, 400, 310, 230]
    if num_groups == 6:
        return [520, 490, 450, 370, 300, 230] if pressure else [560, 520, 470, 390, 310, 230]
    # fallback monotone low-to-high over a similar time span.
    start, end = (520, 230) if pressure else (560, 230)
    if num_groups == 1:
        return [start]
    return torch.linspace(start, end, num_groups).round().int().tolist()

def _find_group_index(anchor: Any, group_names: Sequence[str]) -> int:
    if anchor is None:
        return min(2, len(group_names) - 1)
    if isinstance(anchor, int):
        if 0 <= anchor < len(group_names):
            return int(anchor)
        raise ValueError(f"noise_anchor_band index {anchor} is out of range for {group_names}")
    text = str(anchor).strip().lower()
    if text.isdigit():
        return _find_group_index(int(text), group_names)
    normalized = text if text.startswith("k") or text == "dc" else f"k{text}"
    normalized = normalized.replace(" ", "")
    for idx, name in enumerate(group_names):
        if normalized == name.lower().replace(" ", ""):
            return idx
    raise ValueError(f"noise_anchor_band={anchor!r} did not match groups {list(group_names)}")


def _raw_weights_from_group_model_noise(
    freq_scales: torch.Tensor,
    *,
    channels: int,
    band_edges: Sequence[int],
    group_model_noise: Sequence[float],
    power_normalization: str | None = "raw_mean_square",
) -> torch.Tensor:
    if len(group_model_noise) != len(band_edges) - 1:
        raise ValueError(
            f"Expected {len(band_edges) - 1} group noise values for band_edges={band_edges}, "
            f"got {len(group_model_noise)}."
        )
    raw = torch.empty_like(freq_scales, dtype=torch.float32)
    for value, left, right in zip(group_model_noise, band_edges[:-1], band_edges[1:]):
        raw[_channel_slice(left, right, channels)] = (
            freq_scales[_channel_slice(left, right, channels)].float() * float(value)
        )
    return _normalise_weights(
        raw,
        mode=power_normalization,
        freq_scales=freq_scales,
    )


def _raw_weights_from_group_raw_noise(
    *,
    top_k_freqs: int,
    channels: int,
    band_edges: Sequence[int],
    group_raw_noise: Sequence[float],
    device: torch.device | str,
    freq_scales: torch.Tensor | None = None,
    power_normalization: str | None = "raw_mean_square",
) -> torch.Tensor:
    if len(group_raw_noise) != len(band_edges) - 1:
        raise ValueError(
            f"Expected {len(band_edges) - 1} group noise values for band_edges={band_edges}, "
            f"got {len(group_raw_noise)}."
        )
    raw = torch.empty(top_k_freqs * channels, dtype=torch.float32, device=device)
    for value, left, right in zip(group_raw_noise, band_edges[:-1], band_edges[1:]):
        raw[_channel_slice(left, right, channels)] = float(value)
    return _normalise_weights(
        raw,
        mode=power_normalization,
        freq_scales=freq_scales,
    )


def _band_snr_curves(
    noise_weights: torch.Tensor,
    *,
    channels: int,
    band_edges: Sequence[int],
    num_steps: int,
    shift: float,
    device: torch.device | str,
) -> list[torch.Tensor]:
    base = _base_snr(num_steps, shift, device=device)
    noise_weights = noise_weights.to(device).float().clamp(min=1e-12)
    snr = base[:, None] / noise_weights[None, :].pow(2)
    curves = []
    for left, right in zip(band_edges[:-1], band_edges[1:]):
        part = snr[:, _channel_slice(left, right, channels)]
        curves.append(part.clamp(min=1e-30).log().mean(dim=1).exp())
    return curves


def _first_crossing_below_one(curve: torch.Tensor) -> int | None:
    idx = torch.nonzero(curve.float().flatten() < 1.0, as_tuple=False)
    return None if idx.numel() == 0 else int(idx[0].item())

def _tune_shift_for_anchor(
    effective_noise_weights: torch.Tensor,
    *,
    channels: int,
    band_edges: Sequence[int],
    anchor_group: int,
    target_t: int,
    num_steps: int,
    device: torch.device | str,
    lo: float = -8.0,
    hi: float = 8.0,
) -> float:
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        curves = _band_snr_curves(
            effective_noise_weights,
            channels=channels,
            band_edges=band_edges,
            num_steps=num_steps,
            shift=mid,
            device=device,
        )
        crossing = _first_crossing_below_one(curves[anchor_group])
        crossing = num_steps if crossing is None else crossing
        if crossing < target_t:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def _diagnostics(
    *,
    name: str,
    raw_weights: torch.Tensor | None,
    freq_scales: torch.Tensor | None,
    channels: int,
    band_edges: Sequence[int] | None,
    top_k_freqs: int,
    num_steps: int,
    shift_value: float,
    device: torch.device | str,
    noise_space: str = "raw_gamma",
    power_normalization: str | None = "raw_mean_square",
    group_model_noise: Sequence[float] | None = None,
    target_crossings: Sequence[int] | None = None,
) -> dict[str, Any]:
    diag: dict[str, Any] = {
        "noise_schedule": name,
        "noise_space": noise_space,
        "noise_power_normalization": power_normalization,
        "resolved_shift_value": float(shift_value),
        "has_aniso_weights": raw_weights is not None,
    }
    if raw_weights is None or band_edges is None:
        return diag

    group_names = _group_names(band_edges, top_k_freqs)
    raw_curves = _band_snr_curves(
        raw_weights,
        channels=channels,
        band_edges=band_edges,
        num_steps=num_steps,
        shift=shift_value,
        device=device,
    )
    model_curves = None
    model_noise = None
    if freq_scales is not None:
        model_noise = raw_weights.float() / freq_scales.float().clamp(min=1e-12)
        model_curves = _band_snr_curves(
            model_noise,
            channels=channels,
            band_edges=band_edges,
            num_steps=num_steps,
            shift=shift_value,
            device=device,
        )

    band_rows = []
    for idx, (band, left, right, raw_curve) in enumerate(
        zip(group_names, band_edges[:-1], band_edges[1:], raw_curves)
    ):
        raw_part = raw_weights[_channel_slice(left, right, channels)].float()
        row = {
            "band": band,
            "raw_w_rms": float(raw_part.pow(2).mean().sqrt()),
            "raw_snr_cross_t": _first_crossing_below_one(raw_curve),
        }
        if model_noise is not None and model_curves is not None:
            model_part = model_noise[_channel_slice(left, right, channels)].float()
            model_cross = _first_crossing_below_one(model_curves[idx])
            row["model_noise_rms"] = float(model_part.pow(2).mean().sqrt())
            row["model_snr_cross_t"] = model_cross
        if noise_space == "model_normalized" and "model_snr_cross_t" in row:
            row["snr_cross_t"] = row["model_snr_cross_t"]
        else:
            row["snr_cross_t"] = row["raw_snr_cross_t"]
        if group_model_noise is not None and idx < len(group_model_noise):
            key = "group_model_noise" if noise_space == "model_normalized" else "group_raw_noise"
            row[key] = float(group_model_noise[idx])
        if target_crossings is not None and idx < len(target_crossings):
            row["target_cross_t"] = int(target_crossings[idx])
        band_rows.append(row)

    diag.update(
        {
            "noise_band_edges": tuple(int(edge) for edge in band_edges),
            "noise_bands": group_names,
            "raw_w_min": float(raw_weights.min()),
            "raw_w_max": float(raw_weights.max()),
            "band_rows": band_rows,
        }
    )
    if freq_scales is not None:
        diag["freq_scale_min"] = float(freq_scales.min())
        diag["freq_scale_max"] = float(freq_scales.max())
    return diag


def format_noise_diagnostics(diagnostics: dict[str, Any]) -> str:
    '''Return a concise multi-line string for run logs.'''
    lines = [
        f"Noise schedule: {diagnostics.get('noise_schedule')} "
        f"space={diagnostics.get('noise_space')} "
        f"shift={diagnostics.get('resolved_shift_value')}"
    ]
    if diagnostics.get("raw_w_min") is not None:
        range_line = (
            "  raw_w range="
            f"[{diagnostics['raw_w_min']:.4g}, {diagnostics['raw_w_max']:.4g}]"
        )
        if diagnostics.get("freq_scale_min") is not None:
            range_line += (
                " freq_scale range="
                f"[{diagnostics['freq_scale_min']:.4g}, {diagnostics['freq_scale_max']:.4g}]"
            )
        lines.append(range_line)
    for row in diagnostics.get("band_rows", []):
        parts = [
            f"  {row['band']}:",
            f"raw_rms={row['raw_w_rms']:.4g}",
            f"raw_cross_t={row['raw_snr_cross_t']}",
        ]
        if "model_noise_rms" in row:
            parts.append(f"model_noise_rms={row['model_noise_rms']:.4g}")
            parts.append(f"model_cross_t={row['model_snr_cross_t']}")
        parts.append(f"target_space_cross_t={row['snr_cross_t']}")
        if "target_cross_t" in row:
            parts.append(f"target={row['target_cross_t']}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def resolve_noise_schedule(
    *,
    config: dict[str, Any],
    freq_scales: torch.Tensor | None,
    top_k_freqs: int,
    channels: int,
    num_steps: int,
    device: torch.device | str,
) -> ResolvedNoiseSchedule:
    '''Resolve config fields to ``SpectralDiffusion`` constructor arguments.'''

    preset_raw = config.get("noise_schedule", None)
    preset = str(preset_raw).strip().lower() if preset_raw is not None else ""
    preset = preset.replace("-", "_")
    schedule = str(config.get("schedule", "cosine"))
    shift_config = config.get("shift_value", 0.0)
    noise_space = _canonical_noise_space(config.get("noise_space", "raw_gamma"))
    power_normalization = str(config.get("noise_power_normalization", "raw_mean_square"))

    if preset in {"", "none", "legacy_config"}:
        shift_value = _optional_float(shift_config, default=0.0)
        aniso_gamma = config.get("aniso_gamma")
        aniso_weights = None
        if aniso_gamma is not None and float(aniso_gamma) > 0.0 and freq_scales is not None:
            aniso_weights = make_aniso_weights(
                freq_scales,
                gamma=float(aniso_gamma),
                top_k=top_k_freqs,
                channels=channels,
                legacy_direction=bool(config.get("aniso_legacy_direction", False)),
            )
        diag = _diagnostics(
            name="legacy_config",
            raw_weights=aniso_weights,
            freq_scales=freq_scales,
            channels=channels,
            band_edges=parse_band_edges(config.get("noise_band_edges", DEFAULT_5GROUP_BANDS), top_k_freqs),
            top_k_freqs=top_k_freqs,
            num_steps=num_steps,
            shift_value=float(shift_value or 0.0),
            device=device,
            noise_space=noise_space,
            power_normalization=power_normalization,
        )
        return ResolvedNoiseSchedule(schedule, float(shift_value or 0.0), aniso_weights, diag)

    aliases = {
        "current": "freq_scale_gamma",
        "freq_scale_gamma": "freq_scale_gamma",
        "frequency_scale_gamma": "freq_scale_gamma",
        "anisotropic_gamma": "freq_scale_gamma",
        "gamma": "freq_scale_gamma",
        "legacy": "legacy_high_k",
        "legacy_high_k": "legacy_high_k",
        "cosine_isotropic": "cosine_isotropic",
        "isotropic": "cosine_isotropic",
        "flat_model": "cosine_flat_model",
        "cosine_flat_model": "cosine_flat_model",
        "low_first_mild": "cosine_low_first_mild",
        "cosine_low_first_mild": "cosine_low_first_mild",
        "low_first_strong": "cosine_low_first_strong",
        "cosine_low_first_strong": "cosine_low_first_strong",
        "low_first_targeted": "cosine_low_first_targeted",
        "cosine_low_first_targeted": "cosine_low_first_targeted",
        "cosine_low_first_targeted_pressure": "cosine_low_first_targeted_pressure",
        "low_first_targeted_pressure": "cosine_low_first_targeted_pressure",
        "high_first_ablation": "cosine_high_first_ablation",
        "cosine_high_first_ablation": "cosine_high_first_ablation",
    }
    if preset not in aliases:
        raise ValueError(f"Unknown noise_schedule={preset_raw!r}")
    preset = aliases[preset]

    if preset == "cosine_isotropic":
        shift_value = _optional_float(shift_config, default=0.0)
        return ResolvedNoiseSchedule(
            schedule,
            float(shift_value or 0.0),
            None,
            _diagnostics(
                name=preset,
                raw_weights=None,
                freq_scales=freq_scales,
                channels=channels,
                band_edges=None,
                top_k_freqs=top_k_freqs,
                num_steps=num_steps,
                shift_value=float(shift_value or 0.0),
                device=device,
            ),
        )

    if freq_scales is None and preset in {"freq_scale_gamma", "legacy_high_k"}:
        raise ValueError(
            f"noise_schedule={preset!r} derives gamma weights from frequency scales; "
            "provide freq_scales_path/aniso_source or use another preset."
        )
    if freq_scales is None and noise_space == "model_normalized":
        raise ValueError(
            f"noise_schedule={preset!r} with noise_space='model_normalized' requires "
            "frequency scales/aniso_source. Use noise_space='raw_gamma', "
            "noise_schedule='cosine_isotropic', or provide freq_scales_path/aniso_source."
        )

    if freq_scales is not None:
        freq_scales = freq_scales[: top_k_freqs * channels].float().to(device).clamp(min=1e-8)
    band_spec = config.get("noise_band_edges") or DEFAULT_5GROUP_BANDS
    band_edges = parse_band_edges(band_spec, top_k_freqs)
    group_names = _group_names(band_edges, top_k_freqs)
    group_model_noise: list[float] | None = None
    target_crossings: list[int] | None = None

    if preset == "freq_scale_gamma":
        gamma = float(config.get("aniso_gamma", 0.5) if config.get("aniso_gamma") is not None else 0.5)
        raw_weights = make_aniso_weights(
            freq_scales,
            gamma=gamma,
            top_k=top_k_freqs,
            channels=channels,
            legacy_direction=False,
        ).to(device)
        shift_value = _optional_float(shift_config, default=0.0)
    elif preset == "legacy_high_k":
        gamma = float(config.get("aniso_gamma", 0.5) if config.get("aniso_gamma") is not None else 0.5)
        raw_weights = make_aniso_weights(
            freq_scales,
            gamma=gamma,
            top_k=top_k_freqs,
            channels=channels,
            legacy_direction=True,
        ).to(device)
        shift_value = _optional_float(shift_config, default=0.0)
    elif preset == "cosine_flat_model":
        group_model_noise = [1.0] * (len(band_edges) - 1)
        if noise_space == "model_normalized":
            raw_weights = _raw_weights_from_group_model_noise(
                freq_scales,
                channels=channels,
                band_edges=band_edges,
                group_model_noise=group_model_noise,
                power_normalization=power_normalization,
            ).to(device)
        else:
            raw_weights = _raw_weights_from_group_raw_noise(
                top_k_freqs=top_k_freqs,
                channels=channels,
                band_edges=band_edges,
                group_raw_noise=group_model_noise,
                device=device,
                freq_scales=freq_scales,
                power_normalization=power_normalization,
            )
        shift_value = _optional_float(shift_config, default=0.0)
    elif preset in {"cosine_low_first_mild", "cosine_low_first_strong"}:
        strong = preset.endswith("strong")
        group_model_noise = _as_list(config.get("noise_group_model_multipliers"), cast=float)
        if not group_model_noise:
            group_model_noise = _default_group_multipliers(len(band_edges) - 1, strong=strong)
        if noise_space == "model_normalized":
            raw_weights = _raw_weights_from_group_model_noise(
                freq_scales,
                channels=channels,
                band_edges=band_edges,
                group_model_noise=group_model_noise,
                power_normalization=power_normalization,
            ).to(device)
        else:
            raw_weights = _raw_weights_from_group_raw_noise(
                top_k_freqs=top_k_freqs,
                channels=channels,
                band_edges=band_edges,
                group_raw_noise=group_model_noise,
                device=device,
                freq_scales=freq_scales,
                power_normalization=power_normalization,
            )
        shift_value = _optional_float(shift_config, default=0.0)
    elif preset in {"cosine_low_first_targeted", "cosine_low_first_targeted_pressure", "cosine_high_first_ablation"}:
        pressure = preset == "cosine_low_first_targeted_pressure"
        target_crossings = _as_list(config.get("noise_target_crossings"), cast=int)
        if not target_crossings:
            target_crossings = _default_target_crossings(len(band_edges) - 1, pressure=pressure)
        if preset == "cosine_high_first_ablation":
            target_crossings = list(reversed(target_crossings))
        if len(target_crossings) != len(band_edges) - 1:
            raise ValueError(
                f"noise_target_crossings length {len(target_crossings)} does not match "
                f"{len(band_edges) - 1} noise bands {group_names}."
            )
        base = _base_snr(num_steps, shift=0.0, device=device)
        group_model_noise = [
            float(torch.sqrt(base[int(max(0, min(num_steps - 1, t)))].clamp(min=1e-12)))
            for t in target_crossings
        ]
        if noise_space == "model_normalized":
            raw_weights = _raw_weights_from_group_model_noise(
                freq_scales,
                channels=channels,
                band_edges=band_edges,
                group_model_noise=group_model_noise,
                power_normalization=power_normalization,
            ).to(device)
            effective_noise_for_shift = raw_weights / freq_scales.float().clamp(min=1e-12)
        else:
            raw_weights = _raw_weights_from_group_raw_noise(
                top_k_freqs=top_k_freqs,
                channels=channels,
                band_edges=band_edges,
                group_raw_noise=group_model_noise,
                device=device,
                freq_scales=freq_scales,
                power_normalization=power_normalization,
            )
            effective_noise_for_shift = raw_weights
        if _is_auto(shift_config) or config.get("noise_auto_shift", False):
            anchor_group = _find_group_index(config.get("noise_anchor_band"), group_names)
            shift_value = _tune_shift_for_anchor(
                effective_noise_for_shift,
                channels=channels,
                band_edges=band_edges,
                anchor_group=anchor_group,
                target_t=target_crossings[anchor_group],
                num_steps=num_steps,
                device=device,
            )
        else:
            shift_value = _optional_float(shift_config, default=0.0)
    else:
        raise AssertionError(f"Unhandled noise_schedule preset {preset!r}")

    diag = _diagnostics(
        name=preset,
        raw_weights=raw_weights,
        freq_scales=freq_scales,
        channels=channels,
        band_edges=band_edges,
        top_k_freqs=top_k_freqs,
        num_steps=num_steps,
        shift_value=float(shift_value or 0.0),
        device=device,
        noise_space=noise_space,
        power_normalization=power_normalization,
        group_model_noise=group_model_noise,
        target_crossings=target_crossings,
    )
    return ResolvedNoiseSchedule(schedule, float(shift_value or 0.0), raw_weights, diag)
