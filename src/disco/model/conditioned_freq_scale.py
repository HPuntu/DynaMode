from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def size_bin_label(length: int, cutoffs: list[int]) -> str:
    for cutoff in cutoffs:
        if length <= cutoff:
            return f"le_{cutoff}"
    return f"gt_{cutoffs[-1]}"


def _flatten_tensor(x: Any) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    return x.detach().cpu().float().reshape(-1)


def is_conditioned_freq_scale_payload(obj: Any) -> bool:
    return isinstance(obj, dict) and "global_scale" in obj and "conditioned_scales" in obj


def trim_conditioned_freq_scale_payload(payload: dict[str, Any], expected_dim: int) -> dict[str, Any]:
    meta = dict(payload.get("metadata", {}))
    top_k = int(meta.get("top_k", 0))
    if top_k <= 0:
        raise ValueError("conditioned freq-scale payload metadata missing positive top_k")
    if expected_dim % top_k != 0:
        raise ValueError(
            f"expected_dim={expected_dim} is not divisible by top_k={top_k} for conditioned payload"
        )

    n_channels = expected_dim // top_k
    out = {
        "metadata": meta,
        "global_scale": _flatten_tensor(payload["global_scale"])[:expected_dim].contiguous(),
        "conditioned_scales": {
            str(k): _flatten_tensor(v)[:expected_dim].contiguous()
            for k, v in payload.get("conditioned_scales", {}).items()
        },
        "dc_baselines": {
            str(k): _flatten_tensor(v).contiguous()
            for k, v in payload.get("dc_baselines", {}).items()
        },
    }
    # Preserve per-(protein, temp) per-residue DC table as 2-D tensors
    # (L_full, coord_channels). Keys are strings "f{domain_key}|{temp}".
    prdc = payload.get("per_residue_dc_baselines") or {}
    if prdc:
        out["per_residue_dc_baselines"] = {
            str(k): (v if torch.is_tensor(v) else torch.as_tensor(v)).float().detach().cpu().contiguous()
            for k, v in prdc.items()
        }
    meta["n_channels"] = n_channels
    return out


def load_freq_scale_artifact(path: str, expected_dim: int, map_location: str | torch.device = "cpu"):
    obj = torch.load(path, map_location=map_location)
    if torch.is_tensor(obj):
        if obj.shape[0] < expected_dim:
            raise ValueError(
                f"Loaded freq_scales too short: got {obj.shape[0]}, expected at least {expected_dim}"
            )
        return obj[:expected_dim].contiguous().float(), None

    if isinstance(obj, dict):
        if is_conditioned_freq_scale_payload(obj):
            payload = trim_conditioned_freq_scale_payload(obj, expected_dim)
            return payload["global_scale"], payload

        if "freq_scale" in obj and torch.is_tensor(obj["freq_scale"]):
            scale = obj["freq_scale"]
            if scale.shape[0] < expected_dim:
                raise ValueError(
                    f"Loaded freq_scale too short: got {scale.shape[0]}, expected at least {expected_dim}"
                )
            return scale[:expected_dim].contiguous().float(), None

    raise TypeError(
        f"Unsupported freq-scale artifact at {path!r}; expected tensor or conditioned payload dict."
    )


def build_conditioned_freq_scale_payload(
    stats_payload: dict[str, Any],
    *,
    alpha: float = 0.75,
    stat_name: str = "abs_q75",
    coord_channels: int = 3,
    scale_condition_modes: int = 8,
    scale_condition_channels: int | None = None,
) -> dict[str, Any]:
    meta = dict(stats_payload["metadata"])
    buckets = stats_payload["buckets"]

    top_k = int(meta["top_k"])
    n_channels = int(meta["n_channels"])
    feature_dim = int(meta["feature_dim"])
    scale_condition_channels = int(scale_condition_channels or min(coord_channels, n_channels))
    scale_condition_modes = min(int(scale_condition_modes), top_k - 1)

    overall_feature = buckets["overall"]["feature"]
    global_scale = _flatten_tensor(overall_feature[stat_name])[:feature_dim]
    global_dc = _flatten_tensor(overall_feature["mean"])[:coord_channels]

    conditioned_scales: dict[str, torch.Tensor] = {}
    dc_baselines: dict[str, torch.Tensor] = {"overall": global_dc.contiguous()}

    for bucket_name, bucket_stats in buckets.items():
        if bucket_name == "overall":
            continue
        feature_stats = bucket_stats["feature"]
        bucket_scale = _flatten_tensor(feature_stats[stat_name])[:feature_dim]
        bucket_dc = _flatten_tensor(feature_stats["mean"])[:coord_channels]
        conditioned_scales[bucket_name] = (
            (1.0 - alpha) * global_scale + alpha * bucket_scale
        ).contiguous()
        dc_baselines[bucket_name] = (
            (1.0 - alpha) * global_dc + alpha * bucket_dc
        ).contiguous()

    return {
        "metadata": {
            "source_stats_metadata": meta,
            "scheme": "temp_size_shrunk",
            "alpha": float(alpha),
            "stat_name": stat_name,
            "top_k": top_k,
            "n_channels": n_channels,
            "feature_dim": feature_dim,
            "coords_type": meta.get("coords_type", "unknown"),
            "coords_only": meta.get("coords_only", False),
            "representation": meta.get("representation", "displacement" if meta.get("displacement", False) else "raw_coords"),
            "displacement": meta.get("displacement", meta.get("legacy_displacement", False)),
            "representation_length_min": meta.get("representation_length_min", None),
            "representation_length_max": meta.get("representation_length_max", None),
            "representation_length_residual_max": meta.get("representation_length_residual_max", None),
            "size_bins": list(meta.get("size_bins", [])),
            "temps_seen": list(meta.get("temps_seen", [])),
            "subset_tag": meta.get("subset_tag", None),
            "coord_channels": int(coord_channels),
            "scale_condition_modes": int(scale_condition_modes),
            "scale_condition_channels": int(scale_condition_channels),
        },
        "global_scale": global_scale.contiguous(),
        "conditioned_scales": conditioned_scales,
        "dc_baselines": dc_baselines,
    }


@dataclass
class ConditionedFreqScaleLookup:
    payload: dict[str, Any]

    def __post_init__(self):
        meta = self.payload["metadata"]
        self.top_k = int(meta["top_k"])
        self.n_channels = int(meta["n_channels"])
        self.coord_channels = int(meta.get("coord_channels", min(3, self.n_channels)))
        self.size_bins = [int(x) for x in meta.get("size_bins", [])]
        self.temps_seen = [int(x) for x in meta.get("temps_seen", [])]
        self.scale_condition_modes = int(meta.get("scale_condition_modes", min(8, self.top_k - 1)))
        self.scale_condition_channels = int(
            meta.get("scale_condition_channels", min(self.coord_channels, self.n_channels))
        )
        self.global_scale = _flatten_tensor(self.payload["global_scale"]).contiguous()
        self.conditioned_scales = {
            str(k): _flatten_tensor(v).contiguous()
            for k, v in self.payload.get("conditioned_scales", {}).items()
        }
        self.dc_baselines = {
            str(k): _flatten_tensor(v).contiguous()
            for k, v in self.payload.get("dc_baselines", {}).items()
        }
        # Optional per-(protein, temp) per-residue DC baselines. Each value is a
        # (L_full, coord_channels) tensor in spectrum-space DC units. Looked up
        # by the dataloader and passed through the batch as `dc_baseline_per_res`.
        raw_prdc = self.payload.get("per_residue_dc_baselines", {}) or {}
        self.per_residue_dc_baselines: dict[str, torch.Tensor] = {}
        for k, v in raw_prdc.items():
            t = v if torch.is_tensor(v) else torch.as_tensor(v)
            self.per_residue_dc_baselines[str(k)] = t.detach().cpu().float().contiguous()

    def has_per_residue_dc(self) -> bool:
        return bool(self.per_residue_dc_baselines)

    def lookup_per_residue_dc(
        self,
        domain_key: str,
        temp: int,
        start_res: int,
        n_res: int,
        *,
        coord_channels: int | None = None,
    ) -> torch.Tensor | None:
        """Return the (n_res, coord_channels) per-residue DC slice for this crop.

        Resolves ``f"{domain_key}|{temp}"``; if missing, falls back to the
        nearest observed temperature for the same protein. Returns ``None``
        when the protein is absent from the per-residue table (e.g. ATLAS),
        so the caller can fall back to bucket-level DC."""
        if not self.per_residue_dc_baselines:
            return None
        coord_channels = int(coord_channels or self.coord_channels)
        kt = f"{domain_key}|{int(temp)}"
        row = self.per_residue_dc_baselines.get(kt)
        if row is None:
            candidates = [
                (abs(int(t) - int(temp)), t)
                for k in self.per_residue_dc_baselines
                if k.startswith(f"{domain_key}|")
                for t in [k.split("|", 1)[1]]
            ]
            if not candidates:
                return None
            _, best_temp = min(candidates)
            row = self.per_residue_dc_baselines[f"{domain_key}|{best_temp}"]
        row = row[:, :coord_channels]
        end = start_res + n_res
        if end <= row.shape[0]:
            return row[start_res:end].contiguous()
        pad = torch.zeros(end - row.shape[0], coord_channels, dtype=row.dtype)
        return torch.cat([row[start_res:], pad], dim=0).contiguous()

    def _to_kelvin(self, temp: torch.Tensor) -> torch.Tensor:
        temp = temp.float()
        if temp.numel() == 0:
            return temp
        if temp.max().item() <= 1.5 and temp.min().item() >= -0.5:
            return temp * 200.0 + 250.0
        return temp

    def _nearest_temp(self, temp_value: float) -> int:
        if not self.temps_seen:
            return int(round(temp_value))
        return min(self.temps_seen, key=lambda t: abs(float(t) - float(temp_value)))

    def _bucket_name(self, temp_value: float, length: int) -> tuple[str, str, str]:
        temp_bin = self._nearest_temp(temp_value)
        size_label = size_bin_label(int(length), self.size_bins)
        return (
            f"temp_size:{temp_bin}|{size_label}",
            f"temp:{temp_bin}",
            f"size:{size_label}",
        )

    def _effective_lengths(
        self,
        mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            return mask.float().sum(dim=1)
        if seq_len is None:
            raise ValueError("seq_len is required when mask is None")
        dev = device if device is not None else torch.device("cpu")
        return torch.full((batch_size,), float(seq_len), device=dev)

    def lookup_scales(
        self,
        temp: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        seq_len: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        dev = device or temp.device
        B = temp.shape[0]
        temp_k = self._to_kelvin(temp.detach().cpu())
        lengths = self._effective_lengths(mask.detach().cpu() if mask is not None else None, B, seq_len=seq_len)

        rows = []
        for i in range(B):
            temp_size_name, temp_name, size_name = self._bucket_name(float(temp_k[i].item()), int(lengths[i].item()))
            scale = self.conditioned_scales.get(temp_size_name)
            if scale is None:
                scale = self.conditioned_scales.get(temp_name)
            if scale is None:
                scale = self.conditioned_scales.get(size_name)
            if scale is None:
                scale = self.global_scale
            rows.append(scale)
        return torch.stack(rows, dim=0).to(dev)

    def lookup_dc_baselines(
        self,
        temp: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        coord_channels: int | None = None,
        seq_len: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        coord_channels = int(coord_channels or self.coord_channels)
        dev = device or temp.device
        B = temp.shape[0]
        temp_k = self._to_kelvin(temp.detach().cpu())
        lengths = self._effective_lengths(mask.detach().cpu() if mask is not None else None, B, seq_len=seq_len)

        rows = []
        for i in range(B):
            temp_size_name, temp_name, size_name = self._bucket_name(float(temp_k[i].item()), int(lengths[i].item()))
            baseline = self.dc_baselines.get(temp_size_name)
            if baseline is None:
                baseline = self.dc_baselines.get(temp_name)
            if baseline is None:
                baseline = self.dc_baselines.get(size_name)
            if baseline is None:
                baseline = self.dc_baselines["overall"]
            rows.append(baseline[:coord_channels])
        return torch.stack(rows, dim=0).to(dev)

    def lookup_scale_features(
        self,
        temp: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        seq_len: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        scales = self.lookup_scales(temp, mask, seq_len=seq_len, device=device)
        B = scales.shape[0]
        view = scales.view(B, self.top_k, self.n_channels)
        modes = min(self.scale_condition_modes + 1, self.top_k)
        chans = min(self.scale_condition_channels, self.n_channels)
        return view[:, :modes, :chans].reshape(B, modes * chans)

    def residualise_dc(
        self,
        x: torch.Tensor,
        temp: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        coord_channels: int | None = None,
        seq_len: int | None = None,
        per_residue_baseline: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Subtract the DC baseline from ``x`` at k=0 and return the residual.

        If ``per_residue_baseline`` is provided (shape ``(B, L, coord_channels)``),
        it is subtracted per-residue — this is the dataloader-supplied
        per-(protein, temp) per-residue baseline, which captures the strongly
        temperature-dependent per-residue drift that bucket-level baselines
        average away. The returned ``baseline`` then has shape ``(B, L, C)``
        and ``restore_dc`` will add it back.

        Otherwise falls back to the bucket-level ``(B, C)`` baseline, which is
        broadcast across all residues of each protein.
        """
        coord_channels = int(coord_channels or self.coord_channels)
        if per_residue_baseline is not None:
            per_res = per_residue_baseline.to(device=x.device, dtype=x.dtype)
            per_res = per_res[..., :coord_channels]
            x_out = x.clone()
            x_out[:, :, :coord_channels] = x_out[:, :, :coord_channels] - per_res
            return x_out, per_res
        baseline = self.lookup_dc_baselines(
            temp, mask, coord_channels=coord_channels, seq_len=seq_len, device=x.device
        )
        x_out = x.clone()
        x_out[:, :, :coord_channels] = x_out[:, :, :coord_channels] - baseline.unsqueeze(1)
        return x_out, baseline

    def restore_dc(
        self,
        x: torch.Tensor,
        baseline: torch.Tensor | None,
        *,
        coord_channels: int | None = None,
    ) -> torch.Tensor:
        if baseline is None:
            return x
        coord_channels = int(coord_channels or baseline.shape[-1])
        x_out = x.clone()
        if baseline.dim() == 3:  # per-residue (B, L, C)
            x_out[:, :, :coord_channels] = x_out[:, :, :coord_channels] + baseline[..., :coord_channels]
        else:                    # per-protein (B, C)
            x_out[:, :, :coord_channels] = x_out[:, :, :coord_channels] + baseline[:, None, :coord_channels]
        return x_out
