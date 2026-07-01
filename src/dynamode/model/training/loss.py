'''Training loss helpers for the public DynaMode spectral models.'''

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from dynamode.model.training.metrics import safe_vector_norm
from dynamode.spectral.adapters import DCT


def get_frequency_weights(weighting_type, n_steps, device, tau=32.0, min_weight=0.1):
    '''Generate 1-D frequency-band loss weights.'''
    indices = torch.arange(n_steps, device=device, dtype=torch.float32)
    if weighting_type == "exponential_decay":
        return torch.exp(-indices / tau)
    if weighting_type == "linear_decay":
        return torch.linspace(1.0, min_weight, steps=n_steps, device=device)
    if weighting_type == "quadratic_decay":
        progress = torch.linspace(1.0, math.sqrt(min_weight), steps=n_steps, device=device)
        return progress ** 2
    return torch.ones(n_steps, device=device)


def _prepare_frequency_weights(freq_weights, n_steps, device, dtype):
    '''Return a length-n_steps frequency-weight vector on the requested device/dtype.'''
    if freq_weights is None:
        return torch.ones(n_steps, device=device, dtype=dtype)
    if freq_weights.shape[0] < n_steps:
        raise ValueError(
            f"freq_weights too short: got {freq_weights.shape[0]}, expected at least {n_steps}"
        )
    return freq_weights[:n_steps].to(device=device, dtype=dtype)


def compute_bending_weight(coords, native_coords, mask, bending_lambda, displacement):
    '''Per-residue bending weight from the spatial gradient of deviations.'''
    with torch.no_grad():
        dev = coords if displacement else coords - native_coords.unsqueeze(1)
        diff = dev[:, :, 1:, :] - dev[:, :, :-1, :]
        bend = (diff ** 2).sum(-1).mean(1)
        bend = F.pad(bend, (0, 1))
        bend = bend * mask.float()

        valid_count = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        bend_mean = (bend * mask.float()).sum(dim=1, keepdim=True) / valid_count
        bend_norm = bend / (bend_mean + 1e-8)

        weight = 1.0 + bending_lambda * bend_norm
        weight = weight * mask.float() + (1.0 - mask.float())

    return weight


def local_geometry_loss(
    pred_ca,
    mask,
    target_caca=3.8,
    tol=0.05,
    clash_lambda=0.0,
    clash_threshold=3.5,
    min_sep=2,
    clash_max_pairs=4096,
    clash_pair_chunk=512,
):
    '''CA-CA tolerance-band hinge loss plus optional non-bonded clash hinge.'''
    bsz, n_frames, length, _ = pred_ca.shape

    d_pred = safe_vector_norm(pred_ca[:, :, 1:] - pred_ca[:, :, :-1], dim=-1)
    bond_mask = (mask[:, 1:] * mask[:, :-1]).unsqueeze(1).float()
    violation = (d_pred - target_caca).abs() - tol
    bond_loss = (violation.clamp(min=0) ** 2) * bond_mask
    denom_bond = bond_mask.sum(dim=(1, 2)) * n_frames + 1e-8
    bond_per_sample = bond_loss.sum(dim=(1, 2)) / denom_bond

    if clash_lambda <= 0.0:
        return bond_per_sample

    pair_i, pair_j = torch.triu_indices(length, length, offset=int(min_sep), device=pred_ca.device)
    n_pairs = pair_i.numel()
    max_pairs = min(int(clash_max_pairs), int(n_pairs)) if clash_max_pairs is not None else int(n_pairs)
    if max_pairs <= 0:
        return bond_per_sample
    if max_pairs < n_pairs:
        sel = torch.randint(int(n_pairs), (int(max_pairs),), device=pred_ca.device)
        pair_i = pair_i[sel]
        pair_j = pair_j[sel]

    clash_loss = pred_ca.new_zeros(bsz)
    denom_clash = pred_ca.new_zeros(bsz)
    chunk = max(int(clash_pair_chunk), 1)
    for start in range(0, max_pairs, chunk):
        pi = pair_i[start:start + chunk]
        pj = pair_j[start:start + chunk]
        diff = pred_ca[:, :, pi, :] - pred_ca[:, :, pj, :]
        d = safe_vector_norm(diff, dim=-1)
        pair_valid = (mask[:, pi] * mask[:, pj]).float().unsqueeze(1)
        clash = (clash_threshold - d).clamp(min=0) ** 2
        clash_loss = clash_loss + (clash * pair_valid).sum(dim=(1, 2))
        denom_clash = denom_clash + pair_valid.sum(dim=(1, 2)) * n_frames

    clash_per_sample = clash_loss / (denom_clash + 1e-8)
    return bond_per_sample + clash_lambda * clash_per_sample


GEO_LOSS_ALIASES = {
    "idct_ca-ca": "idct_ca-ca",
    "idct-ca-ca": "idct_ca-ca",
    "idct_caca": "idct_ca-ca",
    "idct-caca": "idct_ca-ca",
    "caca": "idct_ca-ca",
    "ca-ca": "idct_ca-ca",
    "spec_geo": "spec_geo",
    "spectral_geo": "spec_geo",
    "spectral_geometry": "spec_geo",
    "risk_band": "risk_band",
    "risk_bond": "risk_band",
    "band_risk": "risk_band",
}


def parse_geo_loss_modes(value) -> tuple[str, ...]:
    '''Parse comma-delimited geometry auxiliary losses.'''
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
    '''Closest distances between batched 3D line segments.'''
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
    '''Convert selected DCT coeffs (..., K, C) to (..., T, C).'''
    n_modes = coeff.shape[-2]
    w_inv = DCT.get_idct_matrix(int(n_time_steps), coeff.device).to(dtype=coeff.dtype)[:, :n_modes]
    return torch.einsum("...kc,tk->...tc", coeff, w_inv)


def _ca_coeff_view(x_spec, *, top_k_freqs, channels, coord_channels, representation):
    if not getattr(representation, "name", representation) in ("raw_coords", "displacement"):
        return None, None
    bsz, length, dim = x_spec.shape
    n_modes = int(top_k_freqs)
    channels = int(channels)
    if dim < n_modes * channels:
        return None, None
    x = x_spec[..., :n_modes * channels].view(bsz, length, n_modes, channels)
    ca_start = 3 if int(coord_channels) == 12 else 0
    if ca_start + 3 > channels:
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
    rounded_temp = int(round(float(temp)))
    return f"temp_size:{rounded_temp}|{size_label}", f"size:{size_label}", f"temp:{rounded_temp}", "overall"


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
    '''Per-sample lower-bound distance thresholds from the margin artifact.'''
    bsz = mask.shape[0]
    pair = torch.full((bsz,), float(pair_default), device=mask.device, dtype=torch.float32)
    segment = torch.full((bsz,), float(segment_default), device=mask.device, dtype=torch.float32)
    if not margin_artifact:
        return pair, segment

    buckets = margin_artifact.get("buckets", {})
    meta = margin_artifact.get("metadata", {})
    size_bins = meta.get("size_bins", [100, 200, 300, 400, 500, 600])
    temps = raw_temps.detach().float().cpu().tolist()
    lengths = mask.detach().bool().sum(dim=1).cpu().tolist()
    for idx, (temp, length) in enumerate(zip(temps, lengths)):
        for name in _bucket_label_for_sample(temp, length, size_bins):
            stats = buckets.get(name)
            if not stats:
                continue
            nb = stats.get("nonbonded_ca", {})
            seg = stats.get("chain_segment", {})
            nb_val = nb.get(quantile_key, nb.get("q05", nb.get("min")))
            seg_val = seg.get(quantile_key, seg.get("q05", seg.get("min")))
            if nb_val is not None and math.isfinite(float(nb_val)):
                pair[idx] = min(float(pair_default), float(nb_val))
            if seg_val is not None and math.isfinite(float(seg_val)):
                segment[idx] = min(float(segment_default), float(seg_val))
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
    '''Direct spectral-coordinate topology loss on nonlocal pairs and segments.'''
    del target_spec
    pred_coeff, _ = _ca_coeff_view(
        pred_spec,
        top_k_freqs=top_k_freqs,
        channels=channels,
        coord_channels=coord_channels,
        representation=representation,
    )
    if pred_coeff is None:
        z = pred_spec.new_tensor(0.0)
        return z, {"spec_geo_pair": 0.0, "spec_geo_segment": 0.0}

    bsz, length, _n_modes, _ = pred_coeff.shape
    pair_thresh, seg_thresh = resolve_topology_thresholds(
        raw_temps,
        mask,
        margin_artifact,
        pair_default=clash_threshold,
        segment_default=segment_threshold,
    )
    total = pred_spec.new_tensor(0.0)
    metrics: dict[str, float] = {}

    if pair_lambda > 0.0 and length > 2:
        pi, pj = _sample_upper_tri_pairs(length, offset=2, max_pairs=max_pairs, device=pred_spec.device)
        pair_loss = pred_spec.new_zeros(bsz)
        pair_denom = pred_spec.new_zeros(bsz)
        for start in range(0, int(pi.numel()), max(int(pair_chunk), 1)):
            ci = pi[start:start + int(pair_chunk)]
            cj = pj[start:start + int(pair_chunk)]
            coeff_diff = pred_coeff[:, ci, :, :] - pred_coeff[:, cj, :, :]
            pair_time = _dct_time_from_coeffs(coeff_diff, n_time_steps).permute(0, 2, 1, 3)
            if getattr(representation, "is_displacement", False):
                pair_time = pair_time + (native_coords[:, ci, :] - native_coords[:, cj, :]).unsqueeze(1)
            d2 = torch.sum(pair_time * pair_time, dim=-1)
            valid = (mask[:, ci] & mask[:, cj]).float().unsqueeze(1)
            hinge = (pair_thresh.view(bsz, 1, 1).square() - d2).clamp_min(0.0).square()
            pair_loss = pair_loss + (hinge * valid).sum(dim=(1, 2))
            pair_denom = pair_denom + valid.sum(dim=(1, 2)) * int(n_time_steps)
        pair_per = pair_loss / pair_denom.clamp_min(1.0)
        pair_scalar = pair_per.mean()
        total = total + float(pair_lambda) * pair_scalar
        metrics["spec_geo_pair"] = float(pair_scalar.detach().item())

    if segment_lambda > 0.0 and length > 3 and max_segment_pairs is not None and int(max_segment_pairs) > 0:
        si, sj = _sample_upper_tri_pairs(
            length - 1,
            offset=2,
            max_pairs=max_segment_pairs,
            device=pred_spec.device,
        )
        endpoints = torch.stack([si, si + 1, sj, sj + 1], dim=1).reshape(-1)
        unique, inverse = torch.unique(endpoints, sorted=True, return_inverse=True)
        ep_coeff = pred_coeff[:, unique, :, :]
        ep_time = _dct_time_from_coeffs(ep_coeff, n_time_steps).permute(0, 2, 1, 3)
        if getattr(representation, "is_displacement", False):
            ep_time = ep_time + native_coords[:, unique, :].unsqueeze(1)
        ep_time = ep_time[:, :, inverse, :].view(bsz, int(n_time_steps), int(si.numel()), 4, 3)
        dist = _torch_segment_segment_distances(
            ep_time[:, :, :, 0, :],
            ep_time[:, :, :, 1, :],
            ep_time[:, :, :, 2, :],
            ep_time[:, :, :, 3, :],
        )
        valid = (mask[:, si] & mask[:, si + 1] & mask[:, sj] & mask[:, sj + 1]).float().unsqueeze(1)
        hinge = (seg_thresh.view(bsz, 1, 1).square() - dist.square()).clamp_min(0.0).square()
        seg_per = (hinge * valid).sum(dim=(1, 2)) / (
            valid.sum(dim=(1, 2)) * int(n_time_steps)
        ).clamp_min(1.0)
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
    '''Band-wise triangle-inequality risk loss on coefficient error.'''
    pred_coeff, _ = _ca_coeff_view(
        pred_spec,
        top_k_freqs=top_k_freqs,
        channels=channels,
        coord_channels=coord_channels,
        representation=representation,
    )
    target_coeff, _ = _ca_coeff_view(
        target_spec,
        top_k_freqs=top_k_freqs,
        channels=channels,
        coord_channels=coord_channels,
        representation=representation,
    )
    if pred_coeff is None or target_coeff is None:
        z = pred_spec.new_tensor(0.0)
        return z, {"risk_band_pair": 0.0, "risk_band_segment": 0.0}

    bsz, length, n_modes, _ = pred_coeff.shape
    err_coeff = pred_coeff - target_coeff
    pair_thresh, seg_thresh = resolve_topology_thresholds(
        raw_temps,
        mask,
        margin_artifact,
        pair_default=clash_threshold,
        segment_default=segment_threshold,
    )
    edges = [int(edge) for edge in band_edges if int(edge) < n_modes]
    if not edges or edges[0] != 0:
        edges = [0] + edges
    if edges[-1] != n_modes:
        edges.append(n_modes)

    total = pred_spec.new_tensor(0.0)
    metrics: dict[str, float] = {}
    band_count = max(len(edges) - 1, 1)

    if pair_lambda > 0.0 and length > 2:
        pi, pj = _sample_upper_tri_pairs(length, offset=2, max_pairs=max_pairs, device=pred_spec.device)
        target_pair_coeff = target_coeff[:, pi, :, :] - target_coeff[:, pj, :, :]
        target_pair_time = _dct_time_from_coeffs(target_pair_coeff, n_time_steps).permute(0, 2, 1, 3)
        if getattr(representation, "is_displacement", False):
            target_pair_time = target_pair_time + (native_coords[:, pi, :] - native_coords[:, pj, :]).unsqueeze(1)
        target_min = safe_vector_norm(target_pair_time, dim=-1).amin(dim=1)
        margin = (target_min - pair_thresh.view(bsz, 1)).clamp_min(0.0)
        pair_loss_total = pred_spec.new_tensor(0.0)
        for lo, hi in zip(edges[:-1], edges[1:]):
            band_loss = pred_spec.new_zeros(bsz)
            band_denom = pred_spec.new_zeros(bsz)
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

    if segment_lambda > 0.0 and length > 3 and max_segment_pairs is not None and int(max_segment_pairs) > 0:
        si, sj = _sample_upper_tri_pairs(
            length - 1,
            offset=2,
            max_pairs=max_segment_pairs,
            device=pred_spec.device,
        )
        endpoints = torch.stack([si, si + 1, sj, sj + 1], dim=1).reshape(-1)
        unique, inverse = torch.unique(endpoints, sorted=True, return_inverse=True)
        target_ep = _dct_time_from_coeffs(target_coeff[:, unique, :, :], n_time_steps).permute(0, 2, 1, 3)
        if getattr(representation, "is_displacement", False):
            target_ep = target_ep + native_coords[:, unique, :].unsqueeze(1)
        target_ep = target_ep[:, :, inverse, :].view(bsz, int(n_time_steps), int(si.numel()), 4, 3)
        target_dist = _torch_segment_segment_distances(
            target_ep[:, :, :, 0, :],
            target_ep[:, :, :, 1, :],
            target_ep[:, :, :, 2, :],
            target_ep[:, :, :, 3, :],
        )
        margin = (target_dist.amin(dim=1) - seg_thresh.view(bsz, 1)).clamp_min(0.0)
        seg_valid = (mask[:, si] & mask[:, si + 1] & mask[:, sj] & mask[:, sj + 1]).float()
        seg_loss_total = pred_spec.new_tensor(0.0)
        for lo, hi in zip(edges[:-1], edges[1:]):
            err_ep = _dct_time_from_coeffs(err_coeff[:, unique, lo:hi, :], n_time_steps)
            risk_ep = safe_vector_norm(err_ep, dim=-1).amax(dim=-1)
            risk_ep = risk_ep[:, inverse].view(bsz, int(si.numel()), 4).amax(dim=-1)
            hinge = (risk_ep - margin).clamp_min(0.0).square()
            seg_loss_total = seg_loss_total + (
                (hinge * seg_valid).sum(dim=1) / seg_valid.sum(dim=1).clamp_min(1.0)
            ).mean()
        seg_scalar = seg_loss_total / band_count
        total = total + float(segment_lambda) * seg_scalar
        metrics["risk_band_segment"] = float(seg_scalar.detach().item())

    return total, metrics


def backbone_bond_loss(pred_bb_abs, mask):
    '''Constrain backbone covalent bond lengths to ideal values.'''
    _bsz, n_frames, _length, _atoms, _xyz = pred_bb_abs.shape
    pred_n = pred_bb_abs[..., 0, :]
    pred_ca = pred_bb_abs[..., 1, :]
    pred_c = pred_bb_abs[..., 2, :]
    pred_o = pred_bb_abs[..., 3, :]

    delta = 0.1
    res_mask = mask.unsqueeze(1).float()

    def _bond(a, b, ideal, bmask):
        d = safe_vector_norm(a - b, dim=-1)
        err = F.huber_loss(d, torch.full_like(d, ideal), reduction="none", delta=delta)
        return (err * bmask).sum() / (bmask.sum() * n_frames + 1e-8)

    pep_mask = (mask[:, :-1] * mask[:, 1:]).unsqueeze(1).float()
    losses = torch.stack([
        _bond(pred_n, pred_ca, 1.46, res_mask),
        _bond(pred_ca, pred_c, 1.52, res_mask),
        _bond(pred_c, pred_o, 1.23, res_mask),
        _bond(pred_c[:, :, :-1], pred_n[:, :, 1:], 1.33, pep_mask),
    ])
    return losses.mean()


def geometry_schedule_factor(
    epoch,
    warmup_start=50,
    warmup_epochs=10,
    decay_start=200,
    decay_epochs=200,
    min_factor=0.1,
):
    '''Warm up, hold, then slowly decay an auxiliary geometry weight.'''
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
    '''Per-residue signed spectral coefficient matching loss.'''
    bsz, length, dim = x_0_pred.shape
    n_modes = dim // coord_channels

    pred = x_0_pred.view(bsz, length, n_modes, coord_channels)
    gt = x_0_gt.view(bsz, length, n_modes, coord_channels).detach()
    loss = (pred - gt).abs()
    mask_exp = mask.float().unsqueeze(-1).unsqueeze(-1)
    weights = _prepare_frequency_weights(freq_weights, n_modes, x_0_pred.device, loss.dtype)
    weights = weights.view(1, 1, n_modes, 1)
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
    '''MSE on per-residue vector amplitudes for the first n_modes modes.'''
    bsz, length, dim = x_pred.shape
    n_total_modes = dim // coord_channels
    n_modes = min(int(n_modes), n_total_modes)
    pred = x_pred.view(bsz, length, n_total_modes, coord_channels)[:, :, :n_modes, :]
    gt = x_gt.view(bsz, length, n_total_modes, coord_channels)[:, :, :n_modes, :].detach()
    pred_amp = safe_vector_norm(pred, dim=-1)
    gt_amp = safe_vector_norm(gt, dim=-1)
    sq = (pred_amp - gt_amp) ** 2
    if mask is None:
        return sq.mean()
    m = mask.float().unsqueeze(-1)
    return (sq * m).sum() / (m.sum() * n_modes + 1e-8)


def masked_feature_mse(pred, target, mask):
    '''Mean squared error over valid residues for tensors shaped (B, L, C).'''
    mask_exp = mask.float().unsqueeze(-1)
    loss = F.mse_loss(pred, target.detach(), reduction="none")
    return (loss * mask_exp).sum() / (mask.float().sum() * pred.shape[-1] + 1e-8)


def spectral_dc_mse_loss(x_0_pred, x_0_gt, mask, coord_channels=3):
    '''Signed DC-only clean-spectrum MSE on k=0 across coordinate channels.'''
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
    '''Signed low-frequency clean-spectrum MSE on the first n_modes.'''
    bsz, length, dim = x_0_pred.shape
    n_total_modes = dim // coord_channels
    n = min(max(int(n_modes), 1), n_total_modes)

    pred = x_0_pred.view(bsz, length, n_total_modes, coord_channels)[:, :, :n, :]
    gt = x_0_gt.view(bsz, length, n_total_modes, coord_channels)[:, :, :n, :].detach()
    loss = (pred - gt).pow(2)
    mask_exp = mask.float().unsqueeze(-1).unsqueeze(-1)
    weights = _prepare_frequency_weights(freq_weights, n, x_0_pred.device, loss.dtype)
    weights = weights.view(1, 1, n, 1)
    return (loss * mask_exp * weights).sum() / (
        mask.float().sum() * coord_channels * weights.sum() + 1e-8
    )


__all__ = [
    "GEO_LOSS_ALIASES",
    "backbone_bond_loss",
    "compute_bending_weight",
    "geometry_schedule_factor",
    "get_frequency_weights",
    "load_topology_margin_artifact",
    "local_geometry_loss",
    "masked_feature_mse",
    "parse_geo_loss_modes",
    "resolve_topology_thresholds",
    "risk_band_geometry_loss",
    "spectral_amplitude_loss",
    "spectral_dc_mse_loss",
    "spectral_geometry_losses",
    "spectral_low_k_loss",
    "spectral_mode_vector_norm_loss",
]
