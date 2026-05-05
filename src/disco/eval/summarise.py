"""Shared analysis/plotting/table helpers for ICML 2026 model-comparison notebooks.

Loads results from ``results/{version}/`` for each model version (v8, v9, v10, v11)
and produces figure and LaTeX-table wrappers that mirror the styling already used
in ``manuscripts/icml_2026/figures.ipynb`` and ``manuscripts/icml_2026/tables.ipynb``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
except Exception:
    torch = None


# ── Paths / constants ────────────────────────────────────────────────────────

REPO_ROOT = Path("/Users/kebl8116-admin/Projects/pancake")
RESULTS_ROOT = REPO_ROOT / "results"

DEFAULT_VERSIONS = ("v8", "v9", "v10", "v11")
SOURCES = ("mdcath", "atlas")
MDCATH_TEMPS = (320, 348, 379, 413, 450)

# Version labels for figure legends / table headers
VERSION_LABEL = {
    "v8": "Pancake-v8 (prior)",
    "v9": "Pancake-v9 (slow-branch)",
    "v10": "Pancake-v10 (block-mix)",
    "v11": "Pancake-v11 (dual-branch)",
    "v11a": "Pancake-v11a",
    "v11b": "Pancake-v11b",
    "v12a": "Pancake-v12a",
    "v12b": "Pancake-v12b",
}

# Per-version palette — picked from the existing ICML blue/yellow scheme so the
# figures blend with the rest of the manuscript.
VERSION_COLORS = {
    "v8":  "#BDD0E0",   # BLUE_LIGHT
    "v9":  "#5C7FA3",   # BLUE_DARK
    "v10": "#C7A85A",   # YELLOW_DARK
    "v11": "#7D5BA6",   # soft plum, distinct from blues/yellows
    "v11a": "#94A7B8",  # muted slate-blue
    "v11b": "#406784",  # deeper steel blue
    "v12a": "#B8843E",  # warm amber-brown
    "v12b": "#8F6B2F",  # darker amber-brown
}

# Style constants re-exported from the existing figure style (figures.ipynb).
BLUE_DARK   = "#5C7FA3"
BLUE        = "#86A9C6"
BLUE_LIGHT  = "#BDD0E0"
YELLOW_DARK = "#C7A85A"
YELLOW      = "#DEC78A"
YELLOW_LIGHT = "#EFE3BE"
SLATE       = "#6F7B87"
INK         = "#23303B"
BG          = "#FAFBFC"

ICML_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
    "text.usetex": False,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.5,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.0,
}


def apply_icml_style():
    """Apply the shared matplotlib rcParams used across the manuscript figures."""
    plt.rcParams.update(ICML_RC)


# ── Loading ──────────────────────────────────────────────────────────────────

# ── Loading ──────────────────────────────────────────────────────────────────

def _results_dir(version: str) -> Path:
    return RESULTS_ROOT / version


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _extract_source_payload(payload, source: str) -> dict:
    """Extract one source's payload from a combined evaluation JSON.

    New combined layout is a flat dict like:
      {
        "evaluation/run_name": ...,
        "mdcath/per_target_rmsf_r_median": ...,
        "atlas/per_target_rmsf_r_median": ...,
      }

    Returns a per-source dict with the source prefix stripped, optionally
    including shared evaluation/* metadata unchanged.
    """
    if not isinstance(payload, dict):
        raise KeyError(f"Expected dict payload, got {type(payload).__name__}")

    source_prefix = f"{source}/"
    out = {}

    for k, v in payload.items():
        if not isinstance(k, str):
            continue
        if k.startswith(source_prefix):
            out[k[len(source_prefix):]] = v
        elif k.startswith("evaluation/"):
            out[k] = v  # optional; keeps shared metadata available

    if out:
        return out

    raise KeyError(f"Could not find source={source!r} in combined evaluation payload")

def _normalise_summary_keys(d: dict) -> dict:
    """Map new flat evaluation-summary keys to the legacy names expected downstream."""
    out = dict(d)

    rename_map = {
        "pairwise_rmsd_r_across_targets": "pairwise_rmsd_r",
    }
    for new_key, legacy_key in rename_map.items():
        if new_key in d and legacy_key not in out:
            out[legacy_key] = d[new_key]

    explicit_test_map = {
        "global_rmsf_spearman": "global_rmsf_spearman",
        "global_rmsf_r": "global_rmsf_pearson",
        "per_target_rmsf_spearman_median": "per_target_rmsf_spearman",
        "per_target_rmsf_r_median": "per_target_rmsf_pearson",
        "spec_dc_r_mean": "spectral_dc_r",
        "spec_dc_amp_recovery_mean": "spectral_dc_amp_recovery",
        "spec_low_r_mean": "spectral_low_r",
        "spec_mid_r_mean": "spectral_mid_r",
        "spec_high_r_mean": "spectral_high_r",
        "spec_total_r_mean": "spectral_total_r",
        "caca_jsd_mean": "global_caca_jsd",
    }
    for src_key, explicit_key in explicit_test_map.items():
        if src_key in d and f"test/{explicit_key}" not in out:
            out[f"test/{explicit_key}"] = d[src_key]

    legacy_explicit_map = {
        "test/rmsf_spearman": "test/global_rmsf_spearman",
        "test/rmsf_pearson": "test/global_rmsf_pearson",
        "trajectory_benchmark/global_rmsf_r": "test/global_rmsf_pearson",
        "trajectory_benchmark/per_target_rmsf_spearman_median": "test/per_target_rmsf_spearman",
        "trajectory_benchmark/per_target_rmsf_r_median": "test/per_target_rmsf_pearson",
    }
    for legacy_key, explicit_key in legacy_explicit_map.items():
        if legacy_key in out and explicit_key not in out:
            out[explicit_key] = out[legacy_key]

    compat_test_map = {
        "global_rmsf_spearman": "rmsf_spearman",
        "global_rmsf_r": "rmsf_pearson",
        "spec_low_r_mean": "low_band_ratio",
        "spec_mid_r_mean": "mid_band_ratio",
        "spec_high_r_mean": "high_band_ratio",
    }
    for src_key, compat_key in compat_test_map.items():
        if src_key in d and f"test/{compat_key}" not in out:
            out[f"test/{compat_key}"] = d[src_key]

    # trajectory benchmark keys now stored without namespace
    traj_keys = [
        "pairwise_rmsd_r",
        "global_rmsf_r",
        "per_target_rmsf_spearman_median",
        "per_target_rmsf_r_median",
        "rmwd_median",
        "md_pca_w2_median",
        "alphaflow_md_pca_emd_median",
        "pc_sim_gt_0_5_pct",
        "alphaflow_pc_sim_gt_0_5_pct",
        "weak_contacts_j_median",
        "transient_contacts_j_median",
        "dG_fold_r",
        "fnc_mean_pred_median",
        "fnc_mean_ref_median",
        "rg_jsd_median",
    ]
    for k in traj_keys:
        if k in d and f"trajectory_benchmark/{k}" not in out:
            out[f"trajectory_benchmark/{k}"] = d[k]

    # add more mappings here if your "test/..." keys were also flattened
    test_keys = [
        "rmsf_spearman",
        "rmsf_spearman_per_sample_mean",
        "rmsf_pearson",
        "rmsf_pearson_per_sample_mean",
        "lddt_mean",
        "caca_dist_A_mean",
        "spectral_mse_mean",
        "low_band_power_ratio_mean",
        "mid_band_power_ratio_mean",
        "high_band_power_ratio_mean",
        "dc_error_mean",
        "dc_pred_over_gt",
        "frame_disp_pred_over_gt",
    ]
    for k in test_keys:
        if k in d and f"test/{k}" not in out:
            out[f"test/{k}"] = d[k]

    temp_metric_aliases = {
        "spec_dc_r": "spectral_dc_r",
        "spec_dc_amp_recovery": "spectral_dc_amp_recovery",
        "spec_low_r": "spectral_low_r",
        "spec_mid_r": "spectral_mid_r",
        "spec_high_r": "spectral_high_r",
        "spec_total_r": "spectral_total_r",
        "global_rmsf_spearman": "global_rmsf_spearman",
        "global_rmsf_r": "global_rmsf_pearson",
        "per_target_rmsf_spearman": "per_target_rmsf_spearman",
        "per_target_rmsf_r": "per_target_rmsf_pearson",
    }
    for key, value in d.items():
        if not isinstance(key, str) or not key.startswith("temp_"):
            continue
        if f"test/{key}" not in out:
            out[f"test/{key}"] = value
        m = re.fullmatch(r"(temp_\d+)/(.*)", key)
        if not m:
            continue
        temp_prefix, metric_tail = m.groups()
        alias_tail = metric_tail
        for src_metric, alias_metric in temp_metric_aliases.items():
            if metric_tail == src_metric:
                alias_tail = alias_metric
                break
            if metric_tail.startswith(f"{src_metric}_"):
                alias_tail = f"{alias_metric}_{metric_tail[len(src_metric) + 1:]}"
                break
        alias_key = f"test/{temp_prefix}/{alias_tail}"
        if alias_key not in out:
            out[alias_key] = value

    return out

def _load_eval_summary(version: str, source: str) -> dict:
    """Load summary metrics for one version/source."""
    root = _results_dir(version)

    new_path = root / "evaluation_summary.json"
    if new_path.exists():
        payload = _load_json(new_path)
        try:
            extracted = _extract_source_payload(payload, source)
            return _normalise_summary_keys(extracted)
        except KeyError:
            pass

    legacy_candidates = [
        root / f"test_summary_stats_{source}.json",
        root / f"{source}_trajectory_benchmark_summary_stats.json",
    ]

    merged = {}
    found = False
    for path in legacy_candidates:
        if path.exists():
            merged.update(_load_json(path))
            found = True

    if not found:
        raise FileNotFoundError(
            f"No summary evaluation file found for version={version!r}, source={source!r}"
        )
    return _normalise_summary_keys(merged)


def _load_eval_per_target(version: str, source: str) -> dict:
    """Load per-target metrics for one version/source.

    Tries the new combined file first, then falls back to legacy per-source files.
    """
    root = _results_dir(version)

    new_path = root / "evaluation_per_target.json"
    if new_path.exists():
        payload = _load_json(new_path)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict) and row.get("dataset") == source]
        return _extract_source_payload(payload, source)

    legacy_candidates = [
        root / f"{source}_trajectory_benchmark_per_target.json",
    ]

    list_payload = None
    merged = {}
    found = False
    for path in legacy_candidates:
        if path.exists():
            payload = _load_json(path)
            if isinstance(payload, list):
                list_payload = payload
            elif isinstance(payload, dict):
                merged.update(payload)
            else:
                raise TypeError(
                    f"Unsupported per-target payload type {type(payload).__name__} in {path}"
                )
            found = True

    if not found:
        raise FileNotFoundError(
            f"No per-target evaluation file found for version={version!r}, source={source!r}"
        )
    if list_payload is not None:
        return list_payload
    return merged


# def load_test_summary(version: str, source: str) -> dict:
#     """Load test-summary metrics for one version/source."""
#     summary = _load_eval_summary(version, source)
#     return summary

def load_test_raw(version: str, source: str):
    """Load ``test_raw_distributions_{source}.pt`` for one version, or ``None``."""
    path = _results_dir(version) / f"test_raw_distributions_{source}.pt"
    if torch is None or not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False) 

# def load_trajectory_summary(version: str, source: str) -> dict:
#     """Load trajectory-summary metrics for one version/source."""
#     summary = _load_eval_summary(version, source)
#     return summary


def load_evaluation_summary(version: str, source: str) -> dict:
    return _load_eval_summary(version, source)


def load_evaluation_per_target(version: str, source: str) -> dict:
    return _load_eval_per_target(version, source)


def load_test_summary(version: str, source: str) -> dict:
    return load_evaluation_summary(version, source)


def load_trajectory_summary(version: str, source: str) -> dict:
    summary = load_evaluation_summary(version, source)
    try:
        per_target = load_trajectory_per_target(version, source)
    except FileNotFoundError:
        per_target = []

    pair_key = "trajectory_benchmark/pairwise_rmsd_r"
    pair_val = summary.get(pair_key)
    if pair_val is None or (isinstance(pair_val, float) and math.isnan(pair_val)):
        pred = np.asarray(
            [row.get("pairwise_rmsd_pred_mean_mean", np.nan) for row in per_target if isinstance(row, dict)],
            dtype=np.float64,
        )
        ref = np.asarray(
            [row.get("pairwise_rmsd_ref_mean_mean", np.nan) for row in per_target if isinstance(row, dict)],
            dtype=np.float64,
        )
        mask = np.isfinite(pred) & np.isfinite(ref)
        if np.count_nonzero(mask) >= 2:
            summary[pair_key] = float(np.corrcoef(ref[mask], pred[mask])[0, 1])

    if "trajectory_benchmark/pc_sim_gt_0_5_pct" not in summary:
        pc_vals = np.asarray(
            [row.get("pc_sim_mean", np.nan) for row in per_target if isinstance(row, dict)],
            dtype=np.float64,
        )
        mask = np.isfinite(pc_vals)
        if np.any(mask):
            summary["trajectory_benchmark/pc_sim_gt_0_5_pct"] = float(
                100.0 * np.mean(pc_vals[mask] > 0.5)
            )

    if "trajectory_benchmark/alphaflow_pc_sim_gt_0_5_pct" not in summary:
        pc_vals = np.asarray(
            [row.get("alphaflow_pc_sim_mean", np.nan) for row in per_target if isinstance(row, dict)],
            dtype=np.float64,
        )
        mask = np.isfinite(pc_vals)
        if np.any(mask):
            summary["trajectory_benchmark/alphaflow_pc_sim_gt_0_5_pct"] = float(
                100.0 * np.mean(pc_vals[mask] > 0.5)
            )

    if "trajectory_benchmark/alphaflow_pc_sim_concat_gt_0_5_pct" not in summary:
        pc_vals = np.asarray(
            [row.get("alphaflow_pc_sim_concat_mean", np.nan) for row in per_target if isinstance(row, dict)],
            dtype=np.float64,
        )
        mask = np.isfinite(pc_vals)
        if np.any(mask):
            summary["trajectory_benchmark/alphaflow_pc_sim_concat_gt_0_5_pct"] = float(
                100.0 * np.mean(pc_vals[mask] > 0.5)
            )

    if "trajectory_benchmark/alphaflow_md_pca_emd_median" not in summary:
        vals = np.asarray(
            [row.get("alphaflow_md_pca_emd_mean", np.nan) for row in per_target if isinstance(row, dict)],
            dtype=np.float64,
        )
        mask = np.isfinite(vals)
        if np.any(mask):
            summary["trajectory_benchmark/alphaflow_md_pca_emd_median"] = float(np.nanmedian(vals[mask]))

    return summary


def load_trajectory_per_target(version: str, source: str) -> dict:
    return load_evaluation_per_target(version, source)


def load_all(versions: Iterable[str] = DEFAULT_VERSIONS, source: str = "mdcath") -> dict:
    out: dict[str, dict] = {}
    for v in versions:
        entry: dict = {}
        try:
            entry["summary"] = load_test_summary(v, source)
        except FileNotFoundError:
            entry["summary"] = None
        try:
            entry["traj"] = load_trajectory_summary(v, source)
        except FileNotFoundError:
            entry["traj"] = None
        try:
            entry["per_target"] = load_trajectory_per_target(v, source)
        except FileNotFoundError:
            entry["per_target"] = None
        out[v] = entry
    return out


# ── Summary introspection helpers ────────────────────────────────────────────

TEMP_KEY_RE = re.compile(r"(?:test/)?temp_(\d+)/(.*)_(mean|std|median|min|max|count)$")


def build_temp_table(summary: dict) -> pd.DataFrame:
    """Pivot ``test/temp_{T}/...`` metric means into a DataFrame indexed by temp."""
    records = []
    for key, value in summary.items():
        m = TEMP_KEY_RE.fullmatch(key)
        if m:
            temp, metric, stat = m.groups()
            records.append({"temp": int(temp), "metric": metric,
                            "stat": stat, "value": value})
    if not records:
        return pd.DataFrame()
    long = pd.DataFrame(records).sort_values(["temp", "metric", "stat"])
    long = (long.groupby(["temp", "metric", "stat"], as_index=False)
                .agg(value=("value", "last")))
    return (long[long["stat"] == "mean"]
            .pivot(index="temp", columns="metric", values="value")
            .sort_index())


def s(summary: dict, key: str, default=np.nan):
    """``summary.get(key, nan)`` but converts ``None`` to ``nan``."""
    val = summary.get(key, default) if summary else default
    return np.nan if val is None else val


def get_mean_sem(stats: dict, key_prefix: str) -> tuple[float, float]:
    """Return (mean, SEM) from ``_mean``/``_std``/``_count`` keys (NaN-safe)."""
    mean = stats.get(f"{key_prefix}_mean")
    std = stats.get(f"{key_prefix}_std")
    count = stats.get(f"{key_prefix}_count")
    if mean is None or std is None or count in (None, 0):
        return float("nan"), float("nan")
    return mean, std / math.sqrt(count)


def fmt(mean: float, sem: float, dec: int = 3) -> str:
    """Format ``mean ± SEM`` for LaTeX, or ``--`` if NaN."""
    if (isinstance(mean, float) and math.isnan(mean)) or \
       (isinstance(sem, float) and math.isnan(sem)):
        return "--"
    f = f"{{:.{dec}f}}"
    return f"{f.format(mean)} $\\pm$ {f.format(sem)}"


def fmt_ratio(num_m, num_s, num_n, den_m, den_s, den_n, dec=2) -> str:
    """Ratio ``num/den`` with delta-method SEM."""
    if any(x is None or (isinstance(x, float) and math.isnan(x))
           for x in (num_m, num_s, num_n, den_m, den_s, den_n)) or den_m == 0:
        return "--"
    ratio = num_m / den_m
    sem_sq = ratio ** 2 * ((num_s / num_n) / num_m ** 2 + (den_s / den_n) / den_m ** 2)
    return fmt(ratio, math.sqrt(sem_sq), dec=dec)


def caca_dev(stats: dict, prefix: str) -> tuple[float, float]:
    """``|CA-CA mean - 3.8|`` with the underlying SEM."""
    m, se = get_mean_sem(stats, prefix)
    return (abs(m - 3.8) if not math.isnan(m) else float("nan")), se


def dc_error(stats: dict, temp_pfx: str) -> tuple[float, float]:
    """Signed DC amplitude error (pred - gt) with propagated SEM."""
    pred_m, _ = get_mean_sem(stats, f"{temp_pfx}/dc_pred_mean")
    gt_m, _ = get_mean_sem(stats, f"{temp_pfx}/dc_gt_mean")
    count = stats.get(f"{temp_pfx}/dc_pred_mean_count")
    pstd = stats.get(f"{temp_pfx}/dc_pred_mean_std")
    gstd = stats.get(f"{temp_pfx}/dc_gt_mean_std")
    if any(x is None for x in (pred_m, gt_m, count, pstd, gstd)) or count == 0 \
       or math.isnan(pred_m) or math.isnan(gt_m):
        return float("nan"), float("nan")
    return pred_m - gt_m, math.sqrt((pstd ** 2 + gstd ** 2) / count)


# ── Cross-version tidy frame (for plotting) ──────────────────────────────────

HEADLINE_METRICS = [
    # (column name, summary key, direction: +1 higher-better / -1 lower-better / 0 ratio)
    ("global_rmsf_spearman",  "test/global_rmsf_spearman",              +1),
    ("rmsf_spearman",         "test/rmsf_spearman",                     +1),
    ("per_target_rmsf_spearman", "test/per_target_rmsf_spearman",       +1),
    ("rmsf_spearman_per_samp", "test/rmsf_spearman_per_sample_mean",    +1),
    ("global_rmsf_pearson",   "test/global_rmsf_pearson",               +1),
    ("rmsf_pearson",          "test/rmsf_pearson",                      +1),
    ("per_target_rmsf_pearson", "test/per_target_rmsf_pearson",         +1),
    ("rmsf_pearson_per_samp", "test/rmsf_pearson_per_sample_mean",      +1),
    ("lddt",                  "test/lddt_mean",                         +1),
    ("caca_dev",              "test/caca_dist_A_mean",                  -1),  # special: |x-3.83|
    ("global_caca_jsd",       "test/global_caca_jsd",                   -1),
    ("spectral_mse",          "test/spectral_mse_mean",                 -1),
    ("spectral_dc_r",         "test/spectral_dc_r",                     +1),
    ("spectral_low_r",        "test/spectral_low_r",                    +1),
    ("spectral_mid_r",        "test/spectral_mid_r",                    +1),
    ("spectral_high_r",       "test/spectral_high_r",                   +1),
    ("low_band_ratio",        "test/low_band_power_ratio_mean",          0),
    ("mid_band_ratio",        "test/mid_band_power_ratio_mean",          0),
    ("high_band_ratio",       "test/high_band_power_ratio_mean",         0),
    ("dc_error",              "test/dc_error_mean",                      0),
    ("frame_disp_ratio",      "test/frame_disp_pred_over_gt",            0),
]

TRAJECTORY_METRICS = [
    ("pairwise_rmsd_r",    "trajectory_benchmark/pairwise_rmsd_r",         +1),
    ("global_rmsf_r",      "trajectory_benchmark/global_rmsf_r",           +1),
    ("per_target_rmsf_r",  "trajectory_benchmark/per_target_rmsf_r_median", +1),
    ("rmwd",               "trajectory_benchmark/rmwd_median",             -1),
    ("pca_w2",             "trajectory_benchmark/alphaflow_md_pca_emd_median", -1),
    ("pc_sim_pct",         "trajectory_benchmark/alphaflow_pc_sim_concat_gt_0_5_pct", +1),
    ("weak_contacts_j",    "trajectory_benchmark/weak_contacts_j_median",  +1),
    ("transient_contacts_j", "trajectory_benchmark/transient_contacts_j_median", +1),
    ("dG_fold_r",          "trajectory_benchmark/dG_fold_r",               +1),
    ("fnc_mean_pred",      "trajectory_benchmark/fnc_mean_pred_median",    +1),
    ("fnc_mean_ref",       "trajectory_benchmark/fnc_mean_ref_median",     +1),
    ("rg_jsd",             "trajectory_benchmark/rg_jsd_median",           -1),
]


def build_headline_frame(
    versions: Iterable[str] = DEFAULT_VERSIONS,
    source: str = "mdcath",
) -> pd.DataFrame:
    """Tidy DataFrame of headline metrics, rows=versions."""
    rows = []
    for v in versions:
        try:
            summary = load_test_summary(v, source)
        except FileNotFoundError:
            continue
        row = {"version": v, "label": VERSION_LABEL.get(v, v)}
        for col, key, _ in HEADLINE_METRICS:
            row[col] = s(summary, key)
        row["caca_dev"] = (
            abs(row["caca_dev"] - 3.83) if not math.isnan(row["caca_dev"]) else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("version")


def build_trajectory_frame(
    versions: Iterable[str] = DEFAULT_VERSIONS,
    source: str = "atlas",
) -> pd.DataFrame:
    """Tidy DataFrame of trajectory-benchmark medians, rows=versions."""
    rows = []
    for v in versions:
        try:
            tb = load_trajectory_summary(v, source)
        except FileNotFoundError:
            continue
        row = {"version": v, "label": VERSION_LABEL.get(v, v)}
        for col, key, _ in TRAJECTORY_METRICS:
            row[col] = s(tb, key)
        rows.append(row)
    return pd.DataFrame(rows).set_index("version")


def build_per_temp_frame(
    versions: Iterable[str] = DEFAULT_VERSIONS,
    source: str = "mdcath",
    metric: str = "rmsf_spearman",
) -> pd.DataFrame:
    """Wide DataFrame: index=temperature, columns=version, values=metric mean."""
    out = {}
    for v in versions:
        try:
            summary = load_test_summary(v, source)
        except FileNotFoundError:
            continue
        tm = build_temp_table(summary)
        if not tm.empty and metric in tm.columns:
            out[v] = tm[metric]
    return pd.DataFrame(out)


# ── Plotting: multi-panel summary figure ─────────────────────────────────────

def _bar_by_version(ax, frame, col, ylabel, versions, ref=None, title=None,
                    lower_better=False, value_fmt="{:.3f}"):
    xs = np.arange(len(versions))
    vals = np.array([frame.loc[v, col] if v in frame.index else np.nan
                     for v in versions])
    colors = [VERSION_COLORS.get(v, SLATE) for v in versions]
    ax.bar(xs, vals, color=colors, edgecolor="none", width=0.72)
    ax.set_xticks(xs)
    ax.set_xticklabels(versions, fontsize=8)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if ref is not None:
        ax.axhline(ref, color=SLATE, linestyle="--", linewidth=0.9, alpha=0.9)
    # annotate best
    finite = np.where(np.isfinite(vals))[0]
    if len(finite) > 0:
        best = finite[np.argmin(vals[finite])] if lower_better else \
               finite[np.argmax(vals[finite])]
        for i, v in enumerate(vals):
            if not np.isfinite(v):
                continue
            weight = "bold" if i == best else "normal"
            ax.text(i, v, value_fmt.format(v), ha="center", va="bottom",
                    fontsize=7.0, color=INK, fontweight=weight)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 * 1.12 if y1 > 0 else y1 * 0.88)


def _line_per_temp(ax, frame_wide, versions, ylabel, title=None, ref=None,
                   ymin=None, ymax=None):
    for v in versions:
        if v not in frame_wide.columns:
            continue
        col = frame_wide[v].dropna()
        if col.empty:
            continue
        ax.plot(col.index, col.values, marker="o",
                color=VERSION_COLORS.get(v, SLATE),
                label=v, linewidth=1.8)
    if ref is not None:
        ax.axhline(ref, color=SLATE, linestyle="--", linewidth=0.9, alpha=0.8)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)


def plot_model_comparison(
    versions: Iterable[str] = DEFAULT_VERSIONS,
    sources: Iterable[str] = SOURCES,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Multi-panel figure comparing versions on RMSF / lDDT / spectral / ΔG.

    Layout: 4 rows × 4 cols. Row 1 = per-source headline bars. Row 2 = spectral
    band recovery bars. Row 3 = RMSF per-temperature line for each source.
    Row 4 = trajectory-benchmark medians (RMSF-r / pair-RMSD-r / ΔG-r / Rg-JSD).
    """
    apply_icml_style()
    versions = list(versions)
    sources = list(sources)

    fig = plt.figure(figsize=(13, 12), facecolor="white")
    gs = fig.add_gridspec(4, 4, hspace=0.6, wspace=0.35)

    # Row 1: headline metrics per source (RMSF ρ, RMSF per-sample ρ, lDDT, DC err)
    for i, src in enumerate(sources):
        frame = build_headline_frame(versions, src)
        ax = fig.add_subplot(gs[0, i * 2 + 0], facecolor="white")
        _bar_by_version(ax, frame, "rmsf_spearman", "RMSF Spearman ρ", versions,
                        title=f"{src.upper()} — RMSF ρ ↑")
        ax = fig.add_subplot(gs[0, i * 2 + 1], facecolor="white")
        _bar_by_version(ax, frame, "lddt", "lDDT", versions,
                        title=f"{src.upper()} — lDDT ↑")

    # Row 2: spectral band recovery (pred/gt, target=1.0) — low/mid/high
    for i, src in enumerate(sources):
        frame = build_headline_frame(versions, src)
        ax = fig.add_subplot(gs[1, i * 2 + 0], facecolor="white")
        xs = np.arange(len(versions))
        width = 0.24
        for j, (col, label, shade) in enumerate([
            ("low_band_ratio",  "Low",  YELLOW_DARK),
            ("mid_band_ratio",  "Mid",  BLUE_DARK),
            ("high_band_ratio", "High", BLUE),
        ]):
            vals = np.array([frame.loc[v, col] if v in frame.index else np.nan
                             for v in versions])
            ax.bar(xs + (j - 1) * width, vals, width=width, color=shade,
                   edgecolor="none", label=label)
        ax.axhline(1.0, color=SLATE, linestyle="--", linewidth=0.9, alpha=0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels(versions, fontsize=8)
        ax.set_ylabel("Pred / GT power")
        ax.set_title(f"{src.upper()} — spectral band recovery")
        ax.legend(loc="best", ncol=3, fontsize=7)

        ax = fig.add_subplot(gs[1, i * 2 + 1], facecolor="white")
        _bar_by_version(ax, frame, "spectral_mse", "Spectral MSE",
                        versions, lower_better=True,
                        title=f"{src.upper()} — spectral MSE ↓",
                        value_fmt="{:.1f}")

    # Row 3: per-temperature line plots of RMSF ρ (mdCATH has per-temp; ATLAS = single temp)
    ax = fig.add_subplot(gs[2, 0:2], facecolor="white")
    temp_frame = build_per_temp_frame(versions, "mdcath", "rmsf_spearman")
    _line_per_temp(ax, temp_frame, versions, "RMSF Spearman ρ",
                   title="mdCATH — RMSF ρ vs temperature ↑", ymin=0.3, ymax=1.0)
    ax.legend(ncol=len(versions), loc="lower left", fontsize=7)

    ax = fig.add_subplot(gs[2, 2:4], facecolor="white")
    temp_frame = build_per_temp_frame(versions, "mdcath", "lddt")
    _line_per_temp(ax, temp_frame, versions, "lDDT",
                   title="mdCATH — lDDT vs temperature ↑", ymin=0.3, ymax=0.8)
    ax.legend(ncol=len(versions), loc="lower left", fontsize=7)

    # Row 4: trajectory-benchmark metrics (ATLAS + mdCATH side-by-side)
    panel_metrics = [
        ("pairwise_rmsd_r", "Pair. RMSD r ↑", False),
        ("global_rmsf_r",   "Global RMSF r ↑", False),
        ("dG_fold_r",       "ΔG-fold r ↑",    False),
        ("rg_jsd",          "Rg JSD ↓",       True),
    ]
    for j, (col, ylabel, lower) in enumerate(panel_metrics):
        ax = fig.add_subplot(gs[3, j], facecolor="white")
        xs = np.arange(len(versions))
        width = 0.36
        for k, src in enumerate(sources):
            tf = build_trajectory_frame(versions, src)
            vals = np.array([tf.loc[v, col] if v in tf.index else np.nan
                             for v in versions])
            color = BLUE_DARK if src == "mdcath" else YELLOW_DARK
            ax.bar(xs + (k - 0.5) * width, vals, width=width,
                   color=color, edgecolor="none",
                   label=src.upper())
        ax.set_xticks(xs)
        ax.set_xticklabels(versions, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.axhline(0.0 if not lower else 0.0, color=SLATE,
                   linestyle="--", linewidth=0.6, alpha=0.5)
        if j == 0:
            ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        "Pancake model comparison across v8, v9, v10, v11 — mdCATH + ATLAS",
        fontsize=11, fontweight="bold", y=0.995,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, facecolor="white")
    return fig


# ── LaTeX table: trajectory benchmark with all our models + competitors ──────

_TRAJ_COLS: list[tuple[str, bool]] = [
    ("pair_rmsd_r",    True),   # ↑
    ("global_rmsf_r",  True),   # ↑
    ("per_tgt_rmsf_r", True),   # ↑
    ("rmwd",           False),  # ↓
    ("pca_w2",         False),  # ↓
    ("pc_sim_pct",     True),   # ↑
    ("weak_j",         True),   # ↑
    ("trans_j",        True),   # ↑
]

# Published competitor numbers carried over verbatim from tables.ipynb.
COMPETITORS_ATLAS = [
    dict(name="BioEMU",       pair_rmsd_r=-0.02, global_rmsf_r=0.09, per_tgt_rmsf_r=None,
         rmwd=19.23, pca_w2=3.61, pc_sim_pct=14,   weak_j=0.26, trans_j=0.06),
    dict(name="AlphaFlow-MD", pair_rmsd_r=0.48,  global_rmsf_r=0.60, per_tgt_rmsf_r=None,
         rmwd=2.61,  pca_w2=1.52, pc_sim_pct=44,   weak_j=0.51, trans_j=0.29),
    dict(name="MDGen",        pair_rmsd_r=0.48,  global_rmsf_r=0.50, per_tgt_rmsf_r=None,
         rmwd=2.69,  pca_w2=1.89, pc_sim_pct=10,   weak_j=0.62, trans_j=0.41),
    dict(name="Tempo",        pair_rmsd_r=0.91,  global_rmsf_r=0.89, per_tgt_rmsf_r=None,
         rmwd=1.49,  pca_w2=0.60, pc_sim_pct=76,   weak_j=0.74, trans_j=0.38),
]

COMPETITORS_MDCATH = [
    dict(name="BioEMU",       pair_rmsd_r=-0.02, global_rmsf_r=0.13, per_tgt_rmsf_r=None,
         rmwd=10.70, pca_w2=2.49, pc_sim_pct=9.38,  weak_j=0.38, trans_j=0.12),
    dict(name="ESMFlow-MD",   pair_rmsd_r=0.26,  global_rmsf_r=0.34, per_tgt_rmsf_r=None,
         rmwd=4.08,  pca_w2=2.36, pc_sim_pct=25.00, weak_j=0.51, trans_j=0.28),
    dict(name="AlphaFlow-MD", pair_rmsd_r=0.41,  global_rmsf_r=0.41, per_tgt_rmsf_r=None,
         rmwd=5.62,  pca_w2=2.38, pc_sim_pct=21.88, weak_j=0.42, trans_j=0.27),
    dict(name="MDGen",        pair_rmsd_r=0.71,  global_rmsf_r=0.67, per_tgt_rmsf_r=None,
         rmwd=3.36,  pca_w2=2.62, pc_sim_pct=17.19, weak_j=0.41, trans_j=0.20),
    dict(name="MarS-FM",
         pair_rmsd_r=0.65,   pair_rmsd_r_sem=0.004,
         global_rmsf_r=0.71, global_rmsf_r_sem=0.003,
         per_tgt_rmsf_r=0.89, per_tgt_rmsf_r_sem=0.001,
         rmwd=None, pca_w2=None, pc_sim_pct=None, weak_j=None, trans_j=None),
    dict(name="Tempo",        pair_rmsd_r=0.77,  global_rmsf_r=0.67, per_tgt_rmsf_r=None,
         rmwd=4.21,  pca_w2=2.33, pc_sim_pct=7.81,  weak_j=0.43, trans_j=0.20),
]


def pancake_traj_row(version: str, source: str) -> dict:
    """Build a trajectory-benchmark table row for a given Pancake version."""
    tb = load_trajectory_summary(version, source)
    return dict(
        name=f"Pancake-{version}",
        pair_rmsd_r    = tb.get("trajectory_benchmark/pairwise_rmsd_r"),
        global_rmsf_r  = tb.get("trajectory_benchmark/global_rmsf_r"),
        per_tgt_rmsf_r = tb.get("trajectory_benchmark/per_target_rmsf_r_median"),
        rmwd           = tb.get("trajectory_benchmark/rmwd_median"),
        pca_w2         = tb.get(
            "trajectory_benchmark/alphaflow_md_pca_emd_median",
            tb.get("trajectory_benchmark/md_pca_w2_median"),
        ),
        pc_sim_pct     = tb.get(
            "trajectory_benchmark/alphaflow_pc_sim_concat_gt_0_5_pct",
            tb.get(
                "trajectory_benchmark/alphaflow_pc_sim_gt_0_5_pct",
                tb.get("trajectory_benchmark/pc_sim_gt_0_5_pct"),
            ),
        ),
        weak_j         = tb.get("trajectory_benchmark/weak_contacts_j_median"),
        trans_j        = tb.get("trajectory_benchmark/transient_contacts_j_median"),
    )


def _best_per_col(rows: list[dict]) -> dict[str, float]:
    best: dict[str, float] = {}
    for col, higher_better in _TRAJ_COLS:
        vals = [r[col] for r in rows if r.get(col) is not None]
        if vals:
            best[col] = max(vals) if higher_better else min(vals)
    return best


def _fmt_cell(row: dict, col: str, best: dict, pc_dec: int = 1) -> str:
    v = row.get(col)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    sem = row.get(col + "_sem")
    is_best = best.get(col) is not None and abs(v - best[col]) < 1e-9
    if col == "pc_sim_pct":
        body = f"{v:.0f}\\%" if v == int(v) else f"{v:.{pc_dec}f}\\%"
        return f"\\textbf{{{body}}}" if is_best else body
    if col in ("rmwd", "pca_w2"):
        body = f"{v:.2f}"
        return f"\\textbf{{{body}}}" if is_best else body
    if sem is not None:
        sem_str = f"{sem:.3f}".lstrip("0") or "0"
        inner = f"{v:.2f}\\pm{sem_str}"
        return f"$\\mathbf{{{inner}}}$" if is_best else f"${inner}$"
    body = f"{v:.3f}"
    return f"\\textbf{{{body}}}" if is_best else body


def _render_traj_row(row: dict, best: dict, pc_dec: int = 1) -> str:
    cells = [row["name"]] + [_fmt_cell(row, col, best, pc_dec=pc_dec)
                             for col, _ in _TRAJ_COLS]
    return "        " + " & ".join(cells) + r" \\"


def make_trajectory_comparison_table(
    source: str,
    versions: Iterable[str] = DEFAULT_VERSIONS,
    competitors: list[dict] | None = None,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """Return a LaTeX ``table*`` comparing all Pancake versions + competitors.

    Args:
      source: ``"atlas"`` or ``"mdcath"``.
      versions: Pancake versions to include (rows appear in this order).
      competitors: competitor rows; defaults to published numbers for ``source``.
      caption: table caption (defaults to a standard one).
      label: LaTeX label (defaults to ``tab:{source}_trajectory_vs``).
    """
    source = source.lower()
    versions = list(versions)

    if competitors is None:
        competitors = (COMPETITORS_ATLAS if source == "atlas"
                       else COMPETITORS_MDCATH)
    pancake_rows = [pancake_traj_row(v, source) for v in versions]
    all_rows = competitors + pancake_rows
    best = _best_per_col(all_rows)

    pc_dec = 0 if source == "atlas" else 2
    comp_block = "\n".join(_render_traj_row(r, best, pc_dec=pc_dec)
                           for r in competitors)
    pan_block = "\n".join(_render_traj_row(r, best, pc_dec=pc_dec)
                          for r in pancake_rows)

    label = label or f"tab:{source}_trajectory_vs"
    caption = caption or (
        f"Trajectory benchmark on "
        + ("ATLAS (300\\,K)" if source == "atlas" else "mdCATH")
        + ". $\\uparrow$: higher is better; $\\downarrow$: lower is better. "
          "\\textbf{Bold}: best per column among methods reporting that metric. "
          "\\,`--': metric not reported. All Pancake versions share the same "
          "spectral diffusion backbone (v8 = prior, v9 = slow-branch, "
          "v10 = block-mix, v11 = dual-branch). Pancake metrics are medians "
          "over targets (Pair.\\ RMSD $r$ and Global RMSF $r$ are global "
          "Pearson correlations)."
    )

    return (
        r"""
\begin{table*}[t]
  \caption{""" + caption + r"""}
  \label{""" + label + r"""}
  \begin{center}
    \begin{small}
      \begin{sc}
        \begin{tabular}{lcccccccc}
        \toprule
        Method & Pair.\ RMSD $r$ $\uparrow$ & Global RMSF $r$ $\uparrow$ & Per-tgt RMSF $r$ $\uparrow$ & RMWD $\downarrow$ & PCA $\mathcal{W}_2$ $\downarrow$ & PC-sim $\uparrow$ & Weak J $\uparrow$ & Trans.\ J $\uparrow$ \\
        \midrule
"""
        + comp_block + "\n"
        + "        \\midrule\n"
        + pan_block + "\n"
        + r"""        \bottomrule
        \end{tabular}
      \end{sc}
    \end{small}
  \end{center}
  \vskip -0.1in
\end{table*}
"""
    )
