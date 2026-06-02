#!/usr/bin/env python3
"""Sensor-level lag intuition analyses for pooled super-brain decoders.

This script intentionally writes to a separate output folder. It treats the
existing super-brain lag heatmaps as the behavioral target, then asks which
pooled electrodes look short-lag-preferring or long-lag-preferring by several
complementary diagnostics:

  - raw standardized decoder weight norms from super_brain_channel_anatomy.py
  - Haufe-transformed decoder patterns
  - sensor lag-profile clusters
  - per-channel encoding preferred-lag cross-checks
  - short-preferring vs long-preferring sensor-group decoders

The default short/long definitions are short=N 1..3 and long=N 7..10.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    clean_subject,
    row_corr,
    score_predictions,
    subject_label,
    target_indices,
)
from train_decoders import cv_predict_ridge, load_prepared_subject, valid_current_indices  # noqa: E402


TARGET_STEMS = ["word_vectors_pca20", "gpt2_ctx32_layer8_pca20"]
COORDS = ["x", "y", "z"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "prepared_midpoint")
    parser.add_argument("--super-brain-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "super_brain_midpoint")
    parser.add_argument("--preferred-lag-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "preferred_lag_midpoint")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "super_brain_midpoint" / "lag_sensor_intuition",
    )
    parser.add_argument("--target-stems", nargs="+", default=TARGET_STEMS)
    parser.add_argument("--subjects", nargs="+", default=[f"{i:02d}" for i in range(1, 10)])
    parser.add_argument("--lags", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--short-lags", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--long-lags", nargs="+", type=int, default=[7, 8, 9, 10])
    parser.add_argument(
        "--encoding-short-lags",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Past encoding lags to use for the short cross-check, interpreted as -N columns.",
    )
    parser.add_argument(
        "--encoding-long-lags",
        nargs="+",
        type=int,
        default=[4, 5],
        help="Past encoding lags to use for the long cross-check, interpreted as -N columns.",
    )
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--top-n-profiles", type=int, default=24)
    parser.add_argument("--sensor-quantile", type=float, default=0.25)
    parser.add_argument("--min-total-weight-quantile", type=float, default=0.50)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--skip-haufe", action="store_true")
    parser.add_argument("--skip-group-decoders", action="store_true")
    parser.add_argument("--skip-group-ablation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def finite_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3 or len(np.unique(x[mask])) < 2 or len(np.unique(y[mask])) < 2:
        return float("nan"), float("nan"), int(mask.sum())
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), int(mask.sum())


def finite_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3 or len(np.unique(x[mask])) < 2 or len(np.unique(y[mask])) < 2:
        return float("nan"), float("nan"), int(mask.sum())
    r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p), int(mask.sum())


def load_alpha(super_brain_root: Path, target_stem: str) -> float:
    path = super_brain_root / target_stem / "alpha_tune_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return float(json.loads(path.read_text(encoding="utf-8"))["best_alpha"])


def channel_weights_path(super_brain_root: Path, target_stem: str) -> Path:
    return super_brain_root / target_stem / "anatomy" / "channel_decoder_weights.csv"


def require_cols(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")


def add_lag_indices(
    df: pd.DataFrame,
    lags: list[int],
    short_lags: list[int],
    long_lags: list[int],
    *,
    prefix: str = "lag",
    value_suffix: str = "weight_norm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add short/long summary columns to a per-channel lag-norm table."""
    out = df.copy()
    lag_cols = [f"{prefix}_{lag}_{value_suffix}" for lag in lags]
    short_cols = [f"{prefix}_{lag}_{value_suffix}" for lag in short_lags]
    long_cols = [f"{prefix}_{lag}_{value_suffix}" for lag in long_lags]
    require_cols(out, lag_cols, Path("<lag table>"))

    values = out[lag_cols].to_numpy(dtype=float)
    denom = np.maximum(np.nansum(values, axis=1, keepdims=True), 1e-12)
    fractions = values / denom
    frac = pd.DataFrame(
        fractions,
        columns=[f"{prefix}_{lag}_fraction" for lag in lags],
        index=out.index,
    )
    out = pd.concat([out, frac], axis=1)

    short = out[short_cols].mean(axis=1)
    long = out[long_cols].mean(axis=1)
    out["short_weight_mean"] = short
    out["long_weight_mean"] = long
    out["long_minus_short_weight_mean"] = long - short
    out["long_short_norm_index"] = (long - short) / np.maximum(long + short, 1e-12)

    short_frac_cols = [f"{prefix}_{lag}_fraction" for lag in short_lags]
    long_frac_cols = [f"{prefix}_{lag}_fraction" for lag in long_lags]
    out["short_weight_fraction_sum"] = out[short_frac_cols].sum(axis=1)
    out["long_weight_fraction_sum"] = out[long_frac_cols].sum(axis=1)
    out["long_minus_short_fraction"] = out["long_weight_fraction_sum"] - out["short_weight_fraction_sum"]

    lag_arr = np.asarray(lags, dtype=float)
    frac_values = out[[f"{prefix}_{lag}_fraction" for lag in lags]].to_numpy(dtype=float)
    out["lag_fraction_center_of_mass"] = np.nansum(frac_values * lag_arr[None, :], axis=1)
    slopes = []
    for row in frac_values:
        if np.isfinite(row).sum() < 3:
            slopes.append(np.nan)
        else:
            slopes.append(float(stats.linregress(lag_arr, row).slope))
    out["lag_fraction_slope"] = slopes

    profile = pd.DataFrame(frac_values, columns=[str(lag) for lag in lags], index=out.index)
    return out, profile


def roi_summary(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for roi, group in df.groupby("paper_roi", dropna=False):
        values = group[value_col].to_numpy(dtype=float)
        rows.append(
            {
                "paper_roi": roi,
                "n_channels": int(len(group)),
                f"mean_{value_col}": float(np.nanmean(values)),
                f"median_{value_col}": float(np.nanmedian(values)),
                "fraction_long_pref_gt0": float(np.nanmean(values > 0)),
                "median_lag_com": float(np.nanmedian(group["lag_fraction_center_of_mass"])),
                "mean_total_weight_norm": float(np.nanmean(group.get("total_weight_norm", np.nan))),
            }
        )
    return pd.DataFrame(rows).sort_values(f"mean_{value_col}", ascending=False)


def coordinate_summary(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    rows = []
    coord_data = {
        "x": df["x"].to_numpy(dtype=float),
        "abs_x": np.abs(df["x"].to_numpy(dtype=float)),
        "y": df["y"].to_numpy(dtype=float),
        "z": df["z"].to_numpy(dtype=float),
    }
    for value_col in value_cols:
        y = df[value_col].to_numpy(dtype=float)
        for coord, x in coord_data.items():
            rho, p, n = finite_spearman(x, y)
            rows.append({"value": value_col, "coordinate": coord, "spearman_rho": rho, "p": p, "n": n})
    return pd.DataFrame(rows)


def plane_spec(plane: str = "yz") -> tuple[str, str, str, str]:
    if plane == "xy":
        return "x", "y", "right(+)/left(-) x (mm)", "anterior(+)/posterior(-) y (mm)"
    if plane == "xz":
        return "x", "z", "right(+)/left(-) x (mm)", "superior(+)/inferior(-) z (mm)"
    return "y", "z", "anterior(+)/posterior(-) y (mm)", "superior(+)/inferior(-) z (mm)"


def scatter_map(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    label: str,
    *,
    plane: str = "yz",
    cmap: str = "coolwarm",
    symmetric: bool = True,
) -> None:
    xcol, ycol, xlabel, ylabel = plane_spec(plane)
    values = df[value_col].to_numpy(dtype=float)
    finite = np.isfinite(values) & np.isfinite(df[xcol].to_numpy(dtype=float)) & np.isfinite(df[ycol].to_numpy(dtype=float))
    if not finite.any():
        return
    if symmetric:
        vmax = float(np.nanpercentile(np.abs(values[finite]), 98))
        vmax = max(vmax, 1e-6)
        vmin = -vmax
    else:
        vmin = float(np.nanpercentile(values[finite], 2))
        vmax = float(np.nanpercentile(values[finite], 98))
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-6
    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    sc = ax.scatter(
        df.loc[finite, xcol],
        df.loc[finite, ycol],
        c=values[finite],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=48,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.9,
    )
    ax.axhline(0, color="0.85", lw=0.6)
    ax.axvline(0, color="0.85", lw=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linewidth=0.35, alpha=0.2)
    fig.colorbar(sc, ax=ax, label=label)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_roi_box(df: pd.DataFrame, value_col: str, output_path: Path, title: str) -> None:
    rois = (
        df.groupby("paper_roi")[value_col]
        .median()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    data = [df.loc[df["paper_roi"].astype(str) == roi, value_col].dropna().to_numpy(dtype=float) for roi in rois]
    fig, ax = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)
    ax.boxplot(data, tick_labels=rois, showfliers=False)
    ax.axhline(0, color="0.7", lw=0.8, ls="--")
    ax.set_xlabel("paper ROI")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_top_profiles(
    df: pd.DataFrame,
    profile: pd.DataFrame,
    lags: list[int],
    output_path: Path,
    title: str,
    top_n: int,
) -> None:
    top_long = df.sort_values("long_short_norm_index", ascending=False).head(top_n)
    top_short = df.sort_values("long_short_norm_index", ascending=True).head(top_n)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True, sharey=True)
    for ax, sub, panel_title, color in [
        (axes[0], top_short, "short-preferring sensors", "C0"),
        (axes[1], top_long, "long-preferring sensors", "C3"),
    ]:
        mat = profile.loc[sub.index, [str(lag) for lag in lags]].to_numpy(dtype=float)
        for row in mat:
            ax.plot(lags, row, color=color, alpha=0.18, lw=0.9)
        ax.plot(lags, np.nanmean(mat, axis=0), color=color, lw=2.6, marker="o")
        ax.set_title(panel_title)
        ax.set_xlabel("past lag N")
        ax.grid(True, linewidth=0.35, alpha=0.25)
    axes[0].set_ylabel("within-sensor weight fraction")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def cluster_profiles(
    df: pd.DataFrame,
    profile: pd.DataFrame,
    lags: list[int],
    n_clusters: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mat = profile[[str(lag) for lag in lags]].to_numpy(dtype=float)
    mat = np.nan_to_num(mat, nan=0.0)
    centered = mat - mat.mean(axis=1, keepdims=True)
    scale = np.linalg.norm(centered, axis=1, keepdims=True)
    use = np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 0)
    k = min(int(n_clusters), len(df))
    labels = KMeans(n_clusters=k, random_state=int(seed), n_init=25).fit_predict(use)

    tmp = df.copy()
    tmp["_raw_cluster"] = labels
    means = []
    for label in sorted(np.unique(labels)):
        idx = labels == label
        lag_com = float(np.nanmean(tmp.loc[idx, "lag_fraction_center_of_mass"]))
        means.append((label, lag_com))
    remap = {label: i + 1 for i, (label, _) in enumerate(sorted(means, key=lambda x: x[1]))}
    tmp["profile_cluster"] = [remap[label] for label in labels]
    tmp = tmp.drop(columns=["_raw_cluster"])

    rows = []
    for cluster, group in tmp.groupby("profile_cluster"):
        row = {
            "profile_cluster": int(cluster),
            "n_channels": int(len(group)),
            "mean_lag_com": float(np.nanmean(group["lag_fraction_center_of_mass"])),
            "mean_long_short_norm_index": float(np.nanmean(group["long_short_norm_index"])),
        }
        prof = profile.loc[group.index, [str(lag) for lag in lags]].to_numpy(dtype=float)
        for i, lag in enumerate(lags):
            row[f"mean_fraction_lag_{lag}"] = float(np.nanmean(prof[:, i]))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("profile_cluster")
    return tmp, summary


def plot_cluster_profiles(summary: pd.DataFrame, lags: list[int], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    for _, row in summary.iterrows():
        y = [row[f"mean_fraction_lag_{lag}"] for lag in lags]
        ax.plot(lags, y, marker="o", lw=2.0, label=f"cluster {int(row['profile_cluster'])} (n={int(row['n_channels'])})")
    ax.set_xlabel("past lag N")
    ax.set_ylabel("mean within-sensor weight fraction")
    ax.set_title(title)
    ax.grid(True, linewidth=0.35, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def encoding_crosscheck(
    weights: pd.DataFrame,
    target_stem: str,
    preferred_lag_dir: Path,
    out_dir: Path,
    enc_short_lags: list[int],
    enc_long_lags: list[int],
) -> dict[str, object] | None:
    path = preferred_lag_dir / f"preferred_lag__{target_stem}.csv"
    if not path.is_file():
        return None
    enc = pd.read_csv(path)
    need = ["subject", "channel", "preferred_lag_words", "max_abs_corr"]
    require_cols(enc, need, path)
    short_cols = [f"corr_lag_{-lag:+d}" for lag in enc_short_lags]
    long_cols = [f"corr_lag_{-lag:+d}" for lag in enc_long_lags]
    require_cols(enc, short_cols + long_cols, path)
    enc["encoding_past_short_corr_mean"] = enc[short_cols].mean(axis=1)
    enc["encoding_past_long_corr_mean"] = enc[long_cols].mean(axis=1)
    enc["encoding_past_long_minus_short_corr"] = enc["encoding_past_long_corr_mean"] - enc["encoding_past_short_corr_mean"]
    enc["encoding_past_short_abs_mean"] = enc[short_cols].abs().mean(axis=1)
    enc["encoding_past_long_abs_mean"] = enc[long_cols].abs().mean(axis=1)
    enc["encoding_past_long_short_abs_index"] = (
        enc["encoding_past_long_abs_mean"] - enc["encoding_past_short_abs_mean"]
    ) / np.maximum(enc["encoding_past_long_abs_mean"] + enc["encoding_past_short_abs_mean"], 1e-12)
    enc["encoding_prefers_past"] = enc["preferred_lag_words"] < 0
    enc["encoding_prefers_long_past"] = enc["preferred_lag_words"] <= -min(enc_long_lags)

    keep_cols = [
        "subject",
        "channel",
        "paper_roi",
        "preferred_lag_words",
        "corr_at_preferred_lag",
        "max_abs_corr",
        "significant_best",
        "encoding_past_long_minus_short_corr",
        "encoding_past_long_short_abs_index",
        "encoding_prefers_past",
        "encoding_prefers_long_past",
    ]
    merged = weights.merge(enc[[c for c in keep_cols if c in enc.columns]], on=["subject", "channel"], how="left", suffixes=("", "_encoding"))
    merged.to_csv(out_dir / "encoding_crosscheck.csv", index=False)

    rho, p, n = finite_spearman(
        merged["long_short_norm_index"].to_numpy(dtype=float),
        merged["encoding_past_long_short_abs_index"].to_numpy(dtype=float),
    )
    sig = merged[merged.get("significant_best", False).fillna(False)] if "significant_best" in merged.columns else merged.iloc[0:0]
    sig_rho, sig_p, sig_n = finite_spearman(
        sig["long_short_norm_index"].to_numpy(dtype=float) if len(sig) else np.array([]),
        sig["encoding_past_long_short_abs_index"].to_numpy(dtype=float) if len(sig) else np.array([]),
    )
    summary = {
        "preferred_lag_file": str(path),
        "decoder_vs_encoding_abs_long_short_spearman_rho": rho,
        "decoder_vs_encoding_abs_long_short_spearman_p": p,
        "decoder_vs_encoding_abs_long_short_n": n,
        "significant_only_spearman_rho": sig_rho,
        "significant_only_spearman_p": sig_p,
        "significant_only_n": sig_n,
    }
    (out_dir / "encoding_crosscheck_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    x = merged["long_short_norm_index"].to_numpy(dtype=float)
    y = merged["encoding_past_long_short_abs_index"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[mask], y[mask], s=22, alpha=0.45, edgecolor="none")
    ax.axhline(0, color="0.75", lw=0.8, ls="--")
    ax.axvline(0, color="0.75", lw=0.8, ls="--")
    ax.set_xlabel("decoder long-short index")
    ax.set_ylabel("encoding abs long-short index")
    ax.set_title(f"{target_stem}: decoder vs encoding lag preference\nSpearman rho={rho:.3f}, p={p:.3g}")
    fig.savefig(out_dir / "encoding_decoder_long_short_scatter.png", dpi=220)
    plt.close(fig)
    return summary


def make_sensor_groups(
    weights: pd.DataFrame,
    q: float,
    min_total_weight_quantile: float,
) -> pd.DataFrame:
    filt = weights.copy()
    threshold = float(filt["total_weight_norm"].quantile(float(min_total_weight_quantile)))
    high = filt[filt["total_weight_norm"] >= threshold].copy()
    low_cut = float(high["long_short_norm_index"].quantile(float(q)))
    high_cut = float(high["long_short_norm_index"].quantile(1.0 - float(q)))
    rows = []
    for group_name, group in [
        ("short_pref", high[high["long_short_norm_index"] <= low_cut]),
        ("long_pref", high[high["long_short_norm_index"] >= high_cut]),
    ]:
        tmp = group[["subject", "channel", "paper_roi", "long_short_norm_index", "total_weight_norm"]].copy()
        tmp["sensor_group"] = group_name
        tmp["selection_min_total_weight"] = threshold
        tmp["selection_low_cut"] = low_cut
        tmp["selection_high_cut"] = high_cut
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def load_group_matrix(
    prepared_dir: Path,
    subjects: list[str],
    members: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    wanted = {
        subject: set(group["channel"].astype(str).tolist())
        for subject, group in members.groupby("subject")
    }
    parts = []
    rows = []
    common_valid: np.ndarray | None = None
    col = 0
    for subject in subjects:
        sub = subject_label(subject)
        x_all, valid, channels = load_prepared_subject(prepared_dir, subject)
        common_valid = valid if common_valid is None else (common_valid & valid)
        sub_wanted = wanted.get(sub, set())
        cols = [i for i, channel in enumerate(channels) if str(channel) in sub_wanted]
        if not cols:
            continue
        parts.append(x_all[:, cols].astype(np.float32))
        for i in cols:
            rows.append({"subject": sub, "channel": channels[i], "column_idx": col})
            col += 1
    if common_valid is None:
        raise RuntimeError("No subjects loaded.")
    if not parts:
        return np.empty((len(common_valid), 0), dtype=np.float32), common_valid, pd.DataFrame(rows)
    return np.hstack(parts), common_valid, pd.DataFrame(rows)


def run_group_decoders(
    prepared_dir: Path,
    super_brain_root: Path,
    target_stem: str,
    subjects: list[str],
    lags: list[int],
    group_membership: pd.DataFrame,
    output_dir: Path,
    outer_splits: int,
    overwrite: bool,
) -> pd.DataFrame:
    out_path = output_dir / "sensor_group_decoder_scores.csv"
    if out_path.is_file() and not overwrite:
        return pd.read_csv(out_path)
    word_scores = np.load(prepared_dir / f"{target_stem}.npy").astype(np.float32)
    alpha = load_alpha(super_brain_root, target_stem)
    rows = []
    for group_name, members in group_membership.groupby("sensor_group"):
        x_all, valid, selected = load_group_matrix(prepared_dir, subjects, members)
        if x_all.shape[1] == 0:
            continue
        shared_current = valid_current_indices(valid, word_scores, lags, "all_lags")
        x_shared = x_all[shared_current].astype(np.float32)
        finite_x = np.isfinite(x_shared).all(axis=1)
        current = shared_current[finite_x]
        x = x_shared[finite_x]
        print(
            f"    {target_stem} {group_name}: {x.shape[1]} sensors, {len(current)} words",
            flush=True,
        )
        for lag in lags:
            for direction in ("past", "future"):
                target = target_indices(current, lag, direction)
                y = word_scores[target].astype(np.float32)
                finite_y = np.isfinite(y).all(axis=1)
                use_x = x[finite_y]
                use_y = y[finite_y]
                y_pred = cv_predict_ridge(use_x, use_y, alpha, outer_splits)
                point_r = row_corr(use_y, y_pred)
                row = {
                    "target_stem": target_stem,
                    "sensor_group": group_name,
                    "direction": direction,
                    "lag": int(lag),
                    "n_sensors": int(x.shape[1]),
                    "n_samples": int(len(use_y)),
                    "alpha": float(alpha),
                    "mean_point_r_from_point_scores": float(np.nanmean(point_r)),
                }
                row.update(score_predictions(use_y, y_pred))
                rows.append(row)
    scores = pd.DataFrame(rows)
    scores.to_csv(out_path, index=False)
    return scores


def group_column_indices(channel_meta: pd.DataFrame, group_membership: pd.DataFrame) -> dict[str, np.ndarray]:
    lookup = {
        (str(row.subject), str(row.channel)): i
        for i, row in enumerate(channel_meta[["subject", "channel"]].itertuples(index=False))
    }
    out: dict[str, list[int]] = {}
    for group_name, group in group_membership.groupby("sensor_group"):
        cols = []
        for row in group[["subject", "channel"]].itertuples(index=False):
            idx = lookup.get((str(row.subject), str(row.channel)))
            if idx is not None:
                cols.append(idx)
        out[str(group_name)] = sorted(set(cols))
    return {name: np.asarray(cols, dtype=int) for name, cols in out.items()}


def cv_group_ablation_predictions(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    outer_splits: int,
    groups: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    from sklearn.model_selection import KFold

    pred = {"full": np.zeros_like(y, dtype=np.float32)}
    for group_name in groups:
        pred[f"ablated_{group_name}"] = np.zeros_like(y, dtype=np.float32)
    cv = KFold(n_splits=int(outer_splits), shuffle=False)
    for train_idx, test_idx in cv.split(x):
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x[train_idx]).astype(np.float32)
        x_test = x_scaler.transform(x[test_idx]).astype(np.float32)
        y_train = y_scaler.fit_transform(y[train_idx]).astype(np.float32)
        model = Ridge(alpha=float(alpha), fit_intercept=False)
        model.fit(x_train, y_train)
        pred["full"][test_idx] = y_scaler.inverse_transform(model.predict(x_test)).astype(np.float32)
        for group_name, cols in groups.items():
            x_ablate = x_test.copy()
            if len(cols):
                x_ablate[:, cols] = 0.0
            pred[f"ablated_{group_name}"][test_idx] = y_scaler.inverse_transform(model.predict(x_ablate)).astype(np.float32)
    return pred


def run_group_ablation(
    prepared_dir: Path,
    super_brain_root: Path,
    target_stem: str,
    subjects: list[str],
    lags: list[int],
    channel_meta: pd.DataFrame,
    group_membership: pd.DataFrame,
    output_dir: Path,
    outer_splits: int,
    overwrite: bool,
) -> pd.DataFrame:
    out_path = output_dir / "sensor_group_ablation_scores.csv"
    if out_path.is_file() and not overwrite:
        return pd.read_csv(out_path)
    word_scores = np.load(prepared_dir / f"{target_stem}.npy").astype(np.float32)
    alpha = load_alpha(super_brain_root, target_stem)
    x_all, valid = load_pooled_all(prepared_dir, subjects, channel_meta)
    shared_current = valid_current_indices(valid, word_scores, lags, "all_lags")
    x_shared = x_all[shared_current].astype(np.float32)
    finite_x = np.isfinite(x_shared).all(axis=1)
    current = shared_current[finite_x]
    x = x_shared[finite_x]
    groups = group_column_indices(channel_meta, group_membership)
    print(
        f"    {target_stem} full-model group ablation: {x.shape[1]} sensors, {len(current)} words",
        flush=True,
    )

    rows = []
    for lag in lags:
        for direction in ("past", "future"):
            target = target_indices(current, lag, direction)
            y = word_scores[target].astype(np.float32)
            finite_y = np.isfinite(y).all(axis=1)
            use_x = x[finite_y]
            use_y = y[finite_y]
            preds = cv_group_ablation_predictions(use_x, use_y, alpha, outer_splits, groups)
            full_scores = score_predictions(use_y, preds["full"])
            full_point = row_corr(use_y, preds["full"])
            for group_name, cols in groups.items():
                ablated_key = f"ablated_{group_name}"
                ablated_scores = score_predictions(use_y, preds[ablated_key])
                ablated_point = row_corr(use_y, preds[ablated_key])
                rows.append(
                    {
                        "target_stem": target_stem,
                        "direction": direction,
                        "lag": int(lag),
                        "sensor_group": group_name,
                        "n_group_sensors": int(len(cols)),
                        "n_total_sensors": int(x.shape[1]),
                        "n_samples": int(len(use_y)),
                        "alpha": float(alpha),
                        "full_mean_point_r": float(full_scores["mean_point_r"]),
                        "ablated_mean_point_r": float(ablated_scores["mean_point_r"]),
                        "delta_mean_point_r": float(full_scores["mean_point_r"] - ablated_scores["mean_point_r"]),
                        "full_weighted_pc_r": float(full_scores["weighted_pc_r"]),
                        "ablated_weighted_pc_r": float(ablated_scores["weighted_pc_r"]),
                        "delta_weighted_pc_r": float(full_scores["weighted_pc_r"] - ablated_scores["weighted_pc_r"]),
                        "full_mean_point_r_from_point_scores": float(np.nanmean(full_point)),
                        "ablated_mean_point_r_from_point_scores": float(np.nanmean(ablated_point)),
                    }
                )
    scores = pd.DataFrame(rows)
    scores.to_csv(out_path, index=False)
    return scores


def plot_group_ablation_scores(ablation: pd.DataFrame, target_stem: str, output_dir: Path) -> None:
    for metric, ylabel in [
        ("delta_mean_point_r", "full - ablated mean point_r"),
        ("delta_weighted_pc_r", "full - ablated weighted component r"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True, sharey=True)
        for ax, direction in zip(axes, ["past", "future"]):
            sub = ablation[ablation["direction"] == direction]
            for group_name, group in sub.groupby("sensor_group"):
                group = group.sort_values("lag")
                ax.plot(group["lag"], group[metric], marker="o", lw=2.0, label=f"ablate {group_name}")
            ax.axhline(0, color="0.75", lw=0.8, ls="--")
            ax.set_title(direction)
            ax.set_xlabel("lag N")
            ax.grid(True, linewidth=0.35, alpha=0.25)
        axes[0].set_ylabel(ylabel)
        axes[1].legend(frameon=False, fontsize=8, loc="best")
        fig.suptitle(f"{target_stem}: held-out sensor-group ablation")
        fig.savefig(output_dir / f"sensor_group_ablation_{metric}.png", dpi=220)
        plt.close(fig)


def plot_group_decoder_scores(
    group_scores: pd.DataFrame,
    super_brain_root: Path,
    target_stem: str,
    output_dir: Path,
) -> None:
    full_path = super_brain_root / target_stem / "decoders" / "decoder_summary.csv"
    full = pd.read_csv(full_path)
    full = full[full["roi"] == "ALL"].copy()
    full["sensor_group"] = "all_channels_cached"
    plot_df = pd.concat(
        [
            group_scores[["sensor_group", "direction", "lag", "mean_point_r", "weighted_pc_r"]].copy(),
            full[["sensor_group", "direction", "lag", "mean_point_r", "weighted_pc_r"]].copy(),
        ],
        ignore_index=True,
    )
    for metric, ylabel in [
        ("mean_point_r", "mean held-out point_r"),
        ("weighted_pc_r", "weighted component-wise r"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True, sharey=True)
        for ax, direction in zip(axes, ["past", "future"]):
            sub = plot_df[plot_df["direction"] == direction]
            for group_name, group in sub.groupby("sensor_group"):
                group = group.sort_values("lag")
                ax.plot(group["lag"], group[metric], marker="o", lw=2.0, label=group_name)
            ax.set_title(direction)
            ax.set_xlabel("lag N")
            ax.grid(True, linewidth=0.35, alpha=0.25)
        axes[0].set_ylabel(ylabel)
        axes[1].legend(frameon=False, fontsize=8, loc="best")
        fig.suptitle(f"{target_stem}: sensor-group decoders")
        fig.savefig(output_dir / f"sensor_group_decoder_{metric}.png", dpi=220)
        plt.close(fig)


def load_pooled_all(
    prepared_dir: Path,
    subjects: list[str],
    channel_meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    parts = []
    valid: np.ndarray | None = None
    for subject in subjects:
        x_all, v, _ = load_prepared_subject(prepared_dir, subject)
        parts.append(x_all.astype(np.float32))
        valid = v if valid is None else (valid & v)
    if valid is None:
        raise RuntimeError("No subjects loaded.")
    x = np.hstack(parts)
    if "column_idx" in channel_meta.columns:
        cols = channel_meta["column_idx"].to_numpy(dtype=int)
        x = x[:, cols]
    return x, valid


def compute_haufe_patterns(
    prepared_dir: Path,
    super_brain_root: Path,
    target_stem: str,
    subjects: list[str],
    lags: list[int],
    channel_meta: pd.DataFrame,
    output_dir: Path,
    overwrite: bool,
) -> pd.DataFrame:
    out_path = output_dir / "haufe_pattern_indices.csv"
    if out_path.is_file() and not overwrite:
        return pd.read_csv(out_path)
    word_scores = np.load(prepared_dir / f"{target_stem}.npy").astype(np.float32)
    alpha = load_alpha(super_brain_root, target_stem)
    x_all, valid = load_pooled_all(prepared_dir, subjects, channel_meta)
    current = valid_current_indices(valid, word_scores, lags, "all_lags")
    x = x_all[current].astype(np.float32)
    finite_x = np.isfinite(x).all(axis=1)
    current = current[finite_x]
    x = x[finite_x]
    x_scaler = StandardScaler()
    xz = x_scaler.fit_transform(x).astype(np.float32)
    cov_x = (xz.T @ xz) / max(xz.shape[0] - 1, 1)

    rows = channel_meta[["subject", "channel", "x", "y", "z", "paper_roi", "destrieux_label"]].copy()
    summaries = []
    for direction in ("past", "future"):
        norm_cols = []
        for lag in lags:
            y = word_scores[target_indices(current, lag, direction)].astype(np.float32)
            y_scaler = StandardScaler()
            yz = y_scaler.fit_transform(y).astype(np.float32)
            model = Ridge(alpha=float(alpha), fit_intercept=False)
            model.fit(xz, yz)
            weights = model.coef_.T.astype(np.float32)
            cov_y = (yz.T @ yz) / max(yz.shape[0] - 1, 1)
            pattern = cov_x @ weights @ np.linalg.pinv(cov_y)
            col = f"haufe_{direction}_lag_{lag}_pattern_norm"
            rows[col] = np.linalg.norm(pattern, axis=1).astype(np.float32)
            norm_cols.append(col)
            summaries.append(
                {
                    "target_stem": target_stem,
                    "direction": direction,
                    "lag": int(lag),
                    "alpha": float(alpha),
                    "n_samples": int(len(current)),
                    "n_channels": int(xz.shape[1]),
                    "pattern_frobenius_norm": float(np.linalg.norm(pattern)),
                }
            )
        vals = rows[norm_cols].to_numpy(dtype=float)
        frac = vals / np.maximum(vals.sum(axis=1, keepdims=True), 1e-12)
        lag_arr = np.asarray(lags, dtype=float)
        rows[f"haufe_{direction}_lag_center_of_mass"] = frac @ lag_arr
        rows[f"haufe_{direction}_short_mean"] = rows[
            [f"haufe_{direction}_lag_{lag}_pattern_norm" for lag in [lag for lag in lags if lag <= 3]]
        ].mean(axis=1)
        rows[f"haufe_{direction}_long_mean"] = rows[
            [f"haufe_{direction}_lag_{lag}_pattern_norm" for lag in [lag for lag in lags if lag >= 7]]
        ].mean(axis=1)
        rows[f"haufe_{direction}_long_short_norm_index"] = (
            rows[f"haufe_{direction}_long_mean"] - rows[f"haufe_{direction}_short_mean"]
        ) / np.maximum(rows[f"haufe_{direction}_long_mean"] + rows[f"haufe_{direction}_short_mean"], 1e-12)
    rows.to_csv(out_path, index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "haufe_fit_summary.csv", index=False)
    return rows


def plot_haufe_outputs(haufe: pd.DataFrame, target_stem: str, output_dir: Path) -> None:
    for direction in ("past", "future"):
        value_col = f"haufe_{direction}_long_short_norm_index"
        scatter_map(
            haufe,
            value_col,
            output_dir / f"haufe_{direction}_long_short_map_yz.png",
            f"{target_stem}: Haufe pattern long-short index ({direction})",
            "Haufe long-short index",
        )
        plot_roi_box(
            haufe,
            value_col,
            output_dir / f"haufe_{direction}_long_short_by_roi.png",
            f"{target_stem}: Haufe pattern long-short index by ROI ({direction})",
        )
        summary_input = haufe.rename(
            columns={
                value_col: "long_short_norm_index",
                f"haufe_{direction}_lag_center_of_mass": "lag_fraction_center_of_mass",
            }
        ).copy()
        norm_cols = [c for c in haufe.columns if c.startswith(f"haufe_{direction}_lag_") and c.endswith("_pattern_norm")]
        summary_input["total_weight_norm"] = haufe[norm_cols].sum(axis=1)
        roi_summary(summary_input, "long_short_norm_index").to_csv(
            output_dir / f"haufe_{direction}_roi_summary.csv",
            index=False,
        )


def process_target(args: argparse.Namespace, target_stem: str) -> dict[str, object]:
    target_dir = args.output_dir / target_stem
    target_dir.mkdir(parents=True, exist_ok=True)
    weights_file = channel_weights_path(args.super_brain_root, target_stem)
    weights = pd.read_csv(weights_file)
    lag_cols = [f"lag_{lag}_weight_norm" for lag in args.lags]
    require_cols(weights, ["subject", "channel", "x", "y", "z", "paper_roi", "total_weight_norm"] + lag_cols, weights_file)
    weights["subject"] = weights["subject"].astype(str)
    weights["channel"] = weights["channel"].astype(str)
    weights, profile = add_lag_indices(weights, args.lags, args.short_lags, args.long_lags)
    weights["target_stem"] = target_stem
    weights.to_csv(target_dir / "sensor_lag_indices.csv", index=False)

    roi = roi_summary(weights, "long_short_norm_index")
    roi.to_csv(target_dir / "roi_long_short_summary.csv", index=False)
    coord = coordinate_summary(weights, ["long_short_norm_index", "lag_fraction_center_of_mass", "total_weight_norm"])
    coord.to_csv(target_dir / "coordinate_spearman_summary.csv", index=False)
    weights.groupby(["subject", "paper_roi"], dropna=False).agg(
        n_channels=("channel", "count"),
        mean_long_short_norm_index=("long_short_norm_index", "mean"),
        median_long_short_norm_index=("long_short_norm_index", "median"),
        mean_lag_com=("lag_fraction_center_of_mass", "mean"),
        mean_total_weight_norm=("total_weight_norm", "mean"),
    ).reset_index().to_csv(target_dir / "subject_roi_long_short_summary.csv", index=False)

    scatter_map(
        weights,
        "long_short_norm_index",
        target_dir / "decoder_weight_long_short_map_yz.png",
        f"{target_stem}: decoder weight long-short index",
        "long-short index",
    )
    scatter_map(
        weights,
        "lag_fraction_center_of_mass",
        target_dir / "decoder_weight_lag_center_of_mass_map_yz.png",
        f"{target_stem}: decoder weight lag center of mass",
        "lag center of mass",
        cmap="viridis",
        symmetric=False,
    )
    plot_roi_box(
        weights,
        "long_short_norm_index",
        target_dir / "decoder_weight_long_short_by_roi.png",
        f"{target_stem}: decoder weight long-short index by ROI",
    )
    plot_top_profiles(
        weights,
        profile,
        args.lags,
        target_dir / "top_short_long_sensor_profiles.png",
        f"{target_stem}: top short/long sensor lag profiles",
        args.top_n_profiles,
    )

    clustered, cluster_summary = cluster_profiles(weights, profile, args.lags, args.n_clusters, args.random_seed)
    clustered.to_csv(target_dir / "sensor_lag_profile_clusters.csv", index=False)
    cluster_summary.to_csv(target_dir / "profile_cluster_summary.csv", index=False)
    plot_cluster_profiles(
        cluster_summary,
        args.lags,
        target_dir / "profile_cluster_mean_profiles.png",
        f"{target_stem}: sensor lag-profile clusters",
    )
    scatter_map(
        clustered,
        "profile_cluster",
        target_dir / "profile_cluster_map_yz.png",
        f"{target_stem}: sensor lag-profile cluster",
        "cluster",
        cmap="tab10",
        symmetric=False,
    )

    enc_summary = encoding_crosscheck(
        weights,
        target_stem,
        args.preferred_lag_dir,
        target_dir,
        args.encoding_short_lags,
        args.encoding_long_lags,
    )

    group_membership = make_sensor_groups(weights, args.sensor_quantile, args.min_total_weight_quantile)
    group_membership.to_csv(target_dir / "sensor_group_membership.csv", index=False)
    group_scores_path = target_dir / "sensor_group_decoder_scores.csv"
    group_scores = None
    if args.skip_group_decoders:
        group_scores = pd.read_csv(group_scores_path) if group_scores_path.is_file() else None
    else:
        group_scores = run_group_decoders(
            args.prepared_dir,
            args.super_brain_root,
            target_stem,
            [clean_subject(s) for s in args.subjects],
            args.lags,
            group_membership,
            target_dir,
            args.outer_splits,
            args.overwrite,
        )
        plot_group_decoder_scores(group_scores, args.super_brain_root, target_stem, target_dir)

    ablation = None
    ablation_path = target_dir / "sensor_group_ablation_scores.csv"
    if args.skip_group_ablation:
        ablation = pd.read_csv(ablation_path) if ablation_path.is_file() else None
    else:
        ablation = run_group_ablation(
            args.prepared_dir,
            args.super_brain_root,
            target_stem,
            [clean_subject(s) for s in args.subjects],
            args.lags,
            weights,
            group_membership,
            target_dir,
            args.outer_splits,
            args.overwrite,
        )
        plot_group_ablation_scores(ablation, target_stem, target_dir)

    haufe_summary = None
    if not args.skip_haufe:
        haufe = compute_haufe_patterns(
            args.prepared_dir,
            args.super_brain_root,
            target_stem,
            [clean_subject(s) for s in args.subjects],
            args.lags,
            weights,
            target_dir,
            args.overwrite,
        )
        plot_haufe_outputs(haufe, target_stem, target_dir)
        haufe_summary = coordinate_summary(
            haufe,
            ["haufe_past_long_short_norm_index", "haufe_future_long_short_norm_index"],
        ).to_dict(orient="records")

    summary = {
        "target_stem": target_stem,
        "n_channels": int(len(weights)),
        "short_lags": [int(x) for x in args.short_lags],
        "long_lags": [int(x) for x in args.long_lags],
        "mean_decoder_long_short_index": float(np.nanmean(weights["long_short_norm_index"])),
        "median_decoder_long_short_index": float(np.nanmedian(weights["long_short_norm_index"])),
        "fraction_decoder_long_pref_gt0": float(np.nanmean(weights["long_short_norm_index"] > 0)),
        "top_roi_by_mean_long_short": roi.iloc[0].to_dict() if len(roi) else None,
        "encoding_crosscheck": enc_summary,
        "haufe_coordinate_summary": haufe_summary,
    }
    if group_scores is not None and len(group_scores):
        group_summary = (
            group_scores.groupby(["sensor_group", "direction"])
            .agg(
                mean_point_r=("mean_point_r", "mean"),
                max_point_r=("mean_point_r", "max"),
                best_lag=("lag", lambda s: int(s.iloc[np.nanargmax(group_scores.loc[s.index, "mean_point_r"].to_numpy(dtype=float))])),
            )
            .reset_index()
        )
        group_summary.to_csv(target_dir / "sensor_group_decoder_summary.csv", index=False)
        summary["sensor_group_decoder_summary"] = group_summary.to_dict(orient="records")
    if ablation is not None and len(ablation):
        ablation_summary = (
            ablation.groupby(["sensor_group", "direction"])
            .agg(
                mean_delta_point_r=("delta_mean_point_r", "mean"),
                max_delta_point_r=("delta_mean_point_r", "max"),
                best_lag=("lag", lambda s: int(s.iloc[np.nanargmax(ablation.loc[s.index, "delta_mean_point_r"].to_numpy(dtype=float))])),
            )
            .reset_index()
        )
        ablation_summary.to_csv(target_dir / "sensor_group_ablation_summary.csv", index=False)
        summary["sensor_group_ablation_summary"] = ablation_summary.to_dict(orient="records")
    (target_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def cross_target_outputs(output_dir: Path, target_stems: list[str]) -> dict[str, object]:
    tables = []
    for target_stem in target_stems:
        path = output_dir / target_stem / "sensor_lag_indices.csv"
        if path.is_file():
            df = pd.read_csv(path)
            cols = [
                "subject",
                "channel",
                "paper_roi",
                "x",
                "y",
                "z",
                "long_short_norm_index",
                "lag_fraction_center_of_mass",
                "total_weight_norm",
            ]
            df = df[cols].copy()
            df["target_stem"] = target_stem
            tables.append(df)
    if len(tables) < 2:
        return {}
    wide = tables[0]
    for nxt in tables[1:]:
        wide = wide.merge(
            nxt,
            on=["subject", "channel"],
            suffixes=("", "_other"),
        )
    first, second = target_stems[0], target_stems[1]
    wide = tables[0].merge(tables[1], on=["subject", "channel"], suffixes=(f"__{first}", f"__{second}"))
    wide.to_csv(output_dir / "cross_target_sensor_lag_indices.csv", index=False)

    x = wide[f"long_short_norm_index__{first}"].to_numpy(dtype=float)
    y = wide[f"long_short_norm_index__{second}"].to_numpy(dtype=float)
    rho, p, n = finite_spearman(x, y)
    r, pr, _ = finite_pearson(x, y)
    summary = {
        "target_x": first,
        "target_y": second,
        "long_short_spearman_rho": rho,
        "long_short_spearman_p": p,
        "long_short_pearson_r": r,
        "long_short_pearson_p": pr,
        "n": n,
    }
    (output_dir / "cross_target_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[mask], y[mask], s=24, alpha=0.5, edgecolor="none")
    ax.axhline(0, color="0.75", lw=0.8, ls="--")
    ax.axvline(0, color="0.75", lw=0.8, ls="--")
    ax.set_xlabel(f"{first} decoder long-short index")
    ax.set_ylabel(f"{second} decoder long-short index")
    ax.set_title(f"Cross-target sensor lag preference\nSpearman rho={rho:.3f}, p={p:.3g}")
    fig.savefig(output_dir / "cross_target_long_short_scatter.png", dpi=220)
    plt.close(fig)
    return summary


def write_readme(output_dir: Path, summaries: list[dict[str, object]], cross_summary: dict[str, object]) -> None:
    lines = [
        "# Lag Sensor Intuition Outputs",
        "",
        "This folder was generated by `podcast_ecog/clean_code_v2/lag_sensor_intuition.py`.",
        "",
        "Short lags are N=1..3 and long lags are N=7..10 unless noted in the per-target `summary.json` files.",
        "",
        "Per-target folders contain:",
        "",
        "- `sensor_lag_indices.csv`: raw decoder-weight lag indices per sensor.",
        "- `decoder_weight_long_short_map_yz.png`: long-short decoder-weight map.",
        "- `sensor_lag_profile_clusters.csv` and `profile_cluster_mean_profiles.png`: profile clustering.",
        "- `encoding_crosscheck.csv`: merge with per-channel encoding preferred-lag outputs.",
        "- `sensor_group_decoder_scores.csv`: decoders trained from short-preferring and long-preferring sensor sets.",
        "- `sensor_group_ablation_scores.csv`: held-out full-model ablation of those sensor sets.",
        "- `haufe_pattern_indices.csv`: Haufe-transformed pattern-norm lag indices.",
        "- `nilearn_plots/`: glass-brain marker plots generated by `plot_lag_sensor_nilearn.py`.",
        "",
        "Top-level files contain cross-target comparisons.",
        "",
        "## Quick Summary",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"### {summary['target_stem']}",
                "",
                f"- channels: {summary['n_channels']}",
                f"- mean decoder long-short index: {summary['mean_decoder_long_short_index']:.4f}",
                f"- median decoder long-short index: {summary['median_decoder_long_short_index']:.4f}",
                f"- fraction long-preferring (>0): {summary['fraction_decoder_long_pref_gt0']:.3f}",
                "",
            ]
        )
    if cross_summary:
        lines.extend(
            [
                "### Cross Target",
                "",
                f"- Spearman rho: {cross_summary['long_short_spearman_rho']:.4f}",
                f"- p-value: {cross_summary['long_short_spearman_p']:.4g}",
                f"- n: {cross_summary['n']}",
                "",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "prepared_dir": str(args.prepared_dir),
        "super_brain_root": str(args.super_brain_root),
        "preferred_lag_dir": str(args.preferred_lag_dir),
        "target_stems": args.target_stems,
        "subjects": [subject_label(s) for s in args.subjects],
        "lags": args.lags,
        "short_lags": args.short_lags,
        "long_lags": args.long_lags,
        "encoding_short_lags": args.encoding_short_lags,
        "encoding_long_lags": args.encoding_long_lags,
        "n_clusters": args.n_clusters,
        "sensor_quantile": args.sensor_quantile,
        "min_total_weight_quantile": args.min_total_weight_quantile,
        "skip_haufe": bool(args.skip_haufe),
        "skip_group_decoders": bool(args.skip_group_decoders),
        "skip_group_ablation": bool(args.skip_group_ablation),
    }
    (args.output_dir / "analysis_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    summaries = []
    for target_stem in args.target_stems:
        print(f"\n=== {target_stem} ===", flush=True)
        summaries.append(process_target(args, target_stem))
    cross_summary = cross_target_outputs(args.output_dir, args.target_stems)
    write_readme(args.output_dir, summaries, cross_summary)
    print(f"\nWrote lag sensor intuition outputs to: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
