#!/usr/bin/env python3
"""Map super-brain ALL-channel decoder weights onto electrode anatomy by lag.

Fits full-data ridge decoders (same alpha as super_brain_pipeline) from pooled
ALL electrodes to word-vector targets at each lag N, then visualizes the
per-channel weight norm (||beta|| across PCA targets) on cortical coordinates.

Outputs (under <output-root>/<target-stem>/anatomy/):
  - channel_decoder_weights.csv
  - lag_*_weight_norm_maps_{plane}.png   one subplot per lag
  - preferred_lag_map_{plane}.png
  - max_beta_centroids_by_lag.csv + plot
  - lag_weight_map_correlations.csv + distance plot
  - spatial_summary.csv (Spearman coord vs weight; top-k centroid coords)
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    clean_subject,
    subject_label,
    target_indices,
)
from train_decoders import (  # noqa: E402
    channel_indices_for_roi,
    load_prepared_subject,
    load_roi_lookup,
    valid_current_indices,
)

DEFAULT_ROI_METRICS = (
    Path(__file__).resolve().parent.parent / "outputs" / "gpt2_paper_roi_context_layer_exact" / "channel_paper_roi_metrics.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "prepared_midpoint")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "super_brain_midpoint")
    parser.add_argument("--target-stem", default="word_vectors_pca20")
    parser.add_argument("--roi-metrics", type=Path, default=DEFAULT_ROI_METRICS)
    parser.add_argument("--subjects", nargs="+", default=[f"{i:02d}" for i in range(1, 10)])
    parser.add_argument("--lags", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--direction", choices=["past", "future"], default="past")
    parser.add_argument("--ridge-alpha", type=float, default=None, help="Defaults to alpha_tune_summary.json")
    parser.add_argument("--top-k", type=int, default=10, help="Top channels for centroid summaries.")
    parser.add_argument("--map-plane", choices=["yz", "xy", "xz"], default="yz")
    parser.add_argument("--also-xyz-panels", action="store_true", help="Also save 3-panel xyz projection per lag grid.")
    return parser.parse_args()


def load_alpha(output_root: Path, target_stem: str, override: float | None) -> float:
    if override is not None:
        return float(override)
    path = output_root / target_stem / "alpha_tune_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; pass --ridge-alpha explicitly.")
    return float(json.loads(path.read_text(encoding="utf-8"))["best_alpha"])


def load_channel_anatomy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = ["subject", "channel", "x", "y", "z", "paper_roi"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[need + [c for c in ["destrieux_label", "hemisphere"] if c in df.columns]].copy()


def load_pooled_all(
    prepared_dir: Path, subjects: list[str], roi_lookup: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    parts = []
    valid = None
    rows = []
    col = 0
    for subject in subjects:
        x_all, v, ch_names = load_prepared_subject(prepared_dir, subject)
        cols = channel_indices_for_roi(ch_names, roi_lookup, subject, "ALL")
        if len(cols) == 0:
            raise ValueError(f"{subject_label(subject)} ALL: no channels")
        block = x_all[:, cols].astype(np.float32)
        parts.append(block)
        sub = subject_label(subject)
        for local_idx in cols:
            rows.append({"subject": sub, "channel": ch_names[local_idx], "column_idx": col})
            col += 1
        valid = v if valid is None else (valid & v)
    return np.hstack(parts), valid, pd.DataFrame(rows)


def fit_channel_coef(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Return coef with shape (n_channels, n_targets) in standardized feature space."""
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    xz = x_scaler.fit_transform(x).astype(np.float32)
    yz = y_scaler.fit_transform(y).astype(np.float32)
    model = Ridge(alpha=float(alpha), fit_intercept=False)
    model.fit(xz, yz)
    # sklearn: coef_ is (n_targets, n_features)
    return model.coef_.T.astype(np.float32)


def fit_lag_decoders(
    x: np.ndarray,
    word_scores: np.ndarray,
    current: np.ndarray,
    lags: list[int],
    direction: str,
    alpha: float,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    coefs: dict[int, np.ndarray] = {}
    rows = []
    for lag in lags:
        target = target_indices(current, lag, direction)
        y = word_scores[target].astype(np.float32)
        ok = np.isfinite(y).all(axis=1)
        coef = fit_channel_coef(x[ok], y[ok], alpha)
        coefs[int(lag)] = coef
        rows.append(
            {
                "lag": int(lag),
                "direction": direction,
                "n_samples": int(ok.sum()),
                "n_channels": int(coef.shape[0]),
                "n_targets": int(coef.shape[1]),
                "coef_frobenius_norm": float(np.linalg.norm(coef)),
            }
        )
    return coefs, pd.DataFrame(rows)


def add_weight_columns(channels: pd.DataFrame, coefs: dict[int, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    lags = sorted(coefs)
    out = channels.copy()
    norm_matrix = []
    for lag in lags:
        norm = np.linalg.norm(coefs[lag], axis=1)
        out[f"lag_{lag}_weight_norm"] = norm
        norm_matrix.append(norm)
    norm_matrix = np.vstack(norm_matrix).T
    lag_array = np.asarray(lags, dtype=float)
    norm_sum = np.maximum(norm_matrix.sum(axis=1, keepdims=True), 1e-12)
    frac = norm_matrix / norm_sum
    out["preferred_lag"] = [lags[int(i)] for i in np.argmax(norm_matrix, axis=1)]
    out["lag_center_of_mass"] = frac @ lag_array
    out["total_weight_norm"] = norm_matrix.sum(axis=1)
    for i, lag in enumerate(lags):
        out[f"lag_{lag}_weight_fraction"] = frac[:, i]
    profile = pd.DataFrame(norm_matrix, columns=[str(l) for l in lags])
    return out, profile


def plane_columns(plane: str) -> tuple[str, str, str, str]:
    if plane == "yz":
        return "y", "z", "anterior(+)/posterior(-) y (mm)", "superior(+)/inferior(-) z (mm)"
    if plane == "xy":
        return "x", "y", "right(+)/left(-) x (MNI mm)", "anterior(+)/posterior(-) y (mm)"
    return "x", "z", "right(+)/left(-) x (MNI mm)", "superior(+)/inferior(-) z (mm)"


def scatter_lag_grid(
    df: pd.DataFrame,
    lags: list[int],
    output_path: Path,
    plane: str,
    *,
    title: str,
    value_prefix: str = "weight_norm",
) -> None:
    xcol, ycol, xlabel, ylabel = plane_columns(plane)
    ncols = min(5, len(lags))
    nrows = int(math.ceil(len(lags) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.8 * nrows), constrained_layout=True, squeeze=False)
    cols = [f"lag_{lag}_{value_prefix}" for lag in lags]
    vals = df[cols].to_numpy(dtype=float)
    vmin = float(np.nanpercentile(vals, 2))
    vmax = float(np.nanpercentile(vals, 98))
    last_sc = None
    for ax, lag in zip(axes.flat, lags):
        col = f"lag_{lag}_{value_prefix}"
        last_sc = ax.scatter(
            df[xcol],
            df[ycol],
            c=df[col],
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            s=42,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.88,
        )
        ax.set_title(f"lag {lag}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.35, alpha=0.2)
    for ax in axes.flat[len(lags) :]:
        ax.axis("off")
    fig.suptitle(title)
    if last_sc is not None:
        fig.colorbar(last_sc, ax=axes, shrink=0.82, label="||beta|| (standardized features)")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def scatter_xyz_panels(df: pd.DataFrame, lags: list[int], output_path: Path, title: str) -> None:
    ncols = min(5, len(lags))
    nrows = int(math.ceil(len(lags) / ncols))
    pairs = [("x", "y"), ("x", "z"), ("y", "z")]
    fig, axes = plt.subplots(nrows, ncols * 3, figsize=(3.2 * ncols * 3, 3.4 * nrows), constrained_layout=True, squeeze=False)
    cols = [f"lag_{lag}_weight_norm" for lag in lags]
    vals = df[cols].to_numpy(dtype=float)
    vmin = float(np.nanpercentile(vals, 2))
    vmax = float(np.nanpercentile(vals, 98))
    for i, lag in enumerate(lags):
        row, base = divmod(i, ncols)
        col_name = f"lag_{lag}_weight_norm"
        for j, (xc, yc) in enumerate(pairs):
            ax = axes[row, base * 3 + j]
            sc = ax.scatter(
                df[xc], df[yc], c=df[col_name], cmap="magma", vmin=vmin, vmax=vmax,
                s=28, edgecolor="black", linewidth=0.12, alpha=0.85,
            )
            ax.set_title(f"lag {lag} ({xc}-{yc})")
            ax.axhline(0, color="0.85", lw=0.6)
            ax.axvline(0, color="0.85", lw=0.6)
    for ax in axes.flat:
        if not ax.has_data():
            ax.axis("off")
    fig.suptitle(title)
    fig.colorbar(sc, ax=axes, shrink=0.75, label="||beta||")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_preferred_lag(df: pd.DataFrame, output_path: Path, plane: str) -> None:
    xcol, ycol, xlabel, ylabel = plane_columns(plane)
    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    sc = ax.scatter(
        df[xcol],
        df[ycol],
        c=df["preferred_lag"],
        cmap="viridis",
        s=58,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.9,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Electrodes colored by lag with largest ||beta||")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("preferred lag N")
    ax.grid(True, linewidth=0.35, alpha=0.2)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def global_weight_com(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    rows = []
    for lag in lags:
        col = f"lag_{lag}_weight_norm"
        wt = df[col].to_numpy(dtype=float)
        wt = wt / max(wt.sum(), 1e-12)
        rows.append(
            {
                "lag": int(lag),
                "x": float(np.sum(wt * df["x"].to_numpy(dtype=float))),
                "y": float(np.sum(wt * df["y"].to_numpy(dtype=float))),
                "z": float(np.sum(wt * df["z"].to_numpy(dtype=float))),
            }
        )
    return pd.DataFrame(rows)


def _fit_lag_line_points(lags: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.column_stack(
        [
            stats.linregress(lags, pts[:, j]).intercept + stats.linregress(lags, pts[:, j]).slope * lags
            for j in range(3)
        ]
    )
    resid = np.linalg.norm(pts - pred, axis=1)
    return pred, resid


def lag_line_metrics(lags: list[int], pts: np.ndarray, locus: str) -> pd.DataFrame:
    """Test whether spatial positions move linearly with lag N."""
    lag_arr = np.asarray(lags, dtype=float)
    rows = []
    for j, coord in enumerate(["x", "y", "z"]):
        slope, intercept, r, p, se = stats.linregress(lag_arr, pts[:, j])
        rho, p_spearman = stats.spearmanr(lag_arr, pts[:, j])
        rows.append(
            {
                "locus": locus,
                "coord": coord,
                "pearson_r": float(r),
                "pearson_r2": float(r**2),
                "pearson_p": float(p),
                "spearman_rho": float(rho),
                "spearman_p": float(p_spearman),
                "slope_per_lag_mm": float(slope),
                "intercept_mm": float(intercept),
            }
        )
    pred, resid = _fit_lag_line_points(lag_arr, pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    path_len = float(seg.sum())
    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    centered = pts - pts.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = centered @ vt[0]
    pc1_var = float(s[0] ** 2 / max(np.sum(s**2), 1e-12))
    r_pc1, p_pc1 = stats.pearsonr(lag_arr, pc1)
    rows.append(
        {
            "locus": locus,
            "coord": "summary",
            "pearson_r": float(r_pc1),
            "pearson_r2": float(r_pc1**2),
            "pearson_p": float(p_pc1),
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "slope_per_lag_mm": np.nan,
            "intercept_mm": np.nan,
            "path_length_mm": path_len,
            "chord_mm": chord,
            "chord_over_path": chord / max(path_len, 1e-12),
            "mean_perp_dist_mm": float(resid.mean()),
            "max_perp_dist_mm": float(resid.max()),
            "pc1_explained_var": pc1_var,
        }
    )
    return pd.DataFrame(rows)


def mean_subject_com(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    per_subject = [global_weight_com(group, lags) for _, group in df.groupby("subject")]
    return pd.DataFrame(
        {
            "lag": lags,
            "x": np.mean([part["x"].to_numpy(float) for part in per_subject], axis=0),
            "y": np.mean([part["y"].to_numpy(float) for part in per_subject], axis=0),
            "z": np.mean([part["z"].to_numpy(float) for part in per_subject], axis=0),
        }
    )


def com_medial_lateral(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Distance from interhemispheric midline (|x|), comparable across hemispheres."""
    rows = []
    for lag in lags:
        col = f"lag_{lag}_weight_norm"
        wt = df[col].to_numpy(dtype=float)
        wt = wt / max(wt.sum(), 1e-12)
        abs_x = np.abs(df["x"].to_numpy(dtype=float))
        rows.append({"lag": int(lag), "medial_lateral_mm": float(np.sum(wt * abs_x))})
    return pd.DataFrame(rows)


def per_subject_com_slopes(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    rows = []
    lag_arr = np.asarray(lags, dtype=float)
    for subject, group in df.groupby("subject"):
        com = global_weight_com(group, lags)
        row = {"subject": subject}
        for coord in ["x", "y", "z"]:
            slope, intercept, r, p, se = stats.linregress(lag_arr, com[coord].to_numpy(float))
            row[f"{coord}_slope_per_lag"] = float(slope)
            row[f"{coord}_r"] = float(r)
            row[f"{coord}_p"] = float(p)
        rows.append(row)
    return pd.DataFrame(rows)


def direction_audit(df: pd.DataFrame, lags: list[int]) -> dict:
    """Summarize anatomically labeled COM drift with lag (MNI152: +x=R, +y=A, +z=S)."""
    com = global_weight_com(df, lags)
    sub_com = mean_subject_com(df, lags)
    medial = com_medial_lateral(df, lags)
    sub_slopes = per_subject_com_slopes(df, lags)
    lag_arr = np.asarray(lags, dtype=float)
    pts = com[["x", "y", "z"]].to_numpy(float)
    _, _, vt = np.linalg.svd(pts - pts.mean(axis=0, keepdims=True), full_matrices=False)
    pc1 = vt[0]
    med_slope, _, med_r, med_p, _ = stats.linregress(lag_arr, medial["medial_lateral_mm"].to_numpy(float))
    rh_frac = []
    for lag in lags:
        w = df[f"lag_{lag}_weight_norm"].to_numpy(dtype=float)
        rh_frac.append(float(w[df["x"].to_numpy(float) >= 0].sum() / max(w.sum(), 1e-12)))
    return {
        "coordinate_system": "MNI152NLin2009aSym mm; +x=right, +y=anterior, +z=superior",
        "lag_semantics": "past direction: N=1 is one word back, larger N is farther into the past",
        "pooled_com_n1_to_n10_delta_mm": {
            "x_raw": float(com.iloc[-1]["x"] - com.iloc[0]["x"]),
            "y": float(com.iloc[-1]["y"] - com.iloc[0]["y"]),
            "z": float(com.iloc[-1]["z"] - com.iloc[0]["z"]),
            "medial_lateral_abs_x": float(medial.iloc[-1]["medial_lateral_mm"] - medial.iloc[0]["medial_lateral_mm"]),
        },
        "anatomical_interpretation_increasing_lag": {
            "x_raw_mni": "toward midline from left-hemisphere bulk (NOT a right-hemisphere shift)",
            "medial_lateral": "medial (decreasing |x|)" if med_slope < 0 else "lateral (increasing |x|)",
            "y": "posterior (decreasing y)" if stats.linregress(lag_arr, com["y"])[0] < 0 else "anterior",
            "z": "superior (increasing z)" if stats.linregress(lag_arr, com["z"])[0] > 0 else "inferior",
        },
        "pc1_direction_cosines_xyz": [float(v) for v in pc1],
        "right_hemisphere_weight_fraction_by_lag": {int(lags[i]): rh_frac[i] for i in range(len(lags))},
        "right_hemisphere_fraction_range": [float(min(rh_frac)), float(max(rh_frac))],
        "per_subject_y_slope_negative_count": int((sub_slopes["y_slope_per_lag"] < 0).sum()),
        "per_subject_y_slope_n": int(len(sub_slopes)),
        "mean_subject_com_slopes_per_lag": {
            coord: float(stats.linregress(lag_arr, sub_com[coord].to_numpy(float)).slope) for coord in ["x", "y", "z"]
        },
        "medial_lateral_slope_per_lag": float(med_slope),
        "medial_lateral_r": float(med_r),
        "medial_lateral_p": float(med_p),
    }


def plot_lag_line_analysis(
    centroids: pd.DataFrame,
    com_df: pd.DataFrame,
    output_dir: Path,
    dir_tag: str,
    plane: str,
) -> None:
    xcol, ycol, xlabel, ylabel = plane_columns(plane)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True, sharex=True)
    for ax, coord in zip(axes, ["x", "y", "z"]):
        ax.plot(centroids["lag"], centroids[f"weighted_{coord}"], "-o", color="C0", label="top-10 centroid")
        ax.plot(com_df["lag"], com_df[coord], "-s", color="C2", label="all-channel COM")
        ax.plot(centroids["lag"], centroids[f"argmax_{coord}"], "x--", color="C3", alpha=0.7, label="global argmax")
        slope, intercept, *_ = stats.linregress(com_df["lag"].to_numpy(float), com_df[coord].to_numpy(float))
        lag_range = np.array([com_df["lag"].min(), com_df["lag"].max()], dtype=float)
        ax.plot(lag_range, intercept + slope * lag_range, "--", color="C2", alpha=0.55, lw=1.2)
        ax.set_xlabel("lag N")
        ax.set_ylabel(f"{coord} (mm)")
        ax.set_title(coord)
        ax.grid(True, linewidth=0.35, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Spatial position vs lag (does the locus move along a line?)", y=1.02)
    fig.savefig(output_dir / f"lag_{dir_tag}_spatial_line_coords_{plane}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.6), constrained_layout=True)
    wx, wy = f"weighted_{xcol}", f"weighted_{ycol}"
    ax.plot(centroids[wx], centroids[wy], "-o", color="C0", lw=1.8, label="top-10 centroid")
    ax.plot(com_df[xcol], com_df[ycol], "-s", color="C2", lw=1.8, label="all-channel COM")
    ax.scatter(centroids[f"argmax_{xcol}"], centroids[f"argmax_{ycol}"], s=70, facecolors="none", edgecolors="C3", label="global argmax")
    for _, row in com_df.iterrows():
        ax.annotate(f"N={int(row['lag'])}", (row[xcol], row[ycol]), textcoords="offset points", xytext=(3, 3), fontsize=7, color="C2")
    lag_arr = com_df["lag"].to_numpy(float)
    pts = com_df[[xcol, ycol]].to_numpy(float)
    pred2d = np.column_stack(
        [stats.linregress(lag_arr, pts[:, j]).intercept + stats.linregress(lag_arr, pts[:, j]).slope * lag_arr for j in range(2)]
    )
    ax.plot(pred2d[:, 0], pred2d[:, 1], "--", color="C2", alpha=0.6, lw=1.5, label="COM linear fit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Trajectory in brain space (+ linear fit on all-channel COM)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, linewidth=0.35, alpha=0.25)
    fig.savefig(output_dir / f"lag_{dir_tag}_spatial_line_trajectory_{plane}.png", dpi=220)
    plt.close(fig)


def per_subject_lag_line_metrics(argmax_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, group in argmax_df.groupby("subject"):
        group = group.sort_values("lag")
        lags = group["lag"].to_numpy(dtype=int).tolist()
        if len(lags) < 4:
            continue
        pts = group[["x", "y", "z"]].to_numpy(dtype=float)
        lag_arr = group["lag"].to_numpy(dtype=float)
        centered = pts - pts.mean(axis=0, keepdims=True)
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        pc1 = centered @ vt[0]
        r_pc1, p_pc1 = stats.pearsonr(lag_arr, pc1)
        _, resid = _fit_lag_line_points(lag_arr, pts)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        rows.append(
            {
                "subject": subject,
                "n_lags": len(lags),
                "pc1_lag_pearson_r": float(r_pc1),
                "pc1_lag_p": float(p_pc1),
                "chord_over_path": float(np.linalg.norm(pts[-1] - pts[0]) / max(seg.sum(), 1e-12)),
                "mean_perp_dist_mm": float(resid.mean()),
            }
        )
    return pd.DataFrame(rows)


def topk_centroids(df: pd.DataFrame, lags: list[int], top_k: int) -> pd.DataFrame:
    rows = []
    for lag in lags:
        col = f"lag_{lag}_weight_norm"
        sub = df.sort_values(col, ascending=False).head(int(top_k))
        w = sub[col].to_numpy(dtype=float)
        w = w / max(w.sum(), 1e-12)
        rows.append(
            {
                "lag": int(lag),
                "top_k": int(top_k),
                "weighted_x": float(np.sum(w * sub["x"].to_numpy(dtype=float))),
                "weighted_y": float(np.sum(w * sub["y"].to_numpy(dtype=float))),
                "weighted_z": float(np.sum(w * sub["z"].to_numpy(dtype=float))),
                "argmax_value": float(sub[col].iloc[0]),
                "argmax_subject": sub["subject"].iloc[0],
                "argmax_channel": sub["channel"].iloc[0],
                "argmax_x": float(sub["x"].iloc[0]),
                "argmax_y": float(sub["y"].iloc[0]),
                "argmax_z": float(sub["z"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def plot_centroid_trajectory(centroids: pd.DataFrame, output_path: Path, plane: str) -> None:
    xcol, ycol, xlabel, ylabel = plane_columns(plane)
    wx, wy = f"weighted_{xcol}", f"weighted_{ycol}"
    axc, ayc = f"argmax_{xcol}", f"argmax_{ycol}"
    fig, ax = plt.subplots(figsize=(6.8, 5.6), constrained_layout=True)
    ax.plot(centroids[wx], centroids[wy], "-o", color="black", lw=1.8, label="top-k weighted centroid")
    ax.scatter(centroids[axc], centroids[ayc], s=80, facecolors="none", edgecolors="C3", linewidths=1.2, label="global argmax")
    for _, row in centroids.iterrows():
        ax.annotate(f"N={int(row['lag'])}", (row[wx], row[wy]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Spatial locus of peak decoder weights across lags")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, linewidth=0.35, alpha=0.25)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def spearman_coord_table(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    rows = []
    for lag in lags:
        col = f"lag_{lag}_weight_norm"
        for coord in ["x", "y", "z"]:
            sub = df[[col, coord]].dropna()
            if len(sub) < 8 or sub[col].nunique() < 2:
                rho, p = np.nan, np.nan
            else:
                rho, p = stats.spearmanr(sub[coord], sub[col])
            rows.append({"lag": int(lag), "coord": coord, "metric": col, "spearman_rho": rho, "p": p, "n": len(sub)})
    return pd.DataFrame(rows)


def per_subject_argmax(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    rows = []
    for lag in lags:
        col = f"lag_{lag}_weight_norm"
        for subject, group in df.groupby("subject"):
            if group[col].notna().sum() == 0:
                continue
            idx = group[col].idxmax()
            row = group.loc[idx]
            rows.append(
                {
                    "lag": int(lag),
                    "subject": subject,
                    "channel": row["channel"],
                    "weight_norm": float(row[col]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "paper_roi": row.get("paper_roi", np.nan),
                }
            )
    return pd.DataFrame(rows)


def plot_subject_argmax_spread(argmax_df: pd.DataFrame, output_path: Path, plane: str) -> None:
    xcol, ycol, xlabel, ylabel = plane_columns(plane)
    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    for lag, group in argmax_df.groupby("lag"):
        ax.scatter(group[xcol], group[ycol], s=55, label=f"N={lag}", alpha=0.85, edgecolor="white", linewidth=0.35)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Per-subject argmax channel location at each lag")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.grid(True, linewidth=0.35, alpha=0.2)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def map_correlation(profile: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    corr = profile.corr(method="pearson")
    corr.to_csv(output_dir / "lag_weight_map_correlations.csv")
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    im = ax.imshow(corr.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.index)
    ax.set_xlabel("lag N")
    ax.set_ylabel("lag N")
    ax.set_title("Correlation of channel weight maps across lags")
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.savefig(output_dir / "lag_weight_map_correlations.png", dpi=220)
    plt.close(fig)
    return corr


def main() -> int:
    args = parse_args()
    subjects = [clean_subject(s) for s in args.subjects]
    lags = sorted(set(int(l) for l in args.lags if int(l) > 0))
    alpha = load_alpha(args.output_root, args.target_stem, args.ridge_alpha)

    out_dir = args.output_root / args.target_stem / "anatomy"
    out_dir.mkdir(parents=True, exist_ok=True)

    word_scores = np.load(args.prepared_dir / f"{args.target_stem}.npy").astype(np.float32)
    roi_lookup = load_roi_lookup(args.prepared_dir)
    x_all, valid, channel_table = load_pooled_all(args.prepared_dir, subjects, roi_lookup)
    current = valid_current_indices(valid, word_scores, lags, "all_lags")
    x = x_all[current].astype(np.float32)

    anatomy = load_channel_anatomy(args.roi_metrics)
    channels = channel_table.merge(anatomy, on=["subject", "channel"], how="left")
    missing = channels["x"].isna().sum()
    if missing:
        print(f"Warning: {missing} pooled channels missing xyz coordinates; dropping them.", flush=True)
        keep = channels["x"].notna().to_numpy()
        channels = channels.loc[keep].reset_index(drop=True)
        x = x[:, keep]

    print(
        f"Fitting super-brain ALL decoders: {x.shape[1]} channels, {x.shape[0]} words, "
        f"alpha={alpha}, direction={args.direction}",
        flush=True,
    )
    coefs, coef_summary = fit_lag_decoders(x, word_scores, current, lags, args.direction, alpha)
    coef_summary.to_csv(out_dir / "lag_decoder_coef_summary.csv", index=False)

    weights, profile = add_weight_columns(channels, coefs)
    weights.to_csv(out_dir / "channel_decoder_weights.csv", index=False)

    dir_tag = args.direction
    scatter_lag_grid(
        weights,
        lags,
        out_dir / f"lag_{dir_tag}_weight_norm_maps_{args.map_plane}.png",
        args.map_plane,
        title=f"Decoder ||beta|| by lag N ({args.target_stem}, {dir_tag})",
    )
    if args.also_xyz_panels:
        scatter_xyz_panels(
            weights,
            lags,
            out_dir / f"lag_{dir_tag}_weight_norm_maps_xyz_panels.png",
            title=f"Decoder ||beta|| by lag N ({args.target_stem}, {dir_tag})",
        )

    plot_preferred_lag(weights, out_dir / f"preferred_lag_map_{dir_tag}_{args.map_plane}.png", args.map_plane)

    centroids = topk_centroids(weights, lags, args.top_k)
    centroids.to_csv(out_dir / "max_beta_centroids_by_lag.csv", index=False)
    plot_centroid_trajectory(centroids, out_dir / f"max_beta_centroid_trajectory_{dir_tag}_{args.map_plane}.png", args.map_plane)

    com_df = global_weight_com(weights, lags)
    com_df.to_csv(out_dir / "global_weight_com_by_lag.csv", index=False)
    mean_sub_com_df = mean_subject_com(weights, lags)
    mean_sub_com_df.to_csv(out_dir / "mean_subject_com_by_lag.csv", index=False)
    com_medial_lateral(weights, lags).to_csv(out_dir / "global_weight_medial_lateral_by_lag.csv", index=False)
    per_subject_com_slopes(weights, lags).to_csv(out_dir / "per_subject_com_slopes_by_lag.csv", index=False)
    audit = direction_audit(weights, lags)
    with (out_dir / "com_direction_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    line_parts = [
        lag_line_metrics(lags, centroids[["weighted_x", "weighted_y", "weighted_z"]].to_numpy(float), "top10_centroid"),
        lag_line_metrics(lags, centroids[["argmax_x", "argmax_y", "argmax_z"]].to_numpy(float), "global_argmax"),
        lag_line_metrics(lags, com_df[["x", "y", "z"]].to_numpy(float), "all_channel_com"),
        lag_line_metrics(lags, mean_sub_com_df[["x", "y", "z"]].to_numpy(float), "mean_subject_com"),
    ]
    line_metrics = pd.concat(line_parts, ignore_index=True)
    line_metrics.to_csv(out_dir / "lag_spatial_line_metrics.csv", index=False)
    plot_lag_line_analysis(centroids, com_df, out_dir, dir_tag, args.map_plane)

    argmax_df = per_subject_argmax(weights, lags)
    argmax_df.to_csv(out_dir / "per_subject_argmax_channel_by_lag.csv", index=False)
    plot_subject_argmax_spread(argmax_df, out_dir / f"per_subject_argmax_spread_{dir_tag}_{args.map_plane}.png", args.map_plane)
    per_sub_line = per_subject_lag_line_metrics(argmax_df)
    per_sub_line.to_csv(out_dir / "per_subject_argmax_lag_line_metrics.csv", index=False)

    spearman = spearman_coord_table(weights, lags)
    spearman.to_csv(out_dir / "coord_spearman_vs_weight_norm.csv", index=False)

    map_correlation(profile, out_dir)

    summary = {
        "target_stem": args.target_stem,
        "direction": args.direction,
        "ridge_alpha": alpha,
        "n_channels": int(len(weights)),
        "n_words": int(x.shape[0]),
        "lags": lags,
        "top_k_centroid": int(args.top_k),
        "all_channel_com_pc1_lag_r": float(
            line_metrics.loc[(line_metrics["locus"] == "all_channel_com") & (line_metrics["coord"] == "summary"), "pearson_r"].iloc[0]
        ),
        "all_channel_com_chord_over_path": float(
            line_metrics.loc[(line_metrics["locus"] == "all_channel_com") & (line_metrics["coord"] == "summary"), "chord_over_path"].iloc[0]
        ),
        "com_direction_audit": audit,
    }
    with (out_dir / "anatomy_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote super-brain channel anatomy to: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
