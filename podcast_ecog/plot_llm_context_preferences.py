#!/usr/bin/env python3
"""Visualize which ECoG channels prefer which LLM context length."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from nilearn.plotting import plot_markers


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MOUNT_ROOT = Path("/Volumes/ellie/language-decoding-expts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=THIS_DIR / "outputs" / "llm_context_sweep")
    parser.add_argument("--mounted-root", type=Path, default=DEFAULT_MOUNT_ROOT)
    parser.add_argument("--subject", default="03")
    parser.add_argument("--models", nargs="+", default=None, help="Subset model names from the summary CSV.")
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Subset layer indices.")
    parser.add_argument("--reference-results", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "outputs" / "llm_context_preferences")
    parser.add_argument("--short-context-max", type=int, default=2)
    parser.add_argument("--long-context-min", type=int, default=16)
    parser.add_argument("--min-best-r", type=float, default=None, help="Only plot channels with best r at least this value.")
    parser.add_argument("--node-size", type=float, default=35.0)
    parser.add_argument("--display-mode", default="lzr")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def find_reference(args: argparse.Namespace) -> Path:
    if args.reference_results is not None:
        return args.reference_results
    path = (
        args.mounted_root
        / "podcast_ecog"
        / "outputs_all_channels"
        / f"sub-{args.subject}_gpt2-xl_layer-24_encoding_results.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find reference result with coordinates: {path}. "
            "Pass --reference-results explicitly."
        )
    return path


def load_channel_metadata(args: argparse.Namespace) -> pd.DataFrame:
    target_path = args.sweep_dir / "targets" / f"sub-{args.subject}" / "preferred_lag_targets.npz"
    if not target_path.is_file():
        matches = sorted((args.sweep_dir / "targets" / f"sub-{args.subject}" / "cache").glob("*targets.npz"))
        if not matches:
            raise FileNotFoundError(
                f"No target cache found under {args.sweep_dir / 'targets' / f'sub-{args.subject}'}. "
                "Run the sweep with ridge evaluation first, not --extract-only."
            )
        target_path = matches[-1]

    with np.load(target_path, allow_pickle=True) as target:
        channel_names = target["channel_names"].astype(str)
        preferred_lag_s = target["preferred_lag_s"].astype(float)

    reference_path = find_reference(args)
    with np.load(reference_path, allow_pickle=True) as reference:
        ref_names = reference["channel_names"].astype(str)
        coords = reference["coords"].astype(float)

    if np.nanmax(np.abs(coords)) < 1:
        coords = coords * 1000.0

    ref_lookup = {name: i for i, name in enumerate(ref_names)}
    missing = [name for name in channel_names if name not in ref_lookup]
    if missing:
        raise ValueError(f"{len(missing)} target channels are missing from coordinate reference, e.g. {missing[:5]}")

    coord_idx = np.asarray([ref_lookup[name] for name in channel_names], dtype=int)
    table = pd.DataFrame(
        {
            "subject": f"sub-{args.subject}",
            "channel": channel_names,
            "preferred_lag_s": preferred_lag_s,
            "x": coords[coord_idx, 0],
            "y": coords[coord_idx, 1],
            "z": coords[coord_idx, 2],
        }
    )
    return table


def add_long_short_contrast(table: pd.DataFrame, contexts: list[int], score_matrix: np.ndarray, args: argparse.Namespace) -> pd.DataFrame:
    contexts_arr = np.asarray(contexts, dtype=int)
    short_idx = np.flatnonzero(contexts_arr <= int(args.short_context_max))
    long_idx = np.flatnonzero(contexts_arr >= int(args.long_context_min))
    if len(short_idx) == 0 or len(long_idx) == 0:
        table["best_short_context_tokens"] = np.nan
        table["best_long_context_tokens"] = np.nan
        table["best_short_r"] = np.nan
        table["best_long_r"] = np.nan
        table["long_minus_short_r"] = np.nan
        return table

    short_scores = score_matrix[short_idx]
    long_scores = score_matrix[long_idx]
    best_short_pos = short_scores.argmax(axis=0)
    best_long_pos = long_scores.argmax(axis=0)
    best_short_r = short_scores[best_short_pos, np.arange(score_matrix.shape[1])]
    best_long_r = long_scores[best_long_pos, np.arange(score_matrix.shape[1])]
    table["best_short_context_tokens"] = contexts_arr[short_idx][best_short_pos]
    table["best_long_context_tokens"] = contexts_arr[long_idx][best_long_pos]
    table["best_short_r"] = best_short_r
    table["best_long_r"] = best_long_r
    table["long_minus_short_r"] = best_long_r - best_short_r
    return table


def corr_key(model_name: str, context_tokens: int, layer: int, subject: str) -> str:
    return f"{safe_name(model_name)}__ctx{int(context_tokens)}__layer{int(layer)}__sub{subject}"


def load_sweep(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    summary_path = args.sweep_dir / "llm_context_sweep_summary.csv"
    corrs_path = args.sweep_dir / "llm_context_sweep_corrs.npz"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not corrs_path.is_file():
        raise FileNotFoundError(corrs_path)

    summary = pd.read_csv(summary_path)
    summary = summary[summary["subject"] == f"sub-{args.subject}"].copy()
    if args.models is not None:
        summary = summary[summary["model"].isin(args.models)].copy()
    if args.layers is not None:
        summary = summary[summary["layer"].isin(args.layers)].copy()
    if summary.empty:
        raise ValueError("No summary rows matched the requested subject/model/layer filters.")

    with np.load(corrs_path, allow_pickle=True) as data:
        corrs = {key: data[key] for key in data.files}
    return summary, corrs


def preference_table_for_group(
    *,
    metadata: pd.DataFrame,
    group: pd.DataFrame,
    corrs: dict[str, np.ndarray],
    subject: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    group = group.sort_values("context_tokens")
    model = str(group["model"].iloc[0])
    layer = int(group["layer"].iloc[0])
    contexts = group["context_tokens"].to_numpy(dtype=int)
    channel_scores = []
    used_contexts = []

    for context in contexts:
        key = corr_key(model, int(context), layer, subject)
        if key not in corrs:
            raise KeyError(f"Missing correlations for {key}")
        values = corrs[key].mean(axis=0)
        channel_scores.append(values)
        used_contexts.append(int(context))

    score_matrix = np.vstack(channel_scores)
    best_idx = score_matrix.argmax(axis=0)
    best_r = score_matrix[best_idx, np.arange(score_matrix.shape[1])]
    best_context = np.asarray(used_contexts, dtype=int)[best_idx]

    table = metadata.copy()
    table.insert(1, "model", model)
    table.insert(2, "layer", layer)
    table["preferred_context_tokens"] = best_context
    table["best_r"] = best_r
    table["worst_r"] = score_matrix.min(axis=0)
    table["context_range_r"] = table["best_r"] - table["worst_r"]
    if 0 in used_contexts:
        table["delta_vs_context_0"] = table["best_r"] - score_matrix[used_contexts.index(0)]
    for context, scores in zip(used_contexts, score_matrix):
        table[f"r_context_{context}"] = scores
    table = add_long_short_contrast(table, used_contexts, score_matrix, args=args)
    return table


def context_cmap(contexts: np.ndarray) -> tuple[ListedColormap, BoundaryNorm, np.ndarray]:
    contexts = np.asarray(sorted(np.unique(contexts)), dtype=float)
    base = plt.get_cmap("viridis", len(contexts))
    cmap = ListedColormap(base(np.arange(len(contexts))))
    if len(contexts) == 1:
        bounds = np.asarray([contexts[0] - 0.5, contexts[0] + 0.5])
    else:
        mids = (contexts[:-1] + contexts[1:]) / 2.0
        bounds = np.r_[contexts[0] - (mids[0] - contexts[0]), mids, contexts[-1] + (contexts[-1] - mids[-1])]
    return cmap, BoundaryNorm(bounds, cmap.N), contexts


def plot_brain(table: pd.DataFrame, output_path: Path, node_size: float, display_mode: str) -> None:
    plot_table = table.dropna(subset=["preferred_context_tokens", "x", "y", "z"]).copy()
    if plot_table.empty:
        return
    values = plot_table["preferred_context_tokens"].to_numpy(dtype=float)
    coords = plot_table[["x", "y", "z"]].to_numpy(dtype=float)
    order = np.argsort(values)
    display = plot_markers(
        values[order],
        coords[order],
        node_size=node_size,
        display_mode=display_mode,
        node_cmap="viridis",
        node_vmin=float(np.nanmin(values)),
        node_vmax=float(np.nanmax(values)),
        colorbar=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display.savefig(output_path, dpi=250)
    display.close()


def plot_projection(table: pd.DataFrame, output_path: Path) -> None:
    plot_table = table.dropna(subset=["preferred_context_tokens", "x", "y", "z"]).copy()
    if plot_table.empty:
        return
    contexts = plot_table["preferred_context_tokens"].to_numpy(dtype=int)
    cmap, norm, ordered_contexts = context_cmap(contexts)
    coords = plot_table[["x", "y", "z"]]
    pairs = [("x", "y"), ("x", "z"), ("y", "z")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (x_col, y_col) in zip(axes, pairs):
        scatter = ax.scatter(
            coords[x_col],
            coords[y_col],
            c=contexts,
            cmap=cmap,
            norm=norm,
            s=42,
            edgecolor="black",
            linewidth=0.25,
            alpha=0.9,
        )
        ax.set_xlabel(f"{x_col} mm")
        ax.set_ylabel(f"{y_col} mm")
        ax.axhline(0, color="0.85", linewidth=0.7)
        ax.axvline(0, color="0.85", linewidth=0.7)
    cbar = fig.colorbar(scatter, ax=axes, ticks=ordered_contexts, shrink=0.82)
    cbar.set_label("preferred previous GPT tokens")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def contrast_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -0.01, 0.01
    vmax = float(np.nanpercentile(np.abs(finite), 98))
    vmax = max(vmax, 0.01)
    return -vmax, vmax


def plot_contrast_brain(table: pd.DataFrame, output_path: Path, node_size: float, display_mode: str) -> None:
    plot_table = table.dropna(subset=["long_minus_short_r", "x", "y", "z"]).copy()
    if plot_table.empty:
        return
    values = plot_table["long_minus_short_r"].to_numpy(dtype=float)
    coords = plot_table[["x", "y", "z"]].to_numpy(dtype=float)
    vmin, vmax = contrast_limits(values)
    order = np.argsort(values)
    display = plot_markers(
        values[order],
        coords[order],
        node_size=node_size,
        display_mode=display_mode,
        node_cmap="coolwarm",
        node_vmin=vmin,
        node_vmax=vmax,
        colorbar=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display.savefig(output_path, dpi=250)
    display.close()


def plot_contrast_projection(table: pd.DataFrame, output_path: Path) -> None:
    plot_table = table.dropna(subset=["long_minus_short_r", "x", "y", "z"]).copy()
    if plot_table.empty:
        return
    values = plot_table["long_minus_short_r"].to_numpy(dtype=float)
    coords = plot_table[["x", "y", "z"]]
    vmin, vmax = contrast_limits(values)
    pairs = [("x", "y"), ("x", "z"), ("y", "z")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (x_col, y_col) in zip(axes, pairs):
        scatter = ax.scatter(
            coords[x_col],
            coords[y_col],
            c=values,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            s=42,
            edgecolor="black",
            linewidth=0.25,
            alpha=0.9,
        )
        ax.set_xlabel(f"{x_col} mm")
        ax.set_ylabel(f"{y_col} mm")
        ax.axhline(0, color="0.85", linewidth=0.7)
        ax.axvline(0, color="0.85", linewidth=0.7)
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.82)
    cbar.set_label("best long-context r - best short-context r")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_counts(table: pd.DataFrame, output_path: Path) -> None:
    counts = (
        table.groupby(["model", "layer", "preferred_context_tokens"], as_index=False)
        .agg(
            channels=("channel", "count"),
            mean_best_r=("best_r", "mean"),
            median_best_r=("best_r", "median"),
            mean_context_range_r=("context_range_r", "mean"),
        )
        .sort_values(["model", "layer", "preferred_context_tokens"])
    )
    counts.to_csv(output_path, index=False)
    print(counts.to_string(index=False), flush=True)


def write_contrast_summary(table: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for (model, layer), group in table.groupby(["model", "layer"], sort=True):
        delta = group["long_minus_short_r"].dropna()
        rows.append(
            {
                "model": model,
                "layer": int(layer),
                "channels": int(len(delta)),
                "mean_long_minus_short_r": float(delta.mean()),
                "median_long_minus_short_r": float(delta.median()),
                "long_better_channels": int((delta > 0).sum()),
                "long_better_by_0.005": int((delta > 0.005).sum()),
                "long_better_by_0.01": int((delta > 0.01).sum()),
                "short_better_by_0.01": int((delta < -0.01).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False), flush=True)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_channel_metadata(args)
    summary, corrs = load_sweep(args)

    tables = []
    for (model, layer), group in summary.groupby(["model", "layer"], sort=True):
        table = preference_table_for_group(
            metadata=metadata,
            group=group,
            corrs=corrs,
            subject=args.subject,
            args=args,
        )
        if args.min_best_r is not None:
            table = table[table["best_r"] >= args.min_best_r].copy()
        tables.append(table)

        stem = f"sub-{args.subject}_{safe_name(str(model))}_layer-{int(layer)}"
        table.to_csv(args.output_dir / f"{stem}_preferred_context_by_channel.csv", index=False)
        plot_brain(
            table,
            args.output_dir / f"{stem}_preferred_context_brain.png",
            node_size=args.node_size,
            display_mode=args.display_mode,
        )
        plot_projection(table, args.output_dir / f"{stem}_preferred_context_projections.png")
        plot_contrast_brain(
            table,
            args.output_dir / f"{stem}_long_minus_short_context_brain.png",
            node_size=args.node_size,
            display_mode=args.display_mode,
        )
        plot_contrast_projection(table, args.output_dir / f"{stem}_long_minus_short_context_projections.png")

    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(args.output_dir / f"sub-{args.subject}_preferred_context_by_channel.csv", index=False)
    write_counts(combined, args.output_dir / f"sub-{args.subject}_preferred_context_counts.csv")
    write_contrast_summary(combined, args.output_dir / f"sub-{args.subject}_long_minus_short_context_summary.csv")
    print(f"Saved context preference outputs to: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
