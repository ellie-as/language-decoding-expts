#!/usr/bin/env python3
"""Nilearn glass-brain marker plots for short-vs-long lag sensor maps.

This follows the style used in the Podcast ECoG tutorial:

    plot_markers(values[order], coords[order], display_mode="lzr", ...)

The tutorial converts MNE channel locations from meters to millimeters before
calling Nilearn. The clean-code sensor tables already store MNI coordinates in
millimeters, so this script uses x/y/z directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from nilearn.plotting import plot_markers

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from common import DEFAULT_OUTPUT_ROOT  # noqa: E402


TARGET_STEMS = ["word_vectors_pca20", "gpt2_ctx32_layer8_pca20"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "super_brain_midpoint" / "lag_sensor_intuition",
    )
    parser.add_argument("--target-stems", nargs="+", default=TARGET_STEMS)
    parser.add_argument("--display-mode", default="lzr", help="Nilearn display_mode, e.g. lzr, ortho, xz, yz.")
    parser.add_argument("--node-size", type=float, default=28.0)
    parser.add_argument("--group-node-size", type=float, default=46.0)
    parser.add_argument("--alpha", type=float, default=0.78)
    parser.add_argument("--percentile", type=float, default=98.0)
    return parser.parse_args()


def finite_table(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    cols = ["x", "y", "z", value_col]
    out = df[np.isfinite(df[cols].to_numpy(dtype=float)).all(axis=1)].copy()
    return out


def coords_values(df: pd.DataFrame, value_col: str, *, abs_values: bool = False) -> tuple[np.ndarray, np.ndarray]:
    vals = df[value_col].to_numpy(dtype=float)
    if abs_values:
        vals = np.abs(vals)
    coords = df[["x", "y", "z"]].to_numpy(dtype=float)
    order = np.argsort(vals)
    return vals[order], coords[order]


def safe_vlim(values: np.ndarray, percentile: float, *, symmetric: bool) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 0.0, 1.0
    if symmetric:
        vmax = float(np.nanpercentile(np.abs(finite), percentile))
        vmax = max(vmax, 1e-6)
        return -vmax, vmax
    vmin = float(np.nanpercentile(finite, 100.0 - percentile))
    vmax = float(np.nanpercentile(finite, percentile))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return vmin, vmax


def marker_plot(
    df: pd.DataFrame,
    value_col: str,
    output_file: Path,
    *,
    title: str,
    cmap: str,
    display_mode: str,
    node_size: float,
    alpha: float,
    percentile: float,
    symmetric: bool,
    abs_values: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    sub = finite_table(df, value_col)
    if sub.empty:
        return
    values, coords = coords_values(sub, value_col, abs_values=abs_values)
    if vmin is None or vmax is None:
        vmin, vmax = safe_vlim(values, percentile, symmetric=symmetric)
    display = plot_markers(
        values,
        coords,
        node_size=node_size,
        display_mode=display_mode,
        node_vmin=vmin,
        node_vmax=vmax,
        node_cmap=cmap,
        alpha=alpha,
        colorbar=True,
        title=title,
        output_file=str(output_file),
    )
    if display is not None:
        display.close()


def add_group_coordinates(groups: pd.DataFrame, sensors: pd.DataFrame) -> pd.DataFrame:
    cols = ["subject", "channel", "x", "y", "z", "destrieux_label", "lag_fraction_center_of_mass"]
    return groups.merge(sensors[cols], on=["subject", "channel"], how="left")


def write_group_summary(groups: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for group_name, group in groups.groupby("sensor_group"):
        rows.append(
            {
                "sensor_group": group_name,
                "n_sensors": int(len(group)),
                "median_x": float(group["x"].median()),
                "median_y": float(group["y"].median()),
                "median_z": float(group["z"].median()),
                "median_long_short_norm_index": float(group["long_short_norm_index"].median()),
                "median_lag_center_of_mass": float(group["lag_fraction_center_of_mass"].median()),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "nilearn_selected_group_coordinate_summary.csv", index=False)


def process_target(args: argparse.Namespace, target_stem: str) -> None:
    target_dir = args.input_dir / target_stem
    output_dir = target_dir / "nilearn_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    sensors = pd.read_csv(target_dir / "sensor_lag_indices.csv")
    marker_plot(
        sensors,
        "long_short_norm_index",
        output_dir / "all_sensors_long_short_index_lzr.png",
        title=f"{target_stem}: long vs short lag preference",
        cmap="coolwarm",
        display_mode=args.display_mode,
        node_size=args.node_size,
        alpha=args.alpha,
        percentile=args.percentile,
        symmetric=True,
    )
    marker_plot(
        sensors,
        "lag_fraction_center_of_mass",
        output_dir / "all_sensors_lag_center_of_mass_lzr.png",
        title=f"{target_stem}: lag center of mass",
        cmap="viridis",
        display_mode=args.display_mode,
        node_size=args.node_size,
        alpha=args.alpha,
        percentile=args.percentile,
        symmetric=False,
    )

    groups = add_group_coordinates(pd.read_csv(target_dir / "sensor_group_membership.csv"), sensors)
    write_group_summary(groups, output_dir)
    short = groups[groups["sensor_group"] == "short_pref"].copy()
    long = groups[groups["sensor_group"] == "long_pref"].copy()
    marker_plot(
        short,
        "long_short_norm_index",
        output_dir / "selected_short_pref_sensors_lzr.png",
        title=f"{target_stem}: selected short-preferring sensors",
        cmap="Blues",
        display_mode=args.display_mode,
        node_size=args.group_node_size,
        alpha=0.88,
        percentile=args.percentile,
        symmetric=False,
        abs_values=True,
        vmin=0.0,
        vmax=max(float(np.nanpercentile(np.abs(short["long_short_norm_index"]), args.percentile)), 1e-6),
    )
    marker_plot(
        long,
        "long_short_norm_index",
        output_dir / "selected_long_pref_sensors_lzr.png",
        title=f"{target_stem}: selected long-preferring sensors",
        cmap="Reds",
        display_mode=args.display_mode,
        node_size=args.group_node_size,
        alpha=0.88,
        percentile=args.percentile,
        symmetric=False,
        vmin=0.0,
        vmax=max(float(np.nanpercentile(long["long_short_norm_index"], args.percentile)), 1e-6),
    )

    haufe_path = target_dir / "haufe_pattern_indices.csv"
    if haufe_path.is_file():
        haufe = pd.read_csv(haufe_path)
        for direction in ("past", "future"):
            marker_plot(
                haufe,
                f"haufe_{direction}_long_short_norm_index",
                output_dir / f"haufe_{direction}_long_short_index_lzr.png",
                title=f"{target_stem}: Haufe {direction} long vs short",
                cmap="coolwarm",
                display_mode=args.display_mode,
                node_size=args.node_size,
                alpha=args.alpha,
                percentile=args.percentile,
                symmetric=True,
            )

    print(f"Wrote Nilearn plots for {target_stem}: {output_dir}", flush=True)


def main() -> int:
    args = parse_args()
    for target_stem in args.target_stems:
        process_target(args, target_stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
