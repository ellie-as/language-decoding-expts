#!/usr/bin/env python3
"""Plot preferred encoding lag per ECoG channel from saved encoding results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.plotting import plot_markers


DEFAULT_CEPH_ROOT = Path("/Volumes/ellie/language-decoding-expts")


def parse_args() -> argparse.Namespace:
    local_output = Path(__file__).resolve().parent / "outputs" / "preferred_lag"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-npz",
        type=Path,
        default=None,
        help="Encoding result NPZ. If omitted, search under --ceph-root.",
    )
    parser.add_argument(
        "--ceph-root",
        type=Path,
        default=DEFAULT_CEPH_ROOT,
        help="Mounted Ceph clone of language-decoding-expts.",
    )
    parser.add_argument("--output-dir", type=Path, default=local_output)
    parser.add_argument(
        "--metric",
        choices=("corr", "abs-corr"),
        default="corr",
        help="Choose preferred lag by max correlation or max absolute correlation.",
    )
    parser.add_argument("--cmap", default="coolwarm")
    parser.add_argument("--node-size", type=float, default=35.0)
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument(
        "--display-mode",
        default="lzr",
        help="Nilearn display mode, e.g. lzr, x, y, z, ortho.",
    )
    return parser.parse_args()


def find_results_npz(ceph_root: Path) -> Path:
    search_root = ceph_root / "podcast_ecog"
    candidates = sorted(
        search_root.glob("outputs*/**/*encoding_results.npz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No '*encoding_results.npz' files found under {search_root}. "
            "Pass --results-npz explicitly if the result lives elsewhere."
        )
    return candidates[0]


def load_results(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as result:
        corrs = result["corrs"]
        lags = result["lags"]
        coords = result["coords"]
        channel_names = result["channel_names"].astype(str)

    if corrs.ndim != 3:
        raise ValueError(f"Expected corrs with shape (folds, channels, lags), got {corrs.shape}.")
    if len(lags) != corrs.shape[-1]:
        raise ValueError(f"lags length {len(lags)} does not match corrs lag dimension {corrs.shape[-1]}.")
    if len(channel_names) != corrs.shape[1]:
        raise ValueError(
            f"channel_names length {len(channel_names)} does not match corrs channel dimension {corrs.shape[1]}."
        )
    if coords.shape != (corrs.shape[1], 3):
        raise ValueError(f"coords shape {coords.shape} does not match expected {(corrs.shape[1], 3)}.")

    # Nilearn expects MNI coordinates in mm. Older outputs may store meters.
    if np.nanmax(np.abs(coords)) < 1:
        coords = coords * 1000.0
    return corrs, lags, coords, channel_names


def compute_preferred_lag(corrs: np.ndarray, lags: np.ndarray, metric: str) -> pd.DataFrame:
    mean_corr = corrs.mean(axis=0)
    score = np.abs(mean_corr) if metric == "abs-corr" else mean_corr
    preferred_idx = score.argmax(axis=-1)
    preferred_lag = lags[preferred_idx]
    preferred_corr = mean_corr[np.arange(mean_corr.shape[0]), preferred_idx]
    max_corr = mean_corr.max(axis=-1)

    return pd.DataFrame(
        {
            "preferred_lag_s": preferred_lag,
            "preferred_lag_index": preferred_idx,
            "corr_at_preferred_lag": preferred_corr,
            "max_corr": max_corr,
        }
    )


def save_plot(
    values: np.ndarray,
    coords: np.ndarray,
    output_path: Path,
    cmap: str,
    node_size: float,
    display_mode: str,
    vmin: float | None,
    vmax: float | None,
) -> None:
    if vmin is None or vmax is None:
        max_abs = float(np.nanmax(np.abs(values)))
        vmin = -max_abs if vmin is None else vmin
        vmax = max_abs if vmax is None else vmax

    order = np.argsort(values)
    display = plot_markers(
        values[order],
        coords[order],
        node_size=node_size,
        display_mode=display_mode,
        node_vmin=vmin,
        node_vmax=vmax,
        node_cmap=cmap,
        colorbar=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display.savefig(output_path, dpi=250)
    display.close()


def main() -> int:
    args = parse_args()
    results_path = args.results_npz or find_results_npz(args.ceph_root)
    results_path = results_path.expanduser().resolve()
    print(f"Reading encoding results from: {results_path}", flush=True)

    corrs, lags, coords, channel_names = load_results(results_path)
    preferred = compute_preferred_lag(corrs, lags, args.metric)
    preferred.insert(0, "channel", channel_names)
    preferred[["x", "y", "z"]] = coords

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = results_path.stem.replace("_encoding_results", "")
    metric_suffix = "abs_corr" if args.metric == "abs-corr" else "corr"

    csv_path = args.output_dir / f"{stem}_preferred_lag_{metric_suffix}.csv"
    preferred.to_csv(csv_path, index=False)
    print(f"Saved preferred lag table: {csv_path}", flush=True)

    png_path = args.output_dir / f"{stem}_preferred_lag_{metric_suffix}.png"
    save_plot(
        preferred["preferred_lag_s"].to_numpy(),
        coords,
        png_path,
        cmap=args.cmap,
        node_size=args.node_size,
        display_mode=args.display_mode,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    print(f"Saved preferred lag plot: {png_path}", flush=True)

    summary = preferred["preferred_lag_s"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    print(summary.to_string(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
