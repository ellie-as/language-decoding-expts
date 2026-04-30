from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import resample
from tqdm import tqdm

from .data import RunSpec, load_meg_array


def expected_n_times(config: dict[str, Any]) -> int:
    win = config["windows"]
    duration = float(win["meg_tmax_sec"]) - float(win["meg_tmin_sec"])
    return int(round(duration * float(config["features"]["meg_downsample_hz"])))


def extract_meg_window(meg: dict[str, Any], t: float, config: dict[str, Any]) -> np.ndarray | None:
    sfreq = float(meg["sfreq"])
    data = meg["data"]
    win = config["windows"]
    start = int(round((t + float(win["meg_tmin_sec"])) * sfreq))
    stop = int(round((t + float(win["meg_tmax_sec"])) * sfreq))
    if start < 0 or stop > data.shape[1] or stop <= start:
        return None
    window = data[:, start:stop].astype(np.float32, copy=False)
    target_sfreq = float(config["features"]["meg_downsample_hz"])
    target_n = expected_n_times(config)
    if abs(target_sfreq - sfreq) > 1e-6 or window.shape[1] != target_n:
        window = resample(window, target_n, axis=1).astype(np.float32)
    if config["features"].get("standardize_channels", False):
        mean = window.mean(axis=1, keepdims=True)
        std = window.std(axis=1, keepdims=True)
        window = (window - mean) / np.maximum(std, 1e-6)
    return window.astype(np.float32)


def build_meg_feature_memmap(
    windows: pd.DataFrame,
    runs: list[RunSpec],
    config: dict[str, Any],
    output_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_lookup = {r.group_id: r for r in runs}
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first_run = run_lookup[windows["run_group"].iloc[0]]
    first_meg = load_meg_array(first_run, config)
    n_channels = int(first_meg["data"].shape[0])
    n_times = expected_n_times(config)
    valid_rows = []
    windows_by_run = list(windows.groupby("run_group", sort=False))
    for run_group, sub in windows_by_run:
        meg = first_meg if run_group == first_run.group_id else load_meg_array(run_lookup[run_group], config)
        if int(meg["data"].shape[0]) != n_channels:
            raise ValueError(
                f"Run {run_group} has {meg['data'].shape[0]} channels, expected {n_channels}. "
                "Check that events and MEG files were paired correctly and that all runs use the same channel set."
            )
        for _, row in sub.iterrows():
            if extract_meg_window(meg, float(row["t"]), config) is not None:
                valid_rows.append(row)
    metadata = pd.DataFrame(valid_rows).reset_index(drop=True)
    arr = np.memmap(out_path, mode="w+", dtype="float32", shape=(len(metadata), n_channels, n_times))

    row_offset = 0
    for run_group, sub in tqdm(list(metadata.groupby("run_group", sort=False)), desc="MEG windows"):
        meg = first_meg if run_group == first_run.group_id else load_meg_array(run_lookup[run_group], config)
        for _, row in sub.iterrows():
            window = extract_meg_window(meg, float(row["t"]), config)
            if window is None:
                raise RuntimeError("Internal error: window became invalid during second pass")
            if window.shape != (n_channels, n_times):
                raise ValueError(f"Run {run_group} produced window shape {window.shape}, expected {(n_channels, n_times)}")
            arr[row_offset] = window
            row_offset += 1
    arr.flush()
    info = {
        "path": str(out_path),
        "n_examples": int(len(metadata)),
        "n_channels": n_channels,
        "n_timepoints": n_times,
        "raw_sampling_hz": float(first_meg["sfreq"]),
        "downsample_hz": float(config["features"]["meg_downsample_hz"]),
        "tmin_sec": float(config["windows"]["meg_tmin_sec"]),
        "tmax_sec": float(config["windows"]["meg_tmax_sec"]),
        "dtype": "float32",
        "shape": [int(len(metadata)), n_channels, n_times],
    }
    return metadata, info


def load_meg_memmap(path: str | Path, info: dict[str, Any], mode: str = "r") -> np.memmap:
    return np.memmap(path, dtype=info.get("dtype", "float32"), mode=mode, shape=tuple(info["shape"]))
