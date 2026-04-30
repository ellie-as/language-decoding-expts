from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def create_run_splits(windows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    split_cfg = config["split"]
    rng = np.random.default_rng(int(split_cfg.get("seed", config["project"].get("seed", 0))))
    groups = np.array(sorted(windows["run_group"].unique()))
    rng.shuffle(groups)
    if len(groups) == 1:
        assignment = {groups[0]: "train"}
        # Debug mock data often has one run. Split by time while keeping this explicit in summary.
        q1, q2 = windows["t"].quantile([split_cfg.get("train_fraction", 0.8), split_cfg.get("train_fraction", 0.8) + split_cfg.get("val_fraction", 0.1)])
        out = windows[["example_id", "subject", "session", "run", "run_group", "t"]].copy()
        out["split"] = np.where(out["t"] <= q1, "train", np.where(out["t"] <= q2, "val", "test"))
        out["split_grouping"] = "time_within_single_debug_run"
        return out

    n = len(groups)
    n_train = max(1, int(round(n * float(split_cfg.get("train_fraction", 0.8)))))
    n_val = max(1, int(round(n * float(split_cfg.get("val_fraction", 0.1)))))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    assignment = {}
    for g in groups[:n_train]:
        assignment[g] = "train"
    for g in groups[n_train : n_train + n_val]:
        assignment[g] = "val"
    for g in groups[n_train + n_val :]:
        assignment[g] = "test"
    out = windows[["example_id", "subject", "session", "run", "run_group", "t"]].copy()
    out["split"] = out["run_group"].map(assignment)
    out["split_grouping"] = "run_group"
    return out


def split_summary(split_df: pd.DataFrame) -> dict[str, Any]:
    runs_by_split = split_df.groupby("split")["run_group"].nunique().to_dict()
    examples_by_split = split_df["split"].value_counts().to_dict()
    duration_by_split = {}
    for split, sub in split_df.groupby("split"):
        duration_by_split[split] = float(sub.groupby("run_group")["t"].agg(lambda x: x.max() - x.min()).sum() / 3600.0)
    held_out = {
        split: sorted(sub["run_group"].unique().tolist())
        for split, sub in split_df[split_df["split"].isin(["val", "test"])].groupby("split")
    }
    return {
        "runs_by_split": runs_by_split,
        "examples_by_split": examples_by_split,
        "approx_hours_by_split": duration_by_split,
        "held_out_runs": held_out,
        "split_grouping": split_df["split_grouping"].iloc[0] if len(split_df) else None,
    }
