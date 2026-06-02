#!/usr/bin/env python3
"""Analyze clean word-decoding outputs and make prediction/boundary heatmaps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    ROI_ORDER,
    bh_fdr,
    boundary_prefix,
    clean_subject,
    crossed_between,
    identification_rank_accuracy,
    sem,
    signflip_pvalues,
    subject_label,
    word_metric_scores_from_npz,
)

ANALYSIS_METRICS = ("point_r", "rank_identification")


COMBINED_PDF_HEATMAP_STEMS = [
    "raw_directional_prediction_heatmap",
    "future_controlled_prediction_delta_heatmap",
    "constituent_boundary_directional_delta_heatmap",
    "constituent_boundary_future_controlled_delta_heatmap",
    "sentence_boundary_directional_delta_heatmap",
    "sentence_boundary_future_controlled_delta_heatmap",
    "event_boundary_directional_delta_heatmap",
    "event_boundary_future_controlled_delta_heatmap",
]

# Legacy default (point_r) filenames without metric suffix.
COMBINED_PDF_HEATMAPS = [f"{stem}.png" for stem in COMBINED_PDF_HEATMAP_STEMS]


def combined_heatmap_filenames(metric_suffix: str) -> list[str]:
    return [f"{stem}_{metric_suffix}.png" for stem in COMBINED_PDF_HEATMAP_STEMS]


def directional_target_columns(lags: list[int]) -> list[str]:
    """Order target columns on a signed lag axis: past left, future right."""
    ordered = sorted({int(lag) for lag in lags if int(lag) > 0})
    return [f"t-{lag}" for lag in reversed(ordered)] + [f"t+{lag}" for lag in ordered]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "prepared")
    parser.add_argument("--decoder-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "decoders")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "analysis")
    parser.add_argument("--subjects", nargs="+", default=[f"{i:02d}" for i in range(1, 10)])
    parser.add_argument("--rois", nargs="+", default=ROI_ORDER)
    parser.add_argument("--lags", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--boundary-levels", nargs="+", default=["sentence", "event", "constituent"])
    parser.add_argument("--min-condition-samples", type=int, default=20)
    return parser.parse_args()


def prediction_path(decoder_dir: Path, subject: str, roi: str, direction: str, lag: int) -> Path:
    return decoder_dir / "predictions" / f"{subject_label(subject)}__{roi}__{direction}__lag-{int(lag):02d}.npz"


def finite_mean(values: np.ndarray, min_samples: int = 1) -> tuple[float, int]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if int(finite.sum()) < int(min_samples):
        return np.nan, int(finite.sum())
    return float(np.nanmean(values[finite])), int(finite.sum())


def plot_heatmap(
    table: pd.DataFrame,
    *,
    value_col: str,
    p_col: str,
    q_col: str,
    row_col: str,
    col_col: str,
    rows: list[str],
    cols: list[int],
    title: str,
    cbar_label: str,
    output_path: Path,
) -> None:
    values = table.pivot(index=row_col, columns=col_col, values=value_col).reindex(index=rows, columns=cols)
    pvals = table.pivot(index=row_col, columns=col_col, values=p_col).reindex(index=rows, columns=cols)
    qvals = table.pivot(index=row_col, columns=col_col, values=q_col).reindex(index=rows, columns=cols)

    annot = values.copy().astype(object)
    for row in rows:
        for col in cols:
            val = values.loc[row, col]
            if not np.isfinite(val):
                annot.loc[row, col] = ""
                continue
            q = qvals.loc[row, col]
            p = pvals.loc[row, col]
            if np.isfinite(q) and q < 0.05:
                star = "**"
            elif np.isfinite(p) and p < 0.05:
                star = "*"
            else:
                star = ""
            annot.loc[row, col] = f"{val:.3f}{star}"

    vmax = float(np.nanpercentile(np.abs(values.to_numpy(dtype=float)), 95)) if np.isfinite(values.to_numpy(dtype=float)).any() else 0.02
    vmax = max(vmax, 0.02)
    fig, ax = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    sns.heatmap(
        values,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=annot,
        fmt="",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_xlabel("N words")
    ax.set_ylabel("ROI")
    ax.set_title(title)
    ax.tick_params(axis="both", length=0)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def raw_directional_prediction(summary: pd.DataFrame, rois: list[str], lags: list[int], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject = summary[summary["direction"].isin(["past", "future"])].copy()
    subject["signed_target_word"] = np.where(subject["direction"] == "future", subject["lag"], -subject["lag"])
    subject["target_label"] = np.where(
        subject["direction"] == "future",
        "t+" + subject["lag"].astype(str),
        "t-" + subject["lag"].astype(str),
    )
    subject.to_csv(output_dir / "raw_directional_prediction_subject_scores.csv", index=False)

    rows = []
    for roi in rois:
        for direction in ("past", "future"):
            for lag in lags:
                cell = subject[(subject["roi"] == roi) & (subject["direction"] == direction) & (subject["lag"] == lag)]
                vals = cell["mean_point_r"].to_numpy(dtype=float)
                observed, p_one, p_two = signflip_pvalues(vals)
                rows.append(
                    {
                        "roi": roi,
                        "direction": direction,
                        "lag": int(lag),
                        "signed_target_word": int(lag if direction == "future" else -lag),
                        "target_label": f"t+{lag}" if direction == "future" else f"t-{lag}",
                        "mean_point_r": observed,
                        "sem_point_r": sem(pd.Series(vals)),
                        "p_one_sided_gt_zero": p_one,
                        "p_two_sided": p_two,
                        "n_subjects": int(np.isfinite(vals).sum()),
                    }
                )
    group = pd.DataFrame(rows)
    group["q_one_sided_bh"] = bh_fdr(group["p_one_sided_gt_zero"].to_numpy(dtype=float))
    group["q_two_sided_bh"] = bh_fdr(group["p_two_sided"].to_numpy(dtype=float))
    group.to_csv(output_dir / "raw_directional_prediction_group_scores.csv", index=False)
    # Keep the per-word point_r plot as the default raw directional heatmap,
    # because it matches the pointwise metric used in boundary-condition splits.
    plot_raw_directional_prediction_heatmap(
        group,
        rois,
        lags,
        "mean_point_r",
        "mean held-out point_r",
        "Raw word decoding by target lag",
        output_dir / "raw_directional_prediction_heatmap.png",
    )
    pc_group = raw_directional_prediction_metric(
        summary,
        rois,
        lags,
        metric="weighted_pc_r",
        output_dir=output_dir,
        file_stem="raw_directional_prediction_weighted_pc",
    )
    plot_raw_directional_prediction_heatmap(
        pc_group,
        rois,
        lags,
        "weighted_pc_r",
        "weighted component-wise r",
        "Raw word decoding by target lag, component-wise score",
        output_dir / "raw_directional_prediction_weighted_pc_heatmap.png",
    )
    return subject, group


def raw_directional_prediction_metric(
    summary: pd.DataFrame,
    rois: list[str],
    lags: list[int],
    *,
    metric: str,
    output_dir: Path,
    file_stem: str,
) -> pd.DataFrame:
    rows = []
    for roi in rois:
        for direction in ("past", "future"):
            for lag in lags:
                cell = summary[(summary["roi"] == roi) & (summary["direction"] == direction) & (summary["lag"] == lag)]
                vals = cell[metric].to_numpy(dtype=float)
                observed, p_one, p_two = signflip_pvalues(vals)
                rows.append(
                    {
                        "roi": roi,
                        "direction": direction,
                        "lag": int(lag),
                        "signed_target_word": int(lag if direction == "future" else -lag),
                        "target_label": f"t+{lag}" if direction == "future" else f"t-{lag}",
                        metric: observed,
                        f"sem_{metric}": sem(pd.Series(vals)),
                        "p_one_sided_gt_zero": p_one,
                        "p_two_sided": p_two,
                        "n_subjects": int(np.isfinite(vals).sum()),
                    }
                )
    group = pd.DataFrame(rows)
    group["q_one_sided_bh"] = bh_fdr(group["p_one_sided_gt_zero"].to_numpy(dtype=float))
    group["q_two_sided_bh"] = bh_fdr(group["p_two_sided"].to_numpy(dtype=float))
    group.to_csv(output_dir / f"{file_stem}_group_scores.csv", index=False)
    return group


def plot_raw_directional_prediction_heatmap(
    group: pd.DataFrame,
    rois: list[str],
    lags: list[int],
    value_col: str,
    cbar_label: str,
    title: str,
    output_path: Path,
    *,
    center: float = 0.0,
) -> None:
    columns = directional_target_columns(lags)
    values = group.pivot(index="roi", columns="target_label", values=value_col).reindex(index=rois, columns=columns)
    pvals = group.pivot(index="roi", columns="target_label", values="p_one_sided_gt_zero").reindex(index=rois, columns=columns)
    qvals = group.pivot(index="roi", columns="target_label", values="q_one_sided_bh").reindex(index=rois, columns=columns)

    annot = values.copy().astype(object)
    for roi in rois:
        for column in columns:
            val = values.loc[roi, column]
            if not np.isfinite(val):
                annot.loc[roi, column] = ""
                continue
            q = qvals.loc[roi, column]
            p = pvals.loc[roi, column]
            if np.isfinite(q) and q < 0.05:
                star = "**"
            elif np.isfinite(p) and p < 0.05:
                star = "*"
            else:
                star = ""
            annot.loc[roi, column] = f"{val:.3f}{star}"

    arr = values.to_numpy(dtype=float)
    devs = np.abs(arr - center)
    half = float(np.nanpercentile(devs, 95)) if np.isfinite(devs).any() else (0.02 if center == 0 else 0.05)
    half = max(half, 0.02 if center == 0 else 0.05)
    fig, ax = plt.subplots(figsize=(16, 5.2), constrained_layout=True)
    sns.heatmap(
        values,
        cmap="vlag",
        center=center,
        vmin=center - half,
        vmax=center + half,
        annot=annot,
        fmt="",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.axvline(len(lags), color="black", linewidth=1.2)
    ax.text(len(lags) / 2, -0.35, "past targets", ha="center", va="center", fontsize=11)
    ax.text(len(lags) + len(lags) / 2, -0.35, "future targets", ha="center", va="center", fontsize=11)
    ax.set_xlabel("Target word relative to neural activity at word t")
    ax.set_ylabel("ROI")
    ax.set_title(title)
    ax.tick_params(axis="both", length=0)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def future_controlled_prediction(summary: pd.DataFrame, rois: list[str], lags: list[int], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = summary.pivot_table(
        index=["subject", "roi", "lag"],
        columns="direction",
        values="mean_point_r",
        aggfunc="first",
    ).reset_index()
    wide["future_controlled_delta"] = wide["past"] - wide["future"]
    wide.to_csv(output_dir / "future_controlled_prediction_subject_scores.csv", index=False)

    rows = []
    for roi in rois:
        for lag in lags:
            vals = wide[(wide["roi"] == roi) & (wide["lag"] == lag)]["future_controlled_delta"].to_numpy(dtype=float)
            observed, p_one, p_two = signflip_pvalues(vals)
            rows.append(
                {
                    "roi": roi,
                    "lag": int(lag),
                    "mean_delta": observed,
                    "sem_delta": sem(pd.Series(vals)),
                    "p_one_sided": p_one,
                    "p_two_sided": p_two,
                    "n_subjects": int(np.isfinite(vals).sum()),
                }
            )
    group = pd.DataFrame(rows)
    group["q_one_sided_bh"] = bh_fdr(group["p_one_sided"].to_numpy(dtype=float))
    group["q_two_sided_bh"] = bh_fdr(group["p_two_sided"].to_numpy(dtype=float))
    group.to_csv(output_dir / "future_controlled_prediction_group_scores.csv", index=False)
    plot_heatmap(
        group,
        value_col="mean_delta",
        p_col="p_one_sided",
        q_col="q_one_sided_bh",
        row_col="roi",
        col_col="lag",
        rows=rois,
        cols=lags,
        title="Future-controlled past-word prediction",
        cbar_label="mean point_r(past t-N) - mean point_r(future t+N)",
        output_path=output_dir / "future_controlled_prediction_delta_heatmap.png",
    )
    return wide, group


def condition_delta(point_r: np.ndarray, crossed: np.ndarray, min_samples: int) -> dict[str, float | int]:
    within = ~crossed
    across = crossed
    within_mean, n_within = finite_mean(point_r[within], min_samples)
    across_mean, n_across = finite_mean(point_r[across], min_samples)
    delta = within_mean - across_mean if np.isfinite(within_mean) and np.isfinite(across_mean) else np.nan
    return {
        "within_mean_point_r": within_mean,
        "across_mean_point_r": across_mean,
        "delta_within_minus_across": delta,
        "n_within": n_within,
        "n_across": n_across,
    }


def boundary_condition_scores(
    args: argparse.Namespace,
    rois: list[str],
    lags: list[int],
    subjects: list[str],
    boundary_level: str,
    metric_suffix: str = "point_r",
) -> pd.DataFrame:
    words = pd.read_csv(args.prepared_dir / "words.csv")
    column = f"{boundary_level}_boundary_after"
    if column not in words.columns:
        print(f"Skipping boundary level {boundary_level!r}: {column} not found in words.csv", flush=True)
        return pd.DataFrame()
    boundary_after = words[column].astype(bool).to_numpy()
    prefix = boundary_prefix(boundary_after)
    rows = []
    for subject in subjects:
        for roi in rois:
            for lag in lags:
                for direction in ("past", "future"):
                    path = prediction_path(args.decoder_dir, subject, roi, direction, lag)
                    if not path.is_file():
                        continue
                    npz = np.load(path)
                    scores = word_metric_scores_from_npz(npz, metric_suffix)
                    current = npz["current_word_idx"].astype(np.int32)
                    target = npz["target_word_idx"].astype(np.int32)
                    crossed = crossed_between(prefix, current, target)
                    row = {
                        "level": boundary_level,
                        "subject": subject_label(subject),
                        "roi": roi,
                        "lag": int(lag),
                        "direction": direction,
                        "metric": metric_suffix,
                    }
                    row.update(condition_delta(scores, crossed, args.min_condition_samples))
                    rows.append(row)
    return pd.DataFrame(rows)


def future_controlled_boundary(
    condition: pd.DataFrame, rois: list[str], lags: list[int], output_dir: Path, level: str, metric_suffix: str = "point_r"
) -> pd.DataFrame:
    wide = condition.pivot_table(
        index=["level", "subject", "roi", "lag"],
        columns="direction",
        values="delta_within_minus_across",
        aggfunc="first",
    ).reset_index()
    wide["future_controlled_boundary_delta"] = wide["past"] - wide["future"]
    wide.to_csv(output_dir / f"{level}_boundary_future_controlled_subject_scores.csv", index=False)

    rows = []
    for roi in rois:
        for lag in lags:
            vals = wide[(wide["roi"] == roi) & (wide["lag"] == lag)]["future_controlled_boundary_delta"].to_numpy(dtype=float)
            observed, p_one, p_two = signflip_pvalues(vals)
            rows.append(
                {
                    "level": level,
                    "roi": roi,
                    "lag": int(lag),
                    "mean_delta": observed,
                    "sem_delta": sem(pd.Series(vals)),
                    "p_one_sided": p_one,
                    "p_two_sided": p_two,
                    "n_subjects": int(np.isfinite(vals).sum()),
                }
            )
    group = pd.DataFrame(rows)
    group["q_one_sided_bh"] = bh_fdr(group["p_one_sided"].to_numpy(dtype=float))
    group["q_two_sided_bh"] = bh_fdr(group["p_two_sided"].to_numpy(dtype=float))
    group.to_csv(output_dir / f"{level}_boundary_future_controlled_group_scores_{metric_suffix}.csv", index=False)
    plot_heatmap(
        group,
        value_col="mean_delta",
        p_col="p_one_sided",
        q_col="q_one_sided_bh",
        row_col="roi",
        col_col="lag",
        rows=rois,
        cols=lags,
        title=f"{level.title()} boundary effect, future controlled ({metric_suffix})",
        cbar_label="(past within-across) - (future within-across)",
        output_path=output_dir / f"{level}_boundary_future_controlled_delta_heatmap_{metric_suffix}.png",
    )
    return group


def directional_boundary(
    condition: pd.DataFrame, rois: list[str], lags: list[int], output_dir: Path, level: str, metric_suffix: str = "point_r"
) -> pd.DataFrame:
    subject = condition[condition["direction"].isin(["past", "future"])].copy()
    subject["signed_target_word"] = np.where(subject["direction"] == "future", subject["lag"], -subject["lag"])
    subject["target_label"] = np.where(
        subject["direction"] == "future",
        "t+" + subject["lag"].astype(str),
        "t-" + subject["lag"].astype(str),
    )
    subject.to_csv(output_dir / f"{level}_boundary_directional_subject_scores.csv", index=False)

    rows = []
    for roi in rois:
        for direction in ("past", "future"):
            for lag in lags:
                cell = subject[(subject["roi"] == roi) & (subject["direction"] == direction) & (subject["lag"] == lag)]
                vals = cell["delta_within_minus_across"].to_numpy(dtype=float)
                observed, p_one, p_two = signflip_pvalues(vals)
                rows.append(
                    {
                        "level": level,
                        "roi": roi,
                        "direction": direction,
                        "lag": int(lag),
                        "signed_target_word": int(lag if direction == "future" else -lag),
                        "target_label": f"t+{lag}" if direction == "future" else f"t-{lag}",
                        "mean_delta": observed,
                        "sem_delta": sem(pd.Series(vals)),
                        "p_one_sided": p_one,
                        "p_two_sided": p_two,
                        "n_subjects": int(np.isfinite(vals).sum()),
                    }
                )
    group = pd.DataFrame(rows)
    group["q_one_sided_bh"] = bh_fdr(group["p_one_sided"].to_numpy(dtype=float))
    group["q_two_sided_bh"] = bh_fdr(group["p_two_sided"].to_numpy(dtype=float))
    group.to_csv(output_dir / f"{level}_boundary_directional_group_scores_{metric_suffix}.csv", index=False)
    plot_directional_boundary_heatmap(
        group,
        rois,
        lags,
        level,
        output_dir / f"{level}_boundary_directional_delta_heatmap_{metric_suffix}.png",
        metric_suffix=metric_suffix,
    )
    return group


def plot_directional_boundary_heatmap(
    group: pd.DataFrame,
    rois: list[str],
    lags: list[int],
    level: str,
    output_path: Path,
    metric_suffix: str = "point_r",
) -> None:
    columns = directional_target_columns(lags)
    values = group.pivot(index="roi", columns="target_label", values="mean_delta").reindex(index=rois, columns=columns)
    pvals = group.pivot(index="roi", columns="target_label", values="p_one_sided").reindex(index=rois, columns=columns)
    qvals = group.pivot(index="roi", columns="target_label", values="q_one_sided_bh").reindex(index=rois, columns=columns)

    annot = values.copy().astype(object)
    for roi in rois:
        for column in columns:
            val = values.loc[roi, column]
            if not np.isfinite(val):
                annot.loc[roi, column] = ""
                continue
            q = qvals.loc[roi, column]
            p = pvals.loc[roi, column]
            if np.isfinite(q) and q < 0.05:
                star = "**"
            elif np.isfinite(p) and p < 0.05:
                star = "*"
            else:
                star = ""
            annot.loc[roi, column] = f"{val:.3f}{star}"

    vmax = float(np.nanpercentile(np.abs(values.to_numpy(dtype=float)), 95)) if np.isfinite(values.to_numpy(dtype=float)).any() else 0.02
    vmax = max(vmax, 0.02)
    cbar_label = (
        "within-boundary rank accuracy - across-boundary rank accuracy"
        if metric_suffix == "rank_identification"
        else "within-boundary mean point_r - across-boundary mean point_r"
    )
    fig, ax = plt.subplots(figsize=(16, 5.2), constrained_layout=True)
    sns.heatmap(
        values,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=annot,
        fmt="",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.axvline(len(lags), color="black", linewidth=1.2)
    ax.text(len(lags) / 2, -0.35, "past targets", ha="center", va="center", fontsize=11)
    ax.text(len(lags) + len(lags) / 2, -0.35, "future targets", ha="center", va="center", fontsize=11)
    ax.set_xlabel("Target word relative to neural activity at word t")
    ax.set_ylabel("ROI")
    ax.set_title(f"{level.title()} boundary effect by target lag")
    ax.tick_params(axis="both", length=0)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _group_test(values: np.ndarray, null: float = 0.0) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    observed = float(np.mean(values))
    if null == 0.0:
        _, p_one, p_two = signflip_pvalues(values)
    else:
        _, p_one, p_two = signflip_pvalues(values - null)
    return observed, p_one, p_two


def raw_directional_from_npz(
    args: argparse.Namespace,
    subjects: list[str],
    rois: list[str],
    lags: list[int],
    output_dir: Path,
    metric_suffix: str,
) -> pd.DataFrame:
    null = 0.5 if metric_suffix == "rank_identification" else 0.0
    value_col = "mean_point_r" if metric_suffix == "point_r" else "mean_rank_accuracy"
    rows = []
    for roi in rois:
        for direction in ("past", "future"):
            for lag in lags:
                vals = []
                for subject in subjects:
                    path = prediction_path(args.decoder_dir, subject, roi, direction, lag)
                    if not path.is_file():
                        continue
                    npz = np.load(path)
                    if metric_suffix == "point_r":
                        vals.append(float(np.nanmean(npz["point_r"])))
                    else:
                        vals.append(identification_rank_accuracy(npz["y_true"], npz["y_pred"]))
                vals = np.asarray(vals, dtype=float)
                observed, p_one, p_two = _group_test(vals, null=null)
                rows.append(
                    {
                        "roi": roi,
                        "direction": direction,
                        "lag": int(lag),
                        "target_label": f"t+{lag}" if direction == "future" else f"t-{lag}",
                        value_col: observed,
                        "p_one_sided_gt_zero": p_one,
                        "p_two_sided": p_two,
                        "n_subjects": int(np.isfinite(vals).sum()),
                    }
                )
    group = pd.DataFrame(rows)
    group["q_one_sided_bh"] = bh_fdr(group["p_one_sided_gt_zero"].to_numpy(dtype=float))
    group.to_csv(output_dir / f"raw_directional_prediction_group_scores_{metric_suffix}.csv", index=False)
    center = 0.5 if metric_suffix == "rank_identification" else 0.0
    cbar = (
        "rank-identification accuracy (chance 0.5)"
        if metric_suffix == "rank_identification"
        else "mean held-out point_r"
    )
    plot_raw_directional_prediction_heatmap(
        group,
        rois,
        lags,
        value_col,
        cbar,
        f"Raw word decoding by target lag ({metric_suffix})",
        output_dir / f"raw_directional_prediction_heatmap_{metric_suffix}.png",
        center=center,
    )
    return group


def future_controlled_from_npz(
    args: argparse.Namespace,
    subjects: list[str],
    rois: list[str],
    lags: list[int],
    output_dir: Path,
    metric_suffix: str,
) -> pd.DataFrame:
    rows = []
    for roi in rois:
        for lag in lags:
            deltas = []
            for subject in subjects:
                past_p = prediction_path(args.decoder_dir, subject, roi, "past", lag)
                fut_p = prediction_path(args.decoder_dir, subject, roi, "future", lag)
                if not past_p.is_file() or not fut_p.is_file():
                    continue
                past = np.load(past_p)
                future = np.load(fut_p)
                if metric_suffix == "point_r":
                    v_p = float(np.nanmean(past["point_r"]))
                    v_f = float(np.nanmean(future["point_r"]))
                else:
                    v_p = identification_rank_accuracy(past["y_true"], past["y_pred"])
                    v_f = identification_rank_accuracy(future["y_true"], future["y_pred"])
                deltas.append(v_p - v_f)
            deltas = np.asarray(deltas, dtype=float)
            observed, p_one, p_two = _group_test(deltas, null=0.0)
            rows.append(
                {
                    "roi": roi,
                    "lag": int(lag),
                    "mean_delta": observed,
                    "sem_delta": sem(pd.Series(deltas)),
                    "p_one_sided": p_one,
                    "p_two_sided": p_two,
                    "n_subjects": int(np.isfinite(deltas).sum()),
                }
            )
    group = pd.DataFrame(rows)
    group["q_one_sided_bh"] = bh_fdr(group["p_one_sided"].to_numpy(dtype=float))
    ylab = (
        "mean rank_acc(past) - mean rank_acc(future)"
        if metric_suffix == "rank_identification"
        else "mean point_r(past t-N) - mean point_r(future t+N)"
    )
    group.to_csv(output_dir / f"future_controlled_prediction_group_scores_{metric_suffix}.csv", index=False)
    plot_heatmap(
        group,
        value_col="mean_delta",
        p_col="p_one_sided",
        q_col="q_one_sided_bh",
        row_col="roi",
        col_col="lag",
        rows=rois,
        cols=lags,
        title=f"Future-controlled past-word prediction ({metric_suffix})",
        cbar_label=ylab,
        output_path=output_dir / f"future_controlled_prediction_delta_heatmap_{metric_suffix}.png",
    )
    return group


def run_analysis_for_metric(
    args: argparse.Namespace,
    subjects: list[str],
    rois: list[str],
    lags: list[int],
    metric_suffix: str,
) -> None:
    print(f"Analysis metric: {metric_suffix}", flush=True)
    raw_directional_from_npz(args, subjects, rois, lags, args.output_dir, metric_suffix)
    future_controlled_from_npz(args, subjects, rois, lags, args.output_dir, metric_suffix)
    all_condition_rows = []
    for level in args.boundary_levels:
        condition = boundary_condition_scores(args, rois, lags, subjects, level, metric_suffix)
        if condition.empty:
            continue
        condition.to_csv(args.output_dir / f"{level}_boundary_condition_scores_{metric_suffix}.csv", index=False)
        all_condition_rows.append(condition)
        directional_boundary(condition, rois, lags, args.output_dir, level, metric_suffix)
        future_controlled_boundary(condition, rois, lags, args.output_dir, level, metric_suffix)
    if all_condition_rows:
        pd.concat(all_condition_rows, ignore_index=True).to_csv(
            args.output_dir / f"boundary_condition_scores_{metric_suffix}.csv", index=False
        )
    write_combined_heatmap_pdf(args.output_dir, metric_suffix=metric_suffix)


def write_combined_heatmap_pdf(
    output_dir: Path,
    metric_suffix: str = "point_r",
    filenames: list[str] | None = None,
) -> Path:
    if filenames is None:
        filenames = combined_heatmap_filenames(metric_suffix)
    pdf_path = output_dir / f"combined_decoding_heatmaps_{metric_suffix}.pdf"
    missing = [name for name in filenames if not (output_dir / name).is_file()]
    if missing:
        print(f"Skipping missing heatmaps in combined PDF: {missing}", flush=True)

    with PdfPages(pdf_path) as pdf:
        for name in filenames:
            image_path = output_dir / name
            if not image_path.is_file():
                continue
            image = plt.imread(image_path)
            height, width = image.shape[:2]
            dpi = 240
            fig = plt.figure(figsize=(width / dpi, height / dpi))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(image)
            ax.set_axis_off()
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)
    print(f"Combined heatmap PDF written to: {pdf_path}", flush=True)
    return pdf_path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = [clean_subject(s) for s in args.subjects]
    rois = [roi for roi in ROI_ORDER if roi in set(args.rois)]
    lags = sorted(set(int(lag) for lag in args.lags if int(lag) > 0))

    summary_path = args.decoder_dir / "decoder_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    decoder_summary = pd.read_csv(summary_path)
    for metric_suffix in ANALYSIS_METRICS:
        run_analysis_for_metric(args, subjects, rois, lags, metric_suffix)
    print(f"Analysis outputs written to: {args.output_dir}", flush=True)
    for metric_suffix in ANALYSIS_METRICS:
        print(f"Combined PDF ({metric_suffix}): {args.output_dir / f'combined_decoding_heatmaps_{metric_suffix}.pdf'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
