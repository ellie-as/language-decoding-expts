#!/usr/bin/env python3
"""Pooled super-brain decoder: tune alpha, train, analyse, combined PDF.

Concatenates electrodes across subjects (shared stimulus timeline), tunes ridge
alpha on present-word decoding (inner CV), trains lag/direction/ROI decoders,
then writes the same eight heatmaps + combined PDF as decoding_analysis.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    ROI_ORDER,
    bh_fdr,
    boundary_prefix,
    clean_subject,
    crossed_between,
    row_corr,
    word_metric_scores_from_npz,
    score_predictions,
    subject_label,
    target_indices,
)
from decoding_analysis import (  # noqa: E402
    condition_delta,
    plot_heatmap,
    plot_raw_directional_prediction_heatmap,
    plot_directional_boundary_heatmap,
    write_combined_heatmap_pdf,
)
from train_decoders import (  # noqa: E402
    channel_indices_for_roi,
    cv_predict_ridge,
    load_prepared_subject,
    load_roi_lookup,
    valid_current_indices,
)

ANALYSIS_METRICS = ("point_r", "rank_identification")


def p_gt_null(values: np.ndarray, metric_suffix: str) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return float("nan")
    null = 0.5 if metric_suffix == "rank_identification" else 0.0
    _, p = stats.ttest_1samp(finite, null, alternative="greater")
    return float(p)


def _metric_tag(metric_suffix: str) -> str:
    return metric_suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "prepared_midpoint")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "super_brain_midpoint")
    parser.add_argument("--target-stem", default="word_vectors_pca20")
    parser.add_argument("--subjects", nargs="+", default=[f"{i:02d}" for i in range(1, 10)])
    parser.add_argument("--rois", nargs="+", default=ROI_ORDER)
    parser.add_argument("--lags", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--alphas", nargs="+", type=float, default=[100, 300, 1000, 3000, 10000, 30000])
    parser.add_argument("--inner-splits", type=int, default=3, help="Inner CV folds for alpha tuning.")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--boundary-levels", nargs="+", default=["sentence", "event", "constituent"])
    parser.add_argument("--min-condition-samples", type=int, default=20)
    parser.add_argument("--event-min-samples", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip alpha tuning and training; recompute analysis heatmaps/PDFs from saved predictions.",
    )
    return parser.parse_args()


def prediction_path(decoder_dir: Path, roi: str, direction: str, lag: int) -> Path:
    return decoder_dir / "predictions" / f"super_brain__{roi}__{direction}__lag-{int(lag):02d}.npz"


def load_pooled_roi(prepared_dir: Path, subjects: list[str], roi: str, roi_lookup: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    parts = []
    valid = None
    used_subjects = []
    for subject in subjects:
        x_all, v, channels = load_prepared_subject(prepared_dir, subject)
        cols = channel_indices_for_roi(channels, roi_lookup, subject, roi)
        if len(cols) == 0:
            if roi == "ALL":
                raise ValueError(f"{subject_label(subject)} {roi}: no channels")
            continue
        parts.append(x_all[:, cols].astype(np.float32))
        used_subjects.append(subject_label(subject))
        valid = v if valid is None else (valid & v)
    if not parts:
        raise ValueError(f"{roi}: no channels across any subject")
    return np.hstack(parts), valid, used_subjects


def inner_cv_score(x: np.ndarray, y: np.ndarray, alpha: float, splits: int) -> float:
    """Mean held-out correlation (present-word proxy metric for alpha tuning)."""
    cv = KFold(n_splits=int(splits), shuffle=False)
    scores = []
    for train_idx, test_idx in cv.split(x):
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x[train_idx]).astype(np.float32)
        x_test = x_scaler.transform(x[test_idx]).astype(np.float32)
        y_train = y_scaler.fit_transform(y[train_idx]).astype(np.float32)
        model = Ridge(alpha=float(alpha), fit_intercept=False)
        model.fit(x_train, y_train)
        pred = y_scaler.inverse_transform(model.predict(x_test)).astype(np.float32)
        pr = row_corr(y[test_idx], pred)
        scores.append(float(np.nanmean(pr)))
    return float(np.nanmean(scores))


def tune_alpha(x: np.ndarray, y: np.ndarray, alphas: list[float], inner_splits: int) -> tuple[float, pd.DataFrame]:
    rows = []
    best_alpha = float(alphas[0])
    best_score = -np.inf
    for alpha in alphas:
        score = inner_cv_score(x, y, alpha, inner_splits)
        rows.append({"alpha": float(alpha), "inner_cv_mean_point_r": score})
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha, pd.DataFrame(rows)


def p_gt_zero(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return float("nan")
    _, p = stats.ttest_1samp(finite, 0.0, alternative="greater")
    return float(p)


def p_paired_gt_zero(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    diff = a[mask] - b[mask]
    if len(diff) < 2:
        return float("nan")
    _, p = stats.ttest_1samp(diff, 0.0, alternative="greater")
    return float(p)


def train_decoders(
    args: argparse.Namespace,
    subjects: list[str],
    rois: list[str],
    lags: list[int],
    word_scores: np.ndarray,
    alpha: float,
    decoder_dir: Path,
) -> pd.DataFrame:
    decoder_dir.mkdir(parents=True, exist_ok=True)
    (decoder_dir / "predictions").mkdir(parents=True, exist_ok=True)
    roi_lookup = load_roi_lookup(args.prepared_dir)
    rows = []
    for roi in rois:
        x_all, valid, used = load_pooled_roi(args.prepared_dir, subjects, roi, roi_lookup)
        shared_current = valid_current_indices(valid, word_scores, lags, "all_lags")
        print(f"{roi}: {x_all.shape[1]} pooled channels ({len(used)} subjects), {len(shared_current)} shared words", flush=True)
        for lag in lags:
            for direction in ("past", "future"):
                out_path = prediction_path(decoder_dir, roi, direction, lag)
                if out_path.is_file() and not args.overwrite:
                    npz = np.load(out_path)
                    row = {
                        "roi": roi,
                        "lag": int(lag),
                        "direction": direction,
                        "n_channels": int(x_all.shape[1]),
                        "alpha": float(alpha),
                    }
                    row.update(score_predictions(npz["y_true"], npz["y_pred"]))
                    row["n_samples"] = int(len(npz["point_r"]))
                    rows.append(row)
                    continue
                current = shared_current
                x = x_all[current]
                finite_x = np.isfinite(x).all(axis=1)
                current = current[finite_x]
                x = x[finite_x]
                target = target_indices(current, lag, direction)
                y = word_scores[target].astype(np.float32)
                finite_y = np.isfinite(y).all(axis=1)
                use_current = current[finite_y]
                use_target = target[finite_y]
                use_x = x[finite_y]
                use_y = y[finite_y]
                print(f"  {roi} lag {lag:02d} {direction}: {len(use_current)} samples", flush=True)
                y_pred = cv_predict_ridge(use_x, use_y, alpha, args.outer_splits)
                point_r = row_corr(use_y, y_pred)
                np.savez_compressed(
                    out_path,
                    roi=roi,
                    lag=int(lag),
                    direction=direction,
                    current_word_idx=use_current.astype(np.int32),
                    target_word_idx=use_target.astype(np.int32),
                    y_true=use_y.astype(np.float32),
                    y_pred=y_pred.astype(np.float32),
                    point_r=point_r.astype(np.float32),
                    alpha=float(alpha),
                )
                row = {
                    "roi": roi,
                    "lag": int(lag),
                    "direction": direction,
                    "n_channels": int(x_all.shape[1]),
                    "alpha": float(alpha),
                    "n_samples": int(len(use_current)),
                }
                row.update(score_predictions(use_y, y_pred))
                rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(decoder_dir / "decoder_summary.csv", index=False)
    return summary


def raw_directional_pooled(
    decoder_dir: Path, rois: list[str], lags: list[int], output_dir: Path, metric_suffix: str
) -> pd.DataFrame:
    value_col = "mean_point_r" if metric_suffix == "point_r" else "mean_rank_accuracy"
    rows = []
    for roi in rois:
        for direction in ("past", "future"):
            for lag in lags:
                npz = np.load(prediction_path(decoder_dir, roi, direction, lag))
                scores = word_metric_scores_from_npz(npz, metric_suffix)
                rows.append(
                    {
                        "roi": roi,
                        "direction": direction,
                        "lag": int(lag),
                        "target_label": f"t+{lag}" if direction == "future" else f"t-{lag}",
                        value_col: float(np.nanmean(scores)),
                        "p_one_sided_gt_zero": p_gt_null(scores, metric_suffix),
                        "n_words": int(np.isfinite(scores).sum()),
                    }
                )
    group = pd.DataFrame(rows)
    group["q_one_sided_bh"] = bh_fdr(group["p_one_sided_gt_zero"].to_numpy(dtype=float))
    tag = _metric_tag(metric_suffix)
    group.to_csv(output_dir / f"raw_directional_prediction_group_scores_{tag}.csv", index=False)
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
        f"Word decoding by target lag ({metric_suffix})",
        output_dir / f"raw_directional_prediction_heatmap_{tag}.png",
        center=center,
    )
    return group


def future_controlled_pooled(
    decoder_dir: Path, rois: list[str], lags: list[int], output_dir: Path, metric_suffix: str
) -> pd.DataFrame:
    rows = []
    for roi in rois:
        for lag in lags:
            past = np.load(prediction_path(decoder_dir, roi, "past", lag))
            future = np.load(prediction_path(decoder_dir, roi, "future", lag))
            s_p = word_metric_scores_from_npz(past, metric_suffix)
            s_f = word_metric_scores_from_npz(future, metric_suffix)
            delta = float(np.nanmean(s_p) - np.nanmean(s_f))
            rows.append(
                {
                    "roi": roi,
                    "lag": int(lag),
                    "mean_delta": delta,
                    "p_one_sided": p_paired_gt_zero(s_p, s_f),
                    "n_words": int(min(np.isfinite(s_p).sum(), np.isfinite(s_f).sum())),
                }
            )
    group = pd.DataFrame(rows)
    group["q_one_sided_bh"] = bh_fdr(group["p_one_sided"].to_numpy(dtype=float))
    tag = _metric_tag(metric_suffix)
    group.to_csv(output_dir / f"future_controlled_prediction_group_scores_{tag}.csv", index=False)
    ylab = (
        "mean rank_acc(past t-N) - mean rank_acc(future t+N)"
        if metric_suffix == "rank_identification"
        else "mean point_r(past t-N) - mean point_r(future t+N)"
    )
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
        output_path=output_dir / f"future_controlled_prediction_delta_heatmap_{tag}.png",
    )
    return group


def boundary_scores_pooled(
    prepared_dir: Path,
    decoder_dir: Path,
    rois: list[str],
    lags: list[int],
    level: str,
    min_samples: int,
    metric_suffix: str,
) -> pd.DataFrame:
    words = pd.read_csv(prepared_dir / "words.csv")
    column = f"{level}_boundary_after"
    boundary_after = words[column].astype(bool).to_numpy()
    prefix = boundary_prefix(boundary_after)
    rows = []
    for roi in rois:
        for lag in lags:
            for direction in ("past", "future"):
                npz = np.load(prediction_path(decoder_dir, roi, direction, lag))
                scores = word_metric_scores_from_npz(npz, metric_suffix)
                current = npz["current_word_idx"].astype(np.int32)
                target = npz["target_word_idx"].astype(np.int32)
                crossed = crossed_between(prefix, current, target)
                row = {
                    "level": level,
                    "roi": roi,
                    "lag": int(lag),
                    "direction": direction,
                    "metric": metric_suffix,
                }
                row.update(condition_delta(scores, crossed, min_samples))
                rows.append(row)
    return pd.DataFrame(rows)


def p_within_gt_across(scores: np.ndarray, crossed: np.ndarray) -> tuple[float, float]:
    within = scores[~crossed]
    across = scores[crossed]
    if len(within) < 2 or len(across) < 2:
        return float("nan"), float("nan")
    delta = float(np.nanmean(within) - np.nanmean(across))
    _, p = stats.ttest_ind(within, across, equal_var=False, alternative="greater")
    return delta, float(p)


def fc_boundary_p(
    pr_p: np.ndarray,
    crossed_p: np.ndarray,
    pr_f: np.ndarray,
    crossed_f: np.ndarray,
    n_boot: int = 100,
    seed: int = 0,
) -> float:
    w_p, a_p = pr_p[~crossed_p], pr_p[crossed_p]
    w_f, a_f = pr_f[~crossed_f], pr_f[crossed_f]
    if min(len(w_p), len(a_p), len(w_f), len(a_f)) < 2:
        return float("nan")
    obs = (float(np.nanmean(w_p) - np.nanmean(a_p)) - (float(np.nanmean(w_f)) - np.nanmean(a_f)))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        bw_p = w_p[rng.integers(0, len(w_p), len(w_p))]
        ba_p = a_p[rng.integers(0, len(a_p), len(a_p))]
        bw_f = w_f[rng.integers(0, len(w_f), len(w_f))]
        ba_f = a_f[rng.integers(0, len(a_f), len(a_f))]
        boots.append(
            (float(np.nanmean(bw_p) - np.nanmean(ba_p)) - (float(np.nanmean(bw_f)) - np.nanmean(ba_f)))
        )
    return float(np.mean(np.asarray(boots) >= obs))


def directional_boundary_pooled(
    prepared_dir: Path,
    decoder_dir: Path,
    rois: list[str],
    lags: list[int],
    level: str,
    output_dir: Path,
    metric_suffix: str,
) -> pd.DataFrame:
    words = pd.read_csv(prepared_dir / "words.csv")
    boundary_after = words[f"{level}_boundary_after"].astype(bool).to_numpy()
    prefix = boundary_prefix(boundary_after)
    tag = _metric_tag(metric_suffix)
    rows = []
    for roi in rois:
        for direction in ("past", "future"):
            for lag in lags:
                npz = np.load(prediction_path(decoder_dir, roi, direction, lag))
                scores = word_metric_scores_from_npz(npz, metric_suffix)
                crossed = crossed_between(prefix, npz["current_word_idx"], npz["target_word_idx"])
                delta, p_one = p_within_gt_across(scores, crossed)
                rows.append(
                    {
                        "level": level,
                        "roi": roi,
                        "direction": direction,
                        "lag": int(lag),
                        "target_label": f"t+{lag}" if direction == "future" else f"t-{lag}",
                        "mean_delta": delta,
                        "p_one_sided": p_one,
                        "n_words": int(np.isfinite(scores).sum()),
                    }
                )
    group = pd.DataFrame(rows)
    group["q_one_sided_bh"] = bh_fdr(group["p_one_sided"].to_numpy(dtype=float))
    group.to_csv(output_dir / f"{level}_boundary_directional_group_scores_{tag}.csv", index=False)
    plot_directional_boundary_heatmap(
        group,
        rois,
        lags,
        level,
        output_dir / f"{level}_boundary_directional_delta_heatmap_{tag}.png",
        metric_suffix=metric_suffix,
    )
    return group


def future_controlled_boundary_pooled(
    prepared_dir: Path,
    decoder_dir: Path,
    rois: list[str],
    lags: list[int],
    level: str,
    output_dir: Path,
    metric_suffix: str,
) -> pd.DataFrame:
    words = pd.read_csv(prepared_dir / "words.csv")
    boundary_after = words[f"{level}_boundary_after"].astype(bool).to_numpy()
    prefix = boundary_prefix(boundary_after)
    tag = _metric_tag(metric_suffix)
    rows = []
    for roi in rois:
        for lag in lags:
            past = np.load(prediction_path(decoder_dir, roi, "past", lag))
            future = np.load(prediction_path(decoder_dir, roi, "future", lag))
            s_p = word_metric_scores_from_npz(past, metric_suffix)
            s_f = word_metric_scores_from_npz(future, metric_suffix)
            crossed_p = crossed_between(prefix, past["current_word_idx"], past["target_word_idx"])
            crossed_f = crossed_between(prefix, future["current_word_idx"], future["target_word_idx"])
            delta_p, _ = p_within_gt_across(s_p, crossed_p)
            delta_f, _ = p_within_gt_across(s_f, crossed_f)
            fc = delta_p - delta_f if np.isfinite(delta_p) and np.isfinite(delta_f) else np.nan
            p_one = fc_boundary_p(s_p, crossed_p, s_f, crossed_f)
            rows.append(
                {
                    "level": level,
                    "roi": roi,
                    "lag": int(lag),
                    "mean_delta": fc,
                    "p_one_sided": p_one,
                    "n_words": int(min(len(s_p), len(s_f))),
                }
            )
    group = pd.DataFrame(rows)
    group["q_one_sided_bh"] = bh_fdr(group["p_one_sided"].to_numpy(dtype=float))
    group.to_csv(output_dir / f"{level}_boundary_future_controlled_group_scores_{tag}.csv", index=False)
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
        output_path=output_dir / f"{level}_boundary_future_controlled_delta_heatmap_{tag}.png",
    )
    return group


def analyze_metric(
    args: argparse.Namespace,
    decoder_dir: Path,
    analysis_dir: Path,
    rois: list[str],
    lags: list[int],
    metric_suffix: str,
) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    tag = _metric_tag(metric_suffix)
    print(f"Analysis metric: {metric_suffix}", flush=True)
    raw_directional_pooled(decoder_dir, rois, lags, analysis_dir, metric_suffix)
    future_controlled_pooled(decoder_dir, rois, lags, analysis_dir, metric_suffix)

    all_condition = []
    for level in args.boundary_levels:
        min_n = args.event_min_samples if level == "event" else args.min_condition_samples
        condition = boundary_scores_pooled(
            args.prepared_dir, decoder_dir, rois, lags, level, min_n, metric_suffix
        )
        condition.to_csv(analysis_dir / f"{level}_boundary_condition_scores_{tag}.csv", index=False)
        all_condition.append(condition)
        directional_boundary_pooled(
            args.prepared_dir, decoder_dir, rois, lags, level, analysis_dir, metric_suffix
        )
        future_controlled_boundary_pooled(
            args.prepared_dir, decoder_dir, rois, lags, level, analysis_dir, metric_suffix
        )

    if all_condition:
        pd.concat(all_condition, ignore_index=True).to_csv(
            analysis_dir / f"boundary_condition_scores_{tag}.csv", index=False
        )
    write_combined_heatmap_pdf(analysis_dir, metric_suffix=metric_suffix)


def analyze(args: argparse.Namespace, decoder_dir: Path, analysis_dir: Path, rois: list[str], lags: list[int]) -> None:
    for metric_suffix in ANALYSIS_METRICS:
        analyze_metric(args, decoder_dir, analysis_dir, rois, lags, metric_suffix)


def main() -> int:
    args = parse_args()
    subjects = [clean_subject(s) for s in args.subjects]
    rois = [r for r in ROI_ORDER if r in set(args.rois)]
    lags = sorted(set(int(l) for l in args.lags if int(l) > 0))
    alphas = [float(a) for a in args.alphas]

    target_path = args.prepared_dir / f"{args.target_stem}.npy"
    if not target_path.is_file():
        raise FileNotFoundError(target_path)
    word_scores = np.load(target_path).astype(np.float32)

    run_root = args.output_root / args.target_stem
    decoder_dir = run_root / "decoders"
    analysis_dir = run_root / "analysis"
    run_root.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        roi_lookup = load_roi_lookup(args.prepared_dir)
        x_all, valid, _ = load_pooled_roi(args.prepared_dir, subjects, "ALL", roi_lookup)
        finite_y = np.isfinite(word_scores).all(axis=1)
        current = np.flatnonzero(valid & finite_y & np.isfinite(x_all).all(axis=1))
        x = x_all[current]
        y = word_scores[current]
        print(f"Tuning alpha on present word: {len(current)} words, {x.shape[1]} channels", flush=True)
        best_alpha, alpha_table = tune_alpha(x, y, alphas, args.inner_splits)
        alpha_table.to_csv(run_root / "alpha_tune.csv", index=False)
        with (run_root / "alpha_tune_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "target_stem": args.target_stem,
                    "best_alpha": best_alpha,
                    "inner_splits": args.inner_splits,
                    "alphas_tried": alphas,
                    "n_words": int(len(current)),
                    "n_channels": int(x.shape[1]),
                },
                handle,
                indent=2,
            )
        print(f"Best alpha: {best_alpha}", flush=True)
        print(alpha_table.to_string(index=False), flush=True)
        train_decoders(args, subjects, rois, lags, word_scores, best_alpha, decoder_dir)
    else:
        print("Skipping training (--analyze-only); using existing predictions.", flush=True)

    analyze(args, decoder_dir, analysis_dir, rois, lags)
    print(f"\nSuper-brain outputs written to: {run_root}", flush=True)
    for metric_suffix in ANALYSIS_METRICS:
        pdf = analysis_dir / f"combined_decoding_heatmaps_{metric_suffix}.pdf"
        print(f"Combined PDF ({metric_suffix}): {pdf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
