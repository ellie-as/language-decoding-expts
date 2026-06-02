#!/usr/bin/env python3
"""Train clean ridge decoders for word representations at past/future lags."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    ROI_ORDER,
    clean_subject,
    row_corr,
    score_predictions,
    subject_label,
    target_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "prepared")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "decoders")
    parser.add_argument("--subjects", nargs="+", default=[f"{i:02d}" for i in range(1, 10)])
    parser.add_argument("--rois", nargs="+", default=ROI_ORDER)
    parser.add_argument("--lags", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument(
        "--target-stem",
        default=None,
        help=(
            "Stem of the prepared target .npy file, relative to --prepared-dir. "
            "Defaults to word_vectors_pca{n_components}."
        ),
    )
    parser.add_argument(
        "--target-file",
        type=Path,
        default=None,
        help="Explicit target .npy path. Overrides --target-stem.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1000.0)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--min-channels", type=int, default=1)
    parser.add_argument(
        "--sample-mode",
        choices=["all_lags", "per_lag"],
        default="all_lags",
        help="all_lags uses the same current-word positions for every lag and both directions within a subject.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_target_path(args: argparse.Namespace) -> tuple[Path, str]:
    if args.target_file is not None:
        path = args.target_file
        target_name = path.stem
    else:
        target_name = args.target_stem or f"word_vectors_pca{args.n_components}"
        path = args.prepared_dir / f"{target_name}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, target_name


def prediction_path(output_dir: Path, subject: str, roi: str, direction: str, lag: int) -> Path:
    return output_dir / "predictions" / f"{subject_label(subject)}__{roi}__{direction}__lag-{int(lag):02d}.npz"


def load_prepared_subject(prepared_dir: Path, subject: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    sub = subject_label(subject)
    npz_path = prepared_dir / "subjects" / sub / "neural_word_features.npz"
    channel_path = prepared_dir / "subjects" / sub / "channels.csv"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not channel_path.is_file():
        raise FileNotFoundError(channel_path)
    npz = np.load(npz_path)
    channels = pd.read_csv(channel_path)["channel"].astype(str).tolist()
    return npz["x"].astype(np.float32), npz["valid"].astype(bool), channels


def load_roi_lookup(prepared_dir: Path) -> pd.DataFrame:
    path = prepared_dir / "roi_channels.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["subject", "channel", "paper_roi"])
    return pd.read_csv(path)[["subject", "channel", "paper_roi"]].drop_duplicates()


def channel_indices_for_roi(channels: list[str], roi_lookup: pd.DataFrame, subject: str, roi: str) -> np.ndarray:
    if roi == "ALL":
        return np.arange(len(channels), dtype=int)
    sub = subject_label(subject)
    wanted = set(
        roi_lookup[(roi_lookup["subject"] == sub) & (roi_lookup["paper_roi"] == roi)]["channel"].astype(str).tolist()
    )
    return np.asarray([i for i, channel in enumerate(channels) if channel in wanted], dtype=int)


def cv_predict_ridge(x: np.ndarray, y: np.ndarray, alpha: float, outer_splits: int) -> np.ndarray:
    pred = np.zeros_like(y, dtype=np.float32)
    cv = KFold(n_splits=int(outer_splits), shuffle=False)
    for train_idx, test_idx in cv.split(x):
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x[train_idx]).astype(np.float32)
        x_test = x_scaler.transform(x[test_idx]).astype(np.float32)
        y_train = y_scaler.fit_transform(y[train_idx]).astype(np.float32)
        model = Ridge(alpha=float(alpha), fit_intercept=False)
        model.fit(x_train, y_train)
        pred[test_idx] = y_scaler.inverse_transform(model.predict(x_test)).astype(np.float32)
    return pred


def valid_current_indices(valid_neural: np.ndarray, word_scores: np.ndarray, lags: list[int], mode: str, lag: int | None = None) -> np.ndarray:
    if mode == "per_lag":
        use_lags = [int(lag)]
    else:
        use_lags = [int(x) for x in lags]
    current = np.flatnonzero(valid_neural & np.isfinite(word_scores).all(axis=1))
    keep = np.ones(len(current), dtype=bool)
    for use_lag in use_lags:
        for direction in ("past", "future"):
            target = target_indices(current, use_lag, direction)
            in_range = (target >= 0) & (target < len(word_scores))
            finite = np.zeros(len(current), dtype=bool)
            finite[in_range] = np.isfinite(word_scores[target[in_range]]).all(axis=1)
            keep &= in_range & finite
    return current[keep]


def load_existing_summary(path: Path) -> dict[str, object]:
    npz = np.load(path, allow_pickle=False)
    y_true = npz["y_true"].astype(np.float32)
    y_pred = npz["y_pred"].astype(np.float32)
    scores = score_predictions(y_true, y_pred)
    point_r = npz["point_r"].astype(np.float32)
    scores.update(
        {
            "n_samples": int(len(point_r)),
            "n_finite_point_r": int(np.isfinite(point_r).sum()),
        }
    )
    return scores


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    subjects = [clean_subject(s) for s in args.subjects]
    rois = [roi for roi in ROI_ORDER if roi in set(args.rois)]
    lags = sorted(set(int(lag) for lag in args.lags if int(lag) > 0))
    target_path, target_name = resolve_target_path(args)
    word_scores = np.load(target_path).astype(np.float32)
    roi_lookup = load_roi_lookup(args.prepared_dir)
    print(f"Using target features: {target_path} shape={word_scores.shape}", flush=True)

    with (args.output_dir / "train_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "prepared_dir": str(args.prepared_dir),
                "subjects": subjects,
                "rois": rois,
                "lags": lags,
                "n_components": int(args.n_components),
                "target_name": target_name,
                "target_path": str(target_path),
                "ridge_alpha": float(args.ridge_alpha),
                "outer_splits": int(args.outer_splits),
                "sample_mode": args.sample_mode,
            },
            handle,
            indent=2,
        )

    rows = []
    for subject in subjects:
        x_all, valid, channels = load_prepared_subject(args.prepared_dir, subject)
        print(f"{subject_label(subject)}: loaded {x_all.shape[0]} words x {x_all.shape[1]} channels", flush=True)
        shared_current = valid_current_indices(valid, word_scores, lags, args.sample_mode) if args.sample_mode == "all_lags" else None
        for roi in rois:
            cols = channel_indices_for_roi(channels, roi_lookup, subject, roi)
            if len(cols) < int(args.min_channels):
                print(f"{subject_label(subject)} {roi}: skipping, {len(cols)} channels", flush=True)
                continue
            x_roi_all = x_all[:, cols].astype(np.float32)
            for lag in lags:
                current = shared_current if shared_current is not None else valid_current_indices(valid, word_scores, lags, args.sample_mode, lag)
                x = x_roi_all[current]
                finite_x = np.isfinite(x).all(axis=1)
                current = current[finite_x]
                x = x[finite_x]
                for direction in ("past", "future"):
                    out_path = prediction_path(args.output_dir, subject, roi, direction, lag)
                    target = target_indices(current, lag, direction)
                    y = word_scores[target].astype(np.float32)
                    finite_y = np.isfinite(y).all(axis=1)
                    use_current = current[finite_y]
                    use_target = target[finite_y]
                    use_x = x[finite_y]
                    use_y = y[finite_y]

                    row = {
                        "subject": subject_label(subject),
                        "roi": roi,
                        "lag": int(lag),
                        "direction": direction,
                        "n_channels": int(len(cols)),
                        "target_name": target_name,
                    }
                    if out_path.is_file() and not args.overwrite:
                        row.update(load_existing_summary(out_path))
                    else:
                        print(
                            f"{subject_label(subject)} {roi} lag {lag:02d} {direction}: "
                            f"{len(use_current)} samples, {len(cols)} channels",
                            flush=True,
                        )
                        y_pred = cv_predict_ridge(use_x, use_y, args.ridge_alpha, args.outer_splits)
                        point_r = row_corr(use_y, y_pred)
                        np.savez_compressed(
                            out_path,
                            subject=subject_label(subject),
                            roi=roi,
                            lag=int(lag),
                            direction=direction,
                            current_word_idx=use_current.astype(np.int32),
                            target_word_idx=use_target.astype(np.int32),
                            y_true=use_y.astype(np.float32),
                            y_pred=y_pred.astype(np.float32),
                            point_r=point_r.astype(np.float32),
                        )
                        row.update(score_predictions(use_y, y_pred))
                        row["n_samples"] = int(len(use_current))
                        row["n_finite_point_r"] = int(np.isfinite(point_r).sum())
                    rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "decoder_summary.csv", index=False)
    print(f"Decoder outputs written to: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
