#!/usr/bin/env python3
"""Pooled present-word decoding summary for the curated v2 outputs.

All participants heard the same podcast on the same word timeline, so electrodes
can be concatenated across subjects into one feature matrix. This script compares
present-word rank-identification accuracy for:

  - per-subject decoders, averaged across subjects
  - one pooled decoder over the union of all electrodes

Outputs under ``<output-dir>``:

  - ``super_brain_<stem>.json``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from common import DEFAULT_OUTPUT_ROOT, clean_subject, identification_rank_accuracy, subject_label  # noqa: E402
from train_decoders import cv_predict_ridge, load_prepared_subject  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "prepared_midpoint")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "super_brain_midpoint")
    parser.add_argument("--target-stems", nargs="+", default=["word_vectors_pca20", "gpt2_ctx32_layer8_pca20"])
    parser.add_argument("--subjects", nargs="+", default=[f"{i:02d}" for i in range(1, 10)])
    parser.add_argument("--ridge-alpha", type=float, default=1000.0)
    parser.add_argument("--outer-splits", type=int, default=5)
    return parser.parse_args()


def run_stem(prepared_dir: Path, stem: str, subjects: list[str], alpha: float, splits: int) -> dict:
    scores = np.load(prepared_dir / f"{stem}.npy").astype(np.float32)
    finite_y = np.isfinite(scores).all(axis=1)

    per_subject = {}
    x_by_subject = {}
    valid_by_subject = {}
    for subject in subjects:
        x_all, valid, _ = load_prepared_subject(prepared_dir, subject)
        ok = valid & np.isfinite(x_all).all(axis=1)
        x_by_subject[subject] = x_all
        valid_by_subject[subject] = ok
        current = np.flatnonzero(ok & finite_y)
        pred = cv_predict_ridge(x_all[current].astype(np.float32), scores[current], alpha, splits)
        per_subject[subject_label(subject)] = {
            "rank_accuracy": identification_rank_accuracy(scores[current], pred),
            "n_channels": int(x_all.shape[1]),
            "n_words": int(len(current)),
        }

    common = finite_y.copy()
    for subject in subjects:
        common &= valid_by_subject[subject]
    common_idx = np.flatnonzero(common)
    x_pooled = np.hstack([x_by_subject[subject][common_idx] for subject in subjects]).astype(np.float32)
    y = scores[common_idx]
    pred = cv_predict_ridge(x_pooled, y, alpha, splits)
    pooled_acc = identification_rank_accuracy(y, pred)

    subject_scores = [v["rank_accuracy"] for v in per_subject.values()]
    return {
        "stem": stem,
        "per_subject": per_subject,
        "per_subject_mean": float(np.nanmean(subject_scores)),
        "per_subject_best": float(np.nanmax(subject_scores)),
        "pooled_rank_accuracy": float(pooled_acc),
        "pooled_n_channels": int(x_pooled.shape[1]),
        "pooled_n_words": int(len(common_idx)),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = [clean_subject(s) for s in args.subjects]

    for stem in args.target_stems:
        result = run_stem(args.prepared_dir, stem, subjects, args.ridge_alpha, args.outer_splits)
        with (args.output_dir / f"super_brain_{stem}.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"\n=== {stem} (present word, chance 0.5) ===", flush=True)
        for subject, value in result["per_subject"].items():
            print(f"  {subject}: {value['rank_accuracy']:.3f}  ({value['n_channels']} ch)", flush=True)
        print(f"  per-subject mean : {result['per_subject_mean']:.3f}", flush=True)
        print(f"  per-subject best : {result['per_subject_best']:.3f}", flush=True)
        print(
            f"  pooled: {result['pooled_rank_accuracy']:.3f}  "
            f"({result['pooled_n_channels']} ch, {result['pooled_n_words']} words)",
            flush=True,
        )
    print(f"\nWrote present-word pooled summaries to: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
