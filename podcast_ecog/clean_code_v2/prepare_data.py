#!/usr/bin/env python3
"""Prepare cached inputs for the clean Podcast ECoG word-decoding analyses.

This script is intentionally boring: it collects the pieces needed by later
decoding scripts into one local cache. It reads the BIDS dataset, but all
outputs are written under ``podcast_ecog/outputs/clean_code_v2`` by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    DEFAULT_BIDS_ROOT,
    DEFAULT_BOUNDARY_JSON,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_ROI_METRICS,
    boundary_after_from_ends,
    clean_subject,
    compute_word_pca,
    highgamma_fif_path,
    load_boundary_json,
    load_dataset_static_word_vectors,
    segment_ids_from_boundary_after,
    segment_means,
    subject_label,
    word_span_means,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "prepared")
    parser.add_argument("--boundary-json", type=Path, default=DEFAULT_BOUNDARY_JSON)
    parser.add_argument("--roi-metrics", type=Path, default=DEFAULT_ROI_METRICS)
    parser.add_argument("--subjects", nargs="+", default=[f"{i:02d}" for i in range(1, 10)])
    parser.add_argument("--task", default="podcast")
    parser.add_argument("--dataset-feature-space", default="en_core_web_lg")
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--window-start", type=float, default=-0.5)
    parser.add_argument("--window-end", type=float, default=0.5)
    parser.add_argument(
        "--window-mode",
        choices=["onset", "word", "midpoint"],
        default="onset",
        help=(
            "onset: fixed [window_start, window_end] s window around word onset (default, original behavior). "
            "word: mean activity over the present word's own [start, end] span. "
            "midpoint: fixed [window_start, window_end] s window around the word midpoint (start+end)/2."
        ),
    )
    parser.add_argument("--picks-regex", default=None)
    parser.add_argument(
        "--prepare-epochs",
        action="store_true",
        help=(
            "Also cache word-locked high-gamma epochs for time-based encoding analyses "
            "(neural_word_epochs.npz per subject)."
        ),
    )
    parser.add_argument("--epoch-tmin", type=float, default=-2.0, help="Epoch start relative to anchor (seconds).")
    parser.add_argument("--epoch-tmax", type=float, default=2.0, help="Epoch stop relative to anchor (seconds).")
    parser.add_argument("--epoch-resample-sfreq", type=float, default=32.0, help="Resample epochs to this Hz.")
    parser.add_argument(
        "--epoch-anchor",
        choices=["onset", "midpoint"],
        default="onset",
        help="Anchor word-locked epochs on word onset (run_encoding.py default) or midpoint.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute subject neural caches even if they already exist.")
    parser.add_argument(
        "--force-epochs",
        action="store_true",
        help="Recompute neural_word_epochs.npz even if it already exists.",
    )
    parser.add_argument("--force-roi", action="store_true", help="Refresh roi_channels.csv without recomputing the rest of the cache.")
    return parser.parse_args()


def write_words_and_boundaries(words: pd.DataFrame, boundary_data: dict[str, list[int]], output_dir: Path) -> None:
    n_words = len(words)
    sentence_after = boundary_after_from_ends(n_words, boundary_data["sentence_end_word_indices"])
    event_after = boundary_after_from_ends(n_words, boundary_data["event_end_word_indices"])

    out = words.copy()
    out["word_idx"] = out["word_idx"].astype(int)
    out["sentence_id"] = segment_ids_from_boundary_after(sentence_after)
    out["event_id"] = segment_ids_from_boundary_after(event_after)
    out["sentence_boundary_after"] = sentence_after
    out["event_boundary_after"] = event_after
    out.to_csv(output_dir / "words.csv", index=False)

    rows = []
    for level, key in [("sentence", "sentence_end_word_indices"), ("event", "event_end_word_indices")]:
        for end_idx in boundary_data[key]:
            if 0 <= int(end_idx) < n_words:
                rows.append(
                    {
                        "level": level,
                        "end_word_idx": int(end_idx),
                        "time": float(out.loc[int(end_idx), "end"]),
                        "word": str(out.loc[int(end_idx), "word"]),
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "boundaries.csv", index=False)


def prepare_embeddings(args: argparse.Namespace) -> pd.DataFrame:
    words_path = args.output_dir / "words.csv"
    scores_path = args.output_dir / f"word_vectors_pca{args.n_components}.npy"
    if words_path.is_file() and scores_path.is_file() and not args.force:
        print(f"Reusing prepared words and PCA vectors from: {args.output_dir}", flush=True)
        return pd.read_csv(words_path)[["word_idx", "word", "start", "end"]].copy()

    print(f"Loading bundled static word vectors: {args.dataset_feature_space}", flush=True)
    words, vectors, coverage = load_dataset_static_word_vectors(args.bids_root, args.dataset_feature_space)
    scores, explained = compute_word_pca(vectors, args.n_components)

    np.save(args.output_dir / "word_vectors_raw.npy", vectors.astype(np.float32))
    np.save(args.output_dir / f"word_vectors_pca{args.n_components}.npy", scores.astype(np.float32))
    coverage.to_csv(args.output_dir / "embedding_coverage.csv", index=False)
    explained.to_csv(args.output_dir / "embedding_pca_explained_variance.csv", index=False)
    return words


def prepare_roi_lookup(args: argparse.Namespace) -> None:
    if not args.roi_metrics.is_file():
        print(f"ROI metrics not found, skipping ROI lookup: {args.roi_metrics}", flush=True)
        return
    roi = pd.read_csv(args.roi_metrics)
    expected = {"subject", "channel", "paper_roi"}
    missing = expected.difference(roi.columns)
    if missing:
        raise ValueError(f"{args.roi_metrics} is missing columns: {sorted(missing)}")
    subjects = {subject_label(s) for s in args.subjects}
    roi = roi[roi["subject"].isin(subjects)].copy()
    roi[["subject", "channel", "paper_roi"]].drop_duplicates().to_csv(args.output_dir / "roi_channels.csv", index=False)


def load_subject_raw(args: argparse.Namespace, subject: str) -> mne.io.BaseRaw:
    fif = highgamma_fif_path(args.bids_root, subject, args.task)
    if not fif.is_file():
        raise FileNotFoundError(fif)
    raw = mne.io.read_raw_fif(fif, preload=False, verbose=False)
    if args.picks_regex:
        picks = mne.pick_channels_regexp(raw.ch_names, args.picks_regex)
        if len(picks) == 0:
            raise ValueError(f"{subject_label(subject)}: no channels match {args.picks_regex!r}")
        raw.pick(picks)
    return raw


def prepare_subject_neural(args: argparse.Namespace, words: pd.DataFrame, subject: str) -> None:
    subject = clean_subject(subject)
    sub = subject_label(subject)
    subject_dir = args.output_dir / "subjects" / sub
    subject_dir.mkdir(parents=True, exist_ok=True)
    npz_path = subject_dir / "neural_word_features.npz"
    channel_path = subject_dir / "channels.csv"
    need_scalar = args.force or not (npz_path.is_file() and channel_path.is_file())
    epoch_path = subject_dir / "neural_word_epochs.npz"
    need_epochs = args.prepare_epochs and (args.force or args.force_epochs or not epoch_path.is_file())
    if not need_scalar and not need_epochs:
        print(f"{sub}: neural cache exists, skipping", flush=True)
        return

    raw = load_subject_raw(args, subject)
    print(f"{sub}: reading {len(raw.ch_names)} channels at {raw.info['sfreq']:.1f} Hz", flush=True)

    if need_scalar:
        data = raw.get_data().astype(np.float32)
        sfreq = float(raw.info["sfreq"])
        if args.window_mode == "word":
            start_idx = raw.time_as_index(words["start"].to_numpy(dtype=float), use_rounding=True)
            stop_idx = raw.time_as_index(words["end"].to_numpy(dtype=float), use_rounding=True)
            x, valid = word_span_means(data, start_idx, stop_idx)
        elif args.window_mode == "midpoint":
            midpoints = (words["start"].to_numpy(dtype=float) + words["end"].to_numpy(dtype=float)) / 2.0
            centers = raw.time_as_index(midpoints, use_rounding=True)
            x, valid = segment_means(data, centers, sfreq, args.window_start, args.window_end)
        else:
            centers = raw.time_as_index(words["start"].to_numpy(dtype=float), use_rounding=True)
            x, valid = segment_means(data, centers, sfreq, args.window_start, args.window_end)

        np.savez_compressed(
            npz_path,
            x=x.astype(np.float32),
            valid=valid.astype(bool),
            window_mode=str(args.window_mode),
            window_start=float(args.window_start),
            window_end=float(args.window_end),
            sfreq=sfreq,
        )
        print(f"{sub}: saved {x.shape[0]} word rows x {x.shape[1]} channels", flush=True)

    if need_epochs:
        if args.epoch_anchor == "midpoint":
            anchor_times = (words["start"].to_numpy(dtype=float) + words["end"].to_numpy(dtype=float)) / 2.0
        else:
            anchor_times = words["start"].to_numpy(dtype=float)
        events = np.zeros((len(words), 3), dtype=int)
        events[:, 0] = raw.time_as_index(anchor_times, use_rounding=True)
        raw.load_data(verbose=False)
        epochs = mne.Epochs(
            raw,
            events,
            tmin=float(args.epoch_tmin),
            tmax=float(args.epoch_tmax),
            baseline=None,
            proj=False,
            event_id=None,
            preload=True,
            event_repeated="merge",
            verbose=False,
        )
        epochs = epochs.resample(
            sfreq=float(args.epoch_resample_sfreq),
            npad="auto",
            method="fft",
            window="hamming",
            verbose=False,
        )
        epoch_data = epochs.get_data(copy=True).astype(np.float32)
        n_words = len(words)
        n_channels = epoch_data.shape[1]
        n_times = epoch_data.shape[2]
        full_epochs = np.full((n_words, n_channels, n_times), np.nan, dtype=np.float32)
        epoch_valid = np.zeros(n_words, dtype=bool)
        for row, word_idx in enumerate(epochs.selection):
            full_epochs[int(word_idx)] = epoch_data[row]
            epoch_valid[int(word_idx)] = np.isfinite(epoch_data[row]).all()
        np.savez_compressed(
            epoch_path,
            epochs=full_epochs,
            times=epochs.times.astype(np.float32),
            valid=epoch_valid,
            epoch_tmin=float(args.epoch_tmin),
            epoch_tmax=float(args.epoch_tmax),
            epoch_resample_sfreq=float(args.epoch_resample_sfreq),
            epoch_anchor=str(args.epoch_anchor),
        )
        print(
            f"{sub}: saved epochs {full_epochs.shape} "
            f"({int(epoch_valid.sum())} valid words, anchor={args.epoch_anchor})",
            flush=True,
        )

    if need_scalar or not channel_path.is_file():
        pd.DataFrame({"channel": list(raw.ch_names)}).to_csv(channel_path, index=False)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = [clean_subject(s) for s in args.subjects]

    words = prepare_embeddings(args)
    if not (args.output_dir / "boundaries.csv").is_file() or args.force:
        boundaries = load_boundary_json(args.boundary_json)
        write_words_and_boundaries(words, boundaries, args.output_dir)
    if not (args.output_dir / "roi_channels.csv").is_file() or args.force or args.force_roi:
        prepare_roi_lookup(args)

    config = {
        "bids_root": str(args.bids_root),
        "boundary_json": str(args.boundary_json),
        "roi_metrics": str(args.roi_metrics),
        "subjects": subjects,
        "task": args.task,
        "dataset_feature_space": args.dataset_feature_space,
        "n_components": int(args.n_components),
        "window_mode": str(args.window_mode),
        "window_start": float(args.window_start),
        "window_end": float(args.window_end),
        "picks_regex": args.picks_regex,
        "prepare_epochs": bool(args.prepare_epochs),
        "epoch_tmin": float(args.epoch_tmin),
        "epoch_tmax": float(args.epoch_tmax),
        "epoch_resample_sfreq": float(args.epoch_resample_sfreq),
        "epoch_anchor": str(args.epoch_anchor),
    }
    with (args.output_dir / "prepare_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    for subject in subjects:
        prepare_subject_neural(args, words, subject)

    print(f"Prepared cache written to: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
