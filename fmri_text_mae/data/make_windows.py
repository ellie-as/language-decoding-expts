#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "fmri_text_mae" / "src"
DECODING = ROOT / "decoding"
for path in (ROOT, SRC, DECODING):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import config  # noqa: E402
from hrf_alignment import response_tr_times, words_in_lagged_window  # noqa: E402
from utils_resp import get_resp  # noqa: E402
from utils_stim import get_story_wordseqs  # noqa: E402

import run_summaries_encoding as rse  # noqa: E402


TRIM_START = 5 + config.TRIM
TRIM_END = config.TRIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create aligned Huth fMRI/text windows.")
    parser.add_argument("--subject", default="S1")
    parser.add_argument("--stories", nargs="+", default=None)
    parser.add_argument("--sessions", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--holdout-stories", nargs="+", default=None)
    parser.add_argument("--holdout-count", type=int, default=5)
    parser.add_argument("--no-story-holdout", action="store_true")
    parser.add_argument("--val-count", type=int, default=3)
    parser.add_argument("--output-dir", default="fmri_text_mae/outputs/windows/S1")
    parser.add_argument("--tr-sec", type=float, default=2.0)
    parser.add_argument("--hrf-lag-sec", type=float, default=4.0)
    parser.add_argument("--fmri-window-len-tr", type=int, default=8)
    parser.add_argument("--stride-tr", type=int, default=2)
    parser.add_argument("--n-voxels", type=int, default=10000)
    parser.add_argument("--voxel-selection", choices=["all", "top_variance"], default="top_variance")
    parser.add_argument("--response-root", default=None)
    parser.add_argument("--local-compute-mode", action="store_true")
    parser.add_argument("--mounted-project-root", default=str(rse.DEFAULT_MOUNTED_PROJECT_ROOT))
    parser.add_argument("--summaries-dir", default=str(rse.LOCAL_DEFAULT_SUMMARIES_DIR))
    parser.add_argument("--ba-dir", default=str(rse.LOCAL_DEFAULT_BA_DIR))
    return parser.parse_args()


def load_story_split(args: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    if args.local_compute_mode:
        rse.configure_local_compute_mode(args)
    stories = rse.load_story_list(args)
    train_plus_val, test = rse.split_story_list(stories, args)
    if len(train_plus_val) <= args.val_count:
        raise ValueError("Not enough training stories to reserve validation stories.")
    train = train_plus_val[:-args.val_count]
    val = train_plus_val[-args.val_count:]
    return train, val, test


def select_voxels(subject: str, train_stories: list[str], n_voxels: int, response_root: str | None, method: str) -> np.ndarray | None:
    if method == "all" or n_voxels <= 0:
        return None
    resp = get_resp(subject, train_stories, stack=True, vox=None, response_root=response_root).astype(np.float32)
    var = np.nan_to_num(resp.var(axis=0))
    keep = min(int(n_voxels), var.shape[0])
    return np.sort(np.argsort(var)[-keep:]).astype(np.int64)


def fit_train_zscore(resp_by_story: dict[str, np.ndarray], train_stories: list[str]) -> tuple[np.ndarray, np.ndarray]:
    train = np.vstack([resp_by_story[s] for s in train_stories]).astype(np.float32)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def make_examples(
    stories: list[str],
    resp_by_story: dict[str, np.ndarray],
    wordseqs: dict,
    mean: np.ndarray,
    std: np.ndarray,
    tr_sec: float,
    hrf_lag_sec: float,
    fmri_window_len_tr: int,
    stride_tr: int,
) -> dict[str, np.ndarray]:
    bold, text, story_ids, start_trs = [], [], [], []
    for story in stories:
        resp = np.nan_to_num((resp_by_story[story].astype(np.float32) - mean) / std)
        tr_times = response_tr_times(wordseqs[story], resp.shape[0], tr_sec, TRIM_START, TRIM_END)
        max_start = resp.shape[0] - fmri_window_len_tr
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, stride_tr):
            end = start + fmri_window_len_tr
            start_sec = float(tr_times[start])
            end_sec = float(tr_times[end - 1] + tr_sec)
            words = words_in_lagged_window(wordseqs[story], start_sec, end_sec, hrf_lag_sec)
            if not words:
                continue
            bold.append(resp[start:end])
            text.append(words)
            story_ids.append(story)
            start_trs.append(start)
    return {
        "bold": np.asarray(bold, dtype=np.float32),
        "text": np.asarray(text, dtype=object),
        "story": np.asarray(story_ids, dtype=object),
        "start_tr": np.asarray(start_trs, dtype=np.int64),
    }


def save_split(path: Path, data: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **data)
    print(f"wrote {path} with {len(data['text'])} windows")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = load_story_split(args)
    all_stories = train + val + test
    vox = select_voxels(args.subject, train, args.n_voxels, args.response_root, args.voxel_selection)
    resp_by_story = get_resp(args.subject, all_stories, stack=False, vox=vox, response_root=args.response_root)
    wordseqs = get_story_wordseqs(all_stories)
    mean, std = fit_train_zscore(resp_by_story, train)

    common = dict(
        resp_by_story=resp_by_story,
        wordseqs=wordseqs,
        mean=mean,
        std=std,
        tr_sec=args.tr_sec,
        hrf_lag_sec=args.hrf_lag_sec,
        fmri_window_len_tr=args.fmri_window_len_tr,
        stride_tr=args.stride_tr,
    )
    save_split(output_dir / "train_windows.npz", make_examples(train, **common))
    save_split(output_dir / "val_windows.npz", make_examples(val, **common))
    save_split(output_dir / "test_windows.npz", make_examples(test, **common))

    metadata = {
        "subject": args.subject,
        "train_stories": train,
        "val_stories": val,
        "test_stories": test,
        "tr_sec": args.tr_sec,
        "hrf_lag_sec": args.hrf_lag_sec,
        "fmri_window_len_tr": args.fmri_window_len_tr,
        "stride_tr": args.stride_tr,
        "n_features": int(mean.shape[0]),
        "voxel_selection": args.voxel_selection,
        "n_voxels_requested": args.n_voxels,
    }
    if vox is not None:
        np.save(output_dir / "voxel_indices.npy", vox)
    np.savez_compressed(output_dir / "normalization.npz", mean=mean, std=std)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
