#!/usr/bin/env python3
"""Export a decoder-compatible MiniLM combo ridge encoding model.

This bridges the lag-preference combo ridge experiment into
``gpt1_encoding_comparison/decode_and_score.py``. The training experiment fits
``X_combo[t] -> response[t + lag]``. The Huth decoder's ``StimulusModel``
constructs delayed features, so the exported weight matrix is zero everywhere
except the requested lag slice.

The decoder feature extractor cannot use oracle GPT summaries of the held-out
story. It instead uses MiniLM embeddings of candidate text windows
(``local/current``, last 20, last 50, last 200 words), matching the exported
feature dimensionality.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))
sys.path.insert(0, str(REPO_DIR / "mindeye_text"))
sys.path.insert(0, str(THIS_DIR))

import config  # noqa: E402
import run_summaries_encoding as rse  # noqa: E402
from _shared import load_or_build_chunk_embeddings  # noqa: E402
from train_lag_encoding import configure_data_root, load_stories, split_stories, stack_lag  # noqa: E402
from train_summary_combo_encoding import build_combo_embeddings, load_or_build_summary_embeddings  # noqa: E402
from utils_resp import get_resp  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--combo-results-dir", required=True, help="Directory with combo lag_corrs.npz/config.json.")
    p.add_argument("--subject", required=True)
    p.add_argument("--lag", type=int, default=2)
    p.add_argument("--alpha", type=float, default=100000.0)
    p.add_argument("--voxel-count", type=int, default=2000)
    p.add_argument("--sessions", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    p.add_argument("--stories", nargs="+", default=None)
    p.add_argument("--val-story-count", type=int, default=8)
    p.add_argument("--val-stories", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summary-horizons", nargs="+", type=int, default=None)
    p.add_argument("--summary-model", default=None)
    p.add_argument("--summaries-dir", default=str(rse.LOCAL_DEFAULT_SUMMARIES_DIR))
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument("--embedding-cache-dir", default=str(THIS_DIR / "cache"))
    p.add_argument("--one-tr-cache-dir", default=str(REPO_DIR / "27-04-expts" / "cache"))
    p.add_argument("--data-root", default=None)
    p.add_argument("--local-compute-mode", action="store_true")
    p.add_argument("--mounted-project-root", default="/Volumes/ellie/language-decoding-expts")
    p.add_argument("--local-cache-root", default=str(REPO_DIR / "local_compute_cache"))
    p.add_argument("--output", default=None)
    return p.parse_args()


def zscore_fit(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def zscore_apply(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def scalar(value):
    arr = np.asarray(value)
    return arr.item() if arr.shape == () else value


def main():
    args = parse_args()
    combo_dir = Path(args.combo_results_dir).expanduser().resolve()
    saved = np.load(combo_dir / "lag_corrs.npz", allow_pickle=True)
    run_cfg = json.load(open(combo_dir / "config.json", encoding="utf-8"))

    args.summary_horizons = args.summary_horizons or [int(x) for x in saved["summary_horizons"]]
    args.summary_model = args.summary_model or str(scalar(saved["summary_model"]))
    args.embedding_model = run_cfg.get("embedding_model", args.embedding_model)
    args.lags = [int(x) for x in saved["lags"]]
    args.chunk_trs = 1
    args.tag = None
    args.voxel_set = str(scalar(saved["voxel_set"])) if "voxel_set" in saved.files else "full_frontal"
    args.seed = int(run_cfg.get("seed", args.seed))

    mounted_root = configure_data_root(args)
    stories = load_stories(args)
    train_stories, val_stories = split_stories(stories, args)
    if "train_stories" in saved.files and "val_stories" in saved.files:
        train_stories = [str(x) for x in saved["train_stories"]]
        val_stories = [str(x) for x in saved["val_stories"]]

    response_root = config.DATA_TRAIN_DIR
    if args.local_compute_mode and mounted_root is not None:
        response_root = str(
            rse.stage_local_response_cache(
                args.subject,
                stories,
                Path(config.DATA_TRAIN_DIR),
                Path(args.local_cache_root).expanduser().resolve(),
            )
        )

    voxels_all = np.asarray(saved["voxels"], dtype=np.int64)
    lags = [int(x) for x in saved["lags"]]
    li = lags.index(int(args.lag))
    score = np.asarray(saved["corrs"][li], dtype=np.float32)
    order = np.argsort(-score)
    local_sel = np.sort(order[: int(args.voxel_count)])
    voxels = voxels_all[local_sel]

    responses = get_resp(args.subject, stories, stack=False, vox=voxels_all, response_root=response_root)
    responses = {s: arr.astype(np.float32) for s, arr in responses.items()}
    resp_lengths = {s: int(arr.shape[0]) for s, arr in responses.items()}

    max_lag = max(lags)
    one_args = argparse.Namespace(
        subject=args.subject,
        embedding_cache_dir=args.one_tr_cache_dir,
        feature_model="embedding",
        chunk_trs=1,
        lag_trs=max_lag,
        embed_batch_size=args.embed_batch_size,
    )
    one_tr, one_dim, one_cache = load_or_build_chunk_embeddings(
        one_args,
        stories,
        resp_lengths,
        response_root=config.DATA_TRAIN_DIR,
    )
    summary_embs, summary_model, summary_cache = load_or_build_summary_embeddings(args, stories, resp_lengths)
    combo = build_combo_embeddings(one_tr, summary_embs, stories, args.summary_horizons)

    x_train, y_train_all = stack_lag(combo, responses, train_stories, args.lag)
    x_val, y_val_all = stack_lag(combo, responses, val_stories, args.lag)
    y_train = y_train_all[:, local_sel]
    y_val = y_val_all[:, local_sel]

    from sklearn.linear_model import Ridge

    x_mean, x_std = zscore_fit(x_train)
    x_train_z = zscore_apply(x_train, x_mean, x_std)
    x_val_z = zscore_apply(x_val, x_mean, x_std)
    y_mean, y_std = zscore_fit(y_train)
    y_train_z = zscore_apply(y_train, y_mean, y_std)
    y_val_z = zscore_apply(y_val, y_mean, y_std)

    model = Ridge(alpha=float(args.alpha), fit_intercept=False)
    model.fit(x_train_z, y_train_z)
    combo_weights = model.coef_.T.astype(np.float32)  # combo_dim x n_selected
    pred_val = x_val_z @ combo_weights
    residual = (y_val_z - pred_val).astype(np.float32)
    sigma = np.cov(residual, rowvar=False).astype(np.float32)

    combo_dim = combo_weights.shape[0]
    delayed_dim = combo_dim * len(config.STIM_DELAYS)
    weights = np.zeros((delayed_dim, combo_weights.shape[1]), dtype=np.float32)
    delay_idx = list(config.STIM_DELAYS).index(int(args.lag))
    weights[delay_idx * combo_dim : (delay_idx + 1) * combo_dim] = combo_weights

    tr_mean = np.zeros(combo_dim, dtype=np.float32)
    tr_std = np.ones(combo_dim, dtype=np.float32)
    word_mean = x_mean.astype(np.float32)

    out = Path(args.output).expanduser().resolve() if args.output else combo_dir / (
        f"decoder_model_minilm_combo_lag{args.lag}_alpha{int(args.alpha)}_top{len(voxels)}.npz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        weights=weights,
        weights_are_selected=True,
        voxels=voxels,
        noise_model=sigma,
        tr_stats=np.array([tr_mean, tr_std], dtype=object),
        word_stats=np.array([word_mean], dtype=object),
        stories=np.array(train_stories),
        val_stories=np.array(val_stories),
        feature_model="minilm_combo_window_proxy",
        training_feature_model="minilm_combo_summaries",
        embedding_model=args.embedding_model,
        summary_model=summary_model,
        summary_horizons=np.array(args.summary_horizons, dtype=int),
        lag=int(args.lag),
        alpha=float(args.alpha),
        selected_corrs=score[local_sel],
        one_tr_cache=one_cache,
        summary_cache=summary_cache,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
