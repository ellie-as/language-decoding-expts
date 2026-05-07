#!/usr/bin/env python3
"""Encode Podcast ECoG high-gamma responses from word-level language-model surprisal.

Surprisal is computed as -log p(current token | previous tokens), pooled to one
value per word, then evaluated as a full-lag encoding feature. The output tables
show which channels and paper ROIs track surprisal/perplexity.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from compare_encoding_models import corr_per_target  # noqa: E402
from run_llm_context_sweep import load_token_table, resolve_device, resolve_dtype, safe_name  # noqa: E402
from run_text_window_encoding import DEFAULT_MOUNT_ROOT, load_epochs  # noqa: E402


ROI_ORDER = ["EAC", "STG", "IFG", "PRC", "MFG", "TMP"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mounted-root", type=Path, default=DEFAULT_MOUNT_ROOT)
    parser.add_argument("--bids-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "outputs" / "surprisal_encoding")
    parser.add_argument("--subjects", nargs="+", default=[f"{i:02d}" for i in range(1, 10)])
    parser.add_argument("--task", default="podcast")
    parser.add_argument("--picks-regex", default=".*")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--context-tokens", type=int, default=64)
    parser.add_argument(
        "--feature",
        choices=["surprisal_sum", "surprisal_mean", "perplexity"],
        default="surprisal_sum",
        help="Word-level scalar used for encoding. surprisal_sum is usually the cleanest word-level measure.",
    )
    parser.add_argument("--tmin", type=float, default=-2.0)
    parser.add_argument("--tmax", type=float, default=2.0)
    parser.add_argument("--resample-sfreq", type=float, default=32.0)
    parser.add_argument("--outer-splits", type=int, default=2)
    parser.add_argument("--ridge-alpha", type=float, default=100000.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--max-channels", type=int, default=None)
    parser.add_argument(
        "--roi-metrics",
        type=Path,
        default=THIS_DIR / "outputs" / "gpt2_paper_roi_context_layer_exact" / "channel_paper_roi_metrics.csv",
        help="Existing channel table with paper_roi assignments.",
    )
    parser.add_argument("--force-surprisal", action="store_true")
    return parser.parse_args()


def bids_root(args: argparse.Namespace) -> Path:
    return args.bids_root or args.mounted_root / "podcast_ecog" / "data" / "ds005574"


def fif_path(root: Path, subject: str, task: str) -> Path:
    return (
        root
        / "derivatives"
        / "ecogprep"
        / f"sub-{subject}"
        / "ieeg"
        / f"sub-{subject}_task-{task}_desc-highgamma_ieeg.fif"
    )


def surprisal_cache_path(args: argparse.Namespace) -> Path:
    word_tag = f"maxwords{args.max_words}" if args.max_words is not None else "allwords"
    return (
        args.output_dir
        / "features"
        / safe_name(args.model)
        / f"ctx{int(args.context_tokens)}"
        / f"word_surprisal__{word_tag}.csv"
    )


def compute_word_surprisal(args: argparse.Namespace, token_table: pd.DataFrame, model, tokenizer, device: str) -> pd.DataFrame:
    cache_path = surprisal_cache_path(args)
    if cache_path.is_file() and not args.force_surprisal:
        print(f"Loading surprisal cache: {cache_path}", flush=True)
        return pd.read_csv(cache_path)

    if args.max_words is not None:
        token_table = token_table[token_table["word_idx"] < int(args.max_words)].copy()
    token_ids = token_table["token_id"].to_numpy(dtype=np.int64)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    seq_len = int(args.context_tokens) + 2

    input_ids = np.full((len(token_ids), seq_len), int(pad_id), dtype=np.int64)
    attention_mask = np.zeros_like(input_ids, dtype=np.int64)
    pred_pos = np.zeros(len(token_ids), dtype=np.int64)
    for i, token_id in enumerate(token_ids):
        prev = token_ids[max(0, i - int(args.context_tokens)) : i]
        if len(prev) == 0:
            seq = np.asarray([eos_id, token_id], dtype=np.int64)
        else:
            seq = np.concatenate([prev, np.asarray([token_id], dtype=np.int64)])
        start = seq_len - len(seq)
        input_ids[i, start:] = seq
        attention_mask[i, start:] = 1
        pred_pos[i] = seq_len - 2

    nll = np.empty(len(token_ids), dtype=np.float32)
    ranks = np.empty(len(token_ids), dtype=np.int32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(token_ids), int(args.batch_size)):
            stop = min(start + int(args.batch_size), len(token_ids))
            batch_ids = torch.tensor(input_ids[start:stop], dtype=torch.long, device=device)
            batch_mask = torch.tensor(attention_mask[start:stop], dtype=torch.long, device=device)
            output = model(batch_ids, attention_mask=batch_mask)
            pos = torch.tensor(pred_pos[start:stop], dtype=torch.long, device=device)
            logits = output.logits[torch.arange(stop - start, device=device), pos]
            targets = torch.tensor(token_ids[start:stop], dtype=torch.long, device=device)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            nll[start:stop] = (-log_probs[torch.arange(stop - start, device=device), targets]).detach().cpu().numpy()
            target_logits = logits[torch.arange(stop - start, device=device), targets]
            ranks[start:stop] = (logits > target_logits[:, None]).sum(dim=1).detach().cpu().numpy().astype(np.int32)
            del output, batch_ids, batch_mask, logits, log_probs

    token_level = token_table[["word_idx", "word", "start", "end"]].copy()
    token_level["token_nll"] = nll
    token_level["token_rank"] = ranks
    words = (
        token_level.groupby("word_idx", sort=True)
        .agg(
            word=("word", "first"),
            start=("start", "first"),
            end=("end", "last"),
            n_tokens=("token_nll", "count"),
            surprisal_sum=("token_nll", "sum"),
            surprisal_mean=("token_nll", "mean"),
            mean_rank=("token_rank", "mean"),
            top1_any_token=("token_rank", lambda x: float((x == 0).mean())),
        )
        .reset_index(drop=True)
    )
    words["perplexity"] = np.exp(words["surprisal_mean"].clip(upper=50))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    words.to_csv(cache_path, index=False)
    print(f"Saved surprisal cache: {cache_path}", flush=True)
    print(f"Token top-1 accuracy: {(ranks == 0).mean() * 100:.2f}%", flush=True)
    return words


def ridge_predict_svd(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    u, s, vt = np.linalg.svd(x_train, full_matrices=False)
    scale = (s / (s * s + float(alpha))).astype(np.float32)
    coef = vt.T @ (scale[:, None] * (u.T @ y_train))
    return (x_test @ coef).astype(np.float32)


def evaluate_full_lag(x: np.ndarray, epoch_data: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    y = epoch_data.reshape(len(epoch_data), -1).astype(np.float32)
    fold_corrs = []
    cv = KFold(args.outer_splits, shuffle=False)
    for fold, (train_idx, test_idx) in enumerate(cv.split(x), start=1):
        print(f"    Fold {fold}/{args.outer_splits}", flush=True)
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x[train_idx]).astype(np.float32)
        x_test = x_scaler.transform(x[test_idx]).astype(np.float32)
        y_train = y_scaler.fit_transform(y[train_idx]).astype(np.float32)
        y_test = y_scaler.transform(y[test_idx]).astype(np.float32)
        pred = ridge_predict_svd(x_train, y_train, x_test, args.ridge_alpha)
        fold_corrs.append(corr_per_target(y_test, pred).reshape(epoch_data.shape[1:]))
    return np.stack(fold_corrs).astype(np.float32)


def load_roi_lookup(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing ROI metrics file: {path}. Run analyze_paper_roi_context_layer.py first, "
            "or pass --roi-metrics to an existing channel_paper_roi_metrics.csv."
        )
    cols = ["subject", "channel", "paper_roi", "destrieux_label"]
    return pd.read_csv(path)[cols].drop_duplicates(["subject", "channel"])


def append_roi(channel_scores: pd.DataFrame, roi_metrics: pd.DataFrame) -> pd.DataFrame:
    merged = channel_scores.merge(roi_metrics, on=["subject", "channel"], how="left")
    merged["paper_roi"] = merged["paper_roi"].fillna("Other")
    return merged


def save_plots(channel_scores: pd.DataFrame, output_dir: Path, feature: str) -> None:
    roi_df = channel_scores[channel_scores["paper_roi"].isin(ROI_ORDER)].copy()
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    data = [roi_df.loc[roi_df["paper_roi"] == roi, "best_r"].dropna().to_numpy() for roi in ROI_ORDER]
    ax.boxplot(data, labels=ROI_ORDER, showmeans=True, showfliers=False)
    ax.axhline(0, color="0.8", lw=1)
    ax.set_ylabel("best surprisal encoding r")
    ax.set_title(feature)
    fig.savefig(output_dir / "surprisal_encoding_r_by_roi_boxplot.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True, constrained_layout=True)
    bins = np.linspace(float(channel_scores["preferred_lag_s"].min()), float(channel_scores["preferred_lag_s"].max()), 33)
    for ax, roi in zip(axes.ravel(), ROI_ORDER):
        sub = roi_df[roi_df["paper_roi"] == roi]
        ax.hist(sub["preferred_lag_s"], bins=bins, color="#4477aa", alpha=0.85)
        ax.axvline(0, color="0.35", lw=1)
        ax.set_title(f"{roi} n={len(sub)}")
        ax.set_xlabel("best surprisal lag (s)")
        ax.set_ylabel("channels")
    fig.savefig(output_dir / "surprisal_preferred_lag_histograms_by_roi.png", dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = bids_root(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    print(f"Loading {args.model} on {device} dtype={dtype}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=args.local_files_only, torch_dtype=dtype).to(device)
    token_table = load_token_table(root, tokenizer)
    words = compute_word_surprisal(args, token_table, model, tokenizer, device)
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.max_words is not None:
        words = words.iloc[: args.max_words].reset_index(drop=True)
    x_all = words[[args.feature]].to_numpy(dtype=np.float32)

    roi_metrics = load_roi_lookup(args.roi_metrics)
    summary_rows = []
    channel_rows = []
    corr_payloads = {}
    for subject in args.subjects:
        t0 = time.time()
        print(f"=== sub-{subject} ===", flush=True)
        epochs = load_epochs(
            fif_path=fif_path(root, subject, args.task),
            words=words,
            picks_regex=args.picks_regex,
            max_channels=args.max_channels,
            tmin=args.tmin,
            tmax=args.tmax,
            resample_sfreq=args.resample_sfreq,
        )
        selection = epochs.selection.astype(np.int64)
        x = x_all[selection]
        epoch_data = epochs.get_data(copy=True).astype(np.float32)
        corrs = evaluate_full_lag(x, epoch_data, args)
        mean_corrs = corrs.mean(axis=0)
        best_idx = mean_corrs.argmax(axis=-1)
        best_r = mean_corrs[np.arange(mean_corrs.shape[0]), best_idx]
        preferred_lag_s = epochs.times[best_idx]
        channel_names = np.asarray(epochs.info["ch_names"]).astype(str)
        coords = np.vstack([ch["loc"][:3] for ch in epochs.info["chs"]]).astype(float) * 1000.0
        corr_payloads[f"sub{subject}"] = corrs

        for i, channel in enumerate(channel_names):
            channel_rows.append(
                {
                    "subject": f"sub-{subject}",
                    "channel": str(channel),
                    "feature": args.feature,
                    "best_r": float(best_r[i]),
                    "preferred_lag_s": float(preferred_lag_s[i]),
                    "x": float(coords[i, 0]),
                    "y": float(coords[i, 1]),
                    "z": float(coords[i, 2]),
                }
            )
        summary_rows.append(
            {
                "subject": f"sub-{subject}",
                "n_channels": int(len(channel_names)),
                "mean_best_r": float(np.mean(best_r)),
                "median_best_r": float(np.median(best_r)),
                "q75_best_r": float(np.quantile(best_r, 0.75)),
                "max_best_r": float(np.max(best_r)),
                "positive_share": float((best_r > 0).mean()),
            }
        )
        print(f"sub-{subject} mean best r={np.mean(best_r):.4f}; done in {(time.time() - t0) / 60:.1f} min", flush=True)

    channel_scores = append_roi(pd.DataFrame(channel_rows), roi_metrics)
    channel_scores.to_csv(args.output_dir / "surprisal_channel_scores.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "surprisal_subject_summary.csv", index=False)
    roi_summary = (
        channel_scores[channel_scores["paper_roi"].isin(ROI_ORDER)]
        .groupby("paper_roi")
        .agg(
            n_channels=("channel", "count"),
            n_subjects=("subject", "nunique"),
            mean_best_r=("best_r", "mean"),
            median_best_r=("best_r", "median"),
            q75_best_r=("best_r", lambda x: float(np.quantile(x, 0.75))),
            max_best_r=("best_r", "max"),
            positive_share=("best_r", lambda x: float((x > 0).mean())),
            median_preferred_lag_s=("preferred_lag_s", "median"),
        )
        .reindex(ROI_ORDER)
    )
    roi_summary.to_csv(args.output_dir / "surprisal_roi_summary.csv")
    np.savez_compressed(args.output_dir / "surprisal_full_lag_corrs.npz", **corr_payloads)
    save_plots(channel_scores, args.output_dir, args.feature)

    print("\nROI summary", flush=True)
    print(roi_summary.round(4).to_string(), flush=True)
    print(f"\nSaved surprisal encoding outputs to: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
