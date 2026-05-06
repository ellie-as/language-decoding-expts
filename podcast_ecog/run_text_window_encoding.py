#!/usr/bin/env python3
"""Compare podcast transcript text-window embeddings as ECoG encoding features."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
from nilearn.plotting import plot_markers
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold


DEFAULT_MOUNT_ROOT = Path("/Volumes/ellie/language-decoding-expts")


def parse_args() -> argparse.Namespace:
    local_output = Path(__file__).resolve().parent / "outputs" / "text_window_encoding"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mounted-root", type=Path, default=DEFAULT_MOUNT_ROOT)
    parser.add_argument("--bids-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=local_output)
    parser.add_argument("--subject", default="03")
    parser.add_argument("--task", default="podcast")
    parser.add_argument("--picks-regex", default=".*")
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[1, 5, 10, 20, 50, 100, 200, 500])
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--normalize-embeddings", action="store_true")
    parser.add_argument("--tmin", type=float, default=-2.0)
    parser.add_argument("--tmax", type=float, default=2.0)
    parser.add_argument("--resample-sfreq", type=float, default=32.0)
    parser.add_argument(
        "--target-mode",
        choices=("preferred-lag", "full-lag"),
        default="preferred-lag",
        help="preferred-lag predicts one sample per channel; full-lag predicts channel x lag.",
    )
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=None,
        help="GPT-2 encoding NPZ used to select each channel's preferred lag.",
    )
    parser.add_argument("--outer-splits", type=int, default=2)
    parser.add_argument("--alphas", nargs="+", type=float, default=[100.0, 1000.0, 10000.0, 100000.0])
    parser.add_argument("--alpha-tune-frac", type=float, default=0.2)
    parser.add_argument(
        "--alpha-tune-targets",
        type=int,
        default=256,
        help="Number of target columns used to choose alpha. Use 0 to skip tuning.",
    )
    parser.add_argument("--ridge-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--max-channels", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    bids_root = args.bids_root or args.mounted_root / "podcast_ecog" / "data" / "ds005574"
    transcript_path = bids_root / "stimuli" / "gpt2-xl" / "transcript.tsv"
    fif_path = (
        bids_root
        / "derivatives"
        / "ecogprep"
        / f"sub-{args.subject}"
        / "ieeg"
        / f"sub-{args.subject}_task-{args.task}_desc-highgamma_ieeg.fif"
    )
    reference_path = args.reference_results or (
        args.mounted_root
        / "podcast_ecog"
        / "outputs_all_channels"
        / f"sub-{args.subject}_gpt2-xl_layer-24_encoding_results.npz"
    )
    for path in [transcript_path, fif_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.target_mode == "preferred-lag" and not reference_path.is_file():
        raise FileNotFoundError(f"Missing reference result for preferred lags: {reference_path}")
    return transcript_path, fif_path, reference_path


def load_words(transcript_path: Path) -> pd.DataFrame:
    transcript = pd.read_csv(transcript_path, sep="\t", index_col=0)
    words = (
        transcript.groupby("word_idx", sort=True)
        .agg(word=("word", "first"), start=("start", "first"), end=("end", "last"))
        .sort_values("start")
        .reset_index(drop=True)
    )
    words["word"] = words["word"].astype(str)
    return words


def make_text_windows(word_values: list[str], window_size: int) -> list[str]:
    windows = []
    for i in range(len(word_values)):
        start = max(0, i - int(window_size) + 1)
        windows.append(" ".join(w.strip() for w in word_values[start : i + 1] if w.strip()))
    return windows


def embedding_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cache_stem(model_name: str, window_size: int, words: pd.DataFrame, normalize: bool) -> str:
    word_digest = hashlib.sha1("\n".join(words["word"].astype(str).tolist()).encode("utf-8")).hexdigest()[:12]
    payload = {
        "model": model_name,
        "window_size": int(window_size),
        "n_words": int(len(words)),
        "first_start": float(words["start"].iloc[0]),
        "last_end": float(words["end"].iloc[-1]),
        "word_digest": word_digest,
        "normalize": bool(normalize),
        "version": 2,
    }
    key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name).strip("-")
    return f"text_window_w{window_size}__{safe_model}__{key}.npz"


def load_or_embed_windows(
    *,
    model: SentenceTransformer,
    model_name: str,
    words: pd.DataFrame,
    window_size: int,
    cache_dir: Path,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_stem(model_name, window_size, words, normalize)
    if cache_path.is_file():
        print(f"Loading cached embeddings: {cache_path}", flush=True)
        return np.load(cache_path)["embeddings"].astype(np.float32)

    texts = make_text_windows(words["word"].tolist(), window_size)
    print(f"Embedding {len(texts)} text windows, size={window_size}", flush=True)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    ).astype(np.float32)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        window_size=np.array(window_size),
        model=np.array(model_name),
        normalize_embeddings=np.array(normalize),
    )
    print(f"Saved embedding cache: {cache_path}", flush=True)
    return embeddings


def load_epochs(
    *,
    fif_path: Path,
    words: pd.DataFrame,
    picks_regex: str,
    max_channels: int | None,
    tmin: float,
    tmax: float,
    resample_sfreq: float,
) -> mne.Epochs:
    print(f"Reading high-gamma FIF from: {fif_path}", flush=True)
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    picks = mne.pick_channels_regexp(raw.ch_names, picks_regex)
    if max_channels is not None:
        picks = picks[:max_channels]
    if len(picks) == 0:
        raise ValueError(f"No channels matched {picks_regex!r}.")
    raw.pick(picks)
    print(f"Picked {len(raw.ch_names)} channels at {raw.info['sfreq']} Hz.", flush=True)
    raw.resample(sfreq=resample_sfreq, npad="auto", window="hamming", verbose=False)
    print(f"Resampled raw to {raw.info['sfreq']} Hz.", flush=True)

    events = np.zeros((len(words), 3), dtype=int)
    events[:, 0] = raw.time_as_index(words["start"].to_numpy(dtype=float), use_rounding=True) + raw.first_samp
    epochs = mne.Epochs(
        raw,
        events,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        proj=False,
        event_id=None,
        preload=True,
        event_repeated="merge",
        verbose=True,
    )
    print(f"Epoch data shape: {epochs.get_data(copy=False).shape}", flush=True)
    return epochs


def load_preferred_lag_targets(reference_path: Path, epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(reference_path, allow_pickle=True) as ref:
        channel_names = ref["channel_names"].astype(str)
        lags = ref["lags"]
        ref_corrs = ref["corrs"].mean(axis=0)
    ref_lookup = {channel: i for i, channel in enumerate(channel_names)}
    missing = [channel for channel in epochs.info["ch_names"] if channel not in ref_lookup]
    if missing:
        raise ValueError(f"Reference results are missing {len(missing)} epoch channels, e.g. {missing[:5]}.")
    ref_indices = np.asarray([ref_lookup[channel] for channel in epochs.info["ch_names"]], dtype=int)
    ref_corrs = ref_corrs[ref_indices]
    preferred_ref_idx = ref_corrs.argmax(axis=-1)
    preferred_lag_s = lags[preferred_ref_idx]
    epoch_idx = np.asarray([int(np.argmin(np.abs(epochs.times - lag))) for lag in preferred_lag_s], dtype=int)

    data = epochs.get_data(copy=True)
    targets = data[:, np.arange(data.shape[1]), epoch_idx].astype(np.float32)
    return targets, preferred_lag_s.astype(np.float32), epoch_idx


def standardize_train_test(
    train: np.ndarray,
    test: np.ndarray,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return ((train - mean) / std).astype(dtype), ((test - mean) / std).astype(dtype), mean.astype(dtype), std.astype(dtype)


def ridge_predict_svd(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    u, s, vt = np.linalg.svd(x_train, full_matrices=False)
    scale = (s / (s * s + float(alpha))).astype(x_train.dtype, copy=False)
    uy = u.T @ y_train
    coef = vt.T @ (scale[:, None] * uy)
    return x_test @ coef


def corr_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(yt * yt, axis=0) * np.sum(yp * yp, axis=0))
    out = np.divide(np.sum(yt * yp, axis=0), denom, out=np.zeros(y_true.shape[1], dtype=np.float32), where=denom > 0)
    return out.astype(np.float32)


def choose_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alphas: list[float],
    tune_frac: float,
    tune_targets: int,
    rng: np.random.Generator,
) -> float:
    if tune_targets == 0 or len(alphas) == 1:
        return float(alphas[0])
    n_val = max(20, int(round(len(x_train) * float(tune_frac))))
    n_val = min(n_val, len(x_train) // 2)
    train_idx = np.arange(0, len(x_train) - n_val)
    val_idx = np.arange(len(x_train) - n_val, len(x_train))
    if train_idx.size < 20:
        return float(alphas[0])

    targets = rng.choice(y_train.shape[1], size=min(int(tune_targets), y_train.shape[1]), replace=False)
    scores = []
    for alpha in alphas:
        pred = ridge_predict_svd(x_train[train_idx], y_train[train_idx][:, targets], x_train[val_idx], alpha)
        scores.append(float(np.nanmean(corr_per_target(y_train[val_idx][:, targets], pred))))
    best = int(np.nanargmax(scores))
    print("  alpha tune:", {str(a): round(s, 5) for a, s in zip(alphas, scores)}, "->", alphas[best], flush=True)
    return float(alphas[best])


def fit_window_encoding(
    x: np.ndarray,
    y: np.ndarray,
    *,
    outer_splits: int,
    alphas: list[float],
    tune_frac: float,
    tune_targets: int,
    dtype_name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    dtype = np.float32 if dtype_name == "float32" else np.float64
    corrs = []
    chosen_alphas = []
    rng = np.random.default_rng(seed)
    for fold, (train_idx, test_idx) in enumerate(KFold(outer_splits, shuffle=False).split(x), start=1):
        print(f"  outer fold {fold}/{outer_splits}", flush=True)
        x_train, x_test, _, _ = standardize_train_test(x[train_idx], x[test_idx], dtype)
        y_train, y_test, _, _ = standardize_train_test(y[train_idx], y[test_idx], dtype)
        alpha = choose_alpha(x_train, y_train, alphas, tune_frac, tune_targets, rng)
        pred = ridge_predict_svd(x_train, y_train, x_test, alpha)
        corrs.append(corr_per_target(y_test, pred))
        chosen_alphas.append(alpha)
    return np.stack(corrs), np.asarray(chosen_alphas, dtype=np.float32)


def channel_prefix(channel: str) -> str:
    return re.sub(r"\d+$", "", channel)


def channel_family(prefix: str) -> str:
    if prefix in {"LGA", "LGB"}:
        return "left_grid"
    if prefix.startswith("D"):
        return "depth"
    if prefix.endswith("OF") or prefix.endswith("F"):
        return "frontal_orbitofrontal"
    if prefix.endswith("AT") or prefix.endswith("MT") or prefix.endswith("PT"):
        return "temporal_strip"
    if prefix.endswith("O"):
        return "occipital"
    if prefix.endswith("P"):
        return "parietal"
    return "other"


def channel_hemisphere(prefix: str) -> str:
    if prefix.startswith("DL"):
        return "left"
    if prefix.startswith("DR"):
        return "right"
    if prefix.startswith("L"):
        return "left"
    if prefix.startswith("R"):
        return "right"
    return "unknown"


def summarize_and_plot(
    *,
    output_dir: Path,
    window_sizes: np.ndarray,
    corrs: np.ndarray,
    channel_names: np.ndarray,
    coords: np.ndarray,
    preferred_lag_s: np.ndarray | None,
    model_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mean_corr = corrs.mean(axis=1)
    best_window_idx = mean_corr.argmax(axis=0)
    best_window = window_sizes[best_window_idx]
    best_corr = mean_corr[best_window_idx, np.arange(mean_corr.shape[1])]

    rows = []
    for ch_i, channel in enumerate(channel_names):
        prefix = channel_prefix(str(channel))
        row = {
            "channel": str(channel),
            "prefix": prefix,
            "family": channel_family(prefix),
            "hemisphere": channel_hemisphere(prefix),
            "preferred_window_words": int(best_window[ch_i]),
            "best_corr": float(best_corr[ch_i]),
        }
        if preferred_lag_s is not None:
            row["gpt2_preferred_lag_s"] = float(preferred_lag_s[ch_i])
        for w_i, window in enumerate(window_sizes):
            row[f"corr_w{int(window)}"] = float(mean_corr[w_i, ch_i])
        rows.append(row)
    channel_df = pd.DataFrame(rows)
    channel_csv = output_dir / "channel_text_window_scores.csv"
    channel_df.to_csv(channel_csv, index=False)
    print(f"Saved channel table: {channel_csv}", flush=True)

    group_rows = []
    for group_col in ["family", "prefix", "hemisphere"]:
        for group, group_df in channel_df.groupby(group_col, sort=True):
            if len(group_df) < 3:
                continue
            row = {
                "group_type": group_col,
                "group": group,
                "n_channels": int(len(group_df)),
                "mean_best_corr": float(group_df["best_corr"].mean()),
                "median_preferred_window_words": float(group_df["preferred_window_words"].median()),
                "short_window_share_le10": float((group_df["preferred_window_words"] <= 10).mean()),
                "long_window_share_ge100": float((group_df["preferred_window_words"] >= 100).mean()),
            }
            for window in window_sizes:
                row[f"mean_corr_w{int(window)}"] = float(group_df[f"corr_w{int(window)}"].mean())
            group_rows.append(row)
    group_df = pd.DataFrame(group_rows)
    group_csv = output_dir / "group_text_window_summary.csv"
    group_df.to_csv(group_csv, index=False)
    print(f"Saved group table: {group_csv}", flush=True)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(window_sizes, mean_corr.mean(axis=1), marker="o", color="black", label="all channels")
    for family, family_df in channel_df.groupby("family", sort=True):
        if len(family_df) < 4:
            continue
        vals = [family_df[f"corr_w{int(window)}"].mean() for window in window_sizes]
        ax.plot(window_sizes, vals, marker="o", lw=1.3, alpha=0.85, label=family)
    ax.set_xscale("log")
    ax.set_xlabel("trailing text window (words)")
    ax.set_ylabel("encoding correlation r")
    ax.set_title(f"Text-window encoding by context horizon\n{model_name}")
    ax.legend(fontsize=7, ncols=2)
    line_plot = output_dir / "text_window_encoding_by_group.png"
    fig.savefig(line_plot, dpi=220)
    plt.close(fig)
    print(f"Saved group plot: {line_plot}", flush=True)

    log_windows = np.log10(best_window.astype(float))
    order = np.argsort(log_windows)
    display = plot_markers(
        log_windows[order],
        coords[order],
        node_size=35,
        display_mode="lzr",
        node_vmin=float(np.log10(window_sizes.min())),
        node_vmax=float(np.log10(window_sizes.max())),
        node_cmap="viridis",
        colorbar=True,
    )
    preferred_plot = output_dir / "preferred_text_window_brain_log10_words.png"
    display.savefig(preferred_plot, dpi=250)
    display.close()
    print(f"Saved preferred-window brain plot: {preferred_plot}", flush=True)

    order = np.argsort(best_corr)
    display = plot_markers(
        best_corr[order],
        coords[order],
        node_size=35,
        display_mode="lzr",
        node_vmin=0,
        node_cmap="inferno_r",
        colorbar=True,
    )
    best_plot = output_dir / "best_text_window_corr_brain.png"
    display.savefig(best_plot, dpi=250)
    display.close()
    print(f"Saved best-correlation brain plot: {best_plot}", flush=True)


def main() -> int:
    args = parse_args()
    transcript_path, fif_path, reference_path = resolve_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or args.output_dir / "cache"

    words = load_words(transcript_path)
    if args.max_words is not None:
        words = words.iloc[: args.max_words].reset_index(drop=True)
    print(f"Loaded {len(words)} transcript words.", flush=True)

    device = embedding_device(args.embedding_device)
    print(f"Loading sentence encoder {args.embedding_model!r} on {device}.", flush=True)
    encoder = SentenceTransformer(args.embedding_model, device=device)

    embeddings_by_window = []
    for window_size in args.window_sizes:
        embeddings_by_window.append(
            load_or_embed_windows(
                model=encoder,
                model_name=args.embedding_model,
                words=words,
                window_size=int(window_size),
                cache_dir=cache_dir,
                batch_size=args.embed_batch_size,
                normalize=args.normalize_embeddings,
            )
        )

    epochs = load_epochs(
        fif_path=fif_path,
        words=words,
        picks_regex=args.picks_regex,
        max_channels=args.max_channels,
        tmin=args.tmin,
        tmax=args.tmax,
        resample_sfreq=args.resample_sfreq,
    )

    selection = epochs.selection
    if args.max_words is not None:
        selection = selection[selection < len(words)]

    preferred_lag_s = None
    preferred_epoch_idx = None
    if args.target_mode == "preferred-lag":
        targets, preferred_lag_s, preferred_epoch_idx = load_preferred_lag_targets(reference_path, epochs)
        target_shape = np.asarray([len(epochs.info["ch_names"])], dtype=int)
    else:
        data = epochs.get_data(copy=True).astype(np.float32)
        target_shape = np.asarray(data.shape[1:], dtype=int)
        targets = data.reshape(len(epochs), -1)

    channel_names = np.asarray(epochs.info["ch_names"]).astype(str)
    coords = np.vstack([ch["loc"][:3] for ch in epochs.info["chs"]]).astype(float) * 1000.0
    print(f"Targets: {targets.shape}; channels: {len(channel_names)}.", flush=True)

    all_corrs = []
    all_alphas = []
    for w_i, (window_size, embeddings) in enumerate(zip(args.window_sizes, embeddings_by_window), start=1):
        print(f"Fitting window {w_i}/{len(args.window_sizes)}: {window_size} words", flush=True)
        x = embeddings[selection].astype(np.float32)
        y = targets.astype(np.float32)
        corrs, chosen_alphas = fit_window_encoding(
            x,
            y,
            outer_splits=args.outer_splits,
            alphas=[float(a) for a in args.alphas],
            tune_frac=args.alpha_tune_frac,
            tune_targets=args.alpha_tune_targets,
            dtype_name=args.ridge_dtype,
            seed=args.seed + int(window_size),
        )
        all_corrs.append(corrs)
        all_alphas.append(chosen_alphas)

    window_sizes = np.asarray(args.window_sizes, dtype=int)
    corrs = np.stack(all_corrs)
    chosen_alphas = np.stack(all_alphas)

    result_path = args.output_dir / "text_window_encoding_results.npz"
    np.savez_compressed(
        result_path,
        corrs=corrs,
        chosen_alphas=chosen_alphas,
        window_sizes=window_sizes,
        channel_names=channel_names,
        coords=coords,
        target_shape=target_shape,
        target_mode=np.array(args.target_mode),
        preferred_lag_s=np.asarray([] if preferred_lag_s is None else preferred_lag_s),
        preferred_epoch_idx=np.asarray([] if preferred_epoch_idx is None else preferred_epoch_idx),
        embedding_model=np.array(args.embedding_model),
        normalize_embeddings=np.array(args.normalize_embeddings),
        picks_regex=np.array(args.picks_regex),
        reference_results=np.array(str(reference_path)),
    )
    print(f"Saved result NPZ: {result_path}", flush=True)

    if args.target_mode == "preferred-lag":
        summarize_and_plot(
            output_dir=args.output_dir,
            window_sizes=window_sizes,
            corrs=corrs,
            channel_names=channel_names,
            coords=coords,
            preferred_lag_s=preferred_lag_s,
            model_name=args.embedding_model,
        )
    else:
        print("Skipping channel summary plots for full-lag mode; run preferred-lag mode for channel-level plots.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
