#!/usr/bin/env python3
"""Cluster-friendly Python port of the Podcast ECoG encoding tutorial."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
from himalaya.backend import get_backend, set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from nilearn.plotting import plot_markers
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OPENNEURO_BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds005574"


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent / "data" / "ds005574"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=Path("podcast_ecog/outputs"))
    parser.add_argument("--subject", default="03")
    parser.add_argument("--task", default="podcast")
    parser.add_argument("--feature-space", default="gpt2-xl")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--picks-regex", default="LG[AB]*")
    parser.add_argument("--tmin", type=float, default=-2.0)
    parser.add_argument("--tmax", type=float, default=2.0)
    parser.add_argument("--resample-sfreq", type=float, default=32.0)
    parser.add_argument("--outer-splits", type=int, default=2)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--alpha-min-exp", type=float, default=1.0)
    parser.add_argument("--alpha-max-exp", type=float, default=10.0)
    parser.add_argument("--n-alphas", type=int, default=10)
    parser.add_argument(
        "--backend",
        choices=("auto", "torch_cuda", "torch", "numpy"),
        default="auto",
        help="Himalaya backend. auto uses torch_cuda when CUDA is available.",
    )
    parser.add_argument("--max-words", type=int, default=None, help="Smoke-test on the first N epochs.")
    parser.add_argument("--max-channels", type=int, default=None, help="Smoke-test on the first N picked channels.")
    parser.add_argument("--download-missing", action="store_true", help="Fetch missing tutorial files from OpenNeuro S3.")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def maybe_download(relative_path: str, bids_root: Path, enabled: bool) -> Path:
    path = bids_root / relative_path
    if path.exists():
        return path
    if not enabled:
        raise FileNotFoundError(f"Missing {path}. Run download_minimal.py or pass --download-missing.")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    url = f"{OPENNEURO_BASE_URL}/{relative_path}"
    print(f"Downloading {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)
    return path


def configure_backend(name: str) -> str:
    if name == "auto":
        name = "torch_cuda" if torch.cuda.is_available() else "numpy"
    if name == "torch_cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to numpy backend.", flush=True)
        name = "numpy"
    set_backend(name)
    return get_backend().__name__


def load_word_embeddings(features_path: Path, transcript_path: Path, layer: int) -> tuple[np.ndarray, pd.DataFrame]:
    with h5py.File(features_path, "r") as f:
        embeddings = f[f"layer-{layer}"][...]
    print(f"Token embedding matrix: {embeddings.shape}", flush=True)

    transcript = pd.read_csv(transcript_path, sep="\t", index_col=0)
    if len(transcript) != len(embeddings):
        raise ValueError(f"Transcript rows ({len(transcript)}) do not match embeddings ({len(embeddings)}).")
    if "rank" in transcript.columns:
        model_acc = (transcript["rank"] == 0).mean()
        print(f"GPT-2 next-token top-1 accuracy in transcript: {model_acc * 100:.3f}%", flush=True)

    aligned = []
    for _, group in transcript.groupby("word_idx", sort=True):
        aligned.append(embeddings[group.index.to_numpy()].mean(axis=0))
    aligned_embeddings = np.stack(aligned)
    words = transcript.groupby("word_idx", sort=True).agg({"word": "first", "start": "first", "end": "last"})
    print(f"Word embedding matrix: {aligned_embeddings.shape}", flush=True)
    return aligned_embeddings, words


def load_epochs(args: argparse.Namespace, words: pd.DataFrame) -> mne.Epochs:
    relative_path = (
        f"derivatives/ecogprep/sub-{args.subject}/ieeg/"
        f"sub-{args.subject}_task-{args.task}_desc-highgamma_ieeg.fif"
    )
    fif_path = maybe_download(relative_path, args.bids_root, args.download_missing)
    raw = mne.io.read_raw_fif(fif_path, verbose=False)

    picks = mne.pick_channels_regexp(raw.ch_names, args.picks_regex)
    if args.max_channels is not None:
        picks = picks[: args.max_channels]
    if len(picks) == 0:
        raise ValueError(f"No channels matched --picks-regex {args.picks_regex!r}.")
    raw = raw.pick(picks)
    print(f"Picked {len(raw.ch_names)} channels at {raw.info['sfreq']} Hz.", flush=True)

    events = np.zeros((len(words), 3), dtype=int)
    events[:, 0] = (words.start.to_numpy() * raw.info["sfreq"]).astype(int)
    epochs = mne.Epochs(
        raw,
        events,
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=None,
        proj=False,
        event_id=None,
        preload=True,
        event_repeated="merge",
        verbose=True,
    )
    print(f"Epochs before resampling: {epochs.get_data(copy=False).shape}", flush=True)
    epochs = epochs.resample(sfreq=args.resample_sfreq, npad="auto", method="fft", window="hamming")
    print(f"Epochs after resampling: {epochs.get_data(copy=False).shape}", flush=True)
    return epochs


def train_encoding(args: argparse.Namespace, x: np.ndarray, y: np.ndarray, epochs_shape: tuple[int, int]) -> np.ndarray:
    alphas = np.logspace(args.alpha_min_exp, args.alpha_max_exp, args.n_alphas)
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, fit_intercept=True, cv=KFold(n_splits=args.inner_splits, shuffle=False)),
    )

    corrs = []
    outer_cv = KFold(args.outer_splits, shuffle=False)
    for fold, (train_index, test_index) in enumerate(outer_cv.split(x), start=1):
        print(f"Outer fold {fold}/{args.outer_splits}", flush=True)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train)
        y_test = y_scaler.transform(y_test)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        corr = correlation_score(y_test, y_pred).reshape(epochs_shape)
        if "torch" in get_backend().__name__:
            corr = corr.numpy(force=True)
        corrs.append(corr)
    return np.stack(corrs)


def save_outputs(
    args: argparse.Namespace,
    corrs: np.ndarray,
    lags: np.ndarray,
    channel_names: list[str],
    coords: np.ndarray,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sub-{args.subject}_{args.feature_space}_layer-{args.layer}"
    npz_path = args.output_dir / f"{stem}_encoding_results.npz"
    np.savez_compressed(
        npz_path,
        corrs=corrs,
        lags=lags,
        channel_names=np.asarray(channel_names),
        coords=coords,
        mean_max_by_channel=corrs.mean(axis=0).max(axis=-1),
    )
    print(f"Saved results: {npz_path}", flush=True)

    if args.skip_plots:
        return

    mean = corrs.mean(axis=(0, 1))
    err = corrs.std(axis=(0, 1)) / np.sqrt(np.prod(corrs.shape[:2]))

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(lags, mean, color="black")
    ax.fill_between(lags, mean - err, mean + err, alpha=0.1, color="black")
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("encoding performance (r +/- sem)")
    ax.axvline(0, c=(0.9, 0.9, 0.9), ls="--")
    ax.axhline(0, c=(0.9, 0.9, 0.9), ls="--")
    lag_plot = args.output_dir / f"{stem}_lag_profile.png"
    fig.savefig(lag_plot, dpi=200)
    plt.close(fig)
    print(f"Saved lag plot: {lag_plot}", flush=True)

    values = corrs.mean(axis=0).max(axis=-1)
    order = values.argsort()
    display = plot_markers(
        values[order],
        coords[order],
        node_size=30,
        display_mode="lzr",
        node_vmin=0,
        node_cmap="inferno_r",
        colorbar=True,
    )
    brain_plot = args.output_dir / f"{stem}_brain_markers.png"
    display.savefig(brain_plot, dpi=200)
    display.close()
    print(f"Saved brain plot: {brain_plot}", flush=True)


def main() -> int:
    args = parse_args()
    backend = configure_backend(args.backend)
    print(f"Himalaya backend: {backend}", flush=True)

    features_path = maybe_download(f"stimuli/{args.feature_space}/features.hdf5", args.bids_root, args.download_missing)
    transcript_path = maybe_download(f"stimuli/{args.feature_space}/transcript.tsv", args.bids_root, args.download_missing)

    aligned_embeddings, words = load_word_embeddings(features_path, transcript_path, args.layer)
    epochs = load_epochs(args, words)

    selection = epochs.selection
    if args.max_words is not None:
        selection = selection[: args.max_words]
        epochs = epochs[: args.max_words]

    x = aligned_embeddings[selection]
    epochs_data = epochs.get_data(copy=True)
    epochs_shape = epochs_data.shape[1:]
    y = epochs_data.reshape(len(epochs), -1)

    if "torch" in get_backend().__name__:
        x = x.astype(np.float32)
        y = y.astype(np.float32)
    print(f"X: {x.shape}; Y: {y.shape}; target shape: {epochs_shape}", flush=True)

    corrs = train_encoding(args, x, y, epochs_shape)
    print(f"Correlation result shape: {corrs.shape}", flush=True)

    channel_locations = {ch["ch_name"]: ch["loc"][:3] for ch in epochs.info["chs"]}
    coords = np.vstack([channel_locations[ch] for ch in epochs.info["ch_names"]]) * 1000.0
    save_outputs(args, corrs, epochs.times.copy(), epochs.info["ch_names"], coords)
    return 0


if __name__ == "__main__":
    sys.exit(main())
