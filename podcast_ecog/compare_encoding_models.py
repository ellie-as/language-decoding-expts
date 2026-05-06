#!/usr/bin/env python3
"""Compare several encoding model classes on podcast text-window embeddings."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from run_text_window_encoding import (  # noqa: E402
    DEFAULT_MOUNT_ROOT,
    load_epochs,
    load_preferred_lag_targets,
    load_words,
    resolve_paths,
)


def parse_args() -> argparse.Namespace:
    out_dir = THIS_DIR / "outputs" / "encoding_model_comparison"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mounted-root", type=Path, default=DEFAULT_MOUNT_ROOT)
    parser.add_argument("--bids-root", type=Path, default=None)
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=None,
        help="GPT-2 encoding NPZ used to select each channel's preferred lag.",
    )
    parser.add_argument("--output-dir", type=Path, default=out_dir)
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=THIS_DIR / "outputs" / "text_window_encoding" / "cache",
    )
    parser.add_argument("--subject", default="03")
    parser.add_argument("--task", default="podcast")
    parser.add_argument("--target-mode", default="preferred-lag", choices=["preferred-lag"])
    parser.add_argument("--picks-regex", default=".*")
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[1, 10, 100, 200])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "pls10", "rff_ridge", "extra_trees", "mlp"],
        choices=["ridge", "pls10", "pls25", "rff_ridge", "extra_trees", "mlp"],
    )
    parser.add_argument("--outer-splits", type=int, default=2)
    parser.add_argument("--ridge-alpha", type=float, default=100000.0)
    parser.add_argument("--rff-components", type=int, default=256)
    parser.add_argument("--rff-gamma", type=float, default=0.01)
    parser.add_argument("--trees", type=int, default=80)
    parser.add_argument("--tree-max-depth", type=int, default=10)
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--mlp-max-iter", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--max-channels", type=int, default=None)
    parser.add_argument("--resample-sfreq", type=float, default=32.0)
    return parser.parse_args()


def cache_path_for_targets(args: argparse.Namespace) -> Path:
    word_tag = f"maxwords{args.max_words}" if args.max_words else "allwords"
    chan_tag = f"maxch{args.max_channels}" if args.max_channels else "allch"
    picks = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.picks_regex).strip("-") or "all"
    return args.output_dir / "cache" / f"sub-{args.subject}_{word_tag}_{chan_tag}_{picks}_preferred_lag_targets.npz"


def load_or_build_targets(args: argparse.Namespace) -> dict[str, np.ndarray]:
    target_cache = cache_path_for_targets(args)
    if target_cache.is_file():
        print(f"Loading target cache: {target_cache}", flush=True)
        with np.load(target_cache, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}

    transcript_path, fif_path, reference_path = resolve_paths(args)
    words = load_words(transcript_path)
    if args.max_words is not None:
        words = words.iloc[: args.max_words].reset_index(drop=True)

    epochs = load_epochs(
        fif_path=fif_path,
        words=words,
        picks_regex=args.picks_regex,
        max_channels=args.max_channels,
        tmin=-2.0,
        tmax=2.0,
        resample_sfreq=args.resample_sfreq,
    )
    targets, preferred_lag_s, preferred_epoch_idx = load_preferred_lag_targets(reference_path, epochs)
    payload = {
        "targets": targets.astype(np.float32),
        "selection": epochs.selection.astype(np.int64),
        "channel_names": np.asarray(epochs.info["ch_names"]).astype(str),
        "preferred_lag_s": preferred_lag_s.astype(np.float32),
        "preferred_epoch_idx": preferred_epoch_idx.astype(np.int64),
    }
    target_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target_cache, **payload)
    print(f"Saved target cache: {target_cache}", flush=True)
    return payload


def find_embedding_cache(cache_dir: Path, window_size: int) -> Path:
    matches = sorted(cache_dir.glob(f"text_window_w{window_size}__*.npz"))
    if not matches:
        raise FileNotFoundError(f"No cached embeddings found for window {window_size} under {cache_dir}")
    return matches[-1]


def load_window_embeddings(cache_dir: Path, window_size: int, selection: np.ndarray) -> np.ndarray:
    path = find_embedding_cache(cache_dir, window_size)
    embeddings = np.load(path)["embeddings"].astype(np.float32)
    if int(selection.max()) >= len(embeddings):
        raise ValueError(f"Selection max {selection.max()} exceeds embedding rows {len(embeddings)} in {path}")
    print(f"Loading X window={window_size}: {path}", flush=True)
    return embeddings[selection]


def corr_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(yt * yt, axis=0) * np.sum(yp * yp, axis=0))
    return np.divide(
        np.sum(yt * yp, axis=0),
        denom,
        out=np.zeros(y_true.shape[1], dtype=np.float32),
        where=denom > 0,
    ).astype(np.float32)


def make_model(name: str, args: argparse.Namespace, fold: int):
    if name == "ridge":
        return Ridge(alpha=args.ridge_alpha, fit_intercept=False)
    if name == "pls10":
        return PLSRegression(n_components=10, scale=False)
    if name == "pls25":
        return PLSRegression(n_components=25, scale=False)
    if name == "rff_ridge":
        return (
            Nystroem(
                kernel="rbf",
                gamma=args.rff_gamma,
                n_components=args.rff_components,
                random_state=args.seed + fold,
            ),
            Ridge(alpha=args.ridge_alpha, fit_intercept=False),
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=args.trees,
            max_depth=args.tree_max_depth,
            min_samples_leaf=3,
            max_features=0.5,
            random_state=args.seed + fold,
            n_jobs=-1,
        )
    if name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(args.mlp_hidden,),
            activation="relu",
            alpha=1e-3,
            learning_rate_init=1e-3,
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=args.mlp_max_iter,
            random_state=args.seed + fold,
        )
    raise ValueError(name)


def fit_predict_model(name: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, args, fold: int) -> np.ndarray:
    model = make_model(name, args, fold)
    if isinstance(model, tuple):
        featurizer, ridge = model
        z_train = featurizer.fit_transform(x_train)
        z_test = featurizer.transform(x_test)
        ridge.fit(z_train, y_train)
        return ridge.predict(z_test).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    return model.predict(x_test).astype(np.float32)


def evaluate_model_window(name: str, window_size: int, x: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    fold_corrs = []
    cv = KFold(args.outer_splits, shuffle=False)
    for fold, (train_idx, test_idx) in enumerate(cv.split(x), start=1):
        t0 = time.time()
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x[train_idx]).astype(np.float32)
        x_test = x_scaler.transform(x[test_idx]).astype(np.float32)
        y_train = y_scaler.fit_transform(y[train_idx]).astype(np.float32)
        y_test = y_scaler.transform(y[test_idx]).astype(np.float32)
        pred = fit_predict_model(name, x_train, y_train, x_test, args, fold)
        fold_corrs.append(corr_per_target(y_test, pred))
        print(f"  {name} w={window_size} fold {fold}/{args.outer_splits}: {time.time() - t0:.1f}s", flush=True)
    return np.stack(fold_corrs)


def write_outputs(
    *,
    args: argparse.Namespace,
    all_corrs: dict[tuple[str, int], np.ndarray],
    channel_names: np.ndarray,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    channel_rows = []
    for (model, window), corrs in all_corrs.items():
        mean_by_channel = corrs.mean(axis=0)
        rows.append(
            {
                "model": model,
                "window_words": int(window),
                "mean_r": float(mean_by_channel.mean()),
                "median_r": float(np.median(mean_by_channel)),
                "q75_r": float(np.quantile(mean_by_channel, 0.75)),
                "max_r": float(mean_by_channel.max()),
                "positive_share": float((mean_by_channel > 0).mean()),
            }
        )
        for channel, score in zip(channel_names, mean_by_channel):
            channel_rows.append({"model": model, "window_words": int(window), "channel": channel, "r": float(score)})

    summary = pd.DataFrame(rows).sort_values(["mean_r", "median_r"], ascending=False)
    channel_scores = pd.DataFrame(channel_rows)
    summary.to_csv(args.output_dir / "model_window_summary.csv", index=False)
    channel_scores.to_csv(args.output_dir / "model_window_channel_scores.csv", index=False)

    model_names = list(args.models)
    window_sizes = list(args.window_sizes)
    tensor = np.zeros((len(model_names), len(window_sizes), len(channel_names)), dtype=np.float32)
    for i, model in enumerate(model_names):
        for j, window in enumerate(window_sizes):
            tensor[i, j] = all_corrs[(model, window)].mean(axis=0)
    np.savez_compressed(
        args.output_dir / "encoding_model_comparison_results.npz",
        corrs=tensor,
        models=np.asarray(model_names),
        window_sizes=np.asarray(window_sizes),
        channel_names=channel_names,
    )

    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    for model in model_names:
        vals = [summary[(summary.model == model) & (summary.window_words == window)]["mean_r"].iloc[0] for window in window_sizes]
        ax.plot(window_sizes, vals, marker="o", label=model)
    ax.set_xscale("log")
    ax.set_xlabel("trailing text window (words)")
    ax.set_ylabel("mean channel correlation r")
    ax.set_title("Encoding model comparison")
    ax.legend(fontsize=8)
    fig.savefig(args.output_dir / "model_window_mean_corr.png", dpi=220)
    plt.close(fig)

    pivot = summary.pivot(index="model", columns="window_words", values="mean_r").loc[model_names, window_sizes]
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    vmax = float(np.nanmax(np.abs(pivot.to_numpy())))
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(window_sizes)))
    ax.set_xticklabels([str(w) for w in window_sizes])
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel("trailing text window (words)")
    ax.set_title("Mean r by model and window")
    fig.colorbar(im, ax=ax, label="mean r")
    fig.savefig(args.output_dir / "model_window_heatmap.png", dpi=220)
    plt.close(fig)

    best = summary.iloc[0].to_dict()
    (args.output_dir / "model_comparison_summary.md").write_text(
        "\n".join(
            [
                "# Encoding Model Comparison",
                "",
                f"- Best condition: `{best['model']}` at `{int(best['window_words'])}` words.",
                f"- Mean r: `{best['mean_r']:.5f}`; median r: `{best['median_r']:.5f}`.",
                "",
                "## Ranked Conditions",
                "",
                summary.to_markdown(index=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_payload = load_or_build_targets(args)
    y = target_payload["targets"].astype(np.float32)
    selection = target_payload["selection"].astype(np.int64)
    channel_names = target_payload["channel_names"].astype(str)

    all_corrs: dict[tuple[str, int], np.ndarray] = {}
    run_config = vars(args).copy()
    run_config["mounted_root"] = str(args.mounted_root)
    run_config["bids_root"] = None if args.bids_root is None else str(args.bids_root)
    run_config["output_dir"] = str(args.output_dir)
    run_config["embedding_cache_dir"] = str(args.embedding_cache_dir)
    (args.output_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    for window in args.window_sizes:
        x = load_window_embeddings(args.embedding_cache_dir, int(window), selection)
        for model in args.models:
            print(f"Evaluating model={model} window={window}", flush=True)
            all_corrs[(model, int(window))] = evaluate_model_window(model, int(window), x, y, args)

    write_outputs(args=args, all_corrs=all_corrs, channel_names=channel_names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
