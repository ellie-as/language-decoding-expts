#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import RunSpec
from src.embeddings import compute_text_embeddings, embedding_sanity_checks
from src.io_utils import ensure_dir, load_config, output_dir, save_json
from src.meg_features import build_meg_feature_memmap
from src.models import train_ridge_baseline
from src.splits import create_run_splits, split_summary
from src.transcript import build_text_windows, normalize_word_events, write_examples
from src.utils import configure_logging, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download one PNPL LibriBrain run and run a tiny real-data ridge smoke test."
    )
    parser.add_argument("--config", default="configs/debug_local.yaml")
    parser.add_argument("--data-path", default="./data/LibriBrain")
    parser.add_argument("--output-dir", default="outputs/tiny_libribrain_smoketest")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--max-duration-sec", type=float, default=180.0)
    parser.add_argument("--max-examples", type=int, default=96)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--real-gtr", action="store_true", help="Use sentence-transformers/gtr-t5-base instead of mock embeddings.")
    parser.add_argument("--preprocessing-str", default="bads+headpos+sss+notch+bp+ds")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    add_pnpl_checkout_to_path(config)
    config = override_for_smoketest(config, args)
    set_seed(int(config["project"].get("seed", 0)))
    out = output_dir(config)

    ds, run_key, h5_path, events_path = load_one_pnpl_run(args)
    subject, session, task, run = run_key
    smoke_dir = ensure_dir(out / "pnpl_subset")
    slim_h5 = smoke_dir / f"sub-{subject}_ses-{session}_task-{task}_run-{run}_smoke_meg.h5"
    slim_events = smoke_dir / f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv"

    events = pd.read_csv(events_path, sep="\t")
    events = events[pd.to_numeric(events["timemeg"], errors="coerce").fillna(np.inf) <= args.max_duration_sec].copy()
    events.to_csv(slim_events, sep="\t", index=False)
    make_slim_h5(Path(h5_path), slim_h5, args.max_duration_sec)

    run_spec = RunSpec(
        subject=f"sub-{subject}",
        session=f"ses-{session}",
        run=f"{task}_run-{run}",
        transcript_path=str(slim_events),
        meg_path=str(slim_h5),
    )

    raw_words = normalize_word_events(events, run_spec)
    windows = build_text_windows([run_spec], config).head(args.max_examples).reset_index(drop=True)
    if len(windows) < 12:
        raise RuntimeError(
            f"Only built {len(windows)} windows from the tiny subset. Increase --max-duration-sec or lower windows.min_words."
        )

    text_dir = ensure_dir(out / "text_windows")
    text_path = text_dir / "text_windows_10s_1hz.parquet"
    windows.to_parquet(text_path, index=False)
    write_examples(windows, text_dir / "examples.txt")

    embeddings = compute_text_embeddings(windows, config)
    emb_dir = ensure_dir(out / "embeddings")
    np.save(emb_dir / "gtr_base_10s_1hz.npy", embeddings.astype(np.float32))
    windows.drop(columns=["text"], errors="ignore").to_parquet(emb_dir / "gtr_base_10s_1hz_metadata.parquet", index=False)
    save_json(embedding_sanity_checks(embeddings, windows), emb_dir / "gtr_base_10s_1hz_info.json")

    splits = create_run_splits(windows, config)
    split_dir = ensure_dir(out / "splits")
    split_path = split_dir / f"split_by_run_seed{config['split'].get('seed', 0)}.parquet"
    splits.to_parquet(split_path, index=False)
    save_json(split_summary(splits), split_dir / "split_summary.json")

    meg_dir = ensure_dir(out / "meg_features")
    meg_path = meg_dir / "meg_10s_plus1s_50hz.dat"
    meg_metadata, meg_info = build_meg_feature_memmap(windows, [run_spec], config, meg_path)
    meg_metadata.to_parquet(meg_dir / "meg_10s_plus1s_metadata.parquet", index=False)
    save_json(meg_info, meg_dir / "meg_10s_plus1s_info.json")

    result = train_ridge_baseline(
        meg_path,
        meg_info,
        meg_metadata,
        embeddings,
        pd.read_parquet(emb_dir / "gtr_base_10s_1hz_metadata.parquet"),
        splits,
        config,
        out,
        control=None,
    )

    summary: dict[str, Any] = {
        "status": "ok",
        "run_key": list(run_key),
        "pnpl_dataset_len": int(len(ds)),
        "pnpl_h5_path": str(h5_path),
        "pnpl_events_path": str(events_path),
        "slim_h5_path": str(slim_h5),
        "slim_events_path": str(slim_events),
        "raw_event_rows_in_subset": int(len(events)),
        "word_rows_in_subset": int(len(raw_words)),
        "text_windows": int(len(windows)),
        "embedding_shape": list(embeddings.shape),
        "meg_feature_shape": meg_info["shape"],
        "ridge": result,
    }
    save_json(summary, out / "inspection" / "tiny_libribrain_smoketest_summary.json")
    print(json.dumps(summary, indent=2, default=str))


def override_for_smoketest(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = dict(config)
    config["project"] = dict(config.get("project", {}))
    config["project"]["output_dir"] = args.output_dir
    config["project"]["cache_dir"] = str(Path(args.output_dir) / "cache")
    config["data"] = dict(config.get("data", {}))
    config["data"]["mock_data"] = False
    config["data"]["libribrain_root"] = args.data_path
    config["debug"] = dict(config.get("debug", {}))
    config["debug"]["enabled"] = False
    config["debug"]["max_examples"] = args.max_examples
    config["features"] = dict(config.get("features", {}))
    config["features"]["pca_components"] = args.pca_components
    config["features"]["pca_batch_size"] = min(64, max(16, args.max_examples // 2))
    config["features"]["standardize_channels"] = True
    config["embeddings"] = dict(config.get("embeddings", {}))
    config["embeddings"]["mock"] = not args.real_gtr
    config["embeddings"]["mock_dim"] = args.embedding_dim
    config["embeddings"]["batch_size"] = min(16, int(config["embeddings"].get("batch_size", 16)))
    config["ridge"] = dict(config.get("ridge", {}))
    config["ridge"]["controls"] = []
    config["ridge"]["alphas"] = [0.1, 1, 10, 100]
    config["evaluation"] = dict(config.get("evaluation", {}))
    config["evaluation"]["n_random_distractors"] = min(99, max(1, args.max_examples - 1))
    config["evaluation"]["nearby_min_sec"] = 15
    config["evaluation"]["nearby_max_sec"] = 90
    return config


def load_one_pnpl_run(args: argparse.Namespace):
    try:
        from pnpl.datasets import LibriBrainSpeech
        from pnpl.datasets.libribrain2025 import constants
    except ImportError as exc:
        raise SystemExit(
            "Could not import PNPL. Install it with `pip install pnpl` or "
            "`pip install -e /Users/eleanorspens/PycharmProjects/pnpl`."
        ) from exc

    run_keys = list(constants.RUN_KEYS)
    run_key = tuple(run_keys[args.run_index])
    ds = LibriBrainSpeech(
        data_path=str(Path(args.data_path).expanduser()),
        preprocessing_str=args.preprocessing_str,
        include_run_keys=[run_key],
        exclude_run_keys=[],
        exclude_tasks=[],
        tmin=0.0,
        tmax=0.2,
        standardize=False,
        include_info=True,
        preload_files=True,
        download=True,
    )
    subject, session, task, run = run_key
    h5_path = Path(ds._get_h5_path(subject, session, task, run))
    events_path = Path(ds._get_events_path(subject, session, task, run))
    return ds, run_key, h5_path, events_path


def add_pnpl_checkout_to_path(config: dict[str, Any]) -> None:
    candidates = []
    if config.get("data", {}).get("pnpl_root"):
        candidates.append(Path(config["data"]["pnpl_root"]).expanduser())
    candidates.append(Path(__file__).resolve().parents[3] / "pnpl")
    for path in candidates:
        if (path / "pnpl").exists():
            sys.path.insert(0, str(path))
            return


def make_slim_h5(source: Path, dest: Path, max_duration_sec: float) -> None:
    ensure_dir(dest.parent)
    with h5py.File(source, "r") as src, h5py.File(dest, "w") as dst:
        sfreq = float(src.attrs.get("sample_frequency", src["data"].attrs.get("sample_frequency", 250.0)))
        n_samples = min(src["data"].shape[1], int(round((max_duration_sec + 2.0) * sfreq)))
        data = src["data"][:, :n_samples].astype("float32")
        dst.create_dataset("data", data=data, compression="gzip")
        dst.attrs["sample_frequency"] = sfreq


if __name__ == "__main__":
    main()
