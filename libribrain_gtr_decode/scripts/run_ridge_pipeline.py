#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import discover_runs, inspect_libribrain
from src.embeddings import compute_text_embeddings, embedding_sanity_checks
from src.io_utils import cache_is_current, config_hash, ensure_dir, load_config, load_json, output_dir, save_json, write_cache_info
from src.meg_features import build_meg_feature_memmap
from src.models import train_ridge_baseline
from src.splits import create_run_splits, split_summary
from src.transcript import build_text_windows, write_examples
from src.utils import configure_logging, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LibriBrain GTR ridge pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true", help="Force debug mode on for this run.")
    parser.add_argument("--force", action="store_true", help="Recompute cached steps.")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    if args.debug:
        config.setdefault("debug", {})["enabled"] = True
        config.setdefault("data", {})["mock_data"] = True
        config.setdefault("embeddings", {})["mock"] = True
        config["_config_hash"] = config_hash(config)
    set_seed(int(config["project"].get("seed", 0)))
    out = output_dir(config)
    runs = discover_runs(config)

    inspection_path = out / "inspection" / "libribrain_summary.json"
    if args.force or not inspection_path.exists():
        save_json(inspect_libribrain(config), inspection_path)

    text_path = out / "text_windows" / "text_windows_10s_1hz.parquet"
    if args.force or not cache_is_current(text_path, config):
        ensure_dir(text_path.parent)
        windows = build_text_windows(runs, config)
        windows.to_parquet(text_path, index=False)
        write_examples(windows, text_path.parent / "examples.txt")
        write_cache_info(text_path, config, {"n_examples": len(windows)})
    else:
        windows = pd.read_parquet(text_path)

    emb_path = out / "embeddings" / "gtr_base_10s_1hz.npy"
    emb_meta_path = out / "embeddings" / "gtr_base_10s_1hz_metadata.parquet"
    if args.force or not cache_is_current(emb_path, config):
        ensure_dir(emb_path.parent)
        embeddings = compute_text_embeddings(windows, config)
        np.save(emb_path, embeddings.astype(np.float32))
        windows.drop(columns=["text"], errors="ignore").to_parquet(emb_meta_path, index=False)
        info = embedding_sanity_checks(embeddings, windows)
        save_json(info, out / "embeddings" / "gtr_base_10s_1hz_info.json")
        write_cache_info(emb_path, config, info)
    else:
        embeddings = np.load(emb_path)

    split_path = out / "splits" / f"split_by_run_seed{config['split'].get('seed', 0)}.parquet"
    if args.force or not cache_is_current(split_path, config):
        ensure_dir(split_path.parent)
        splits = create_run_splits(windows, config)
        splits.to_parquet(split_path, index=False)
        summary = split_summary(splits)
        save_json(summary, split_path.parent / "split_summary.json")
        write_cache_info(split_path, config, summary)
    else:
        splits = pd.read_parquet(split_path)

    meg_path = out / "meg_features" / "meg_10s_plus1s_50hz.dat"
    meg_info_path = out / "meg_features" / "meg_10s_plus1s_info.json"
    meg_meta_path = out / "meg_features" / "meg_10s_plus1s_metadata.parquet"
    if args.force or not cache_is_current(meg_path, config):
        ensure_dir(meg_path.parent)
        meg_metadata, meg_info = build_meg_feature_memmap(windows, runs, config, meg_path)
        meg_metadata.to_parquet(meg_meta_path, index=False)
        save_json(meg_info, meg_info_path)
        write_cache_info(meg_path, config, meg_info)
    else:
        meg_metadata = pd.read_parquet(meg_meta_path)
        meg_info = load_json(meg_info_path)

    embeddings_metadata = pd.read_parquet(emb_meta_path)
    result = train_ridge_baseline(
        meg_path,
        meg_info,
        meg_metadata,
        embeddings,
        embeddings_metadata,
        splits,
        config,
        out,
        control=None,
    )
    for control in config.get("ridge", {}).get("controls", []):
        train_ridge_baseline(meg_path, meg_info, meg_metadata, embeddings, embeddings_metadata, splits, config, out, control=control)
    print("Pipeline complete")
    print(result)


if __name__ == "__main__":
    main()
