#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import evaluate_all, make_plots
from src.io_utils import ensure_dir, load_config, output_dir, save_json
from src.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval from predicted and true embeddings.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.config)
    out = output_dir(config)
    pred_path = Path(args.predictions) if args.predictions else out / "predictions" / f"ridge_{args.split}_predictions.npy"
    preds = np.load(pred_path)
    embeddings = np.load(out / "embeddings" / "gtr_base_10s_1hz.npy")
    emb_meta = pd.read_parquet(out / "embeddings" / "gtr_base_10s_1hz_metadata.parquet")
    meg_meta = pd.read_parquet(out / "meg_features" / "meg_10s_plus1s_metadata.parquet")
    splits = pd.read_parquet(out / "splits" / f"split_by_run_seed{config['split'].get('seed', 0)}.parquet")
    meta = meg_meta.merge(splits[["example_id", "split"]], on="example_id")
    meta = meta[meta["split"] == args.split].reset_index(drop=True)
    emb_idx = meta[["example_id"]].merge(
        emb_meta.reset_index().rename(columns={"index": "embedding_index"})[["example_id", "embedding_index"]],
        on="example_id",
    )["embedding_index"].to_numpy()
    true = embeddings[emb_idx]
    metrics, by_type, ranks = evaluate_all(preds, true, meta, config)
    ensure_dir(out / "results")
    save_json(metrics, out / "results" / "retrieval_metrics.json")
    by_type.to_csv(out / "results" / "retrieval_by_distractor_type.csv", index=False)
    make_plots(ranks, preds, true, by_type, out / "plots")
    print(metrics)


if __name__ == "__main__":
    main()
