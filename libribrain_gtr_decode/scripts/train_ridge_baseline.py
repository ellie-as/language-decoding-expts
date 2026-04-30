#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import load_config, load_json, output_dir
from src.models import train_ridge_baseline
from src.utils import configure_logging, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PCA + ridge MEG to GTR baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--control", choices=["time_shift", "shuffled_labels"], default=None)
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.config)
    set_seed(int(config["project"].get("seed", 0)))
    out = output_dir(config)
    result = train_ridge_baseline(
        out / "meg_features" / "meg_10s_plus1s_50hz.dat",
        load_json(out / "meg_features" / "meg_10s_plus1s_info.json"),
        pd.read_parquet(out / "meg_features" / "meg_10s_plus1s_metadata.parquet"),
        np.load(out / "embeddings" / "gtr_base_10s_1hz.npy"),
        pd.read_parquet(out / "embeddings" / "gtr_base_10s_1hz_metadata.parquet"),
        pd.read_parquet(out / "splits" / f"split_by_run_seed{config['split'].get('seed', 0)}.parquet"),
        config,
        out,
        control=args.control,
    )
    print(result)


if __name__ == "__main__":
    main()
