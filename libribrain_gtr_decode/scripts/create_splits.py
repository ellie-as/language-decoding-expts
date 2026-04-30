#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import ensure_dir, load_config, output_dir, save_json, write_cache_info
from src.splits import create_run_splits, split_summary
from src.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-safe run-level splits.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--text-windows", default=None)
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.config)
    windows_path = Path(args.text_windows) if args.text_windows else output_dir(config) / "text_windows" / "text_windows_10s_1hz.parquet"
    windows = pd.read_parquet(windows_path)
    splits = create_run_splits(windows, config)
    out_dir = ensure_dir(output_dir(config) / "splits")
    split_path = out_dir / f"split_by_run_seed{config['split'].get('seed', 0)}.parquet"
    splits.to_parquet(split_path, index=False)
    summary = split_summary(splits)
    save_json(summary, out_dir / "split_summary.json")
    write_cache_info(split_path, config, summary)
    print(f"Saved splits to {split_path}")


if __name__ == "__main__":
    main()
