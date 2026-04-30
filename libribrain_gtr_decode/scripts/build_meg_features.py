#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import discover_runs
from src.io_utils import ensure_dir, load_config, output_dir, save_json, write_cache_info
from src.meg_features import build_meg_feature_memmap
from src.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build memmapped MEG window features.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--text-windows", default=None)
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.config)
    windows_path = Path(args.text_windows) if args.text_windows else output_dir(config) / "text_windows" / "text_windows_10s_1hz.parquet"
    windows = pd.read_parquet(windows_path)
    out_dir = ensure_dir(output_dir(config) / "meg_features")
    dat_path = out_dir / "meg_10s_plus1s_50hz.dat"
    metadata, info = build_meg_feature_memmap(windows, discover_runs(config), config, dat_path)
    metadata.to_parquet(out_dir / "meg_10s_plus1s_metadata.parquet", index=False)
    save_json(info, out_dir / "meg_10s_plus1s_info.json")
    write_cache_info(dat_path, config, info)
    print(f"Saved MEG features {info['shape']} to {dat_path}")


if __name__ == "__main__":
    main()
