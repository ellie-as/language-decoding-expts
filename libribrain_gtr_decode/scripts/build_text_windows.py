#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import discover_runs
from src.io_utils import ensure_dir, load_config, output_dir, write_cache_info
from src.transcript import build_text_windows, write_examples
from src.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 10-second transcript windows.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.config)
    out = Path(args.output) if args.output else output_dir(config) / "text_windows" / "text_windows_10s_1hz.parquet"
    ensure_dir(out.parent)
    windows = build_text_windows(discover_runs(config), config)
    windows.to_parquet(out, index=False)
    write_examples(windows, out.parent / "examples.txt")
    write_cache_info(out, config, {"n_examples": len(windows)})
    print(f"Saved {len(windows)} text windows to {out}")


if __name__ == "__main__":
    main()
