#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import inspect_libribrain
from src.io_utils import ensure_dir, load_config, output_dir, save_json
from src.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PNPL/LibriBrain data exposure.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.config)
    summary = inspect_libribrain(config)
    out_path = Path(args.output) if args.output else output_dir(config) / "inspection" / "libribrain_summary.json"
    ensure_dir(out_path.parent)
    save_json(summary, out_path)
    print(f"Saved inspection summary to {out_path}")
    print(summary)


if __name__ == "__main__":
    main()
