#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings import compute_text_embeddings, embedding_sanity_checks
from src.io_utils import ensure_dir, load_config, output_dir, save_json, write_cache_info
from src.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute GTR-base embeddings for transcript windows.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--text-windows", default=None)
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.config)
    out_dir = ensure_dir(output_dir(config) / "embeddings")
    text_path = Path(args.text_windows) if args.text_windows else output_dir(config) / "text_windows" / "text_windows_10s_1hz.parquet"
    windows = pd.read_parquet(text_path)
    embeddings = compute_text_embeddings(windows, config)
    emb_path = out_dir / "gtr_base_10s_1hz.npy"
    meta_path = out_dir / "gtr_base_10s_1hz_metadata.parquet"
    info_path = out_dir / "gtr_base_10s_1hz_info.json"
    np.save(emb_path, embeddings.astype(np.float32))
    windows.drop(columns=["text"], errors="ignore").to_parquet(meta_path, index=False)
    info = embedding_sanity_checks(embeddings, windows)
    save_json(info, info_path)
    write_cache_info(emb_path, config, info)
    print(f"Saved embeddings {embeddings.shape} to {emb_path}")


if __name__ == "__main__":
    main()
