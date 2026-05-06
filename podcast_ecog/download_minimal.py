#!/usr/bin/env python3
"""Download the minimal files needed for the Podcast ECoG encoding tutorial."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds005574"
REQUIRED_FILES = (
    "stimuli/gpt2-xl/features.hdf5",
    "stimuli/gpt2-xl/transcript.tsv",
    "derivatives/ecogprep/sub-03/ieeg/sub-03_task-podcast_desc-highgamma_ieeg.fif",
)


def download_file(relative_path: str, bids_root: Path, overwrite: bool = False) -> None:
    destination = bids_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        print(f"exists: {destination}")
        return

    url = f"{BASE_URL}/{relative_path}"
    tmp = destination.with_suffix(destination.suffix + ".part")
    print(f"download: {url}")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(destination)
    print(f"wrote: {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "ds005574",
        help="Where the OpenNeuro ds005574 files should live.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Redownload existing files.")
    args = parser.parse_args()

    args.bids_root.mkdir(parents=True, exist_ok=True)
    for relative_path in REQUIRED_FILES:
        download_file(relative_path, args.bids_root, overwrite=args.overwrite)

    return 0


if __name__ == "__main__":
    sys.exit(main())
