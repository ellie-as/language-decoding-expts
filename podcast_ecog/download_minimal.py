#!/usr/bin/env python3
"""Download Podcast ECoG files from OpenNeuro ds005574."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from xml.etree import ElementTree as ET
from pathlib import Path


BUCKET_URL = "https://s3.amazonaws.com/openneuro.org"
DATASET = "ds005574"
BASE_URL = f"{BUCKET_URL}/{DATASET}"
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


def list_dataset_files() -> list[str]:
    """List all public S3 object keys under ds005574/ and return dataset-relative paths."""
    files: list[str] = []
    continuation = None
    while True:
        url = f"{BUCKET_URL}?list-type=2&prefix={DATASET}/"
        if continuation:
            url += f"&continuation-token={urllib.parse.quote(continuation)}"

        with urllib.request.urlopen(url) as response:
            root = ET.fromstring(response.read())

        namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for key_element in root.findall(".//s3:Key", namespace):
            key = key_element.text
            if not key or key.endswith("/"):
                continue
            if key.startswith(f"{DATASET}/"):
                files.append(key[len(DATASET) + 1 :])

        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=namespace)
        if truncated.lower() != "true":
            break
        continuation = root.findtext("s3:NextContinuationToken", namespaces=namespace)
        if not continuation:
            raise RuntimeError("S3 listing was truncated but no continuation token was returned.")

    return sorted(set(files))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "ds005574",
        help="Where the OpenNeuro ds005574 files should live.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all files in OpenNeuro ds005574 instead of only the tutorial subset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be downloaded without downloading them.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Redownload existing files.")
    args = parser.parse_args()

    args.bids_root.mkdir(parents=True, exist_ok=True)
    relative_paths = list_dataset_files() if args.all else list(REQUIRED_FILES)
    print(f"{'All dataset' if args.all else 'Minimal tutorial'} mode: {len(relative_paths)} files.")
    for relative_path in relative_paths:
        if args.dry_run:
            print(relative_path)
            continue
        download_file(relative_path, args.bids_root, overwrite=args.overwrite)

    return 0


if __name__ == "__main__":
    sys.exit(main())
