#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import ensure_dir, load_config, output_dir, save_json
from src.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LibriBrain run files through PNPL.")
    parser.add_argument("--config", default="configs/ridge_cluster.yaml")
    parser.add_argument("--data-path", default=None)
    parser.add_argument(
        "--partition",
        choices=["all", "train", "validation", "test", "sherlock1_reference"],
        default="all",
        help="Which PNPL run keys to download.",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for staged downloads.")
    parser.add_argument("--run-index", type=int, action="append", default=None, help="Specific RUN_KEYS index; can repeat.")
    parser.add_argument("--preprocessing-str", default="bads+headpos+sss+notch+bp+ds")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    add_pnpl_checkout_to_path(config)
    data_path = Path(args.data_path or config.get("data", {}).get("libribrain_root") or "./data/LibriBrain").expanduser()
    ensure_dir(data_path)

    try:
        from pnpl.datasets.libribrain2025 import constants
        from pnpl.datasets.libribrain2025.base import LibriBrainBase
    except ImportError as exc:
        raise SystemExit(
            "Could not import PNPL. Install it with `pip install pnpl` or set data.pnpl_root to a PNPL checkout."
        ) from exc

    run_keys = select_run_keys(constants, args.partition, args.run_index)
    if args.max_runs is not None:
        run_keys = run_keys[: args.max_runs]

    downloaded = []
    for i, run_key in enumerate(run_keys, start=1):
        subject, session, task, run = run_key
        h5_path = h5_file_path(data_path, subject, session, task, run, args.preprocessing_str)
        events_path = events_file_path(data_path, subject, session, task, run)
        print(f"[{i}/{len(run_keys)}] {run_key}")
        LibriBrainBase.ensure_file_download(str(events_path), str(data_path))
        LibriBrainBase.ensure_file_download(str(h5_path), str(data_path))
        downloaded.append(
            {
                "run_key": list(run_key),
                "events_path": str(events_path),
                "h5_path": str(h5_path),
                "events_exists": events_path.exists(),
                "h5_exists": h5_path.exists(),
            }
        )

    summary: dict[str, Any] = {
        "partition": args.partition,
        "data_path": str(data_path),
        "preprocessing_str": args.preprocessing_str,
        "n_runs": len(run_keys),
        "runs": downloaded,
    }
    out_path = output_dir(config) / "inspection" / f"download_libribrain_{args.partition}_summary.json"
    save_json(summary, out_path)
    print(json.dumps(summary, indent=2))
    print(f"Saved download summary to {out_path}")


def select_run_keys(constants, partition: str, run_indices: list[int] | None) -> list[tuple[str, str, str, str]]:
    all_keys = [tuple(k) for k in constants.RUN_KEYS]
    if run_indices:
        return [all_keys[i] for i in run_indices]
    val = {tuple(k) for k in constants.VALIDATION_RUN_KEYS}
    test = {tuple(k) for k in constants.TEST_RUN_KEYS}
    if partition == "all":
        return all_keys
    if partition == "train":
        return [k for k in all_keys if k not in val and k not in test]
    if partition == "validation":
        return list(val)
    if partition == "test":
        return list(test)
    if partition == "sherlock1_reference":
        return [("0", str(i), "Sherlock1", "1") for i in range(1, 10)] + sorted(val) + sorted(test)
    raise ValueError(f"Unknown partition: {partition}")


def h5_file_path(data_path: Path, subject: str, session: str, task: str, run: str, preprocessing_str: str) -> Path:
    fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}_proc-{preprocessing_str}_meg.h5"
    return data_path / task / "derivatives" / "serialised" / fname


def events_file_path(data_path: Path, subject: str, session: str, task: str, run: str) -> Path:
    fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv"
    return data_path / task / "derivatives" / "events" / fname


def add_pnpl_checkout_to_path(config: dict[str, Any]) -> None:
    candidates = []
    if config.get("data", {}).get("pnpl_root"):
        candidates.append(Path(config["data"]["pnpl_root"]).expanduser())
    candidates.append(Path(__file__).resolve().parents[3] / "pnpl")
    for path in candidates:
        if (path / "pnpl").exists():
            sys.path.insert(0, str(path))
            return


if __name__ == "__main__":
    main()
