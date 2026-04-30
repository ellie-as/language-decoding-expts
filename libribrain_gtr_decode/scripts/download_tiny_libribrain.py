#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import ensure_dir, load_config, output_dir, save_json
from src.utils import configure_logging


def shape_of(x: Any) -> list[int] | None:
    if hasattr(x, "shape"):
        return [int(v) for v in x.shape]
    try:
        return [int(v) for v in np.asarray(x).shape]
    except Exception:
        return None


def jsonable_info(info: Any) -> Any:
    try:
        json.dumps(info)
        return info
    except TypeError:
        return repr(info)


def filtered_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(cls).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download/load one tiny LibriBrain run through PNPL and save a smoke-test summary."
    )
    parser.add_argument("--config", default="configs/debug_local.yaml")
    parser.add_argument("--data-path", default=None, help="Local PNPL/LibriBrain cache directory.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dataset", choices=["speech", "phoneme"], default="speech")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=0.2)
    parser.add_argument("--preprocessing-str", default="bads+headpos+sss+notch+bp+ds")
    parser.add_argument("--partition", default=None, choices=[None, "train", "validation", "test"])
    parser.add_argument("--max-items", type=int, default=1)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--lazy", action="store_true", help="Let PNPL download files on first item access.")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    add_pnpl_checkout_to_path(config)
    data_path = Path(args.data_path or config.get("data", {}).get("libribrain_root") or "./data/LibriBrain").expanduser()
    ensure_dir(data_path)

    try:
        from pnpl.datasets import LibriBrainPhoneme, LibriBrainSpeech
        from pnpl.datasets.libribrain2025 import constants
    except ImportError as exc:
        raise SystemExit(
            "Could not import PNPL. Install it with `pip install pnpl` or "
            "`pip install git+https://github.com/neural-processing-lab/pnpl.git`."
        ) from exc

    run_keys = list(constants.RUN_KEYS)
    if not run_keys:
        raise RuntimeError("PNPL constants.RUN_KEYS is empty")
    if args.run_index < 0 or args.run_index >= len(run_keys):
        raise ValueError(f"--run-index must be between 0 and {len(run_keys) - 1}")

    dataset_cls = LibriBrainSpeech if args.dataset == "speech" else LibriBrainPhoneme
    kwargs = filtered_kwargs(
        dataset_cls,
        {
            "data_path": str(data_path),
            "partition": args.partition,
            "preprocessing_str": args.preprocessing_str,
            "include_run_keys": [run_keys[args.run_index]],
            "tmin": args.tmin,
            "tmax": args.tmax,
            "standardize": not args.no_standardize,
            "include_info": True,
            "exclude_run_keys": [],
            "exclude_tasks": [],
            "preload_files": not args.lazy,
            "download": True,
        },
    )
    ds = dataset_cls(**kwargs)

    n_items = min(int(args.max_items), len(ds))
    examples = []
    for i in range(n_items):
        sample = ds[i]
        if not isinstance(sample, tuple):
            sample = (sample,)
        row = {"index": i, "n_returned_values": len(sample)}
        if len(sample) >= 1:
            row["x_shape"] = shape_of(sample[0])
            row["x_dtype"] = str(getattr(sample[0], "dtype", type(sample[0]).__name__))
        if len(sample) >= 2:
            row["y_shape"] = shape_of(sample[1])
            row["y_dtype"] = str(getattr(sample[1], "dtype", type(sample[1]).__name__))
        if len(sample) >= 3:
            row["info"] = jsonable_info(sample[2])
        examples.append(row)

    summary = {
        "dataset": args.dataset,
        "dataset_class": dataset_cls.__name__,
        "data_path": str(data_path),
        "partition": args.partition,
        "run_index": args.run_index,
        "run_key": str(run_keys[args.run_index]),
        "constructor_kwargs": kwargs,
        "n_dataset_items": int(len(ds)),
        "n_examples_inspected": n_items,
        "examples": examples,
    }
    out_path = Path(args.output) if args.output else output_dir(config) / "inspection" / "tiny_libribrain_summary.json"
    save_json(summary, out_path)
    print(json.dumps(summary, indent=2, default=str))
    print(f"Saved tiny LibriBrain summary to {out_path}")


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
