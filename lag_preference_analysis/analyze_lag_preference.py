#!/usr/bin/env python3
"""Analyse per-lag MiniLM encoding correlations.

Loads ``lag_corrs.npz`` produced by ``train_lag_encoding.py`` and reports:
  - Mean / median per-voxel r at each lag, restricted to each Brodmann sub-ROI.
  - Within-lag z-scored r per ROI (positive = ROI does better than the
    full-frontal mean for that lag).
  - Distribution of preferred lags (argmax over lags) per ROI.
  - Tuning sharpness (argmax r minus mean of other lags) per ROI.

Sub-ROIs are loaded from ``--ba-dir/<UTSxx>/<roi>.json``.

Outputs:
  - ``lag_preference_breakdown.csv``: long-form CSV with one row per
    (metric, roi, lag).
  - Console-friendly tables for quick reading.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent

sys.path.insert(0, str(REPO_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("lag_pref_analysis")

SUBJECT_TO_UTS = {"S1": "UTS01", "S2": "UTS02", "S3": "UTS03"}
SUB_ROIS = ["BA_10", "BA_6", "BA_8", "BA_9_46", "BROCA"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True,
                   help="Directory containing lag_corrs.npz produced by train_lag_encoding.py.")
    p.add_argument("--subject", default=None,
                   help="Subject id (S1/S2/S3). If omitted, read from lag_corrs.npz.")
    p.add_argument("--ba-dir", default=str(REPO_DIR / "ba_indices"),
                   help="Directory of Brodmann ROI JSON files.")
    p.add_argument("--csv-out", default=None,
                   help="Output CSV path (default: <results-dir>/lag_preference_breakdown.csv).")
    p.add_argument("--top-k", type=int, default=2000,
                   help="Top-k voxels by best-lag r to include in the 'top_k_*' breakdowns.")
    return p.parse_args()


def load_sub_roi_masks(ba_dir: Path, uts_id: str, vox: np.ndarray) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {"full_frontal": np.ones_like(vox, dtype=bool)}
    for name in SUB_ROIS:
        path = ba_dir / uts_id / f"{name}.json"
        if not path.is_file():
            log.warning("Missing ROI file: %s", path)
            continue
        with open(path, encoding="utf-8") as f:
            ids = np.asarray(next(iter(json.load(f).values())), dtype=int)
        mask = np.isin(vox, ids)
        if mask.sum() == 0:
            log.warning("ROI %s has no overlap with full_frontal voxels.", name)
            continue
        masks[name] = mask
    return masks


def fmt_table(
    title: str,
    lags: Iterable[int],
    rows: Dict[str, List[float]],
    fmt: str = "{:+.3f}",
) -> str:
    lags = list(lags)
    header = f"{'ROI':<14} | " + " | ".join(f"lag={l:<3}" for l in lags)
    lines = [title, header, "-" * len(header)]
    for name, vals in rows.items():
        cells = " | ".join(fmt.format(v) for v in vals)
        lines.append(f"{name:<14} | {cells}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    npz_path = results_dir / "lag_corrs.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Expected lag_corrs.npz at {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    corrs: np.ndarray = np.asarray(data["corrs"], dtype=np.float32)
    lags: np.ndarray = np.asarray(data["lags"], dtype=int)
    voxels: np.ndarray = np.asarray(data["voxels"], dtype=int)

    subject = args.subject
    if subject is None and "subject" in data.files:
        subject = str(data["subject"])
    if subject is None:
        config_path = results_dir / "config.json"
        if config_path.is_file():
            with open(config_path, encoding="utf-8") as f:
                subject = json.load(f).get("subject")
    if subject is None:
        raise ValueError("Cannot resolve subject. Pass --subject or include it in the npz/config.")
    uts_id = SUBJECT_TO_UTS[subject]
    log.info("Subject %s -> %s", subject, uts_id)
    log.info("corrs shape: %s, lags: %s, voxels: %d", corrs.shape, lags.tolist(), voxels.size)

    ba_dir = Path(args.ba_dir).expanduser().resolve()
    masks = load_sub_roi_masks(ba_dir, uts_id, voxels)
    for name, mask in masks.items():
        log.info("  ROI %-13s -> %d voxels", name, int(mask.sum()))

    # Within-lag z (across full_frontal voxels) - relative anatomy summary.
    z_corrs = np.zeros_like(corrs)
    for li in range(corrs.shape[0]):
        m = float(corrs[li].mean())
        s = float(corrs[li].std()) or 1e-8
        z_corrs[li] = (corrs[li] - m) / s

    preferred = np.argmax(corrs, axis=0)
    preferred_lag = lags[preferred]
    best_r = corrs[preferred, np.arange(corrs.shape[1])]
    other_mean = (corrs.sum(axis=0) - best_r) / max(1, len(lags) - 1)
    sharpness = best_r - other_mean

    # Top-k sharpness selection (across all full_frontal voxels)
    k = int(min(max(1, args.top_k), best_r.size))
    top_idx = np.argsort(-best_r)[:k]

    rows: List[dict] = []

    mean_r_per = {}
    median_r_per = {}
    z_per = {}
    sharp_per = {}
    pref_frac_per: Dict[str, List[float]] = {}
    for name, mask in masks.items():
        if mask.sum() == 0:
            continue
        c = corrs[:, mask]
        mean_r_per[name] = c.mean(axis=1).tolist()
        median_r_per[name] = np.median(c, axis=1).tolist()
        z_per[name] = z_corrs[:, mask].mean(axis=1).tolist()
        sharp_per[name] = float(sharpness[mask].mean())
        pref_in_roi = preferred_lag[mask]
        pref_frac = [float((pref_in_roi == l).mean()) for l in lags]
        pref_frac_per[name] = pref_frac

        for li, lag in enumerate(lags):
            rows.append({
                "metric": "mean_r", "roi": name, "lag": int(lag),
                "value": float(mean_r_per[name][li]),
            })
            rows.append({
                "metric": "median_r", "roi": name, "lag": int(lag),
                "value": float(median_r_per[name][li]),
            })
            rows.append({
                "metric": "mean_within_z", "roi": name, "lag": int(lag),
                "value": float(z_per[name][li]),
            })
            rows.append({
                "metric": "preferred_frac", "roi": name, "lag": int(lag),
                "value": float(pref_frac[li]),
            })
        rows.append({
            "metric": "sharpness_mean", "roi": name, "lag": -1,
            "value": float(sharp_per[name]),
        })
        rows.append({
            "metric": "n_voxels", "roi": name, "lag": -1,
            "value": int(mask.sum()),
        })

    # Top-k voxel breakdown - which lag dominates among the strongest voxels.
    top_pref = preferred_lag[top_idx]
    top_fracs = [float((top_pref == l).mean()) for l in lags]
    for lag, frac in zip(lags, top_fracs):
        rows.append({
            "metric": f"top{k}_preferred_frac", "roi": "top_voxels", "lag": int(lag),
            "value": float(frac),
        })

    log.info("\n%s", fmt_table("== Mean per-voxel r per ROI per lag ==", lags, mean_r_per, fmt="{:+.4f}"))
    log.info("\n%s", fmt_table("== Median per-voxel r per ROI per lag ==", lags, median_r_per, fmt="{:+.4f}"))
    log.info("\n%s", fmt_table(
        "== Within-lag z (vs full-frontal mean for that lag) ==",
        lags,
        z_per,
        fmt="{:+.3f}",
    ))
    log.info("\n%s", fmt_table(
        "== Preferred-lag fractions per ROI (% of voxels) ==",
        lags,
        {name: [v * 100.0 for v in fracs] for name, fracs in pref_frac_per.items()},
        fmt="{:5.1f}",
    ))
    log.info("\n== Mean tuning sharpness (best_r - mean_other_lag_r) ==")
    for name, val in sharp_per.items():
        log.info("  %-14s %+.4f", name, val)
    log.info("\n== Top-%d voxels (by best-lag r): preferred-lag fractions ==", k)
    log.info("  %s", " | ".join(f"lag={l}: {f*100:5.1f}%" for l, f in zip(lags, top_fracs)))

    csv_path = Path(args.csv_out) if args.csv_out else results_dir / "lag_preference_breakdown.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "roi", "lag", "value"])
        writer.writeheader()
        writer.writerows(rows)
    log.info("\nWrote %s", csv_path)


if __name__ == "__main__":
    main()
