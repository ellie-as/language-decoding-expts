#!/usr/bin/env python3
"""Render pycortex flatmaps from voxel_timescale.npz.

Loads the per-voxel timescale metrics produced by ``compute_voxel_timescale.py``
and writes one flatmap per metric (in seconds by default). Voxels outside the
chosen scope (``all`` or ``full_frontal``) and voxels with NaN metrics render
as the curvature background.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("voxel_timescale.flatmap")

SUBJECT_TO_UTS = {"S1": "UTS01", "S2": "UTS02", "S3": "UTS03"}
N_TOTAL_VOXELS_DEFAULT = {"S1": 81126, "S2": 94251, "S3": 95556}

DEFAULT_METRICS = ["half_life_seconds", "exp_tau_seconds", "integrated_ac_seconds"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True,
                   help="Directory containing voxel_timescale.npz.")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                   help=f"Metric names from the npz to plot (default: {DEFAULT_METRICS}).")
    p.add_argument("--pycortex-subject", default=None,
                   help="Pycortex db subject id (UTS01/UTS02/UTS03).")
    p.add_argument("--pycortex-filestore", default=None,
                   help="Pycortex filestore (e.g. <repo>/pycortex-db).")
    p.add_argument("--n-total-voxels", type=int, default=None,
                   help="Full-volume voxel count (default: read from npz/per-subject fallback).")
    p.add_argument("--cmap", default="magma",
                   help="Colormap for the timescale flatmaps.")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--clip-quantiles", nargs=2, type=float, default=(0.05, 0.95),
                   metavar=("LOW", "HIGH"),
                   help="Auto vmin/vmax = these quantiles of finite values (default 0.05 0.95).")
    p.add_argument("--out-dir", default=None,
                   help="Default: <results-dir>/flatmaps")
    return p.parse_args()


def resolve_subject(npz_data, results_dir: Path) -> str:
    if "subject" in npz_data.files:
        return str(np.asarray(npz_data["subject"]).reshape(-1)[0])
    cfg = results_dir / "summary.json"
    if cfg.is_file():
        with open(cfg, encoding="utf-8") as f:
            subj = json.load(f).get("subject")
            if subj:
                return subj
    raise ValueError("Cannot resolve subject. Pass --pycortex-subject UTS0X.")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    npz_path = results_dir / "voxel_timescale.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Expected voxel_timescale.npz at {npz_path}")

    raw = np.load(npz_path, allow_pickle=True)
    voxels = np.asarray(raw["voxels"], dtype=int)
    subject = resolve_subject(raw, results_dir)
    pycortex_subject = args.pycortex_subject or SUBJECT_TO_UTS[subject]
    n_total = int(args.n_total_voxels) if args.n_total_voxels else (
        int(raw["n_total_voxels"]) if "n_total_voxels" in raw.files else N_TOTAL_VOXELS_DEFAULT[subject]
    )
    log.info("Subject=%s pycortex=%s n_total_voxels=%d voxels=%d",
             subject, pycortex_subject, n_total, voxels.size)

    out_dir = Path(args.out_dir) if args.out_dir else (results_dir / "flatmaps")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pycortex_filestore:
        os.environ.setdefault("PYCORTEX_FILESTORE", args.pycortex_filestore)
    try:
        import cortex  # noqa: F401
    except ImportError:
        log.error("pycortex is not installed. `pip install pycortex` first.")
        sys.exit(1)
    if args.pycortex_filestore:
        try:
            cortex.database.default_filestore = args.pycortex_filestore
            log.info("pycortex filestore -> %s", args.pycortex_filestore)
        except Exception as err:
            log.warning("Could not override pycortex filestore: %s", err)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    low_q, high_q = args.clip_quantiles
    written: List[Path] = []
    for name in args.metrics:
        if name not in raw.files:
            log.warning("Metric '%s' not found in npz; skipping", name)
            continue
        values = np.asarray(raw[name], dtype=np.float32)
        if values.size != voxels.size:
            log.warning("Metric '%s' has wrong size (%d vs %d); skipping", name, values.size, voxels.size)
            continue
        full = np.full(int(n_total), np.nan, dtype=np.float32)
        full[voxels] = values

        finite = values[np.isfinite(values)]
        if finite.size == 0:
            log.warning("Metric '%s' has no finite values; skipping", name)
            continue
        vmin = float(args.vmin) if args.vmin is not None else float(np.quantile(finite, low_q))
        vmax = float(args.vmax) if args.vmax is not None else float(np.quantile(finite, high_q))
        if vmax <= vmin:
            vmax = float(vmin + max(1e-3, abs(vmin) * 0.1))

        vol = cortex.Volume(full, pycortex_subject, "fullhead", vmin=vmin, vmax=vmax, cmap=args.cmap)
        fig = cortex.quickflat.make_figure(vol, with_curvature=True, with_colorbar=True)
        title = f"{name}    vmin={vmin:.2g}, vmax={vmax:.2g}"
        fig.suptitle(title, fontsize=13)
        out_path = out_dir / f"{name}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(
            "  %-25s mean=%.3g median=%.3g  ->  %s",
            name, float(np.nanmean(values)), float(np.nanmedian(values)), out_path,
        )
        written.append(out_path)

    if not written:
        log.error("No flatmaps written. Available metrics: %s", list(raw.files))
        sys.exit(1)
    log.info("Done. %d flatmaps in %s", len(written), out_dir)


if __name__ == "__main__":
    main()
