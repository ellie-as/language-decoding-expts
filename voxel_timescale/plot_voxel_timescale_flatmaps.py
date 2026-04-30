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

DEFAULT_METRICS = ["half_life_seconds", "exp_tau_seconds", "positive_integrated_ac_seconds"]


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
    p.add_argument("--xfm-name", default=None,
                   help="Pycortex transform/xfm name (default: auto-detect by mask size).")
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


def configure_pycortex_filestore(explicit_path: str | None) -> str | None:
    """Point pycortex at the repo filestore unless the user overrides it."""
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser().resolve())
    candidates.append(REPO_DIR / "pycortex-db")
    candidates.append(Path.cwd() / "pycortex-db")

    chosen: Path | None = None
    for candidate in candidates:
        if candidate.is_dir():
            chosen = candidate
            break
    if chosen is None:
        log.warning(
            "No pycortex-db directory found (looked at: %s). Falling back to pycortex defaults.",
            ", ".join(str(c) for c in candidates),
        )
        return None

    store = str(chosen)
    os.environ["PYCORTEX_FILESTORE"] = store

    try:
        import cortex  # noqa: F401
    except ImportError:
        log.error("pycortex is not installed. `pip install pycortex` first.")
        raise

    try:
        from cortex.options import config as cortex_config
        if not cortex_config.has_section("basic"):
            cortex_config.add_section("basic")
        cortex_config.set("basic", "filestore", store)
    except Exception as err:
        log.warning("Could not update cortex.options.config: %s", err)

    if hasattr(cortex, "database"):
        try:
            cortex.database.default_filestore = store
        except Exception:
            pass

    try:
        cortex.db.filestore = store
        for attr in ("_subjects", "subjects_cache"):
            if hasattr(cortex.db, attr):
                try:
                    setattr(cortex.db, attr, None)
                except Exception:
                    pass
    except Exception as err:
        log.warning("Could not patch cortex.db filestore directly: %s", err)

    try:
        cortex.db = cortex.database.Database()
        cortex.db.filestore = store
    except Exception as err:
        log.warning("Could not recreate cortex.db: %s", err)

    log.info("Pycortex filestore -> %s", getattr(cortex.db, "filestore", store))
    log.info("Pycortex subjects   -> %s", sorted(cortex.db.subjects.keys()))
    return store


def list_subject_transforms(filestore: str | None, subject: str) -> list[str]:
    if not filestore:
        return []
    base = Path(filestore) / subject / "transforms"
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def find_matching_xfm(
    cortex_module,
    filestore: str | None,
    subject: str,
    n_total: int,
    preferred: str | None,
) -> tuple[str, int]:
    """Find a transform whose thick-mask voxel count matches ``n_total``."""
    candidates: list[str] = []
    seen: set[str] = set()

    def add(name: str | None) -> None:
        if name and name not in seen:
            candidates.append(name)
            seen.add(name)

    add(preferred)
    add("fullhead")
    add(f"{subject}_auto")
    for name in list_subject_transforms(filestore, subject):
        add(name)

    if not candidates:
        raise RuntimeError(f"No transforms found for pycortex subject {subject!r}.")

    matches: list[tuple[str, int]] = []
    for name in candidates:
        try:
            mask = cortex_module.db.get_mask(subject, name, "thick")
            n_mask = int(np.asarray(mask).sum())
        except Exception as err:
            log.debug("xfm %s/%s: could not load thick mask (%s)", subject, name, err)
            continue
        log.info("Transform candidate %s/%s -> thick mask voxels=%d", subject, name, n_mask)
        matches.append((name, n_mask))
        if n_mask == int(n_total):
            return name, n_mask

    if matches:
        name, n_mask = matches[0]
        log.warning(
            "No transform exactly matched n_total=%d. Using %s (mask voxels=%d).",
            n_total, name, n_mask,
        )
        return name, n_mask

    raise RuntimeError(
        f"None of {candidates!r} returned a usable thick mask for subject {subject!r}. "
        "Try --xfm-name with one of the transform directory names in pycortex-db."
    )


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
    log.info("Subject=%s n_total_voxels=%d voxels=%d", subject, n_total, voxels.size)

    out_dir = Path(args.out_dir) if args.out_dir else (results_dir / "flatmaps")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        filestore = configure_pycortex_filestore(args.pycortex_filestore)
        import cortex  # noqa: F401
    except ImportError:
        sys.exit(1)

    available = sorted(cortex.db.subjects.keys())
    if args.pycortex_subject:
        candidates = [args.pycortex_subject]
    else:
        candidates = [
            SUBJECT_TO_UTS.get(subject, ""),
            subject,
            f"sub-{SUBJECT_TO_UTS.get(subject, '')}",
            f"sub-{subject}",
        ]
    pycortex_subject = next((name for name in candidates if name and name in available), None)
    if pycortex_subject is None:
        raise SystemExit(
            f"pycortex does not know about any of {candidates!r} for subject {subject!r}. "
            f"Subjects available in the filestore: {available}. Pass --pycortex-subject, "
            "--pycortex-filestore, or run download_pycortex_files.py."
        )

    xfm_name, mask_voxels = find_matching_xfm(
        cortex, filestore, pycortex_subject, int(n_total), args.xfm_name
    )
    if mask_voxels != int(n_total):
        log.warning(
            "Mask size %d != n_total_voxels %d; using mask size for projection.",
            mask_voxels, int(n_total),
        )
        n_total = mask_voxels
    log.info("Using pycortex subject=%s xfm=%s", pycortex_subject, xfm_name)

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

        vol = cortex.Volume(full, pycortex_subject, xfm_name, vmin=vmin, vmax=vmax, cmap=args.cmap)
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
