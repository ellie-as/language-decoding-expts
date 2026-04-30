#!/usr/bin/env python3
"""Plot pycortex flatmaps of preferred lag and best-lag r.

Loads ``lag_corrs.npz`` produced by ``train_lag_encoding.py`` and renders:

  - ``preferred_lag.png``         argmax_lag corrs per voxel.
  - ``preferred_lag_masked.png``  same, but voxels with best-lag r below
                                  ``--mask-r-threshold`` are NaN-masked so the
                                  gradient is dominated by reliable voxels.
  - ``com_lag.png``               r-weighted center-of-mass over lags
                                  (smoother than argmax).
  - ``com_lag_masked.png``        center-of-mass with the same mask applied.
  - ``best_lag_r.png``            best-lag Pearson r per voxel (intensity).
  - ``per_lag/lag<lag>.png``      one r flatmap per lag (off by default).

Voxels outside ``BA_full_frontal`` are NaN so they show curvature only.

Requires pycortex to be installed and configured for the subject (``UTS01``,
``UTS02`` or ``UTS03``). Pass ``--pycortex-filestore`` if your repo's
``pycortex-db`` directory is not the default filestore.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("lag_pref_flatmaps")

SUBJECT_TO_UTS = {"S1": "UTS01", "S2": "UTS02", "S3": "UTS03"}
N_TOTAL_VOXELS_DEFAULT = {"S1": 81126, "S2": 94251, "S3": 95556}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True,
                   help="Directory containing lag_corrs.npz (output of train_lag_encoding.py).")
    p.add_argument("--subject", default=None, help="Subject S1/S2/S3 (defaults to value saved in npz/config).")
    p.add_argument("--pycortex-subject", default=None, help="Override pycortex db subject (UTS01/UTS02/UTS03).")
    p.add_argument("--pycortex-filestore", default=None,
                   help="Pycortex filestore directory (default: <repo>/pycortex-db when present).")
    p.add_argument("--n-total-voxels", type=int, default=None,
                   help="Total voxels in the subject volume (auto-detected from response file otherwise).")
    p.add_argument("--mask-r-threshold", type=float, default=0.05,
                   help="In '_masked' maps, voxels with best-lag r < threshold are NaN-masked.")
    p.add_argument("--cmap-pref", default="viridis",
                   help="Colormap for lag-preference maps.")
    p.add_argument("--cmap-r", default="inferno",
                   help="Colormap for best-lag-r intensity map.")
    p.add_argument("--per-lag", action="store_true",
                   help="Also write per-lag r flatmaps under per_lag/.")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: <results-dir>/flatmaps).")
    return p.parse_args()


def resolve_subject(args: argparse.Namespace, results_dir: Path, data: Dict[str, np.ndarray]) -> str:
    if args.subject:
        return args.subject
    if "subject" in data:
        return str(np.asarray(data["subject"]).reshape(-1)[0])
    config_path = results_dir / "config.json"
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            subj = json.load(f).get("subject")
            if subj:
                return subj
    raise ValueError("Cannot resolve subject. Pass --subject S1/S2/S3.")


def detect_n_total_voxels(subject: str) -> int:
    """Try to read the full voxel count from one response file."""
    try:
        import config  # noqa: F401
        from utils_resp import get_resp  # noqa: F401
        sess_to_story_path = Path(config.DATA_TRAIN_DIR) / "sess_to_story.json"
        if not sess_to_story_path.is_file():
            raise FileNotFoundError(sess_to_story_path)
        with open(sess_to_story_path, encoding="utf-8") as f:
            sess_to_story = json.load(f)
        first_story = next(iter(sess_to_story.values()))[0]
        sample = get_resp(subject, [first_story], stack=True, vox=None)
        return int(sample.shape[1])
    except Exception as err:
        log.warning("Could not auto-detect n_total_voxels (%s); using fallback.", err)
        return int(N_TOTAL_VOXELS_DEFAULT[subject])


def configure_pycortex_filestore(explicit_path: str | None) -> str | None:
    """Make pycortex use the repo-local pycortex-db before any cortex import.

    Returns the resolved filestore directory or None if no override is applied.
    """
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser().resolve())
    candidates.append(REPO_DIR / "pycortex-db")
    candidates.append(Path.cwd() / "pycortex-db")

    for candidate in candidates:
        if candidate.is_dir():
            store = str(candidate)
            os.environ["PYCORTEX_FILESTORE"] = store
            try:
                import cortex  # noqa: F401
                from cortex.options import config as cortex_config
                cortex_config.set("basic", "filestore", store)
                if hasattr(cortex, "database"):
                    try:
                        cortex.database.default_filestore = store
                    except Exception:  # pragma: no cover - older pycortex
                        pass
                    if hasattr(cortex, "db") and hasattr(cortex.db, "reload_subjects"):
                        cortex.db.reload_subjects()
                    elif hasattr(cortex, "db"):
                        cortex.db = cortex.database.Database()
                log.info("Pycortex filestore -> %s", store)
                return store
            except ImportError:
                log.error("pycortex is not installed. Run `pip install pycortex` first.")
                raise
    log.warning(
        "No pycortex-db directory found (looked at: %s). "
        "Falling back to pycortex's default config.",
        ", ".join(str(c) for c in candidates),
    )
    return None


def lag_center_of_mass(corrs: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """r-weighted average lag (positive r only). NaN where total weight is 0."""
    weights = np.clip(corrs, 0.0, None)
    s = weights.sum(axis=0)
    com = np.divide(
        (weights * lags[:, None].astype(np.float64)).sum(axis=0),
        s,
        where=s > 0,
        out=np.zeros_like(s, dtype=np.float64),
    )
    com = com.astype(np.float32)
    com[s <= 0] = np.nan
    return com


def project_to_full_brain(values: np.ndarray, voxels: np.ndarray, n_total: int) -> np.ndarray:
    full = np.full(int(n_total), np.nan, dtype=np.float32)
    full[voxels] = values
    return full


def make_flatmap(
    cortex_module,
    plt_module,
    full_volume: np.ndarray,
    *,
    pycortex_subject: str,
    vmin: float,
    vmax: float,
    cmap: str,
    title: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vol = cortex_module.Volume(full_volume, pycortex_subject, "fullhead", vmin=vmin, vmax=vmax, cmap=cmap)
    fig = cortex_module.quickflat.make_figure(vol, with_curvature=True, with_colorbar=True)
    fig.suptitle(title, fontsize=13)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt_module.close(fig)
    log.info("Saved %s", out_path)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    npz_path = results_dir / "lag_corrs.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Expected lag_corrs.npz at {npz_path}")

    raw = np.load(npz_path, allow_pickle=True)
    data: Dict[str, np.ndarray] = {key: raw[key] for key in raw.files}

    corrs = np.asarray(data["corrs"], dtype=np.float32)  # (n_lags, n_voxels)
    lags = np.asarray(data["lags"], dtype=int)
    voxels = np.asarray(data["voxels"], dtype=int)

    subject = resolve_subject(args, results_dir, data)
    pycortex_subject = args.pycortex_subject or SUBJECT_TO_UTS[subject]
    n_total = args.n_total_voxels or detect_n_total_voxels(subject)
    log.info(
        "Subject=%s pycortex=%s n_total_voxels=%d corrs=%s lags=%s",
        subject, pycortex_subject, n_total, corrs.shape, lags.tolist(),
    )

    out_dir = Path(args.out_dir) if args.out_dir else (results_dir / "flatmaps")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        configure_pycortex_filestore(args.pycortex_filestore)
        import cortex  # noqa: F401
    except ImportError:
        sys.exit(1)

    if pycortex_subject not in cortex.db.subjects:
        raise SystemExit(
            f"pycortex does not know about subject {pycortex_subject!r} after pointing it at the "
            "filestore. Subjects available: "
            f"{sorted(cortex.db.subjects.keys())}. Either pass --pycortex-filestore explicitly "
            f"or run download_pycortex_files.py to populate {REPO_DIR / 'pycortex-db'}."
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pref_idx = np.argmax(corrs, axis=0)
    preferred_lag = lags[pref_idx].astype(np.float32)
    best_r = corrs[pref_idx, np.arange(corrs.shape[1])].astype(np.float32)
    com = lag_center_of_mass(corrs, lags)

    mask = best_r >= args.mask_r_threshold
    masked_pref = preferred_lag.copy(); masked_pref[~mask] = np.nan
    masked_com = com.copy(); masked_com[~mask] = np.nan
    log.info(
        "best-lag r: mean=%.4f median=%.4f p95=%.4f -- mask >= %.3f keeps %d/%d voxels",
        float(best_r.mean()), float(np.median(best_r)), float(np.quantile(best_r, 0.95)),
        args.mask_r_threshold, int(mask.sum()), int(mask.size),
    )

    np.savez(
        out_dir / "lag_preference_maps.npz",
        voxels=voxels,
        preferred_lag=preferred_lag,
        preferred_lag_masked=masked_pref,
        center_of_mass_lag=com,
        center_of_mass_lag_masked=masked_com,
        best_r=best_r,
        mask_threshold=float(args.mask_r_threshold),
        lags=lags,
        subject=subject,
        n_total_voxels=int(n_total),
    )

    pref_vmin, pref_vmax = float(lags.min()), float(lags.max())
    r_top = max(0.10, float(np.quantile(best_r, 0.99)))

    make_flatmap(
        cortex, plt,
        project_to_full_brain(masked_pref, voxels, n_total),
        pycortex_subject=pycortex_subject,
        vmin=pref_vmin, vmax=pref_vmax, cmap=args.cmap_pref,
        title=f"Preferred lag (TR), masked r >= {args.mask_r_threshold:g}",
        out_path=out_dir / "preferred_lag_masked.png",
    )
    make_flatmap(
        cortex, plt,
        project_to_full_brain(preferred_lag, voxels, n_total),
        pycortex_subject=pycortex_subject,
        vmin=pref_vmin, vmax=pref_vmax, cmap=args.cmap_pref,
        title="Preferred lag (TR), all full_frontal voxels",
        out_path=out_dir / "preferred_lag.png",
    )
    make_flatmap(
        cortex, plt,
        project_to_full_brain(masked_com, voxels, n_total),
        pycortex_subject=pycortex_subject,
        vmin=pref_vmin, vmax=pref_vmax, cmap=args.cmap_pref,
        title=f"r-weighted center-of-mass lag, masked r >= {args.mask_r_threshold:g}",
        out_path=out_dir / "com_lag_masked.png",
    )
    make_flatmap(
        cortex, plt,
        project_to_full_brain(com, voxels, n_total),
        pycortex_subject=pycortex_subject,
        vmin=pref_vmin, vmax=pref_vmax, cmap=args.cmap_pref,
        title="r-weighted center-of-mass lag",
        out_path=out_dir / "com_lag.png",
    )
    make_flatmap(
        cortex, plt,
        project_to_full_brain(best_r, voxels, n_total),
        pycortex_subject=pycortex_subject,
        vmin=0.0, vmax=r_top, cmap=args.cmap_r,
        title=f"Best-lag encoding r per voxel (vmax={r_top:.2f})",
        out_path=out_dir / "best_lag_r.png",
    )

    if args.per_lag:
        per_lag_dir = out_dir / "per_lag"
        per_lag_dir.mkdir(parents=True, exist_ok=True)
        diverging_top = max(0.10, float(np.quantile(np.abs(corrs), 0.99)))
        for li, lag in enumerate(lags):
            make_flatmap(
                cortex, plt,
                project_to_full_brain(corrs[li], voxels, n_total),
                pycortex_subject=pycortex_subject,
                vmin=-diverging_top, vmax=diverging_top, cmap="RdBu_r",
                title=f"Encoding r at lag={int(lag)} TR",
                out_path=per_lag_dir / f"lag{int(lag):02d}.png",
            )

    log.info("All maps written to %s", out_dir)


if __name__ == "__main__":
    main()
