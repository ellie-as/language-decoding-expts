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
    p.add_argument("--pycortex-filestore", default=str(REPO_DIR / "pycortex-db"),
                   help="Pycortex filestore (only used to load mask_thick+reference for the "
                        "within-ROI spatial-gradient analysis).")
    p.add_argument("--xfm-name", default=None,
                   help="Pycortex transform name (default: <UTS>_auto when present).")
    p.add_argument("--gradient-r-threshold", type=float, default=0.05,
                   help="Restrict spatial gradient analysis to voxels with best-lag r >= this.")
    p.add_argument("--no-spatial-gradient", action="store_true",
                   help="Skip the within-ROI spatial-gradient table.")
    return p.parse_args()


def load_voxel_xyz(filestore: Path, uts_id: str, xfm_name: str | None) -> np.ndarray | None:
    """Return MNI mm coordinates for every voxel in the response volume (HDF5 column order).

    Returns ``None`` if the required pycortex files are not present.
    """
    try:
        import nibabel as nib
    except ImportError:
        log.warning("nibabel not installed; skipping spatial gradient analysis.")
        return None

    xfm_dir = filestore / uts_id / "transforms"
    if not xfm_dir.is_dir():
        log.warning("No transforms dir under %s; skipping spatial gradient analysis.", xfm_dir)
        return None
    xfm = xfm_name
    if xfm is None:
        candidates = [p.name for p in xfm_dir.iterdir() if p.is_dir()]
        preferred = f"{uts_id}_auto"
        xfm = preferred if preferred in candidates else (candidates[0] if candidates else None)
    if xfm is None:
        log.warning("No transform under %s; skipping spatial gradient analysis.", xfm_dir)
        return None
    base = xfm_dir / xfm
    mask_path = base / "mask_thick.nii.gz"
    ref_path = base / "reference.nii.gz"
    if not (mask_path.is_file() and ref_path.is_file()):
        log.warning("Missing %s or %s; skipping spatial gradient analysis.", mask_path, ref_path)
        return None
    mask = nib.load(str(mask_path)).get_fdata() > 0
    ref_affine = nib.load(str(ref_path)).affine
    ijk = np.array(np.where(mask)).T
    return nib.affines.apply_affine(ref_affine, ijk).astype(np.float64)


def fit_axis(coord: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return (slope_per_mm, pearson_r) for target ~= a + b*coord."""
    if coord.std() < 1e-9 or target.std() < 1e-9:
        return float("nan"), float("nan")
    slope = float(np.polyfit(coord, target, 1)[0])
    r = float(np.corrcoef(coord, target)[0, 1])
    return slope, r


def spatial_gradient_table(
    masks: Dict[str, np.ndarray],
    xyz: np.ndarray,
    target: np.ndarray,
    keep: np.ndarray,
    *,
    target_name: str,
) -> tuple[str, list[dict]]:
    """Compute LR/PA/DV slopes per ROI per hemisphere. Returns (text_table, rows)."""
    AXES = [("x_LR(+R)", 0), ("y_PA(+A)", 1), ("z_DV(+D)", 2)]
    HEMIS = [
        ("ALL", np.ones(len(xyz), bool)),
        ("LH", xyz[:, 0] < 0),
        ("RH", xyz[:, 0] > 0),
    ]
    rows: list[dict] = []
    lines = [f"== Spatial gradient of {target_name} (TR per metre, Pearson r) =="]
    header = f"{'hemi':<4} {'ROI':<14} {'n':>6}  " + "  ".join(
        f"{ax} slope/m  r" for ax, _ in AXES
    )
    lines.append(header)
    lines.append("-" * len(header))
    for hemi_name, hemi_mask in HEMIS:
        for roi_name, roi_mask in masks.items():
            mask = roi_mask & hemi_mask & keep & ~np.isnan(target)
            n = int(mask.sum())
            if n < 50:
                lines.append(f"{hemi_name:<4} {roi_name:<14} {n:>6}  (n<50, skipped)")
                continue
            cells: list[str] = []
            for ax_name, ax_idx in AXES:
                slope, r = fit_axis(xyz[mask, ax_idx], target[mask])
                rows.append({
                    "metric": f"gradient_{target_name}",
                    "roi": roi_name,
                    "hemi": hemi_name,
                    "axis": ax_name,
                    "slope_per_mm": slope,
                    "slope_per_m": slope * 1000.0,
                    "pearson_r": r,
                    "n_voxels": n,
                })
                cells.append(f"{slope*1000:>+10.2f}  {r:>+6.3f}")
            lines.append(f"{hemi_name:<4} {roi_name:<14} {n:>6}  " + "  ".join(cells))
    return "\n".join(lines), rows


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

    if args.no_spatial_gradient:
        return

    filestore = Path(args.pycortex_filestore).expanduser().resolve()
    xyz = load_voxel_xyz(filestore, uts_id, args.xfm_name)
    if xyz is None:
        log.info("Skipping spatial gradient analysis (filestore lookup failed).")
        return
    if int(xyz.shape[0]) < int(np.max(voxels)) + 1:
        log.warning(
            "Mask voxel count (%d) is smaller than max voxel id (%d). Skipping spatial gradient.",
            xyz.shape[0], int(np.max(voxels)),
        )
        return

    voxel_xyz = xyz[voxels]
    com_target = np.zeros(corrs.shape[1], dtype=np.float64)
    weights = np.clip(corrs, 0.0, None)
    denom = weights.sum(axis=0)
    valid = denom > 0
    com_target[valid] = (weights[:, valid] * lags[:, None].astype(np.float64)).sum(axis=0) / denom[valid]
    com_target[~valid] = np.nan

    keep = best_r >= float(args.gradient_r_threshold)
    log.info(
        "\nSpatial gradient analysis: %d/%d voxels with best-lag r >= %.3f",
        int(keep.sum()), int(keep.size), args.gradient_r_threshold,
    )

    pref_text, pref_rows = spatial_gradient_table(
        masks, voxel_xyz, preferred_lag.astype(np.float64), keep,
        target_name="preferred_lag",
    )
    com_text, com_rows = spatial_gradient_table(
        masks, voxel_xyz, com_target, keep,
        target_name="com_lag",
    )
    log.info("\n%s", pref_text)
    log.info("\n%s", com_text)

    grad_csv = csv_path.with_name("lag_preference_spatial_gradient.csv")
    fieldnames = ["metric", "roi", "hemi", "axis", "slope_per_mm", "slope_per_m", "pearson_r", "n_voxels"]
    with open(grad_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pref_rows + com_rows)
    log.info("Wrote %s", grad_csv)


if __name__ == "__main__":
    main()
