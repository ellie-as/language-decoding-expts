#!/usr/bin/env python3
"""Map and summarize peak frontal clusters from combo LGBM voxel importances."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(THIS_DIR))

from plot_lag_flatmaps import (  # noqa: E402
    SUBJECT_TO_UTS,
    configure_pycortex_filestore,
    find_matching_xfm,
    make_flatmap,
    project_to_full_brain,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("combo_lgbm_peak_clusters")

BLOCKS = ["1TR", "h20", "h50", "h200"]
CONTEXT_WEIGHTS = {"1TR": 1.0, "h20": 20.0, "h50": 50.0, "h200": 200.0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"])
    p.add_argument(
        "--results-root",
        default=str(REPO_DIR / "lag_preference_analysis" / "results"),
        help="Root containing <SUB>__embedding-summary-combo... result directories.",
    )
    p.add_argument("--tag-template", default="{subject}__embedding-summary-combo-h20-50-200__lags1-10__chunk1tr__seed0")
    p.add_argument("--lgbm-prefix", default="combo_lgbm_block_importance_lag2_top200")
    p.add_argument("--out-dir", default=str(THIS_DIR / "results" / "combo_lgbm_peak_clusters"))
    p.add_argument("--pycortex-filestore", default=str(REPO_DIR / "pycortex-db"))
    p.add_argument("--ba-dir", default=str(REPO_DIR / "ba_indices"))
    p.add_argument("--xfm-name", default=None)
    p.add_argument("--top-n", type=int, default=0, help="Optionally keep only the top N rows by saved_combo_best_r.")
    p.add_argument("--min-cluster-size", type=int, default=3)
    p.add_argument("--with-rois", action="store_true")
    return p.parse_args()


def load_mask_ijk(pycortex_filestore: Path, pycortex_subject: str) -> tuple[np.ndarray, np.ndarray]:
    mask_path = pycortex_filestore / pycortex_subject / "transforms" / f"{pycortex_subject}_auto" / "mask_thick.nii.gz"
    mask = nib.load(str(mask_path)).get_fdata() > 0
    return mask, np.asarray(np.where(mask)).T


def load_roi_lookup(ba_dir: Path, pycortex_subject: str) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    subject_dir = ba_dir / pycortex_subject
    for roi in ["BA_10", "BA_9_46", "BA_8", "BA_6", "BROCA", "BA_full_frontal"]:
        path = subject_dir / f"{roi}.json"
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            out[roi] = set(map(int, next(iter(json.load(f).values()))))
    return out


def primary_roi(cluster_voxels: np.ndarray, roi_lookup: dict[str, set[int]]) -> str:
    counts = {
        roi: int(sum(int(v) in ids for v in cluster_voxels))
        for roi, ids in roi_lookup.items()
        if roi != "BA_full_frontal"
    }
    if not counts:
        return "unknown"
    best_roi, best_count = max(counts.items(), key=lambda item: item[1])
    return best_roi if best_count > 0 else "outside_named_rois"


def load_lgbm_voxel_rows(path: Path, top_n: int) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if top_n > 0:
        rows = sorted(rows, key=lambda r: float(r["saved_combo_best_r"]), reverse=True)[: int(top_n)]

    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["global_voxel_index"])].append(row)

    out: list[dict] = []
    for voxel, voxel_rows in grouped.items():
        item = {
            "global_voxel_index": voxel,
            "source_rois": ";".join(sorted({r["roi"] for r in voxel_rows})),
            "saved_combo_best_r": float(np.mean([float(r["saved_combo_best_r"]) for r in voxel_rows])),
            "saved_combo_lag_r": float(np.mean([float(r["saved_combo_lag_r"]) for r in voxel_rows])),
            "lgbm_val_r": float(np.mean([float(r["lgbm_val_r"]) for r in voxel_rows])),
        }
        for block in BLOCKS:
            item[f"{block}_gain_fraction"] = float(np.mean([float(r[f"{block}_gain_fraction"]) for r in voxel_rows]))
        item["long_context_share"] = item["h50_gain_fraction"] + item["h200_gain_fraction"]
        item["context_horizon_index"] = sum(
            item[f"{block}_gain_fraction"] * CONTEXT_WEIGHTS[block] for block in BLOCKS
        )
        out.append(item)
    return sorted(out, key=lambda row: row["global_voxel_index"])


def label_clusters(rows: Sequence[dict], mask_shape: tuple[int, ...], mask_ijk: np.ndarray) -> np.ndarray:
    volume_mask = np.zeros(mask_shape, dtype=bool)
    for row in rows:
        voxel = int(row["global_voxel_index"])
        if voxel >= len(mask_ijk):
            raise IndexError(f"Voxel index {voxel} exceeds mask size {len(mask_ijk)}")
        volume_mask[tuple(mask_ijk[voxel])] = True
    labels_3d, _n = ndimage.label(volume_mask, structure=np.ones((3, 3, 3), dtype=bool))
    labels = np.zeros(len(rows), dtype=np.int32)
    for i, row in enumerate(rows):
        labels[i] = int(labels_3d[tuple(mask_ijk[int(row["global_voxel_index"])])])
    return labels


def summarize_clusters(
    *,
    subject: str,
    rows: Sequence[dict],
    labels: np.ndarray,
    mask_ijk: np.ndarray,
    min_cluster_size: int,
    roi_lookup: dict[str, set[int]],
) -> list[dict]:
    out: list[dict] = []
    voxels = np.asarray([int(r["global_voxel_index"]) for r in rows], dtype=np.int64)
    for cluster_id in [int(c) for c in np.unique(labels) if int(c) > 0]:
        idx = np.nonzero(labels == cluster_id)[0]
        if idx.size < int(min_cluster_size):
            continue
        cluster_rows = [rows[i] for i in idx]
        cluster_voxels = voxels[idx]
        peak_i = idx[int(np.argmax([float(rows[i]["saved_combo_best_r"]) for i in idx]))]
        centroid_ijk = mask_ijk[cluster_voxels].mean(axis=0)
        source_counts = defaultdict(int)
        for row in cluster_rows:
            for roi in str(row["source_rois"]).split(";"):
                source_counts[roi] += 1
        item = {
            "subject": subject,
            "cluster_id": cluster_id,
            "n_voxels": int(idx.size),
            "primary_roi": primary_roi(cluster_voxels, roi_lookup),
            "source_roi_counts": ";".join(f"{roi}:{count}" for roi, count in sorted(source_counts.items())),
            "mean_lgbm_val_r": float(np.mean([float(r["lgbm_val_r"]) for r in cluster_rows])),
            "mean_saved_combo_best_r": float(np.mean([float(r["saved_combo_best_r"]) for r in cluster_rows])),
            "peak_saved_combo_best_r": float(rows[peak_i]["saved_combo_best_r"]),
            "peak_global_voxel": int(rows[peak_i]["global_voxel_index"]),
            "centroid_i": float(centroid_ijk[0]),
            "centroid_j": float(centroid_ijk[1]),
            "centroid_k": float(centroid_ijk[2]),
        }
        for block in BLOCKS:
            item[f"{block}_gain_fraction"] = float(np.mean([float(r[f"{block}_gain_fraction"]) for r in cluster_rows]))
        item["long_context_share"] = float(item["h50_gain_fraction"] + item["h200_gain_fraction"])
        item["context_horizon_index"] = float(
            sum(float(item[f"{block}_gain_fraction"]) * CONTEXT_WEIGHTS[block] for block in BLOCKS)
        )
        out.append(item)
    return sorted(out, key=lambda row: (-row["peak_saved_combo_best_r"], -row["n_voxels"]))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_maps(
    *,
    subject: str,
    pycortex_subject: str,
    pycortex_filestore: Path,
    rows: Sequence[dict],
    labels: np.ndarray,
    mask_ijk: np.ndarray,
    out_dir: Path,
    xfm_name: str | None,
    with_rois: bool,
) -> None:
    configure_pycortex_filestore(str(pycortex_filestore))
    import cortex
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_total = len(mask_ijk)
    xfm, mask_voxels = find_matching_xfm(cortex, str(pycortex_filestore), pycortex_subject, n_total, xfm_name)
    if mask_voxels != n_total:
        log.warning("%s: pycortex mask size %d != expected %d; using mask size.", subject, mask_voxels, n_total)
        n_total = mask_voxels

    voxels = np.asarray([int(r["global_voxel_index"]) for r in rows], dtype=np.int64)
    lgbm_r = np.asarray([float(r["lgbm_val_r"]) for r in rows], dtype=np.float32)
    context_index = np.asarray([float(r["context_horizon_index"]) for r in rows], dtype=np.float32)
    cluster_values = labels.astype(np.float32)
    cluster_values[cluster_values <= 0] = np.nan

    subject_out = out_dir / subject
    make_flatmap(
        cortex,
        plt,
        project_to_full_brain(lgbm_r, voxels, n_total),
        pycortex_subject=pycortex_subject,
        xfm_name=xfm,
        vmin=float(np.nanmin(lgbm_r)),
        vmax=float(np.nanmax(lgbm_r)),
        cmap="inferno",
        title=f"{subject}: LGBM sampled voxel val r",
        out_path=subject_out / "lgbm_voxel_r.png",
        with_rois=with_rois,
    )
    make_flatmap(
        cortex,
        plt,
        project_to_full_brain(cluster_values, voxels, n_total),
        pycortex_subject=pycortex_subject,
        xfm_name=xfm,
        vmin=1.0,
        vmax=max(1.0, float(np.nanmax(cluster_values))),
        cmap="tab20",
        title=f"{subject}: connected LGBM sampled voxel clusters",
        out_path=subject_out / "lgbm_voxel_clusters.png",
        with_rois=with_rois,
    )
    make_flatmap(
        cortex,
        plt,
        project_to_full_brain(context_index, voxels, n_total),
        pycortex_subject=pycortex_subject,
        xfm_name=xfm,
        vmin=0.0,
        vmax=60.0,
        cmap="viridis",
        title=f"{subject}: LGBM gain horizon index",
        out_path=subject_out / "lgbm_context_horizon_index.png",
        with_rois=with_rois,
    )


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    pycortex_filestore = Path(args.pycortex_filestore).expanduser().resolve()
    all_rows: list[dict] = []

    for subject in args.subjects:
        result_dir = results_root / args.tag_template.format(subject=subject)
        voxel_csv = result_dir / f"{args.lgbm_prefix}_voxels.csv"
        if not voxel_csv.is_file():
            log.warning("%s: missing %s; skipping.", subject, voxel_csv)
            continue
        pycortex_subject = SUBJECT_TO_UTS[subject]
        mask, mask_ijk = load_mask_ijk(pycortex_filestore, pycortex_subject)
        rows = load_lgbm_voxel_rows(voxel_csv, args.top_n)
        labels = label_clusters(rows, mask.shape, mask_ijk)
        roi_lookup = load_roi_lookup(Path(args.ba_dir).expanduser().resolve(), pycortex_subject)
        cluster_rows = summarize_clusters(
            subject=subject,
            rows=rows,
            labels=labels,
            mask_ijk=mask_ijk,
            min_cluster_size=args.min_cluster_size,
            roi_lookup=roi_lookup,
        )
        write_csv(out_dir / subject / "lgbm_peak_cluster_summary.csv", cluster_rows)
        all_rows.extend(cluster_rows)

        np.savez(
            out_dir / subject / "lgbm_peak_cluster_maps.npz",
            voxels=np.asarray([int(r["global_voxel_index"]) for r in rows], dtype=np.int64),
            lgbm_val_r=np.asarray([float(r["lgbm_val_r"]) for r in rows], dtype=np.float32),
            saved_combo_best_r=np.asarray([float(r["saved_combo_best_r"]) for r in rows], dtype=np.float32),
            cluster_labels=labels,
            context_horizon_index=np.asarray([float(r["context_horizon_index"]) for r in rows], dtype=np.float32),
            **{f"{block}_gain_fraction": np.asarray([float(r[f"{block}_gain_fraction"]) for r in rows], dtype=np.float32) for block in BLOCKS},
        )
        render_maps(
            subject=subject,
            pycortex_subject=pycortex_subject,
            pycortex_filestore=pycortex_filestore,
            rows=rows,
            labels=labels,
            mask_ijk=mask_ijk,
            out_dir=out_dir,
            xfm_name=args.xfm_name,
            with_rois=args.with_rois,
        )
        log.info("%s: wrote %d clusters from %d LGBM sampled voxels.", subject, len(cluster_rows), len(rows))

    write_csv(out_dir / "lgbm_peak_cluster_summary_all_subjects.csv", all_rows)
    log.info("Wrote combined summary to %s", out_dir / "lgbm_peak_cluster_summary_all_subjects.csv")


if __name__ == "__main__":
    main()
