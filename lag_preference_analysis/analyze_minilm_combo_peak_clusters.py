#!/usr/bin/env python3
"""Map and summarize peak frontal clusters from exported MiniLM combo encoders.

This consumes ``gpt1_encoding_comparison/train_minilm_combo_encoding.py`` outputs,
especially ``encoding_model_minilm_summary_combo.npz``. The exported model keeps
the selected global voxel ids, validation correlations, and ridge weights, which
lets us move from ROI averages to spatial clusters of high-performing voxels.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
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
log = logging.getLogger("minilm_combo_peak_clusters")

BLOCKS = ["1TR", "h20", "h50", "h200"]
CONTEXT_WEIGHTS = {"1TR": 1.0, "h20": 20.0, "h50": 50.0, "h200": 200.0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"])
    p.add_argument("--condition", default="minilm_summary_combo")
    p.add_argument("--encoding-output-root", default=str(REPO_DIR / "gpt1_encoding_comparison" / "outputs"))
    p.add_argument("--out-dir", default=str(THIS_DIR / "results" / "minilm_combo_peak_clusters"))
    p.add_argument("--pycortex-filestore", default=str(REPO_DIR / "pycortex-db"))
    p.add_argument("--ba-dir", default=str(REPO_DIR / "ba_indices"))
    p.add_argument("--xfm-name", default=None)
    p.add_argument("--top-n", type=int, default=1000, help="Keep at most this many highest-r selected voxels per subject.")
    p.add_argument("--r-threshold", type=float, default=0.10, help="High-r threshold applied before clustering.")
    p.add_argument("--min-cluster-size", type=int, default=5)
    p.add_argument("--with-rois", action="store_true")
    return p.parse_args()


def load_encoding(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"].reshape(-1)[0]))
    return {
        "metadata": metadata,
        "weights": np.asarray(data["weights"], dtype=np.float32),
        "voxels": np.asarray(data["voxels"], dtype=np.int64),
        "corrs": np.asarray(data["selected_corrs"], dtype=np.float32),
        "alphas": np.asarray(data["alphas"], dtype=np.float32),
    }


def active_lag_weights(weights: np.ndarray, metadata: dict) -> np.ndarray:
    """Return feature-by-voxel weights for the exported active lag only."""
    feature_blocks = list(metadata["feature_blocks"])
    if feature_blocks != BLOCKS:
        raise ValueError(f"Expected feature blocks {BLOCKS}, got {feature_blocks}")
    lag = int(metadata["lag"])
    delays = [1, 2, 3, 4]
    block_dim = weights.shape[0] // len(delays)
    if lag not in delays:
        raise ValueError(f"Cannot resolve lag {lag}; expected one of {delays}")
    start = delays.index(lag) * block_dim
    return weights[start : start + block_dim]


def block_fractions(active_weights: np.ndarray) -> dict[str, np.ndarray]:
    block_dim = active_weights.shape[0] // len(BLOCKS)
    norms = {}
    total = np.zeros(active_weights.shape[1], dtype=np.float32)
    for i, block in enumerate(BLOCKS):
        vals = np.linalg.norm(active_weights[i * block_dim : (i + 1) * block_dim], axis=0).astype(np.float32)
        norms[block] = vals
        total += vals
    total[total == 0] = np.nan
    return {block: norms[block] / total for block in BLOCKS}


def load_mask_ijk(pycortex_filestore: Path, pycortex_subject: str) -> tuple[np.ndarray, np.ndarray]:
    mask_path = pycortex_filestore / pycortex_subject / "transforms" / f"{pycortex_subject}_auto" / "mask_thick.nii.gz"
    mask = nib.load(str(mask_path)).get_fdata() > 0
    return mask, np.asarray(np.where(mask)).T


def high_r_selection(corrs: np.ndarray, top_n: int, r_threshold: float) -> np.ndarray:
    order = np.argsort(-corrs)
    if top_n > 0:
        order = order[: min(int(top_n), order.size)]
    keep = order[corrs[order] >= float(r_threshold)]
    if keep.size == 0 and top_n > 0:
        log.warning("No voxels passed r >= %.3f; falling back to top %d voxels.", r_threshold, len(order))
        keep = order
    return np.sort(keep)


def label_clusters(voxels: np.ndarray, selected_local: np.ndarray, mask_shape: tuple[int, ...], mask_ijk: np.ndarray) -> np.ndarray:
    volume_mask = np.zeros(mask_shape, dtype=bool)
    for local_idx in selected_local:
        flat_voxel = int(voxels[local_idx])
        if flat_voxel >= len(mask_ijk):
            raise IndexError(f"Voxel index {flat_voxel} exceeds mask size {len(mask_ijk)}")
        volume_mask[tuple(mask_ijk[flat_voxel])] = True
    labels_3d, _n = ndimage.label(volume_mask, structure=np.ones((3, 3, 3), dtype=bool))
    cluster_labels = np.zeros(voxels.shape[0], dtype=np.int32)
    for local_idx in selected_local:
        flat_voxel = int(voxels[local_idx])
        cluster_labels[local_idx] = int(labels_3d[tuple(mask_ijk[flat_voxel])])
    return cluster_labels


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


def summarize_clusters(
    *,
    subject: str,
    voxels: np.ndarray,
    corrs: np.ndarray,
    cluster_labels: np.ndarray,
    fractions: dict[str, np.ndarray],
    mask_ijk: np.ndarray,
    min_cluster_size: int,
    roi_lookup: dict[str, set[int]],
) -> list[dict]:
    rows: list[dict] = []
    cluster_ids = [int(c) for c in np.unique(cluster_labels) if int(c) > 0]
    for cluster_id in cluster_ids:
        idx = np.nonzero(cluster_labels == cluster_id)[0]
        if idx.size < int(min_cluster_size):
            continue
        cluster_voxels = voxels[idx]
        peak_local = idx[int(np.argmax(corrs[idx]))]
        centroid_ijk = mask_ijk[cluster_voxels].mean(axis=0)
        row = {
            "subject": subject,
            "cluster_id": cluster_id,
            "n_voxels": int(idx.size),
            "primary_roi": primary_roi(cluster_voxels, roi_lookup),
            "mean_r": float(corrs[idx].mean()),
            "median_r": float(np.median(corrs[idx])),
            "peak_r": float(corrs[peak_local]),
            "peak_global_voxel": int(voxels[peak_local]),
            "centroid_i": float(centroid_ijk[0]),
            "centroid_j": float(centroid_ijk[1]),
            "centroid_k": float(centroid_ijk[2]),
        }
        for block in BLOCKS:
            row[f"{block}_fraction"] = float(np.nanmean(fractions[block][idx]))
        row["long_context_share"] = float(row["h50_fraction"] + row["h200_fraction"])
        row["context_horizon_index"] = float(
            sum(float(row[f"{block}_fraction"]) * CONTEXT_WEIGHTS[block] for block in BLOCKS)
        )
        rows.append(row)
    return sorted(rows, key=lambda row: (-row["peak_r"], -row["n_voxels"]))


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
    n_total: int,
    voxels: np.ndarray,
    corrs: np.ndarray,
    selected_local: np.ndarray,
    cluster_labels: np.ndarray,
    fractions: dict[str, np.ndarray],
    out_dir: Path,
    xfm_name: str | None,
    with_rois: bool,
) -> None:
    configure_pycortex_filestore(str(pycortex_filestore))
    import cortex
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xfm, mask_voxels = find_matching_xfm(cortex, str(pycortex_filestore), pycortex_subject, n_total, xfm_name)
    if mask_voxels != n_total:
        log.warning("%s: pycortex mask size %d != expected %d; using mask size.", subject, mask_voxels, n_total)
        n_total = mask_voxels

    top_values = np.full(corrs.shape, np.nan, dtype=np.float32)
    top_values[selected_local] = corrs[selected_local]
    cluster_values = cluster_labels.astype(np.float32)
    cluster_values[cluster_values <= 0] = np.nan
    context_index = sum(fractions[block] * CONTEXT_WEIGHTS[block] for block in BLOCKS).astype(np.float32)

    subject_out = out_dir / subject
    make_flatmap(
        cortex,
        plt,
        project_to_full_brain(top_values, voxels, n_total),
        pycortex_subject=pycortex_subject,
        xfm_name=xfm,
        vmin=float(np.nanmin(top_values)),
        vmax=float(np.nanmax(top_values)),
        cmap="inferno",
        title=f"{subject}: high-r MiniLM combo voxels",
        out_path=subject_out / "top_voxel_r.png",
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
        title=f"{subject}: connected high-r voxel clusters",
        out_path=subject_out / "top_voxel_clusters.png",
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
        title=f"{subject}: ridge coefficient horizon index",
        out_path=subject_out / "context_horizon_index.png",
        with_rois=with_rois,
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.encoding_output_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    pycortex_filestore = Path(args.pycortex_filestore).expanduser().resolve()
    all_rows: list[dict] = []

    for subject in args.subjects:
        model_path = output_root / subject / f"encoding_model_{args.condition}.npz"
        if not model_path.is_file():
            log.warning("%s: missing %s; skipping.", subject, model_path)
            continue

        pycortex_subject = SUBJECT_TO_UTS[subject]
        mask, mask_ijk = load_mask_ijk(pycortex_filestore, pycortex_subject)
        enc = load_encoding(model_path)
        active = active_lag_weights(enc["weights"], enc["metadata"])
        fractions = block_fractions(active)
        selected_local = high_r_selection(enc["corrs"], args.top_n, args.r_threshold)
        cluster_labels = label_clusters(enc["voxels"], selected_local, mask.shape, mask_ijk)
        roi_lookup = load_roi_lookup(Path(args.ba_dir).expanduser().resolve(), pycortex_subject)

        rows = summarize_clusters(
            subject=subject,
            voxels=enc["voxels"],
            corrs=enc["corrs"],
            cluster_labels=cluster_labels,
            fractions=fractions,
            mask_ijk=mask_ijk,
            min_cluster_size=args.min_cluster_size,
            roi_lookup=roi_lookup,
        )
        write_csv(out_dir / subject / "peak_cluster_summary.csv", rows)
        all_rows.extend(rows)

        np.savez(
            out_dir / subject / "peak_cluster_maps.npz",
            voxels=enc["voxels"],
            selected_local_indices=selected_local,
            selected_global_voxels=enc["voxels"][selected_local],
            selected_corrs=enc["corrs"][selected_local],
            cluster_labels=cluster_labels,
            context_horizon_index=sum(fractions[block] * CONTEXT_WEIGHTS[block] for block in BLOCKS).astype(np.float32),
            **{f"{block}_fraction": fractions[block].astype(np.float32) for block in BLOCKS},
        )
        render_maps(
            subject=subject,
            pycortex_subject=pycortex_subject,
            pycortex_filestore=pycortex_filestore,
            n_total=len(mask_ijk),
            voxels=enc["voxels"],
            corrs=enc["corrs"],
            selected_local=selected_local,
            cluster_labels=cluster_labels,
            fractions=fractions,
            out_dir=out_dir,
            xfm_name=args.xfm_name,
            with_rois=args.with_rois,
        )
        log.info("%s: wrote %d clusters from %d high-r voxels.", subject, len(rows), len(selected_local))

    write_csv(out_dir / "peak_cluster_summary_all_subjects.csv", all_rows)
    log.info("Wrote combined summary to %s", out_dir / "peak_cluster_summary_all_subjects.csv")


if __name__ == "__main__":
    main()
