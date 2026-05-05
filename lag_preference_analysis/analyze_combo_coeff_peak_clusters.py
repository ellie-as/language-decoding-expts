#!/usr/bin/env python3
"""Map and cluster voxelwise coefficient block fractions for combo ridge models.

This is the coefficient analogue of ``analyze_combo_lgbm_peak_clusters.py``.
It refits one combined model, computes per-voxel L2 coefficient fractions for

    1TR | h20 | h50 | h200 | h500

then renders flatmaps and labels connected clusters among reliable voxels with
high h500 or high long-context (h200 + h500) fraction.
"""
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
sys.path.insert(0, str(REPO_DIR / "decoding"))
sys.path.insert(0, str(REPO_DIR / "mindeye_text"))
sys.path.insert(0, str(THIS_DIR))

import config  # noqa: E402
import run_summaries_encoding as rse  # noqa: E402
from _shared import load_or_build_chunk_embeddings  # noqa: E402
from analyze_combo_coeff_sweep import fit_coefficients, model_tag  # noqa: E402
from plot_lag_flatmaps import (  # noqa: E402
    SUBJECT_TO_UTS,
    configure_pycortex_filestore,
    find_matching_xfm,
    make_flatmap,
    project_to_full_brain,
)
from train_lag_encoding import (  # noqa: E402
    configure_data_root,
    load_full_frontal_voxels,
    load_stories,
    split_stories,
    stack_lag,
)
from train_summary_combo_encoding import build_combo_embeddings, load_or_build_summary_embeddings  # noqa: E402
from utils_resp import get_resp  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("combo_coeff_peak_clusters")

BLOCKS = ["1TR", "h20", "h50", "h200", "h500"]
CONTEXT_WEIGHTS = {"1TR": 1.0, "h20": 20.0, "h50": 50.0, "h200": 200.0, "h500": 500.0}
SUB_ROIS = ["BA_10", "BA_9_46", "BA_8", "BA_6", "BROCA", "BA_full_frontal"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True, choices=sorted(rse.SUBJECT_TO_UTS))
    p.add_argument("--lag", type=int, default=2)
    p.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--summary-horizons", nargs="+", type=int, default=[20, 50, 200, 500])
    p.add_argument("--ridge-alphas", type=float, nargs="+", default=[1_000.0, 10_000.0, 100_000.0, 300_000.0, 1_000_000.0, 3_000_000.0, 10_000_000.0])
    p.add_argument("--voxel-chunk-size", type=int, default=1000)

    p.add_argument("--sessions", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    p.add_argument("--stories", nargs="+", default=None)
    p.add_argument("--val-story-count", type=int, default=8)
    p.add_argument("--val-stories", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summary-model", default=None)
    p.add_argument("--summaries-dir", default=str(rse.LOCAL_DEFAULT_SUMMARIES_DIR))
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument("--embedding-device", default="auto")
    p.add_argument("--embedding-cache-dir", default=str(THIS_DIR / "cache"))
    p.add_argument("--one-tr-cache-dir", default=str(REPO_DIR / "27-04-expts" / "cache"))
    p.add_argument("--ba-dir", default=str(rse.LOCAL_DEFAULT_BA_DIR))
    p.add_argument("--data-root", default=None)
    p.add_argument("--local-compute-mode", action="store_true")
    p.add_argument("--mounted-project-root", default="/Volumes/ellie/language-decoding-expts")
    p.add_argument("--local-cache-root", default=str(REPO_DIR / "local_compute_cache"))

    p.add_argument("--reliable-r-threshold", type=float, default=0.05)
    p.add_argument("--h500-quantile", type=float, default=0.90)
    p.add_argument("--long-quantile", type=float, default=0.90)
    p.add_argument("--min-cluster-size", type=int, default=3)
    p.add_argument("--out-dir", default=str(THIS_DIR / "results" / "combo_coeff_peak_clusters" / "S1_bge_highalpha"))
    p.add_argument("--pycortex-filestore", default=str(REPO_DIR / "pycortex-db"))
    p.add_argument("--pycortex-subject", default=None)
    p.add_argument("--xfm-name", default=None)
    p.add_argument("--with-rois", action="store_true")
    return p.parse_args()


def resolve_embedding_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_mask_ijk(pycortex_filestore: Path, pycortex_subject: str) -> tuple[np.ndarray, np.ndarray]:
    mask_path = pycortex_filestore / pycortex_subject / "transforms" / f"{pycortex_subject}_auto" / "mask_thick.nii.gz"
    mask = nib.load(str(mask_path)).get_fdata() > 0
    return mask, np.asarray(np.where(mask)).T


def load_roi_lookup(ba_dir: Path, pycortex_subject: str) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    subject_dir = ba_dir / pycortex_subject
    for roi in SUB_ROIS:
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


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def block_fraction_arrays(coef: np.ndarray, block_slices: dict[str, slice]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    norms = {block: np.linalg.norm(coef[:, sl], axis=1).astype(np.float32) for block, sl in block_slices.items()}
    total = np.zeros(coef.shape[0], dtype=np.float32)
    for block in block_slices:
        total += norms[block]
    total[total == 0] = np.nan
    fractions = {block: (norms[block] / total).astype(np.float32) for block in block_slices}
    return norms, fractions


def voxel_rows(
    *,
    voxels: np.ndarray,
    corrs: np.ndarray,
    alphas: np.ndarray,
    norms: dict[str, np.ndarray],
    fractions: dict[str, np.ndarray],
    subject: str,
    lag: int,
    embedding_model: str,
) -> list[dict]:
    rows = []
    for i, voxel in enumerate(voxels):
        row = {
            "subject": subject,
            "lag": int(lag),
            "embedding_model": embedding_model,
            "embedding_tag": model_tag(embedding_model),
            "global_voxel_index": int(voxel),
            "full_model_r": float(corrs[i]),
            "selected_alpha": float(alphas[i]),
        }
        for block in BLOCKS:
            row[f"{block}_l2"] = float(norms[block][i])
            row[f"{block}_fraction"] = float(fractions[block][i])
        row["summary_fraction"] = float(sum(fractions[block][i] for block in BLOCKS if block != "1TR"))
        row["long_context_fraction"] = float(fractions["h200"][i] + fractions["h500"][i])
        row["context_horizon_index"] = float(sum(fractions[block][i] * CONTEXT_WEIGHTS[block] for block in BLOCKS))
        rows.append(row)
    return rows


def label_selected_clusters(
    *,
    selected_global_voxels: np.ndarray,
    all_global_voxels: np.ndarray,
    mask_shape: tuple[int, ...],
    mask_ijk: np.ndarray,
) -> np.ndarray:
    volume_mask = np.zeros(mask_shape, dtype=bool)
    for voxel in selected_global_voxels:
        if int(voxel) < len(mask_ijk):
            volume_mask[tuple(mask_ijk[int(voxel)])] = True
    labels_3d, _n = ndimage.label(volume_mask, structure=np.ones((3, 3, 3), dtype=bool))
    labels = np.zeros(len(all_global_voxels), dtype=np.int32)
    for i, voxel in enumerate(all_global_voxels):
        if int(voxel) < len(mask_ijk):
            labels[i] = int(labels_3d[tuple(mask_ijk[int(voxel)])])
    return labels


def summarize_clusters(
    *,
    name: str,
    rows: Sequence[dict],
    labels: np.ndarray,
    mask_ijk: np.ndarray,
    min_cluster_size: int,
    roi_lookup: dict[str, set[int]],
) -> list[dict]:
    out = []
    voxels = np.asarray([int(r["global_voxel_index"]) for r in rows], dtype=np.int64)
    for cluster_id in [int(c) for c in np.unique(labels) if int(c) > 0]:
        idx = np.nonzero(labels == cluster_id)[0]
        if idx.size < int(min_cluster_size):
            continue
        cluster_rows = [rows[i] for i in idx]
        cluster_voxels = voxels[idx]
        peak_i = idx[int(np.argmax([float(rows[i]["h500_fraction"]) for i in idx]))]
        centroid_ijk = mask_ijk[cluster_voxels].mean(axis=0)
        item = {
            "cluster_type": name,
            "cluster_id": cluster_id,
            "n_voxels": int(idx.size),
            "primary_roi": primary_roi(cluster_voxels, roi_lookup),
            "mean_full_model_r": float(np.mean([float(r["full_model_r"]) for r in cluster_rows])),
            "peak_full_model_r": float(np.max([float(r["full_model_r"]) for r in cluster_rows])),
            "peak_h500_fraction": float(rows[peak_i]["h500_fraction"]),
            "peak_global_voxel": int(rows[peak_i]["global_voxel_index"]),
            "centroid_i": float(centroid_ijk[0]),
            "centroid_j": float(centroid_ijk[1]),
            "centroid_k": float(centroid_ijk[2]),
        }
        for block in BLOCKS:
            item[f"{block}_fraction"] = float(np.mean([float(r[f"{block}_fraction"]) for r in cluster_rows]))
        item["long_context_fraction"] = float(np.mean([float(r["long_context_fraction"]) for r in cluster_rows]))
        item["context_horizon_index"] = float(np.mean([float(r["context_horizon_index"]) for r in cluster_rows]))
        out.append(item)
    return sorted(out, key=lambda row: (-row["peak_h500_fraction"], -row["n_voxels"]))


def render_maps(
    *,
    subject: str,
    pycortex_subject: str,
    pycortex_filestore: Path,
    voxels: np.ndarray,
    full_corrs: np.ndarray,
    fractions: dict[str, np.ndarray],
    h500_labels: np.ndarray,
    long_labels: np.ndarray,
    out_dir: Path,
    reliable_mask: np.ndarray,
    xfm_name: str | None,
    with_rois: bool,
) -> None:
    configure_pycortex_filestore(str(pycortex_filestore))
    import cortex
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_total = int(pycortex_filestore.joinpath(pycortex_subject).exists() and len(load_mask_ijk(pycortex_filestore, pycortex_subject)[1]))
    xfm, mask_voxels = find_matching_xfm(cortex, str(pycortex_filestore), pycortex_subject, n_total, xfm_name)
    if mask_voxels != n_total:
        n_total = mask_voxels

    h500 = fractions["h500"].copy()
    long = (fractions["h200"] + fractions["h500"]).astype(np.float32)
    summary = sum(fractions[block] for block in BLOCKS if block != "1TR").astype(np.float32)
    context_index = sum(fractions[block] * CONTEXT_WEIGHTS[block] for block in BLOCKS).astype(np.float32)
    for arr in (h500, long, summary, context_index):
        arr[~reliable_mask] = np.nan

    h500_cluster_values = h500_labels.astype(np.float32)
    h500_cluster_values[h500_cluster_values <= 0] = np.nan
    long_cluster_values = long_labels.astype(np.float32)
    long_cluster_values[long_cluster_values <= 0] = np.nan

    maps = [
        ("full_model_r.png", full_corrs, "inferno", float(np.nanmin(full_corrs)), max(0.1, float(np.nanquantile(full_corrs, 0.99))), f"{subject}: BGE ridge full-model validation r"),
        ("h500_fraction_masked.png", h500, "viridis", 0.0, max(0.25, float(np.nanquantile(h500, 0.99))), f"{subject}: h500 coefficient fraction, reliable voxels"),
        ("long_h200_h500_fraction_masked.png", long, "viridis", 0.0, max(0.50, float(np.nanquantile(long, 0.99))), f"{subject}: h200+h500 coefficient fraction, reliable voxels"),
        ("summary_fraction_masked.png", summary, "viridis", 0.0, 1.0, f"{subject}: summary coefficient fraction, reliable voxels"),
        ("context_horizon_index_masked.png", context_index, "magma", 0.0, max(150.0, float(np.nanquantile(context_index, 0.99))), f"{subject}: coefficient-weighted horizon index, reliable voxels"),
        ("h500_high_clusters.png", h500_cluster_values, "tab20", 1.0, max(1.0, float(np.nanmax(h500_cluster_values))), f"{subject}: high-h500 connected clusters"),
        ("long_high_clusters.png", long_cluster_values, "tab20", 1.0, max(1.0, float(np.nanmax(long_cluster_values))), f"{subject}: high h200+h500 connected clusters"),
    ]
    for filename, values, cmap, vmin, vmax, title in maps:
        make_flatmap(
            cortex,
            plt,
            project_to_full_brain(values.astype(np.float32), voxels, n_total),
            pycortex_subject=pycortex_subject,
            xfm_name=xfm,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            title=title,
            out_path=out_dir / filename,
            with_rois=with_rois,
        )


def main() -> None:
    args = parse_args()
    args.embedding_device = resolve_embedding_device(args.embedding_device)
    args.summary_horizons = sorted({int(h) for h in args.summary_horizons})
    args.lags = [int(args.lag)]
    args.chunk_trs = 1
    args.feature_model = "embedding"
    args.embedding_model = args.embedding_model

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mounted_root = configure_data_root(args)
    stories = load_stories(args)
    train_stories, val_stories = split_stories(stories, args)
    log.info("%s: %d stories | %d train | %d val", args.subject, len(stories), len(train_stories), len(val_stories))

    response_root = config.DATA_TRAIN_DIR
    if args.local_compute_mode and mounted_root is not None:
        response_root = str(
            rse.stage_local_response_cache(
                args.subject,
                stories,
                Path(config.DATA_TRAIN_DIR),
                Path(args.local_cache_root).expanduser().resolve(),
            )
        )

    sample_resp = get_resp(args.subject, [stories[0]], stack=True, vox=None, response_root=response_root)
    voxels = load_full_frontal_voxels(args.subject, int(sample_resp.shape[1]), args.ba_dir)
    responses_by_story = get_resp(args.subject, stories, stack=False, vox=voxels, response_root=response_root)
    responses_by_story = {s: arr.astype(np.float32) for s, arr in responses_by_story.items()}
    resp_lengths = {s: int(arr.shape[0]) for s, arr in responses_by_story.items()}

    one_tr_args = argparse.Namespace(
        subject=args.subject,
        embedding_cache_dir=args.one_tr_cache_dir,
        feature_model="embedding",
        embedding_model=args.embedding_model,
        chunk_trs=1,
        lag_trs=int(args.lag),
        embed_batch_size=int(args.embed_batch_size),
        embedding_device=args.embedding_device,
    )
    one_tr, one_dim, _one_cache = load_or_build_chunk_embeddings(
        one_tr_args,
        stories,
        resp_lengths,
        response_root=config.DATA_TRAIN_DIR,
    )
    summary_embs, _summary_model, _summary_cache = load_or_build_summary_embeddings(args, stories, resp_lengths)
    combo = build_combo_embeddings(one_tr, summary_embs, stories, args.summary_horizons)
    block_names = ["1TR"] + [f"h{h}" for h in args.summary_horizons]
    block_slices = {name: slice(i * one_dim, (i + 1) * one_dim) for i, name in enumerate(block_names)}

    x_train, y_train = stack_lag(combo, responses_by_story, train_stories, args.lag)
    x_val, y_val = stack_lag(combo, responses_by_story, val_stories, args.lag)
    log.info("X_train=%s X_val=%s Y_train=%s", x_train.shape, x_val.shape, y_train.shape)

    coef, alphas, full_corrs = fit_coefficients(
        regressor="ridge",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        args=args,
    )
    if full_corrs is None:
        raise RuntimeError("Expected validation correlations from ridge fit.")
    norms, fractions = block_fraction_arrays(coef, block_slices)
    rows = voxel_rows(
        voxels=voxels,
        corrs=full_corrs,
        alphas=alphas,
        norms=norms,
        fractions=fractions,
        subject=args.subject,
        lag=args.lag,
        embedding_model=args.embedding_model,
    )
    write_csv(out_dir / "voxel_coeff_fractions.csv", rows)

    pycortex_filestore = Path(args.pycortex_filestore).expanduser().resolve()
    pycortex_subject = args.pycortex_subject or SUBJECT_TO_UTS[args.subject]
    mask, mask_ijk = load_mask_ijk(pycortex_filestore, pycortex_subject)
    roi_lookup = load_roi_lookup(Path(args.ba_dir).expanduser().resolve(), pycortex_subject)

    reliable_mask = full_corrs >= float(args.reliable_r_threshold)
    if not np.any(reliable_mask):
        raise ValueError(f"No voxels passed reliable-r threshold {args.reliable_r_threshold}")
    h500_cutoff = float(np.nanquantile(fractions["h500"][reliable_mask], float(args.h500_quantile)))
    long_fraction = (fractions["h200"] + fractions["h500"]).astype(np.float32)
    long_cutoff = float(np.nanquantile(long_fraction[reliable_mask], float(args.long_quantile)))
    h500_selected = reliable_mask & (fractions["h500"] >= h500_cutoff)
    long_selected = reliable_mask & (long_fraction >= long_cutoff)
    log.info(
        "Reliable r>=%.3f keeps %d/%d voxels; h500 cutoff %.4f keeps %d; long cutoff %.4f keeps %d",
        float(args.reliable_r_threshold),
        int(reliable_mask.sum()),
        int(reliable_mask.size),
        h500_cutoff,
        int(h500_selected.sum()),
        long_cutoff,
        int(long_selected.sum()),
    )

    h500_labels = label_selected_clusters(
        selected_global_voxels=voxels[h500_selected],
        all_global_voxels=voxels,
        mask_shape=mask.shape,
        mask_ijk=mask_ijk,
    )
    long_labels = label_selected_clusters(
        selected_global_voxels=voxels[long_selected],
        all_global_voxels=voxels,
        mask_shape=mask.shape,
        mask_ijk=mask_ijk,
    )
    h500_clusters = summarize_clusters(
        name="h500",
        rows=rows,
        labels=h500_labels,
        mask_ijk=mask_ijk,
        min_cluster_size=args.min_cluster_size,
        roi_lookup=roi_lookup,
    )
    long_clusters = summarize_clusters(
        name="h200_h500",
        rows=rows,
        labels=long_labels,
        mask_ijk=mask_ijk,
        min_cluster_size=args.min_cluster_size,
        roi_lookup=roi_lookup,
    )
    write_csv(out_dir / "h500_high_cluster_summary.csv", h500_clusters)
    write_csv(out_dir / "long_h200_h500_high_cluster_summary.csv", long_clusters)

    np.savez(
        out_dir / "coeff_peak_cluster_maps.npz",
        voxels=voxels.astype(np.int64),
        full_model_r=full_corrs.astype(np.float32),
        selected_alpha=alphas.astype(np.float32),
        reliable_mask=reliable_mask.astype(bool),
        h500_selected=h500_selected.astype(bool),
        long_selected=long_selected.astype(bool),
        h500_cluster_labels=h500_labels.astype(np.int32),
        long_cluster_labels=long_labels.astype(np.int32),
        h500_cutoff=np.float32(h500_cutoff),
        long_cutoff=np.float32(long_cutoff),
        **{f"{block}_fraction": fractions[block].astype(np.float32) for block in BLOCKS},
        **{f"{block}_l2": norms[block].astype(np.float32) for block in BLOCKS},
    )

    render_maps(
        subject=args.subject,
        pycortex_subject=pycortex_subject,
        pycortex_filestore=pycortex_filestore,
        voxels=voxels,
        full_corrs=full_corrs,
        fractions=fractions,
        h500_labels=h500_labels,
        long_labels=long_labels,
        out_dir=out_dir,
        reliable_mask=reliable_mask,
        xfm_name=args.xfm_name,
        with_rois=args.with_rois,
    )
    log.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
