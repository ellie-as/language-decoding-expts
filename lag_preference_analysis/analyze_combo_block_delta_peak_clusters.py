#!/usr/bin/env python3
"""Per-voxel block-ablation delta_r maps and clusters for combo ridge models.

Fits the full combo ridge model (1TR | h20 | h50 | h200 | h500 | ...) plus one
leave-one-block-out variant per block, then computes the unique contribution of
each block in each voxel:

    delta_r[block][voxel] = r_full[voxel] - r_drop_block[voxel]

This is the variance-partitioning analogue of analyze_combo_coeff_peak_clusters.
Whereas the coefficient-fraction view is sensitive to ridge attribution between
correlated blocks, ``delta_r`` directly measures how much each block uniquely
contributes to validation prediction in each voxel and is therefore the cleaner
test of whether a region "needs" a long-context feature.

For each block we then identify reliable voxels with high unique ``delta_r``,
spatially cluster them, label each cluster's primary ROI, render flatmaps and
emit per-block / per-ROI counts.
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
sys.path.insert(0, str(REPO_DIR / "decoding"))
sys.path.insert(0, str(REPO_DIR / "mindeye_text"))
sys.path.insert(0, str(THIS_DIR))

import config  # noqa: E402
import run_summaries_encoding as rse  # noqa: E402
from _shared import load_or_build_chunk_embeddings  # noqa: E402
from analyze_combo_block_ablation import fit_ridge_corrs  # noqa: E402
from analyze_combo_coeff_sweep import model_tag  # noqa: E402
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
log = logging.getLogger("combo_block_delta_peak_clusters")

SUB_ROIS = ["BA_10", "BA_9_46", "BA_8", "BA_6", "BROCA", "BA_full_frontal"]
NAMED_ROIS = ["BA_10", "BA_9_46", "BA_8", "BA_6", "BROCA"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True, choices=sorted(rse.SUBJECT_TO_UTS))
    p.add_argument("--lag", type=int, default=2)
    p.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--summary-horizons", nargs="+", type=int, default=[20, 50, 200, 500])
    p.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=[1_000.0, 10_000.0, 100_000.0, 300_000.0, 1_000_000.0, 3_000_000.0, 10_000_000.0],
    )
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

    p.add_argument(
        "--reliable-r-threshold",
        type=float,
        default=0.05,
        help="Reliable-voxel threshold on full-model validation r.",
    )
    p.add_argument(
        "--delta-quantile",
        type=float,
        default=0.90,
        help="Per-block quantile (computed within reliable voxels) used to pick high-delta voxels for clustering.",
    )
    p.add_argument(
        "--delta-min",
        type=float,
        default=0.0,
        help="Lower bound on delta_r for a voxel to be eligible for clustering, in addition to the per-block quantile.",
    )
    p.add_argument("--min-cluster-size", type=int, default=3)
    p.add_argument(
        "--from-voxel-deltas-csv",
        default=None,
        help=(
            "If set, skip ridge fitting and load r_full / drop_<block>_r from this voxel_block_deltas.csv. "
            "Use to redo clustering / maps after a successful fit."
        ),
    )
    p.add_argument(
        "--out-dir",
        default=str(THIS_DIR / "results" / "combo_block_delta_peak_clusters" / "S1_bge"),
    )
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
        if roi in NAMED_ROIS
    }
    if not counts:
        return "unknown"
    best_roi, best_count = max(counts.items(), key=lambda item: item[1])
    return best_roi if best_count > 0 else "outside_named_rois"


def voxel_roi_label(voxel: int, roi_lookup: dict[str, set[int]]) -> str:
    for roi in NAMED_ROIS:
        if roi in roi_lookup and int(voxel) in roi_lookup[roi]:
            return roi
    return "outside_named_rois"


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_voxel_deltas_csv(
    path: Path, block_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Reload voxels / r_full / selected_alpha / r_drops from a previously written CSV."""
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"voxel deltas CSV is empty: {path}")
    voxels = np.asarray([int(r["global_voxel_index"]) for r in rows], dtype=np.int64)
    r_full = np.asarray([float(r["full_model_r"]) for r in rows], dtype=np.float32)
    full_alphas = np.asarray([float(r["selected_alpha"]) for r in rows], dtype=np.float32)
    r_drops: dict[str, np.ndarray] = {}
    for block in block_names:
        col = f"drop_{block}_r"
        if col not in rows[0]:
            raise ValueError(f"CSV {path} is missing column {col!r}")
        r_drops[block] = np.asarray([float(r[col]) for r in rows], dtype=np.float32)
    return voxels, r_full, full_alphas, r_drops


def fit_full_and_drops(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    block_slices: dict[str, slice],
    alphas: Sequence[float],
    voxel_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    log.info("Fitting full combo ridge with %d features", x_train.shape[1])
    r_full, full_alphas = fit_ridge_corrs(
        x_train,
        y_train,
        x_val,
        y_val,
        list(alphas),
        int(voxel_chunk_size),
    )
    r_drops: dict[str, np.ndarray] = {}
    for block, sl in block_slices.items():
        keep = np.concatenate(
            [np.arange(s.start, s.stop) for name, s in block_slices.items() if name != block]
        )
        log.info("Fitting drop-%s ridge with %d features", block, keep.size)
        r_drop, _ = fit_ridge_corrs(
            x_train[:, keep],
            y_train,
            x_val[:, keep],
            y_val,
            list(alphas),
            int(voxel_chunk_size),
        )
        r_drops[block] = r_drop
    return r_full, full_alphas, r_drops


def voxel_delta_rows(
    *,
    voxels: np.ndarray,
    r_full: np.ndarray,
    full_alphas: np.ndarray,
    r_drops: dict[str, np.ndarray],
    block_names: Sequence[str],
    subject: str,
    lag: int,
    embedding_model: str,
    roi_lookup: dict[str, set[int]],
    reliable_mask: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    deltas = {block: (r_full - r_drops[block]).astype(np.float32) for block in block_names}
    best_block_idx = np.full(len(voxels), -1, dtype=np.int32)
    stacked = np.stack([deltas[b] for b in block_names], axis=0)
    if reliable_mask.any():
        best_block_idx[reliable_mask] = np.argmax(stacked[:, reliable_mask], axis=0)
    for i, voxel in enumerate(voxels):
        row: dict = {
            "subject": subject,
            "lag": int(lag),
            "embedding_model": embedding_model,
            "embedding_tag": model_tag(embedding_model),
            "global_voxel_index": int(voxel),
            "roi": voxel_roi_label(int(voxel), roi_lookup),
            "full_model_r": float(r_full[i]),
            "selected_alpha": float(full_alphas[i]),
            "reliable": bool(reliable_mask[i]),
        }
        for block in block_names:
            row[f"drop_{block}_r"] = float(r_drops[block][i])
            row[f"delta_drop_{block}"] = float(deltas[block][i])
        if int(best_block_idx[i]) >= 0:
            row["best_block"] = block_names[int(best_block_idx[i])]
            row["best_delta"] = float(deltas[row["best_block"]][i])
        else:
            row["best_block"] = ""
            row["best_delta"] = float("nan")
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
    block: str,
    deltas: dict[str, np.ndarray],
    voxels: np.ndarray,
    full_corrs: np.ndarray,
    labels: np.ndarray,
    mask_ijk: np.ndarray,
    min_cluster_size: int,
    roi_lookup: dict[str, set[int]],
) -> list[dict]:
    out = []
    block_delta = deltas[block]
    for cluster_id in [int(c) for c in np.unique(labels) if int(c) > 0]:
        idx = np.nonzero(labels == cluster_id)[0]
        if idx.size < int(min_cluster_size):
            continue
        cluster_voxels = voxels[idx]
        peak_local = idx[int(np.argmax(block_delta[idx]))]
        centroid_ijk = mask_ijk[cluster_voxels].mean(axis=0)
        item: dict = {
            "block": block,
            "cluster_id": cluster_id,
            "n_voxels": int(idx.size),
            "primary_roi": primary_roi(cluster_voxels, roi_lookup),
            "mean_full_model_r": float(np.mean(full_corrs[idx])),
            "peak_full_model_r": float(np.max(full_corrs[idx])),
            "peak_delta_focal": float(block_delta[peak_local]),
            "peak_global_voxel": int(voxels[peak_local]),
            "centroid_i": float(centroid_ijk[0]),
            "centroid_j": float(centroid_ijk[1]),
            "centroid_k": float(centroid_ijk[2]),
        }
        for other in deltas:
            item[f"mean_delta_drop_{other}"] = float(np.mean(deltas[other][idx]))
        out.append(item)
    return sorted(out, key=lambda row: (-row["peak_delta_focal"], -row["n_voxels"]))


def roi_block_contingency(
    *,
    voxels: np.ndarray,
    block_labels: dict[str, np.ndarray],
    roi_lookup: dict[str, set[int]],
    min_cluster_size: int,
) -> list[dict]:
    rows = []
    for roi in NAMED_ROIS + ["outside_named_rois"]:
        for block, labels in block_labels.items():
            n_clusters = 0
            n_clustered_voxels = 0
            for cluster_id in [int(c) for c in np.unique(labels) if int(c) > 0]:
                idx = np.nonzero(labels == cluster_id)[0]
                if idx.size < int(min_cluster_size):
                    continue
                cluster_voxels = voxels[idx]
                if roi == "outside_named_rois":
                    matched = sum(
                        1
                        for v in cluster_voxels
                        if all(int(v) not in roi_lookup.get(r, set()) for r in NAMED_ROIS)
                    )
                else:
                    matched = sum(int(v) in roi_lookup.get(roi, set()) for v in cluster_voxels)
                if matched > 0:
                    n_clusters += 1
                    n_clustered_voxels += matched
            rows.append(
                {
                    "roi": roi,
                    "block": block,
                    "n_clusters_with_roi_voxels": n_clusters,
                    "n_clustered_voxels": n_clustered_voxels,
                }
            )
    return rows


def render_maps(
    *,
    subject: str,
    pycortex_subject: str,
    pycortex_filestore: Path,
    voxels: np.ndarray,
    full_corrs: np.ndarray,
    deltas: dict[str, np.ndarray],
    block_labels: dict[str, np.ndarray],
    block_names: Sequence[str],
    out_dir: Path,
    reliable_mask: np.ndarray,
    best_block_idx: np.ndarray,
    xfm_name: str | None,
    with_rois: bool,
) -> None:
    configure_pycortex_filestore(str(pycortex_filestore))
    import cortex
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_total = int(
        pycortex_filestore.joinpath(pycortex_subject).exists()
        and len(load_mask_ijk(pycortex_filestore, pycortex_subject)[1])
    )
    xfm, mask_voxels = find_matching_xfm(cortex, str(pycortex_filestore), pycortex_subject, n_total, xfm_name)
    if mask_voxels != n_total:
        n_total = mask_voxels

    full = full_corrs.copy()
    make_flatmap(
        cortex,
        plt,
        project_to_full_brain(full.astype(np.float32), voxels, n_total),
        pycortex_subject=pycortex_subject,
        xfm_name=xfm,
        vmin=float(np.nanmin(full)),
        vmax=max(0.1, float(np.nanquantile(full, 0.99))),
        cmap="inferno",
        title=f"{subject}: BGE ridge full-model validation r",
        out_path=out_dir / "full_model_r.png",
        with_rois=with_rois,
    )

    for block in block_names:
        masked = deltas[block].copy()
        masked[~reliable_mask] = np.nan
        vmax_delta = max(0.02, float(np.nanquantile(masked, 0.99)))
        make_flatmap(
            cortex,
            plt,
            project_to_full_brain(masked.astype(np.float32), voxels, n_total),
            pycortex_subject=pycortex_subject,
            xfm_name=xfm,
            vmin=0.0,
            vmax=vmax_delta,
            cmap="viridis",
            title=f"{subject}: delta_r drop {block}, reliable voxels",
            out_path=out_dir / f"delta_drop_{block}_masked.png",
            with_rois=with_rois,
        )
        cluster_values = block_labels[block].astype(np.float32)
        cluster_values[cluster_values <= 0] = np.nan
        if np.isfinite(cluster_values).any():
            make_flatmap(
                cortex,
                plt,
                project_to_full_brain(cluster_values.astype(np.float32), voxels, n_total),
                pycortex_subject=pycortex_subject,
                xfm_name=xfm,
                vmin=1.0,
                vmax=float(np.nanmax(cluster_values)),
                cmap="tab20",
                title=f"{subject}: high delta_r drop {block} clusters",
                out_path=out_dir / f"delta_drop_{block}_clusters.png",
                with_rois=with_rois,
            )

    best = best_block_idx.astype(np.float32)
    best[best < 0] = np.nan
    if np.isfinite(best).any():
        make_flatmap(
            cortex,
            plt,
            project_to_full_brain(best, voxels, n_total),
            pycortex_subject=pycortex_subject,
            xfm_name=xfm,
            vmin=0.0,
            vmax=float(len(block_names) - 1),
            cmap="tab10",
            title=f"{subject}: best block (argmax delta_r) reliable voxels",
            out_path=out_dir / "best_block_argmax_delta.png",
            with_rois=with_rois,
        )


def main() -> None:
    args = parse_args()
    args.embedding_device = resolve_embedding_device(args.embedding_device)
    args.summary_horizons = sorted({int(h) for h in args.summary_horizons})
    args.lags = [int(args.lag)]
    args.chunk_trs = 1
    args.feature_model = "embedding"

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    block_names = ["1TR"] + [f"h{h}" for h in args.summary_horizons]

    if args.from_voxel_deltas_csv:
        csv_path = Path(args.from_voxel_deltas_csv).expanduser().resolve()
        log.info("Resuming from %s (skipping ridge fits)", csv_path)
        voxels, r_full, full_alphas, r_drops = load_voxel_deltas_csv(csv_path, block_names)
        configure_data_root(args)
    else:
        mounted_root = configure_data_root(args)
        stories = load_stories(args)
        train_stories, val_stories = split_stories(stories, args)
        log.info(
            "%s: %d stories | %d train | %d val", args.subject, len(stories), len(train_stories), len(val_stories)
        )

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

        block_slices = {name: slice(i * one_dim, (i + 1) * one_dim) for i, name in enumerate(block_names)}

        x_train, y_train = stack_lag(combo, responses_by_story, train_stories, args.lag)
        x_val, y_val = stack_lag(combo, responses_by_story, val_stories, args.lag)
        log.info("X_train=%s X_val=%s Y_train=%s", x_train.shape, x_val.shape, y_train.shape)

        r_full, full_alphas, r_drops = fit_full_and_drops(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            block_slices=block_slices,
            alphas=args.ridge_alphas,
            voxel_chunk_size=args.voxel_chunk_size,
        )

    pycortex_filestore = Path(args.pycortex_filestore).expanduser().resolve()
    pycortex_subject = args.pycortex_subject or SUBJECT_TO_UTS[args.subject]
    mask, mask_ijk = load_mask_ijk(pycortex_filestore, pycortex_subject)
    roi_lookup = load_roi_lookup(Path(args.ba_dir).expanduser().resolve(), pycortex_subject)

    deltas = {block: (r_full - r_drops[block]).astype(np.float32) for block in block_names}
    reliable_mask = r_full >= float(args.reliable_r_threshold)
    if not np.any(reliable_mask):
        raise ValueError(f"No voxels passed reliable-r threshold {args.reliable_r_threshold}")

    best_block_idx = np.full(len(voxels), -1, dtype=np.int32)
    stacked = np.stack([deltas[b] for b in block_names], axis=0)
    if reliable_mask.any():
        best_block_idx[reliable_mask] = np.argmax(stacked[:, reliable_mask], axis=0)

    rows = voxel_delta_rows(
        voxels=voxels,
        r_full=r_full,
        full_alphas=full_alphas,
        r_drops=r_drops,
        block_names=block_names,
        subject=args.subject,
        lag=args.lag,
        embedding_model=args.embedding_model,
        roi_lookup=roi_lookup,
        reliable_mask=reliable_mask,
    )
    write_csv(out_dir / "voxel_block_deltas.csv", rows)

    block_labels: dict[str, np.ndarray] = {}
    delta_cutoffs: dict[str, float] = {}
    selected_per_block: dict[str, np.ndarray] = {}
    cluster_summaries: list[dict] = []
    for block in block_names:
        block_delta = deltas[block]
        reliable_block = block_delta[reliable_mask]
        cutoff = float(np.nanquantile(reliable_block, float(args.delta_quantile)))
        delta_cutoffs[block] = cutoff
        selected = (
            reliable_mask
            & (block_delta >= cutoff)
            & (block_delta >= float(args.delta_min))
        )
        selected_per_block[block] = selected
        labels = label_selected_clusters(
            selected_global_voxels=voxels[selected],
            all_global_voxels=voxels,
            mask_shape=mask.shape,
            mask_ijk=mask_ijk,
        )
        block_labels[block] = labels
        log.info(
            "block=%s reliable=%d cutoff=%.4f selected=%d clustered=%d",
            block,
            int(reliable_mask.sum()),
            cutoff,
            int(selected.sum()),
            int((labels > 0).sum()),
        )
        cluster_summaries.extend(
            summarize_clusters(
                block=block,
                deltas=deltas,
                voxels=voxels,
                full_corrs=r_full,
                labels=labels,
                mask_ijk=mask_ijk,
                min_cluster_size=args.min_cluster_size,
                roi_lookup=roi_lookup,
            )
        )
    write_csv(out_dir / "block_high_cluster_summary.csv", cluster_summaries)

    contingency = roi_block_contingency(
        voxels=voxels,
        block_labels=block_labels,
        roi_lookup=roi_lookup,
        min_cluster_size=args.min_cluster_size,
    )
    write_csv(out_dir / "roi_block_cluster_counts.csv", contingency)

    np.savez(
        out_dir / "block_delta_peak_cluster_maps.npz",
        voxels=voxels.astype(np.int64),
        full_model_r=r_full.astype(np.float32),
        selected_alpha=full_alphas.astype(np.float32),
        reliable_mask=reliable_mask.astype(bool),
        best_block_idx=best_block_idx.astype(np.int32),
        block_names=np.array(block_names, dtype=object),
        **{f"drop_{b}_r": r_drops[b].astype(np.float32) for b in block_names},
        **{f"delta_drop_{b}": deltas[b].astype(np.float32) for b in block_names},
        **{f"selected_{b}": selected_per_block[b].astype(bool) for b in block_names},
        **{f"cluster_labels_{b}": block_labels[b].astype(np.int32) for b in block_names},
        **{f"delta_cutoff_{b}": np.float32(delta_cutoffs[b]) for b in block_names},
    )

    render_maps(
        subject=args.subject,
        pycortex_subject=pycortex_subject,
        pycortex_filestore=pycortex_filestore,
        voxels=voxels,
        full_corrs=r_full,
        deltas=deltas,
        block_labels=block_labels,
        block_names=block_names,
        out_dir=out_dir,
        reliable_mask=reliable_mask,
        best_block_idx=best_block_idx,
        xfm_name=args.xfm_name,
        with_rois=args.with_rois,
    )
    log.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
