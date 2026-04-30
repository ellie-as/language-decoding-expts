#!/usr/bin/env python3
"""Run a small MindEye-encoding hyperparameter sweep.

The sweep trains several shared-text-latent / subject-head neural encoders and
prints their held-out-story validation correlations. Each training run also
fits a ridge baseline on the same train stories and scores it on the same
validation stories, so the summary includes a like-for-like neural-vs-ridge
comparison.

Run from the repo root on the cluster.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_DIR / "mindeye_text" / "train_mindeye_encoding.py"


VARIANTS = [
    {
        "name": "latent4096_blocks4_drop015_wd001",
        "args": ["--latent-dim", "4096", "--n-blocks", "4", "--dropout", "0.15", "--weight-decay", "1e-2", "--lr", "3e-4"],
    },
    {
        "name": "latent2048_blocks2_drop030_wd005",
        "args": ["--latent-dim", "2048", "--n-blocks", "2", "--dropout", "0.30", "--weight-decay", "5e-2", "--lr", "3e-4"],
    },
    {
        "name": "latent2048_blocks4_drop020_wd002",
        "args": ["--latent-dim", "2048", "--n-blocks", "4", "--dropout", "0.20", "--weight-decay", "2e-2", "--lr", "2e-4"],
    },
    {
        "name": "latent4096_blocks2_drop030_wd005",
        "args": ["--latent-dim", "4096", "--n-blocks", "2", "--dropout", "0.30", "--weight-decay", "5e-2", "--lr", "2e-4"],
    },
    {
        "name": "latent1024_blocks2_drop040_wd010",
        "args": ["--latent-dim", "1024", "--n-blocks", "2", "--dropout", "0.40", "--weight-decay", "1e-1", "--lr", "5e-4"],
    },
    {
        "name": "latent3072_blocks3_drop025_wd003",
        "args": ["--latent-dim", "3072", "--n-blocks", "3", "--dropout", "0.25", "--weight-decay", "3e-2", "--lr", "2e-4"],
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"])
    p.add_argument("--torch-device", default="cuda", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--output-dir", default="mindeye_text/encoding_results_sweep")
    p.add_argument("--baseline-dir", default="gpt1_encoding_comparison/outputs")
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--val-story-count", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="sweep")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--extra-train-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args appended to every train_mindeye_encoding.py call.",
    )
    return p.parse_args()


def ridge_reference(subjects: list[str], baseline_dir: Path) -> dict:
    rows = {}
    vals = []
    for subject in subjects:
        path = baseline_dir / subject / "encoding_model_finetuned.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        corrs = np.asarray(data["bootstrap_corrs"], dtype=np.float32)
        vox = np.asarray(data["voxels"], dtype=np.int64)
        selected_mean = float(corrs[vox].mean())
        selected_median = float(np.median(corrs[vox]))
        rows[subject] = {
            "ridge_selected_mean_r": selected_mean,
            "ridge_selected_median_r": selected_median,
        }
        vals.append(selected_mean)
    rows["ALL"] = {"ridge_selected_mean_r": float(np.mean(vals)) if vals else ""}
    return rows


def load_summary(out_dir: Path, tag: str) -> dict:
    path = out_dir / tag / "summary.json"
    if not path.exists():
        return {"status": "missing_summary"}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    refs = ridge_reference(args.subjects, baseline_dir)
    ridge_all = refs.get("ALL", {}).get("ridge_selected_mean_r", "")
    print(f"Ridge GPT-1 selected-voxel CV mean r reference (not like-for-like): {ridge_all}")

    rows = []
    for variant in VARIANTS:
        tag = f"{args.prefix}__{variant['name']}__seed{args.seed}"
        summary_path = out_dir / tag / "summary.json"
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--subjects",
            *args.subjects,
            "--torch-device",
            args.torch_device,
            "--output-dir",
            str(out_dir),
            "--tag",
            tag,
            "--max-epochs",
            str(args.max_epochs),
            "--patience",
            str(args.patience),
            "--batch-size",
            str(args.batch_size),
            "--val-mode",
            "story",
            "--val-story-count",
            str(args.val_story_count),
            "--selection-metric",
            "val_mean_r",
            "--seed",
            str(args.seed),
            *variant["args"],
            *args.extra_train_args,
        ]
        print("\n===", tag, "===")
        print(" ".join(cmd))
        if args.dry_run:
            status = "dry_run"
        elif args.skip_existing and summary_path.exists():
            status = "reused"
            print(f"Reusing existing {summary_path}")
        else:
            completed = subprocess.run(cmd, cwd=REPO_DIR, check=False)
            status = "ok" if completed.returncode == 0 else f"failed_{completed.returncode}"

        summary = load_summary(out_dir, tag)
        all_summary = summary.get("ALL", {})
        best_r = all_summary.get("best_r_epoch_val_mean_r", "")
        row = {
            "variant": variant["name"],
            "tag": tag,
            "status": status,
            "best_r_epoch": all_summary.get("best_r_epoch", ""),
            "best_val_mean_r_all": best_r,
            "best_val_max_r_all": all_summary.get("best_r_epoch_val_max_r", ""),
            "ridge_like_for_like_mean_r_all": all_summary.get("ridge_like_for_like_mean_r", ""),
            "delta_vs_like_for_like_ridge": all_summary.get("delta_best_r_minus_ridge_like_for_like", ""),
            "ridge_selected_cv_mean_r_all": ridge_all,
            "delta_val_r_minus_ridge_cv_r": (
                float(best_r) - float(ridge_all)
                if best_r != "" and ridge_all != ""
                else ""
            ),
            "summary_path": str(summary_path),
            "checkpoint": str(out_dir / tag / "model.pt"),
        }
        for subject in args.subjects:
            subj_summary = summary.get(subject, {})
            row[f"best_val_mean_r_{subject}"] = subj_summary.get("best_r_epoch_val_mean_r", "")
            row[f"ridge_like_for_like_mean_r_{subject}"] = subj_summary.get("ridge_like_for_like_mean_r", "")
            row[f"delta_vs_like_for_like_ridge_{subject}"] = subj_summary.get(
                "delta_best_r_minus_ridge_like_for_like", ""
            )
            row[f"ridge_selected_cv_mean_r_{subject}"] = refs.get(subject, {}).get("ridge_selected_mean_r", "")
        rows.append(row)

        if best_r != "":
            print(
                f"{variant['name']}: heldout-story mean r={float(best_r):.4f} "
                f"(like-for-like ridge={row['ridge_like_for_like_mean_r_all']}, "
                f"delta={row['delta_vs_like_for_like_ridge']})"
            )

    csv_path = out_dir / f"{args.prefix}__summary.csv"
    fields = sorted({key for row in rows for key in row})
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {csv_path}")
    print("\nSummary:")
    for row in sorted(
        rows,
        key=lambda r: float(r["best_val_mean_r_all"]) if r["best_val_mean_r_all"] != "" else -999,
        reverse=True,
    ):
        print(
            f"{row['variant']:<36} status={row['status']:<10} "
            f"heldout_r={row['best_val_mean_r_all']} "
            f"ridge_lfl={row['ridge_like_for_like_mean_r_all']} "
            f"delta_lfl={row['delta_vs_like_for_like_ridge']}"
        )


if __name__ == "__main__":
    main()
