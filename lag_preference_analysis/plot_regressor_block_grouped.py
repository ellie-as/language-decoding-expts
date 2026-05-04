#!/usr/bin/env python3
"""Plot grouped regressor-block bars from REGRESSORS_BY_REGION.md tables."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_NOTE = REPO_DIR / "experiment_notes" / "REGRESSORS_BY_REGION.md"
DEFAULT_OUT_DIR = REPO_DIR / "lag_preference_analysis" / "figures"

BLOCKS = ["1TR", "h20", "h50", "h200"]
BLOCK_COLORS = {
    "1TR": "#4C78A8",
    "h20": "#F58518",
    "h50": "#54A24B",
    "h200": "#B279A2",
}
ROI_ORDER = ["full_frontal", "BA_10", "BA_9_46", "BA_8", "BA_6", "BROCA"]
CONTEXT_WEIGHTS = {"1TR": 1.0, "h20": 20.0, "h50": 50.0, "h200": 200.0}


def sort_roi_rows(rows: list[dict[str, float | str | int]]) -> list[dict[str, float | str | int]]:
    return sorted(rows, key=lambda row: ROI_ORDER.index(str(row["roi"])))


def parse_block_table(table: str, subject: str) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for line in table.strip().splitlines():
        if "---" in line or line.startswith("| ROI"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 7:
            continue
        rows.append(
            {
                "subject": subject,
                "roi": cells[0],
                "n_voxels": int(cells[1].replace(",", "")),
                "1TR": float(cells[2]),
                "h20": float(cells[3]),
                "h50": float(cells[4]),
                "h200": float(cells[5]),
                "largest_block": cells[6],
            }
        )
    return sort_roi_rows(rows)


def parse_ridge_tables(note_path: Path) -> dict[str, dict[str, list[dict[str, float | str | int]]]]:
    """Extract documented ridge coefficient-fraction tables by subset and subject."""
    text = note_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"##\s+(S\d), Lag 2, Voxels With r > (0\.\d)\n\n"
        r"(?P<table>(?:\|.*\|\n)+)",
        flags=re.MULTILINE,
    )

    tables: dict[str, dict[str, list[dict[str, float | str | int]]]] = {}
    for match in pattern.finditer(text):
        subject = match.group(1)
        threshold = match.group(2).replace(".", "p")
        subset = f"r_gt_{threshold}"
        tables.setdefault(subset, {})[subject] = parse_block_table(match.group("table"), subject)
    if not tables:
        raise ValueError(f"No ridge coefficient tables found in {note_path}")
    return tables


def parse_lgbm_tables(note_path: Path) -> dict[str, list[dict[str, float | str | int]]]:
    """Extract the documented S1/S2 LGBM top-200 summary tables."""
    text = note_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"###\s+(S\d)\s+LGBM, Lag 2, Top 200 Voxels Per ROI\n\n"
        r"(?P<table>(?:\|.*\|\n)+)",
        flags=re.MULTILINE,
    )

    tables: dict[str, list[dict[str, float | str | int]]] = {}
    for match in pattern.finditer(text):
        subject = match.group(1)
        rows: list[dict[str, float | str | int]] = []
        for line in match.group("table").strip().splitlines():
            if "---" in line or line.startswith("| ROI"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) < 9:
                continue
            rows.append(
                {
                    "subject": subject,
                    "roi": cells[0],
                    "n_voxels": int(cells[1].replace(",", "")),
                    "1TR": float(cells[2]),
                    "h20": float(cells[3]),
                    "h50": float(cells[4]),
                    "h200": float(cells[5]),
                    "largest_block": cells[6],
                    "mean_lgbm_val_r": float(cells[7]),
                    "mean_saved_ridge_best_r": float(cells[8]),
                }
            )
        tables[subject] = sort_roi_rows(rows)
    if not tables:
        raise ValueError(f"No LGBM top-200 tables found in {note_path}")
    return tables


def add_context_metrics(rows: list[dict[str, float | str | int]]) -> None:
    for row in rows:
        row["long_context_share"] = float(row["h50"]) + float(row["h200"])
        row["context_horizon_index"] = sum(float(row[block]) * CONTEXT_WEIGHTS[block] for block in BLOCKS)


def plot_grouped_blocks(
    tables: dict[str, list[dict[str, float | str | int]]],
    out_path: Path,
    *,
    panel_label: str,
    y_label: str,
    suptitle: str,
    y_max: float = 0.55,
) -> None:
    subjects = sorted(tables)
    fig, axes = plt.subplots(1, len(subjects), figsize=(5.2 * len(subjects), 4.2), sharey=True)
    if len(subjects) == 1:
        axes = [axes]

    for ax, subject in zip(axes, subjects):
        rows = tables[subject]
        rois = [str(row["roi"]) for row in rows]
        x = np.arange(len(rois))
        width = 0.18
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(BLOCKS))
        for block, offset in zip(BLOCKS, offsets):
            vals = [float(row[block]) for row in rows]
            ax.bar(x + offset, vals, width=width, label=block, color=BLOCK_COLORS[block])

        ax.set_title(f"{subject} {panel_label}")
        ax.set_xticks(x)
        ax.set_xticklabels(rois, rotation=35, ha="right")
        ax.set_ylim(0.0, y_max)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_ylabel(y_label)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(BLOCKS), frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(suptitle, y=0.93)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_context_metrics(tables: dict[str, list[dict[str, float | str | int]]], out_path: Path) -> None:
    subjects = sorted(tables)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    metrics = [
        ("long_context_share", "h50 + h200 gain share"),
        ("context_horizon_index", "Gain-weighted horizon index"),
    ]

    max_context_index = 0.0
    for ax, (metric, ylabel) in zip(axes, metrics):
        for subject in subjects:
            rows = tables[subject]
            rois = [str(row["roi"]) for row in rows]
            vals = [float(row[metric]) for row in rows]
            if metric == "context_horizon_index":
                max_context_index = max(max_context_index, max(vals))
            ax.plot(rois, vals, marker="o", linewidth=2, label=subject)
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(len(rois)))
        ax.set_xticklabels(rois, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylim(0.0, 0.40)
    axes[1].set_ylim(0.0, max(45.0, max_context_index * 1.10))
    axes[1].legend(frameon=False)
    fig.suptitle("Simple context-gradient summaries from LGBM gain shares")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    tables = parse_lgbm_tables(args.note)
    for rows in tables.values():
        add_context_metrics(rows)

    plot_grouped_blocks(
        tables,
        args.out_dir / "regressor_block_grouped_lgbm_top200.png",
        panel_label="LGBM top 200 voxels/ROI",
        y_label="Mean gain-importance share",
        suptitle="LGBM block importance by ROI, all analyzed ROIs retained",
    )
    plot_context_metrics(tables, args.out_dir / "regressor_context_metrics_lgbm_top200.png")

    ridge_tables = parse_ridge_tables(args.note)
    for subset, subset_tables in ridge_tables.items():
        threshold = subset.replace("r_gt_", "r > ").replace("p", ".")
        plot_grouped_blocks(
            subset_tables,
            args.out_dir / f"regressor_block_grouped_ridge_{subset}.png",
            panel_label=f"ridge coefficient fractions ({threshold})",
            y_label="Mean coefficient-norm fraction",
            suptitle=f"Ridge block coefficient fractions by ROI, {threshold}",
            y_max=0.40,
        )


if __name__ == "__main__":
    main()
