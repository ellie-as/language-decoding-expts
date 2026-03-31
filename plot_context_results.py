#!/usr/bin/env python3
"""
Plot results from run_context_encoding.py.

Loads all completed condition .npz files and produces:
  1. ROI bar/line plots comparing context lengths per frontal subregion
  2. (Optional) pycortex flatmaps if the pycortex database is available

Usage:
  python plot_context_results.py --subject S1 --rois frontal_rois_UTS01.json

  # With pycortex brain maps (requires pycortex + pycortex-db for the subject)
  python plot_context_results.py --subject S1 --rois frontal_rois_UTS01.json \
      --pycortex-subject UTS01
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "decoding"))
import config

REGION_ORDER = ["posterior_frontal", "middle_frontal", "anterior_frontal"]
REGION_LABELS = ["Posterior\nfrontal", "Middle\nfrontal", "Anterior\nfrontal"]
MODEL_CMAP = {
    "gpt1": "Blues", "gpt2": "Oranges", "gpt2-pool": "Reds",
    "embedding": "Greens",
}
MODEL_COLOR = {
    "gpt1": "#1f77b4", "gpt2": "#ff7f0e", "gpt2-pool": "#d62728",
    "embedding": "#2ca02c",
}


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_all_conditions(results_dir):
    """Load all per-condition .npz files. Returns dict[label] -> dict."""
    results = {}
    for path in sorted(glob(os.path.join(results_dir, "*.npz"))):
        name = Path(path).stem
        if name == "summary":
            continue
        d = np.load(path, allow_pickle=True)
        results[name] = {k: d[k] for k in d.files}
    return results


def build_roi_table(results, rois, vox):
    """Build a table: rows = conditions, cols = ROIs, values = mean corr."""
    global_to_local = {int(g): i for i, g in enumerate(vox)}
    local_rois = {}
    for rn in REGION_ORDER:
        if rn in rois:
            local_rois[rn] = np.array(
                [global_to_local[v] for v in rois[rn] if v in global_to_local],
                dtype=int,
            )
        else:
            local_rois[rn] = np.array([], dtype=int)

    table = {}
    for label, data in results.items():
        corrs = data["corrs"]
        table[label] = {}
        for rn in REGION_ORDER:
            idx = local_rois[rn]
            table[label][rn] = corrs[idx].mean() if len(idx) > 0 else np.nan
        table[label]["all"] = corrs.mean()
    return table, local_rois


def parse_label(label):
    """Parse 'gpt1_ctx20' -> ('gpt1', 20)."""
    parts = label.rsplit("_ctx", 1)
    return parts[0], int(parts[1])


# ---------------------------------------------------------------------------
# ROI plots
# ---------------------------------------------------------------------------

def plot_roi_lines(table, results, save_path):
    """Line plot: x = frontal subregion, separate line per context length,
    separate panel per model."""
    models = sorted({parse_label(l)[0] for l in table})
    ctx_lengths = sorted({parse_label(l)[1] for l in table})

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ai, model in enumerate(models):
        ax = axes[ai]
        cmap = plt.get_cmap(MODEL_CMAP.get(model, "viridis"))
        colors = cmap(np.linspace(0.35, 0.85, len(ctx_lengths)))

        for ci, ctx in enumerate(ctx_lengths):
            label = f"{model}_ctx{ctx}"
            if label not in table:
                continue
            vals = [table[label][rn] for rn in REGION_ORDER]
            ax.plot(range(len(REGION_ORDER)), vals, "o-", color=colors[ci],
                    label=f"ctx {ctx}", linewidth=2, markersize=6)

        ax.set_xticks(range(len(REGION_ORDER)))
        ax.set_xticklabels(REGION_LABELS, fontsize=9)
        ax.set_title(model.upper(), fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, title="Context\nlength", title_fontsize=8)
        ax.set_ylabel("Mean encoding correlation (r)" if ai == 0 else "")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Encoding correlation by frontal subregion and context length",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {save_path}")
    return fig


def plot_roi_bars(table, results, save_path):
    """Grouped bar chart: groups = context lengths, bars = frontal subregions,
    separate panel per model."""
    models = sorted({parse_label(l)[0] for l in table})
    ctx_lengths = sorted({parse_label(l)[1] for l in table})

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    region_colors = ["#4c72b0", "#dd8452", "#55a868"]

    for ai, model in enumerate(models):
        ax = axes[ai]
        n_ctx = len(ctx_lengths)
        n_reg = len(REGION_ORDER)
        bar_w = 0.8 / n_reg
        x = np.arange(n_ctx)

        for ri, rn in enumerate(REGION_ORDER):
            vals = []
            for ctx in ctx_lengths:
                label = f"{model}_ctx{ctx}"
                vals.append(table[label][rn] if label in table else np.nan)
            offset = (ri - (n_reg - 1) / 2) * bar_w
            ax.bar(x + offset, vals, bar_w * 0.9, color=region_colors[ri],
                   label=REGION_LABELS[ri].replace("\n", " "))

        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in ctx_lengths])
        ax.set_xlabel("Context length (words)")
        ax.set_title(model.upper(), fontsize=12, fontweight="bold")
        if ai == 0:
            ax.set_ylabel("Mean encoding correlation (r)")
            ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Encoding correlation by context length and frontal subregion",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {save_path}")
    return fig


def plot_per_voxel_hist(results, local_rois, save_path):
    """Per-voxel correlation histograms, one panel per condition, colored by ROI."""
    labels = sorted(results.keys(), key=lambda l: (parse_label(l)[0], parse_label(l)[1]))
    n = len(labels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)
    region_colors = ["#4c72b0", "#dd8452", "#55a868"]

    for i, label in enumerate(labels):
        ax = axes[i // ncols, i % ncols]
        corrs = results[label]["corrs"]
        for ri, rn in enumerate(REGION_ORDER):
            idx = local_rois.get(rn, np.array([], dtype=int))
            if len(idx) == 0:
                continue
            ax.hist(corrs[idx], bins=30, alpha=0.5, color=region_colors[ri],
                    label=rn.replace("_", " "), density=True)
        model, ctx = parse_label(label)
        ax.set_title(f"{model.upper()} ctx={ctx}", fontsize=10)
        ax.set_xlabel("r")
        if i == 0:
            ax.legend(fontsize=7)
        ax.axvline(0, color="gray", linewidth=0.5)

    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.suptitle("Per-voxel encoding correlation distributions by ROI",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Pycortex brain maps (optional)
# ---------------------------------------------------------------------------

def plot_pycortex_flatmaps(results, vox, pycortex_subject, n_voxels_total,
                           save_dir):
    """Generate pycortex flatmap PNGs for each condition."""
    try:
        import cortex
    except ImportError:
        print("  pycortex not installed — skipping brain maps")
        print("  Install with: pip install pycortex")
        return

    for label, data in sorted(results.items()):
        corrs = data["corrs"]
        full = np.zeros(n_voxels_total)
        full[vox] = corrs

        vol = cortex.Volume(full, pycortex_subject, "fullhead",
                            vmin=-0.1, vmax=0.3, cmap="inferno")
        fig = cortex.quickflat.make_figure(vol, with_curvature=True,
                                           with_colorbar=True)
        fig.suptitle(label, fontsize=14)
        out_path = save_dir / f"flatmap_{label}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--rois", required=True,
                        help="Frontal ROI JSON (e.g. frontal_rois_UTS01.json)")
    parser.add_argument("--results-dir", default=None,
                        help="Override results directory "
                             "(default: context_results/<subject>)")
    parser.add_argument("--pycortex-subject", default=None,
                        help="Pycortex subject ID for flatmap brain maps "
                             "(e.g. UTS01)")
    parser.add_argument("--n-total-voxels", type=int, default=81126,
                        help="Total voxels in full response matrix "
                             "(for pycortex, default 81126)")
    args = parser.parse_args()

    results_dir = args.results_dir or str(
        REPO_DIR / "context_results" / args.subject)

    if not os.path.isdir(results_dir):
        print(f"No results directory: {results_dir}")
        print("Run run_context_encoding.py first.")
        sys.exit(1)

    # Load all completed conditions
    results = load_all_conditions(results_dir)
    if not results:
        print(f"No .npz files found in {results_dir}")
        sys.exit(1)

    print(f"\n  Found {len(results)} completed conditions:")
    for label in sorted(results.keys()):
        print(f"    {label}")

    # Get voxel indices (should be same across conditions)
    first = next(iter(results.values()))
    vox = first["voxels"]

    # Load ROIs
    with open(args.rois) as f:
        rois = json.load(f)

    # Build ROI table
    table, local_rois = build_roi_table(results, rois, vox)

    # Print table
    print(f"\n  Mean encoding correlation per ROI:")
    hdr = f"    {'condition':<25s}"
    for rn in REGION_ORDER:
        hdr += f"  {rn:>20s}"
    hdr += f"  {'all':>8s}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for label in sorted(table.keys(), key=lambda l: (parse_label(l)[0], parse_label(l)[1])):
        row = f"    {label:<25s}"
        for rn in REGION_ORDER:
            row += f"  {table[label][rn]:20.4f}"
        row += f"  {table[label]['all']:8.4f}"
        print(row)

    # Output directory
    plot_dir = Path(results_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)

    # ROI plots
    print(f"\n  Generating plots...")
    plot_roi_lines(table, results, plot_dir / "roi_lines.png")
    plot_roi_bars(table, results, plot_dir / "roi_bars.png")
    plot_per_voxel_hist(results, local_rois, plot_dir / "voxel_histograms.png")

    # Pycortex flatmaps
    if args.pycortex_subject:
        print(f"\n  Generating pycortex flatmaps...")
        plot_pycortex_flatmaps(results, vox, args.pycortex_subject,
                               args.n_total_voxels, plot_dir)

    print(f"\n  All plots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
