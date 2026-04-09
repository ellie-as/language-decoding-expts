#!/usr/bin/env python3
"""
Plot results from run_summaries_encoding.py.

Loads all completed summary-horizon .npz files and produces:
  1. ROI line plots comparing summary horizons per Brodmann area
  2. ROI bar plots
  3. ROI-by-horizon trend plots
  4. (Optional) pycortex flatmaps if pycortex is available

Usage:
  python plot_summaries_encoding_corrs.py --subject S1

  python plot_summaries_encoding_corrs.py \
      --subject S1 \
      --results-dir /path/to/summaries_encoding_results/S1 \
      --pycortex-subject UTS01
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "decoding"))

import config  # noqa: E402

SUBJECT_TO_UTS = {"S1": "UTS01", "S2": "UTS02", "S3": "UTS03"}
REGION_ORDER = ["BA_6", "BA_8", "BA_9_46", "BA_10", "BROCA"]
REGION_LABELS = [
    "BA 6\n(premotor)",
    "BA 8\n(FEF)",
    "BA 9/46\n(DLPFC)",
    "BA 10\n(frontopolar)",
    "Broca's\narea",
]
REGION_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]


def load_all_conditions(results_dir):
    """Load all per-condition .npz files."""
    results = {}
    for path in sorted(glob(os.path.join(results_dir, "*.npz"))):
        name = Path(path).stem
        if name == "summary":
            continue
        data = np.load(path, allow_pickle=True)
        results[name] = {key: data[key] for key in data.files}
    return results


def load_ba_rois(ba_subject_dir):
    """Load individual BA ROI files, skipping BA_full_frontal."""
    rois = {}
    for path in sorted(glob(os.path.join(ba_subject_dir, "*.json"))):
        fname = os.path.basename(path)
        if fname == "BA_full_frontal.json":
            continue
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for key, indices in d.items():
            rois[key] = indices
    return rois


def build_roi_table(results, rois, vox):
    """Build a table: rows = conditions, cols = ROIs, values = mean corr."""
    global_to_local = {int(g): i for i, g in enumerate(vox)}
    local_rois = {}
    for region_name in REGION_ORDER:
        if region_name in rois:
            local_rois[region_name] = np.array(
                [global_to_local[v] for v in rois[region_name] if v in global_to_local],
                dtype=int,
            )
        else:
            local_rois[region_name] = np.array([], dtype=int)

    table = {}
    for label, data in results.items():
        corrs = data["corrs"]
        table[label] = {}
        for region_name in REGION_ORDER:
            idx = local_rois[region_name]
            table[label][region_name] = corrs[idx].mean() if len(idx) > 0 else np.nan
        table[label]["all"] = corrs.mean()
    return table, local_rois


def plot_pycortex_flatmaps(results, vox, pycortex_subject, n_voxels_total, save_dir, meta_by_label):
    """Generate pycortex flatmap PNGs for each condition."""
    try:
        import cortex
    except ImportError:
        print("  pycortex not installed — skipping brain maps")
        print("  Install with: pip install pycortex")
        return

    for label, data in sorted(results.items(), key=lambda item: sort_key(item[0], meta_by_label)):
        corrs = data["corrs"]
        full = np.zeros(n_voxels_total)
        full[vox] = corrs

        vol = cortex.Volume(full, pycortex_subject, "fullhead", vmin=-0.1, vmax=0.3, cmap="inferno")
        fig = cortex.quickflat.make_figure(vol, with_curvature=True, with_colorbar=True)
        fig.suptitle(describe_condition(meta_by_label[label]), fontsize=14)
        out_path = save_dir / f"flatmap_{label}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


def as_scalar_str(value, default="unknown"):
    """Convert 0-d / 1-element numpy arrays to plain strings."""
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return str(arr.reshape(-1)[0])


def as_scalar_int(value, default=None):
    """Convert 0-d / 1-element numpy arrays to plain ints."""
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return int(arr.reshape(-1)[0])


def build_meta_by_label(results):
    """Extract plotting metadata from saved result files."""
    meta = {}
    for label, data in results.items():
        feature_model = as_scalar_str(data.get("feature_model"), default="embedding")
        feature_backend = as_scalar_str(
            data.get("feature_backend"),
            default=as_scalar_str(data.get("embedding_model"), default="unknown"),
        )
        meta[label] = {
            "label": label,
            "summary_horizon": as_scalar_int(data.get("summary_horizon"), default=-1),
            "summary_model": as_scalar_str(data.get("summary_model")),
            "feature_model": feature_model,
            "feature_backend": feature_backend,
            "embedding_model": as_scalar_str(data.get("embedding_model")),
            "summary_words": as_scalar_int(data.get("summary_words"), default=-1),
        }
    return meta


def pretty_feature_name(feature_model):
    """Readable feature-model label."""
    mapping = {
        "gpt1": "GPT1",
        "gpt2": "GPT2",
        "gpt2-pool": "GPT2 pool",
        "embedding": "Embedding",
    }
    return mapping.get(feature_model, feature_model)


def short_backend_name(value):
    """Compact backend name for plot titles."""
    if not value or value == "unknown":
        return value
    parts = [part for part in str(value).replace("\\", "/").split("/") if part]
    if len(parts) >= 2:
        short = "/".join(parts[-2:])
    else:
        short = str(value)
    return short if len(short) <= 28 else short[-28:]


def should_show_backend(meta):
    """Whether the backend string adds information beyond the feature model."""
    feature_model = meta["feature_model"]
    feature_backend = meta["feature_backend"]
    if feature_backend in {"", "unknown"}:
        return False
    if feature_model == "gpt1" and feature_backend == "perceived":
        return False
    return True


def condition_short_label(meta):
    """Compact condition label for tables and histogram titles."""
    label = f"{meta['feature_model']} h={meta['summary_horizon']}"
    if should_show_backend(meta):
        label += f" [{short_backend_name(meta['feature_backend'])}]"
    return label


def group_key(meta):
    """Group conditions by summary source model and feature extractor."""
    return meta["summary_model"], meta["feature_model"], meta["feature_backend"]


def sort_key(label, meta_by_label):
    """Stable ordering for conditions in tables and plots."""
    meta = meta_by_label[label]
    return (
        meta["summary_model"],
        meta["feature_model"],
        meta["feature_backend"],
        meta["summary_horizon"],
        label,
    )


def describe_group(group):
    """Readable panel title."""
    summary_model, feature_model, feature_backend = group
    title = f"Summary: {summary_model}\nFeatures: {pretty_feature_name(feature_model)}"
    meta = {
        "feature_model": feature_model,
        "feature_backend": feature_backend,
        "embedding_model": feature_backend,
    }
    if should_show_backend(meta):
        title += f"\nBackend: {short_backend_name(feature_backend)}"
    return title


def describe_condition(meta):
    """Readable condition label."""
    label = f"{pretty_feature_name(meta['feature_model'])} h={meta['summary_horizon']}"
    if should_show_backend(meta):
        label += f"\n{short_backend_name(meta['feature_backend'])}"
    return label


def get_groups(meta_by_label):
    """Sorted unique condition groups."""
    return sorted({group_key(meta) for meta in meta_by_label.values()})


def get_group_labels(meta_by_label, group):
    """Condition labels belonging to one group, sorted by horizon."""
    labels = [label for label, meta in meta_by_label.items() if group_key(meta) == group]
    return sorted(labels, key=lambda label: sort_key(label, meta_by_label))


def plot_roi_lines(table, meta_by_label, save_path):
    """Line plot: x = frontal subregion, separate line per summary horizon."""
    groups = get_groups(meta_by_label)
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4.5), sharey=True, squeeze=False)
    axes = axes[0]

    for axis_index, group in enumerate(groups):
        ax = axes[axis_index]
        labels = get_group_labels(meta_by_label, group)
        colors = plt.get_cmap("viridis")(np.linspace(0.25, 0.9, len(labels)))

        for color, label in zip(colors, labels):
            horizon = meta_by_label[label]["summary_horizon"]
            vals = [table[label][region_name] for region_name in REGION_ORDER]
            ax.plot(
                range(len(REGION_ORDER)),
                vals,
                "o-",
                color=color,
                label=f"h {horizon}",
                linewidth=2,
                markersize=6,
            )

        ax.set_xticks(range(len(REGION_ORDER)))
        ax.set_xticklabels(REGION_LABELS, fontsize=9)
        ax.set_title(describe_group(group), fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, title="Summary\nhorizon", title_fontsize=8)
        ax.set_ylabel("Mean encoding correlation (r)" if axis_index == 0 else "")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Encoding correlation by frontal subregion and summary horizon", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_roi_bars(table, meta_by_label, save_path):
    """Grouped bar chart: groups = summary horizons, bars = Brodmann areas."""
    groups = get_groups(meta_by_label)
    fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 5), sharey=True, squeeze=False)
    axes = axes[0]

    for axis_index, group in enumerate(groups):
        ax = axes[axis_index]
        labels = get_group_labels(meta_by_label, group)
        horizons = [meta_by_label[label]["summary_horizon"] for label in labels]
        n_horizons = len(horizons)
        n_regions = len(REGION_ORDER)
        bar_width = 0.8 / n_regions
        x = np.arange(n_horizons)

        for region_index, region_name in enumerate(REGION_ORDER):
            vals = [table[label][region_name] for label in labels]
            offset = (region_index - (n_regions - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                vals,
                bar_width * 0.9,
                color=REGION_COLORS[region_index],
                label=REGION_LABELS[region_index].replace("\n", " "),
            )

        ax.set_xticks(x)
        ax.set_xticklabels([str(horizon) for horizon in horizons])
        ax.set_xlabel("Summary horizon (words)")
        ax.set_title(describe_group(group), fontsize=11, fontweight="bold")
        if axis_index == 0:
            ax.set_ylabel("Mean encoding correlation (r)")
            ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Encoding correlation by summary horizon and Brodmann area", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_horizon_trends(table, meta_by_label, save_path):
    """Direct ROI trend plot with x = summary horizon."""
    groups = get_groups(meta_by_label)
    fig, axes = plt.subplots(1, len(groups), figsize=(5.25 * len(groups), 4.5), sharey=True, squeeze=False)
    axes = axes[0]

    for axis_index, group in enumerate(groups):
        ax = axes[axis_index]
        labels = get_group_labels(meta_by_label, group)
        horizons = [meta_by_label[label]["summary_horizon"] for label in labels]
        x = np.arange(len(horizons))

        for region_index, region_name in enumerate(REGION_ORDER):
            vals = [table[label][region_name] for label in labels]
            ax.plot(
                x,
                vals,
                "o-",
                color=REGION_COLORS[region_index],
                linewidth=2,
                markersize=6,
                label=REGION_LABELS[region_index].replace("\n", " "),
            )

        all_vals = [table[label]["all"] for label in labels]
        ax.plot(
            x,
            all_vals,
            "o--",
            color="#333333",
            linewidth=1.8,
            markersize=5,
            label="All frontal voxels",
        )

        ax.set_xticks(x)
        ax.set_xticklabels([str(horizon) for horizon in horizons])
        ax.set_xlabel("Summary horizon (words)")
        ax.set_title(describe_group(group), fontsize=11, fontweight="bold")
        if axis_index == 0:
            ax.set_ylabel("Mean encoding correlation (r)")
            ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Encoding correlation trends across summary horizons", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_per_voxel_hist(results, local_rois, meta_by_label, save_path):
    """Per-voxel correlation histograms, one panel per condition, colored by BA."""
    labels = sorted(results.keys(), key=lambda label: sort_key(label, meta_by_label))
    n = len(labels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for i, label in enumerate(labels):
        ax = axes[i // ncols, i % ncols]
        corrs = results[label]["corrs"]
        for region_index, region_name in enumerate(REGION_ORDER):
            idx = local_rois.get(region_name, np.array([], dtype=int))
            if len(idx) == 0:
                continue
            ax.hist(
                corrs[idx],
                bins=30,
                alpha=0.45,
                color=REGION_COLORS[region_index],
                label=REGION_LABELS[region_index].replace("\n", " "),
                density=True,
            )
        ax.set_title(describe_condition(meta_by_label[label]), fontsize=10)
        ax.set_xlabel("r")
        if i == 0:
            ax.legend(fontsize=6)
        ax.axvline(0, color="gray", linewidth=0.5)

    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.suptitle("Per-voxel encoding correlation distributions by Brodmann area", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--ba-dir",
        default=str(REPO_DIR / "ba_indices"),
        help="Directory containing per-subject Brodmann area indices.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override results directory (default: summaries_encoding_results/<subject>)",
    )
    parser.add_argument(
        "--pycortex-subject",
        default=None,
        help="Pycortex subject ID for flatmap brain maps, e.g. UTS01.",
    )
    parser.add_argument(
        "--n-total-voxels",
        type=int,
        default=81126,
        help="Total voxels in the full response matrix for pycortex flatmaps.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir or str(REPO_DIR / "summaries_encoding_results" / args.subject)
    if not os.path.isdir(results_dir):
        print(f"No results directory: {results_dir}")
        print("Run run_summaries_encoding.py first.")
        sys.exit(1)

    uts_id = SUBJECT_TO_UTS.get(args.subject)
    ba_subject_dir = os.path.join(args.ba_dir, uts_id) if uts_id else None
    if not ba_subject_dir or not os.path.isdir(ba_subject_dir):
        print(f"No BA directory found at {ba_subject_dir} (subject {args.subject} -> {uts_id})")
        sys.exit(1)

    results = load_all_conditions(results_dir)
    if not results:
        print(f"No .npz files found in {results_dir}")
        sys.exit(1)

    meta_by_label = build_meta_by_label(results)

    print(f"\n  Found {len(results)} completed conditions:")
    for label in sorted(results.keys(), key=lambda item: sort_key(item, meta_by_label)):
        meta = meta_by_label[label]
        print(
            "    "
            f"{condition_short_label(meta):<34s} "
            f"summary_model={meta['summary_model']}"
        )

    first = next(iter(results.values()))
    vox = first["voxels"]
    rois = load_ba_rois(ba_subject_dir)
    table, local_rois = build_roi_table(results, rois, vox)

    print("\n  Mean encoding correlation per ROI:")
    hdr = f"    {'condition':<34s}"
    for region_name in REGION_ORDER:
        hdr += f"  {region_name:>20s}"
    hdr += f"  {'all':>8s}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for label in sorted(table.keys(), key=lambda item: sort_key(item, meta_by_label)):
        meta = meta_by_label[label]
        row = f"    {condition_short_label(meta):<34s}"
        for region_name in REGION_ORDER:
            row += f"  {table[label][region_name]:20.4f}"
        row += f"  {table[label]['all']:8.4f}"
        print(row)

    plot_dir = Path(results_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)

    print("\n  Generating plots...")
    plot_roi_lines(table, meta_by_label, plot_dir / "roi_lines.png")
    plot_roi_bars(table, meta_by_label, plot_dir / "roi_bars.png")
    plot_horizon_trends(table, meta_by_label, plot_dir / "horizon_trends.png")
    plot_per_voxel_hist(results, local_rois, meta_by_label, plot_dir / "voxel_histograms.png")

    if args.pycortex_subject:
        print("\n  Generating pycortex flatmaps...")
        plot_pycortex_flatmaps(
            results,
            vox,
            args.pycortex_subject,
            args.n_total_voxels,
            plot_dir,
            meta_by_label,
        )

    print(f"\n  All plots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
