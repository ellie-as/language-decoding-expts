#!/usr/bin/env python3
"""
Plot both frontal ROI schemes on an inflated fsaverage brain surface:
  • Scheme 1 — 3 frontal strips  (posterior / middle / anterior)
  • Scheme 2 — 4 Brodmann areas  (BA6 / BA8 / BA9-46 / BA10, approximate)

Both use the Destrieux surface atlas from nilearn.  The Brodmann mapping
mirrors the DK-based grouping in create_frontal_rois.py --scheme brodmann.

A pair of overlap heatmaps (% of each 3-ROI in each BA, and vice-versa) is
drawn below the brain panels.

Usage
-----
  micromamba run -n language_decoding_env python plot_frontal_rois.py
  micromamba run -n language_decoding_env python plot_frontal_rois.py --out my_figure.png

Requirements (in language_decoding_env)
-----------------------------------------
  pip install matplotlib    # only missing dependency
  # nibabel and nilearn are already installed
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from nilearn import datasets, surface, plotting


# ── Scheme 1: Destrieux index → 3-ROI frontal strips ──────────────────────
# (posterior / middle / anterior — original grouping)
DESTRIEUX_TO_ROI = {
    # --- 1: posterior frontal (motor / premotor) ---
    3:  1,   # G_and_S_paracentral
    29: 1,   # G_precentral
    46: 1,   # S_central
    69: 1,   # S_precentral-inf-part
    70: 1,   # S_precentral-sup-part
    # --- 2: middle frontal (DLPFC / IFG) ---
    7:  2,   # G_and_S_cingul-Mid-Ant  (caudalACC)
    12: 2,   # G_front_inf-Opercular   (parsopercularis)
    14: 2,   # G_front_inf-Triangul    (parstriangularis)
    15: 2,   # G_front_middle          (rostralmiddlefrontal + caudalmiddlefrontal)
    53: 2,   # S_front_inf
    54: 2,   # S_front_middle
    56: 2,   # S_interm_prim-Jensen
    # --- 3: anterior frontal (frontopolar / OFC / vmPFC) ---
    1:  3,   # G_and_S_frontomargin
    5:  3,   # G_and_S_transv_frontopol
    6:  3,   # G_and_S_cingul-Ant      (rostralACC)
    13: 3,   # G_front_inf-Orbital     (parsorbitalis)
    16: 3,   # G_front_sup             (superiorfrontal)
    24: 3,   # G_orbital               (lateralorbitofrontal)
    31: 3,   # G_rectus                (medialorbitofrontal / gyrus rectus)
    55: 3,   # S_front_sup
    63: 3,   # S_orbital_lateral
    64: 3,   # S_orbital_med-olfact
    65: 3,   # S_orbital-H_Shaped
    71: 3,   # S_suborbital
}

ROI_COLORS = ['#4472C4', '#70AD47', '#FF6B35']   # blue, green, orange
ROI_LABELS = [
    'Posterior frontal\n(motor / premotor)',
    'Middle frontal\n(DLPFC / IFG)',
    'Anterior frontal\n(frontopolar / OFC / vmPFC)',
]
ROI_SHORT = ['Posterior', 'Middle', 'Anterior']


# ── Scheme 2: Destrieux index → 4-ROI Brodmann approximation ──────────────
# Mirrors create_frontal_rois.py --scheme brodmann (DK → Destrieux mapping).
# Note: Destrieux G_front_middle spans both DK rostralmiddlefrontal (BA9/46)
# and caudalmiddlefrontal (BA8) — it is assigned to BA9/46 here (DLPFC proxy).
# The surrounding sulci (S_front_inf, S_front_middle, S_interm_prim-Jensen)
# serve as the BA8 proxy for the caudal / premotor-adjacent territory.
DESTRIEUX_TO_BRODMANN = {
    # --- 1: BA 6  premotor / SMA  (DK: precentral + paracentral) ---
    3:  1,   # G_and_S_paracentral     → DK paracentral
    29: 1,   # G_precentral            → DK precentral
    46: 1,   # S_central
    69: 1,   # S_precentral-inf-part
    70: 1,   # S_precentral-sup-part
    # --- 2: BA 8  caudal MFG / FEF  (DK: caudalmiddlefrontal) ---
    53: 2,   # S_front_inf             caudal border of MFG
    54: 2,   # S_front_middle          within MFG (caudal)
    56: 2,   # S_interm_prim-Jensen    caudal premotor territory
    # --- 3: BA 9/46  DLPFC + medial BA 9  (DK: rostralmiddlefrontal + superiorfrontal) ---
    15: 3,   # G_front_middle          DLPFC (lateral)
    16: 3,   # G_front_sup             medial BA 9 / SFG medial wall  ← added
    55: 3,   # S_front_sup             dorsal MFG border
    # --- 4: BA 10  frontopolar + medial  (DK: frontalpole + rostralACC + medialorbitofrontal) ---
    1:  4,   # G_and_S_frontomargin
    5:  4,   # G_and_S_transv_frontopol
    6:  4,   # G_and_S_cingul-Ant      medial BA 10/32 (rostral ACC / vmPFC)  ← added
    31: 4,   # G_rectus                ventromedial BA 10/11 (gyrus rectus)   ← added
    64: 4,   # S_orbital_med-olfact    medial orbital sulcus                   ← added
    71: 4,   # S_suborbital            suborbital region                       ← added
}

BA_COLORS = ['#C0392B', '#8E44AD', '#2980B9', '#D4AC0D']  # red, purple, blue, gold
BA_LABELS = [
    'BA 6\n(premotor / SMA)',
    'BA 8\n(caudal MFG / FEF)',
    'BA 9/46\n(DLPFC / medial BA 9)',
    'BA 10\n(frontopolar / vmPFC)',
]
BA_SHORT = ['BA6', 'BA8', 'BA9/46', 'BA10']


# ── Helpers ────────────────────────────────────────────────────────────────

def make_roi_map(parc: np.ndarray, mapping: dict) -> np.ndarray:
    """Return a per-vertex array using the given label→value mapping."""
    out = np.zeros(len(parc), dtype=float)
    for label_idx, val in mapping.items():
        out[parc == label_idx] = val
    return out



def compute_overlap(roi_lh, roi_rh, ba_lh, ba_rh, n_roi=3, n_ba=4):
    """Return (count matrix, row-normalised %, col-normalised %) arrays."""
    roi_all = np.concatenate([roi_lh, roi_rh])
    ba_all  = np.concatenate([ba_lh,  ba_rh])

    counts = np.zeros((n_roi, n_ba), dtype=float)
    for i in range(n_roi):
        for j in range(n_ba):
            counts[i, j] = np.sum((roi_all == i + 1) & (ba_all == j + 1))

    row_totals = counts.sum(axis=1, keepdims=True)
    col_totals = counts.sum(axis=0, keepdims=True)
    row_pct = np.where(row_totals > 0, 100 * counts / row_totals, 0)
    col_pct = np.where(col_totals > 0, 100 * counts / col_totals, 0)
    return counts, row_pct, col_pct


def draw_heatmap(ax, data, row_labels, col_labels, title, row_colors, col_colors):
    """Draw an annotated heatmap with coloured row/column label patches."""
    im = ax.imshow(data, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=10, pad=8)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            color = 'white' if v > 55 else 'black'
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                    fontsize=8.5, color=color, fontweight='bold')

    # Colour-coded row and column tick labels
    for tick, color in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(color)
        tick.set_fontweight('bold')
    for tick, color in zip(ax.get_xticklabels(), col_colors):
        tick.set_color(color)
        tick.set_fontweight('bold')

    plt.colorbar(im, ax=ax, label='% of vertices', shrink=0.8, pad=0.02)


# ── Main ───────────────────────────────────────────────────────────────────

def main(out_path: str):
    print("Fetching fsaverage5 surfaces …")
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')

    print("Fetching Destrieux atlas …")
    destrieux = datasets.fetch_atlas_surf_destrieux()

    parc_lh = surface.load_surf_data(destrieux['map_left'])
    parc_rh = surface.load_surf_data(destrieux['map_right'])

    # Build per-vertex ROI maps for both schemes
    roi_lh = make_roi_map(parc_lh, DESTRIEUX_TO_ROI)
    roi_rh = make_roi_map(parc_rh, DESTRIEUX_TO_ROI)
    ba_lh  = make_roi_map(parc_lh, DESTRIEUX_TO_BRODMANN)
    ba_rh  = make_roi_map(parc_rh, DESTRIEUX_TO_BRODMANN)

    # Colormaps
    roi_cmap = mcolors.LinearSegmentedColormap.from_list(
        'frontal_rois', ROI_COLORS, N=3)
    ba_cmap  = mcolors.LinearSegmentedColormap.from_list(
        'brodmann',     BA_COLORS,  N=4)

    # Surface data shortcuts
    surfs = {
        'left':  {'surf': fsaverage['infl_left'],  'sulc': fsaverage['sulc_left']},
        'right': {'surf': fsaverage['infl_right'], 'sulc': fsaverage['sulc_right']},
    }
    view_order = [('left', 'lateral'), ('left', 'medial'),
                  ('right', 'lateral'), ('right', 'medial')]

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('white')
    gs = GridSpec(3, 4, figure=fig,
                  height_ratios=[1.1, 1.1, 0.9],
                  hspace=0.35, wspace=0.04)

    axes_s1 = [fig.add_subplot(gs[0, i], projection='3d') for i in range(4)]
    axes_s2 = [fig.add_subplot(gs[1, i], projection='3d') for i in range(4)]
    ax_heat1 = fig.add_subplot(gs[2, :2])   # row-normalised overlap
    ax_heat2 = fig.add_subplot(gs[2, 2:])   # col-normalised overlap

    # ── Plot scheme 1 (3-ROI strips) ───────────────────────────────────────
    print("Plotting scheme 1 (frontal strips) …")
    for ax, (hemi, view_name) in zip(axes_s1, view_order):
        roi_data = roi_lh if hemi == 'left' else roi_rh
        plotting.plot_surf_stat_map(
            surf_mesh=surfs[hemi]['surf'],
            stat_map=roi_data,
            hemi=hemi,
            view=view_name,
            bg_map=surfs[hemi]['sulc'],
            bg_on_data=True,
            threshold=0.5,
            cmap=roi_cmap,
            vmin=0.5, vmax=3.5,
            colorbar=False,
            axes=ax,
            figure=fig,
            title=f"{hemi[0].upper()}H  {view_name}",
            title_font_size=10,
        )

    # Row 1 legend
    handles_s1 = [mpatches.Patch(facecolor=c, label=l)
                  for c, l in zip(ROI_COLORS, ROI_LABELS)]
    fig.legend(handles=handles_s1, loc='upper center',
               bbox_to_anchor=(0.5, 0.98), ncol=3,
               fontsize=9, frameon=True,
               title='Scheme 1 — Frontal strips (3 ROIs)', title_fontsize=9)

    # ── Plot scheme 2 (Brodmann) ────────────────────────────────────────────
    print("Plotting scheme 2 (Brodmann areas) …")
    for ax, (hemi, view_name) in zip(axes_s2, view_order):
        ba_data = ba_lh if hemi == 'left' else ba_rh
        plotting.plot_surf_stat_map(
            surf_mesh=surfs[hemi]['surf'],
            stat_map=ba_data,
            hemi=hemi,
            view=view_name,
            bg_map=surfs[hemi]['sulc'],
            bg_on_data=True,
            threshold=0.5,
            cmap=ba_cmap,
            vmin=0.5, vmax=4.5,
            colorbar=False,
            axes=ax,
            figure=fig,
            title=f"{hemi[0].upper()}H  {view_name}",
            title_font_size=10,
        )

    # Row 2 legend
    handles_s2 = [mpatches.Patch(facecolor=c, label=l)
                  for c, l in zip(BA_COLORS, BA_LABELS)]
    fig.legend(handles=handles_s2, loc='upper center',
               bbox_to_anchor=(0.5, 0.635), ncol=4,
               fontsize=9, frameon=True,
               title='Scheme 2 — Brodmann areas (4 ROIs, approximate)', title_fontsize=9)

    # ── Overlap heatmaps ────────────────────────────────────────────────────
    print("Computing overlap …")
    counts, row_pct, col_pct = compute_overlap(roi_lh, roi_rh, ba_lh, ba_rh)

    # Print summary to console
    print("\n  Vertex counts (3-ROI rows × BA columns):")
    header = f"{'':>12s}" + "".join(f"  {b:>8s}" for b in BA_SHORT)
    print(header)
    for i, r in enumerate(ROI_SHORT):
        row_str = f"  {r:>10s}" + "".join(f"  {int(counts[i,j]):>8d}" for j in range(4))
        print(row_str)

    draw_heatmap(
        ax_heat1,
        row_pct,
        row_labels=ROI_SHORT,
        col_labels=BA_SHORT,
        title='% of each 3-ROI covered by each Brodmann area\n(rows sum to 100%)',
        row_colors=ROI_COLORS,
        col_colors=BA_COLORS,
    )

    draw_heatmap(
        ax_heat2,
        col_pct.T,   # transpose: rows=BA, cols=3-ROI
        row_labels=BA_SHORT,
        col_labels=ROI_SHORT,
        title='% of each Brodmann area covered by each 3-ROI\n(rows sum to 100%)',
        row_colors=BA_COLORS,
        col_colors=ROI_COLORS,
    )

    fig.suptitle(
        'Frontal ROIs: 3-strip scheme (top) vs Brodmann approximation (middle)\n'
        'Destrieux surface atlas on fsaverage5',
        fontsize=12, y=1.005,
    )

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out',
        default=str(Path(__file__).parent / 'frontal_rois_figure.png'),
        help='Output PNG path',
    )
    args = parser.parse_args()
    main(args.out)
