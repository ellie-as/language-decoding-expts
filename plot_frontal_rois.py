#!/usr/bin/env python3
"""
Plot the 3 frontal ROIs (posterior / middle / anterior) on an inflated
fsaverage brain surface using nilearn's Destrieux atlas.

The Destrieux parcels are mapped to the same three groups defined in
create_frontal_rois.py (Desikan-Killiany grouping logic).

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
from nilearn import datasets, surface, plotting


# ── Destrieux label index → ROI group (0 = not frontal) ───────────────────
# Destrieux atlas parcels mapped to the three frontal strips from
# create_frontal_rois.py (precentral/paracentral/caudalmiddlefrontal →
# posterior; rostralmiddlefrontal/IFG/caudACC → middle;
# superiorfrontal/OFC/frontalpole/rostACC → anterior).
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


def make_roi_map(parc: np.ndarray) -> np.ndarray:
    """Return a per-vertex array: 0 = not frontal, 1/2/3 = ROI group."""
    roi = np.zeros(len(parc), dtype=float)
    for label_idx, roi_val in DESTRIEUX_TO_ROI.items():
        roi[parc == label_idx] = roi_val
    return roi


def main(out_path: str):
    print("Fetching fsaverage5 surfaces …")
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')

    print("Fetching Destrieux atlas …")
    destrieux = datasets.fetch_atlas_surf_destrieux()

    parc_left  = surface.load_surf_data(destrieux['map_left'])
    parc_right = surface.load_surf_data(destrieux['map_right'])

    roi_left  = make_roi_map(parc_left)
    roi_right = make_roi_map(parc_right)

    # Discrete 3-colour map; vmin/vmax centre each integer on one colour
    cmap = mcolors.LinearSegmentedColormap.from_list('frontal_rois', ROI_COLORS, N=3)

    views = [
        # (hemi, view, roi_map, surf_mesh, sulc_map, title)
        ('left',  'lateral', roi_left,  fsaverage['infl_left'],  fsaverage['sulc_left'],  'LH  lateral'),
        ('left',  'medial',  roi_left,  fsaverage['infl_left'],  fsaverage['sulc_left'],  'LH  medial'),
        ('right', 'lateral', roi_right, fsaverage['infl_right'], fsaverage['sulc_right'], 'RH  lateral'),
        ('right', 'medial',  roi_right, fsaverage['infl_right'], fsaverage['sulc_right'], 'RH  medial'),
    ]

    fig, axes = plt.subplots(
        1, 4,
        subplot_kw={'projection': '3d'},
        figsize=(20, 5),
    )
    fig.patch.set_facecolor('white')

    print("Plotting …")
    for ax, (hemi, view, roi, surf, sulc, title) in zip(axes, views):
        plotting.plot_surf_stat_map(
            surf_mesh=surf,
            stat_map=roi,
            hemi=hemi,
            view=view,
            bg_map=sulc,
            bg_on_data=True,
            threshold=0.5,        # mask non-frontal vertices
            cmap=cmap,
            vmin=0.5, vmax=3.5,   # maps 1→blue, 2→green, 3→orange
            colorbar=False,
            axes=ax,
            figure=fig,
            title=title,
            title_font_size=11,
        )

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=c, label=l)
        for c, l in zip(ROI_COLORS, ROI_LABELS)
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=3,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.suptitle('Frontal ROIs — Desikan-Killiany grouping (fsaverage5)',
                 fontsize=13, y=1.02)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out',
        default=str(Path(__file__).parent / 'frontal_rois_figure.png'),
        help='Output PNG path',
    )
    args = parser.parse_args()
    main(args.out)
