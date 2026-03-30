#!/usr/bin/env python3
"""
Create frontal-subregion ROI JSON files for the attribution analysis.

This script maps FreeSurfer Desikan-Killiany atlas labels to the flat voxel
indices used in the HDF5 response files.  Two labelling schemes are supported:

  --scheme frontal_strips   (default)
      Groups frontal labels into posterior / middle / anterior strips.

  --scheme brodmann
      Groups frontal labels into four approximate Brodmann-area ROIs:
        ba10    frontopolar cortex
        ba9_46  dorsolateral prefrontal cortex (DLPFC)
        ba8     frontal eye field / dorsal premotor
        ba6     premotor cortex / SMA

      Note: the Desikan-Killiany atlas follows gyral/sulcal boundaries that
      do not perfectly align with cytoarchitectonic Brodmann-area borders.
      The mapping used here is a standard anatomical approximation widely
      applied in the neuroimaging literature.

Required inputs (from the OpenNeuro dataset ds003020):
  1. A brain mask in functional space — a NIfTI file where non-zero voxels
     correspond to the columns in the HDF5 response arrays.
  2. The FreeSurfer aparc+aseg parcellation resampled to functional space.

Obtaining these files
---------------------
The OpenNeuro dataset includes FreeSurfer outputs for each subject.  To get the
parcellation in functional space you typically:

  (a) Download the subject's FreeSurfer directory from OpenNeuro
      (derivatives/freesurfer/sub-{SUBJECT}/).

  (b) Convert aparc+aseg.mgz → functional space.  If the preprocessing used
      a particular brain mask (e.g. from fMRIPrep), resample to that space:

        mri_label2vol \\
          --seg $SUBJECTS_DIR/sub-S1/mri/aparc+aseg.mgz \\
          --temp functional_mask.nii.gz \\
          --reg register.dat \\
          --o aparc_in_func.nii.gz

      Or, if the data were processed with pycortex, you can extract the
      parcellation and mask directly from the pycortex database:

        import cortex
        aparc = cortex.get_roi_masks("S1", "atlas")
        # ... see pycortex docs

  (c) The brain mask is the set of voxels stored in the HDF5 files.  If you
      have the preprocessed NIfTI, threshold it to create a mask.  Or if a
      mask.nii.gz is provided in the OpenNeuro derivatives, use that directly.

If the parcellation and mask are already in the same 3D space, this script
maps parcellation labels → 3D voxels → flat HDF5 indices → ROI JSON.

Alternatively, if you already have per-voxel coordinates (e.g. from pycortex
or an exported CSV), use --coords mode (see below).

Usage
-----
  # Default scheme (posterior / middle / anterior strips)
  python create_frontal_rois.py \\
      --aparc aparc_in_func.nii.gz \\
      --mask  brain_mask.nii.gz \\
      --out   frontal_rois_S1.json

  # Brodmann-area scheme
  python create_frontal_rois.py \\
      --aparc aparc_in_func.nii.gz \\
      --mask  brain_mask.nii.gz \\
      --scheme brodmann \\
      --out   frontal_rois_ba_S1.json

  # From a CSV of voxel coordinates (flat_index, x, y, z, label)
  python create_frontal_rois.py \\
      --coords voxel_info.csv \\
      --scheme brodmann \\
      --out frontal_rois_ba_S1.json

  # Then use with the attribution script
  python run_attribution.py --subject S1 --experiment perceived_speech \\
      --task wheretheressmoke --use-saved --rois frontal_rois_ba_S1.json
"""

import json
import argparse
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Labelling scheme 1 — frontal strips (original)
# FreeSurfer Desikan-Killiany label IDs (left hemisphere; right = left + 1000)
# ─────────────────────────────────────────────────────────────────────────────

FRONTAL_LABELS = {
    # --- Posterior frontal (motor / premotor) ---
    "posterior_frontal": {
        "precentral":               (1024, 2024),
        "paracentral":              (1017, 2017),
        "caudalmiddlefrontal":      (1003, 2003),
    },
    # --- Middle frontal (DLPFC / inferior frontal) ---
    "middle_frontal": {
        "rostralmiddlefrontal":     (1027, 2027),
        "parsopercularis":          (1018, 2018),
        "parstriangularis":         (1020, 2020),
        "caudalanteriorcingulate":  (1002, 2002),
    },
    # --- Anterior frontal (frontopolar / orbitofrontal / vmPFC) ---
    "anterior_frontal": {
        "superiorfrontal":          (1028, 2028),
        "parsorbitalis":            (1019, 2019),
        "lateralorbitofrontal":     (1012, 2012),
        "medialorbitofrontal":      (1014, 2014),
        "frontalpole":              (1032, 2032),
        "rostralanteriorcingulate": (1026, 2026),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Labelling scheme 2 — Brodmann areas (approximate)
#
# The Desikan-Killiany atlas follows gyral/sulcal anatomy and does not align
# perfectly with cytoarchitectonic Brodmann borders.  The mapping below uses
# the standard anatomical approximation from the neuroimaging literature:
#
#   BA 6   precentral gyrus (spans BA 4 + BA 6) + paracentral (SMA)
#   BA 8   caudal middle frontal gyrus (FEF / dorsal premotor)
#   BA 9/46 rostral middle frontal gyrus (core DLPFC)
#   BA 10  frontal pole (frontopolar cortex)
# ─────────────────────────────────────────────────────────────────────────────

BRODMANN_LABELS = {
    # --- BA 10: frontopolar cortex + medial BA 10/32 + ventromedial BA 10/11 ---
    # DK 'superiorfrontal' covers the medial wall of SFG (BA 9 medial); added
    # to ba9_46 rather than here.  rostralanteriorcingulate (rostral ACC /
    # vmPFC) and medialorbitofrontal (gyrus rectus) fill the medial/orbital
    # BA 10 territory that frontalpole alone misses.
    "ba10": {
        "frontalpole":              (1032, 2032),
        "rostralanteriorcingulate": (1026, 2026),  # medial BA 10/32 (vmPFC)
        "medialorbitofrontal":      (1014, 2014),  # ventromedial BA 10/11
    },
    # --- BA 9/46: dorsolateral prefrontal cortex (DLPFC) + medial BA 9 ---
    # DK 'superiorfrontal' includes the medial wall of the superior frontal
    # gyrus, which corresponds to medial BA 9 / pre-SMA territory (pre-SMA
    # proper sits at the BA 6/8/9 border on the medial wall).
    "ba9_46": {
        "rostralmiddlefrontal":     (1027, 2027),
        "superiorfrontal":          (1028, 2028),  # medial BA 9 / SFG medial wall
    },
    # --- BA 8: frontal eye field / dorsal premotor ---
    "ba8": {
        "caudalmiddlefrontal":      (1003, 2003),
    },
    # --- BA 6: premotor cortex / SMA ---
    # DK 'precentral' spans BA 4 (primary motor) and posterior BA 6;
    # DK 'paracentral' spans the BA 4 leg area and the SMA (BA 6).
    "ba6": {
        "precentral":               (1024, 2024),
        "paracentral":              (1017, 2017),
    },
}

SCHEMES = {
    "frontal_strips": FRONTAL_LABELS,
    "brodmann":       BRODMANN_LABELS,
}


def build_label_index(labels_dict):
    """Return a flat_id → (group, label_name) mapping for a labels dict."""
    all_ids = {}
    for group, labels in labels_dict.items():
        for name, (lh, rh) in labels.items():
            all_ids[lh] = (group, f"lh_{name}")
            all_ids[rh] = (group, f"rh_{name}")
    return all_ids


def from_nifti(aparc_path, mask_path, labels_dict):
    """Build ROIs from a NIfTI parcellation and brain mask.

    The mask defines which voxels are stored in the HDF5 files.  The flat
    index of a voxel in the mask is its column index in the HDF5 data array.
    """
    try:
        import nibabel as nib
    except ImportError:
        print("  nibabel is required:  pip install nibabel")
        raise SystemExit(1)

    print(f"  Loading parcellation: {aparc_path}")
    aparc_img = nib.load(str(aparc_path))
    aparc_data = np.asarray(aparc_img.dataobj).astype(int)

    print(f"  Loading brain mask:   {mask_path}")
    mask_img = nib.load(str(mask_path))
    mask_data = np.asarray(mask_img.dataobj) > 0

    if aparc_data.shape != mask_data.shape:
        print(f"  WARNING: shape mismatch — aparc {aparc_data.shape} vs mask {mask_data.shape}")
        print(f"  Make sure both are in the same functional space.")
        raise SystemExit(1)

    n_mask_voxels = mask_data.sum()
    print(f"  Mask contains {n_mask_voxels} voxels")

    all_ids = build_label_index(labels_dict)

    # flat_index[i] = column index in HDF5 for the i-th True voxel in the mask
    mask_indices = np.where(mask_data.ravel())[0]
    aparc_flat = aparc_data.ravel()

    rois = {group: [] for group in labels_dict}
    label_counts = {}

    for flat_pos, mask_idx in enumerate(mask_indices):
        label_id = aparc_flat[mask_idx]
        if label_id in all_ids:
            group, label_name = all_ids[label_id]
            rois[group].append(int(flat_pos))
            label_counts[label_name] = label_counts.get(label_name, 0) + 1

    print(f"\n  Voxels per sub-label:")
    for name in sorted(label_counts):
        print(f"    {name:<35s}  {label_counts[name]:5d}")

    return rois


def from_coords(csv_path, labels_dict):
    """Build ROIs from a CSV with columns: flat_index, x, y, z, label.

    The 'label' column should contain FreeSurfer aparc label IDs (integers).
    If no label column exists, we use the y-coordinate to split frontal
    cortex into posterior/middle/anterior thirds (frontal_strips only).
    """
    import csv

    print(f"  Loading coordinates: {csv_path}")
    flat_indices = []
    labels = []
    y_coords = []

    all_ids = build_label_index(labels_dict)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        has_label = "label" in fields
        for row in reader:
            flat_indices.append(int(row["flat_index"]))
            y_coords.append(float(row["y"]))
            if has_label:
                labels.append(int(row["label"]))

    if has_label:
        rois = {group: [] for group in labels_dict}
        for idx, label_id in zip(flat_indices, labels):
            if label_id in all_ids:
                group, _ = all_ids[label_id]
                rois[group].append(idx)
        return rois
    else:
        if len(labels_dict) != 3 or list(labels_dict.keys()) != list(FRONTAL_LABELS.keys()):
            print("  WARNING: coordinate-based y-split is only meaningful for the "
                  "'frontal_strips' scheme.  Provide a CSV with a 'label' column "
                  "to use the 'brodmann' scheme.")
            raise SystemExit(1)

        # No labels — split by y-coordinate (anterior-posterior axis)
        y_arr = np.array(y_coords)
        frontal_mask = y_arr > np.median(y_arr)  # rough frontal selection
        frontal_y = y_arr[frontal_mask]
        frontal_idx = np.array(flat_indices)[frontal_mask]

        terciles = np.percentile(frontal_y, [33.3, 66.7])
        rois = {
            "posterior_frontal": frontal_idx[frontal_y <= terciles[0]].tolist(),
            "middle_frontal":   frontal_idx[(frontal_y > terciles[0]) & (frontal_y <= terciles[1])].tolist(),
            "anterior_frontal": frontal_idx[frontal_y > terciles[1]].tolist(),
        }
        print(f"  Split {len(frontal_idx)} frontal voxels by y-coordinate into terciles")
        return rois


def main():
    parser = argparse.ArgumentParser(
        description="Create frontal-subregion ROI JSON for attribution analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--aparc",  type=str, help="FreeSurfer aparc+aseg NIfTI in functional space")
    parser.add_argument("--mask",   type=str, help="Brain mask NIfTI (non-zero = included in HDF5)")
    parser.add_argument("--coords", type=str, help="CSV with flat_index,x,y,z[,label] columns")
    parser.add_argument("--out",    type=str, default="frontal_rois.json", help="Output JSON path")
    parser.add_argument(
        "--scheme",
        choices=list(SCHEMES.keys()),
        default="frontal_strips",
        help=(
            "Labelling scheme: "
            "'frontal_strips' = posterior/middle/anterior (default); "
            "'brodmann' = BA10 / BA9-46 / BA8 / BA6 (approximate)"
        ),
    )
    args = parser.parse_args()

    labels_dict = SCHEMES[args.scheme]

    print()
    print("=" * 55)
    print("  Create Frontal Subregion ROIs")
    print(f"  Scheme: {args.scheme}")
    print("=" * 55)

    if args.aparc and args.mask:
        rois = from_nifti(args.aparc, args.mask, labels_dict)
    elif args.coords:
        rois = from_coords(args.coords, labels_dict)
    else:
        parser.print_help()
        print("\n  Provide either --aparc + --mask, or --coords.\n")
        print(f"  Grouping used for scheme '{args.scheme}':\n")
        for group, labels in labels_dict.items():
            print(f"    {group}:")
            for name in labels:
                print(f"      {name}")
            print()
        return

    # Summary
    print(f"\n  ROI summary:")
    total = 0
    for group in labels_dict:
        n = len(rois[group])
        total += n
        print(f"    {group:<25s}  {n:5d} voxels")
    print(f"    {'total':<25s}  {total:5d} voxels")

    # Save
    out_path = Path(args.out)
    with open(str(out_path), "w") as f:
        json.dump(rois, f, indent=2)
    print(f"\n  Saved to {out_path}")
    print(f"  Use with:  python run_attribution.py ... --rois {out_path}\n")


if __name__ == "__main__":
    main()
