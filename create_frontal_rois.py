#!/usr/bin/env python3
"""
Create frontal-subregion ROI JSON files for the attribution analysis.

This script maps FreeSurfer Desikan-Killiany atlas labels to the flat voxel
indices used in the HDF5 response files, then groups frontal labels into
posterior / middle / anterior strips.

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
  # From NIfTI parcellation + mask
  python create_frontal_rois.py \\
      --aparc aparc_in_func.nii.gz \\
      --mask  brain_mask.nii.gz \\
      --out   frontal_rois_S1.json

  # From a CSV of voxel coordinates (flat_index, x, y, z, label)
  python create_frontal_rois.py \\
      --coords voxel_info.csv \\
      --out frontal_rois_S1.json

  # Then use with the attribution script
  python run_attribution.py --subject S1 --experiment perceived_speech \\
      --task wheretheressmoke --use-saved --rois frontal_rois_S1.json
"""

import json
import argparse
import numpy as np
from pathlib import Path

# -------------------------------------------------------------------------
# FreeSurfer Desikan-Killiany label IDs for frontal regions
# (from FreeSurferColorLUT.txt — left hemisphere IDs; right = left + 1000)
# -------------------------------------------------------------------------

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

ALL_FRONTAL_IDS = {}
for group, labels in FRONTAL_LABELS.items():
    for name, (lh, rh) in labels.items():
        ALL_FRONTAL_IDS[lh] = (group, f"lh_{name}")
        ALL_FRONTAL_IDS[rh] = (group, f"rh_{name}")


def from_nifti(aparc_path, mask_path):
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

    # flat_index[i] = column index in HDF5 for the i-th True voxel in the mask
    mask_indices = np.where(mask_data.ravel())[0]
    aparc_flat = aparc_data.ravel()

    rois = {group: [] for group in FRONTAL_LABELS}
    label_counts = {}

    for flat_pos, mask_idx in enumerate(mask_indices):
        label_id = aparc_flat[mask_idx]
        if label_id in ALL_FRONTAL_IDS:
            group, label_name = ALL_FRONTAL_IDS[label_id]
            rois[group].append(int(flat_pos))
            label_counts[label_name] = label_counts.get(label_name, 0) + 1

    print(f"\n  Voxels per sub-label:")
    for name in sorted(label_counts):
        print(f"    {name:<35s}  {label_counts[name]:5d}")

    return rois


def from_coords(csv_path):
    """Build ROIs from a CSV with columns: flat_index, x, y, z, label.

    The 'label' column should contain FreeSurfer aparc label IDs (integers).
    If no label column exists, we use the y-coordinate to split frontal
    cortex into posterior/middle/anterior thirds.
    """
    import csv

    print(f"  Loading coordinates: {csv_path}")
    flat_indices = []
    labels = []
    y_coords = []

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
        rois = {group: [] for group in FRONTAL_LABELS}
        for idx, label_id in zip(flat_indices, labels):
            if label_id in ALL_FRONTAL_IDS:
                group, _ = ALL_FRONTAL_IDS[label_id]
                rois[group].append(idx)
        return rois
    else:
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
    parser.add_argument("--aparc", type=str, help="FreeSurfer aparc+aseg NIfTI in functional space")
    parser.add_argument("--mask", type=str, help="Brain mask NIfTI (non-zero = included in HDF5)")
    parser.add_argument("--coords", type=str, help="CSV with flat_index,x,y,z[,label] columns")
    parser.add_argument("--out", type=str, default="frontal_rois.json", help="Output JSON path")
    args = parser.parse_args()

    print()
    print("=" * 55)
    print("  Create Frontal Subregion ROIs")
    print("=" * 55)

    if args.aparc and args.mask:
        rois = from_nifti(args.aparc, args.mask)
    elif args.coords:
        rois = from_coords(args.coords)
    else:
        parser.print_help()
        print("\n  Provide either --aparc + --mask, or --coords.\n")
        print("  Grouping used (Desikan-Killiany atlas → 3 strips):\n")
        for group, labels in FRONTAL_LABELS.items():
            print(f"    {group}:")
            for name in labels:
                print(f"      {name}")
            print()
        return

    # Summary
    print(f"\n  ROI summary:")
    total = 0
    for group in ["posterior_frontal", "middle_frontal", "anterior_frontal"]:
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
