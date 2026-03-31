#!/usr/bin/env python3

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ET
import h5py

# ---------------------------------------
# CONFIG
# ---------------------------------------

SUBJECTS = ["UTS01", "UTS02", "UTS03"]
BASE = Path("pycortex-db")

ATLAS_DIR = Path("/ceph/behrens/svenja/language-decoding-expts/atlas_fsaverage/atlases/HarvardOxford")
ATLAS_PATH = ATLAS_DIR / "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz"
XML_PATH   = ATLAS_DIR / "HarvardOxford-Cortical.xml"

OUTDIR  = Path("ba_indices")
PLOTDIR = Path("ba_plots")

# ---------------------------------------
# LOAD ATLAS
# ---------------------------------------

print("Loading Harvard-Oxford atlas...")

atlas_img    = nib.load(str(ATLAS_PATH))
atlas_data   = atlas_img.get_fdata()
atlas_affine = atlas_img.affine

def load_labels(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return {int(l.attrib["index"]): l.text for l in root.findall(".//label")}

labels = load_labels(XML_PATH)

# ---------------------------------------
# DEFINE FRONTAL REGIONS
# ---------------------------------------

def build_frontal_map(labels):
    mapping = {
        "BA_10":  [],
        "BA_9_46": [],
        "BA_8":   [],
        "BA_6":   [],
        "BROCA":  []
    }

    for idx, name in labels.items():
        if name is None:
            continue

        # XML uses 0-based label indices; NIfTI atlas stores 1-based values.
        nifti_val = idx + 1
        name = name.lower()

        if "frontal pole" in name or "frontal medial cortex" in name:
            mapping["BA_10"].append(nifti_val)

        if "middle frontal gyrus" in name:
            mapping["BA_9_46"].append(nifti_val)

        if "superior frontal gyrus" in name:
            mapping["BA_8"].append(nifti_val)

        if "precentral gyrus" in name:
            mapping["BA_6"].append(nifti_val)

        if "inferior frontal gyrus, pars opercularis" in name:
            mapping["BROCA"].append(nifti_val)

        if "inferior frontal gyrus, pars triangularis" in name:
            mapping["BROCA"].append(nifti_val)

    return mapping

TARGET = build_frontal_map(labels)

print("Frontal mapping:")
for k, v in TARGET.items():
    print(k, v)

# ---------------------------------------
# BUILD KD-TREE ON ATLAS
# ---------------------------------------

atlas_vox_coords  = np.array(np.nonzero(atlas_data)).T
atlas_world_coords = nib.affines.apply_affine(atlas_affine, atlas_vox_coords)
atlas_tree = cKDTree(atlas_world_coords)

def voxel_to_atlas_label(mni_xyz):
    """Return the HarvardOxford atlas label (NIfTI value) nearest to mni_xyz."""
    _, idx = atlas_tree.query(mni_xyz)
    voxel = atlas_vox_coords[idx]
    return int(atlas_data[tuple(voxel)])

# ---------------------------------------
# SUBJECT HELPERS
# ---------------------------------------

def get_xfm_dir(subject):
    return BASE / subject / "transforms" / f"{subject}_auto"

def load_coord_and_ref(subject):
    """Load the pycortex coord transform and reference volume affine.

    The 'coord' matrix maps surface mm coordinates to EPI voxel indices:
        epi_ijk = coord @ [sx, sy, sz, 1]^T

    ref_affine maps EPI voxel indices to world (MNI) mm:
        mni_xyz = ref_affine @ [i, j, k, 1]^T
    """
    xfm_dir = get_xfm_dir(subject)
    with open(xfm_dir / "matrices.xfm") as f:
        coord = np.array(json.load(f)["coord"])
    ref_affine = nib.load(str(xfm_dir / "reference.nii.gz")).affine
    return coord, ref_affine

def load_mask(subject):
    """Load mask_thick.nii.gz — defines exactly which voxels are HDF5 columns."""
    mask_path = get_xfm_dir(subject) / "mask_thick.nii.gz"
    return (nib.load(str(mask_path)).get_fdata() > 0)

def load_ref_shape(subject):
    xfm_dir = get_xfm_dir(subject)
    return nib.load(str(xfm_dir / "reference.nii.gz")).shape

# ---------------------------------------
# BUILD ROIS  (works entirely in EPI voxel space)
# ---------------------------------------

def build_rois(subject):
    """For each brain-masked EPI voxel, look up its atlas label and assign
    it to a frontal ROI.  Stores HDF5 flat column indices in the output dict.

    Index convention:
      flat_hdf5_idx  = position of the voxel among the nonzero entries of
                       mask_thick, in C (row-major) ravel order.
      This matches exactly the column ordering of the HDF5 response arrays.
    """
    print(f"\nProcessing {subject}")

    _, ref_affine = load_coord_and_ref(subject)
    mask_data     = load_mask(subject)

    # Get [i,j,k] for every brain-masked voxel, in ravel order
    mask_ijk = np.array(np.where(mask_data)).T   # (n_vox, 3)
    n_vox    = len(mask_ijk)

    # Convert all to MNI world coordinates at once
    mni_coords = nib.affines.apply_affine(ref_affine, mask_ijk)  # (n_vox, 3)

    rois = {k: [] for k in TARGET}

    for flat_idx in range(n_vox):
        label = voxel_to_atlas_label(mni_coords[flat_idx])

        for k, ids in TARGET.items():
            if label in ids:
                rois[k].append(flat_idx)

        if flat_idx % 10000 == 0:
            print(f"  {flat_idx}/{n_vox}")

    # combined full-frontal ROI
    full = set()
    for v in rois.values():
        full.update(v)
    rois["BA_full_frontal"] = sorted(full)

    return rois


def make_rois_exclusive(rois):
    """Remove overlapping assignments, giving priority to more anterior regions."""
    priority = ["BA_10", "BA_9_46", "BA_8", "BA_6", "BROCA"]

    assigned  = set()
    new_rois  = {}

    for k in priority:
        idx = set(map(int, rois.get(k, [])))
        idx = idx - assigned
        new_rois[k] = sorted(idx)
        assigned.update(idx)

    # recompute full frontal from exclusive subsets
    full = set()
    for v in new_rois.values():
        full.update(v)
    new_rois["BA_full_frontal"] = sorted(full)

    return new_rois

# ---------------------------------------
# SAVE
# ---------------------------------------

COLORS = {
    "BA_10":   "red",
    "BA_9_46": "blue",
    "BA_8":    "green",
    "BA_6":    "orange",
    "BROCA":   "purple"
}

def save_json(subject, rois):
    out = OUTDIR / subject
    out.mkdir(parents=True, exist_ok=True)
    for k, v in rois.items():
        json.dump({k: v}, open(out / f"{k}.json", "w"), indent=2)

# ---------------------------------------
# SURFACE HELPERS  (used only for plotting)
# ---------------------------------------

def load_npz_pts(path):
    return np.load(path)["pts"]

def load_surface_coords(subject, kind):
    """Load inflated or flat surface vertex coordinates.
    kind: 'inflated' or 'flat'
    Returns (lh_pts, rh_pts).
    """
    surf = BASE / subject / "surfaces"
    if (surf / f"{kind}_lh.gii").exists():
        lh = nib.load(str(surf / f"{kind}_lh.gii")).darrays[0].data
        rh = nib.load(str(surf / f"{kind}_rh.gii")).darrays[0].data
    else:
        lh = load_npz_pts(surf / f"{kind}_lh.npz")
        rh = load_npz_pts(surf / f"{kind}_rh.npz")
    return lh, rh

def load_json_rois(subject):
    rois = {}
    for f in (OUTDIR / subject).glob("*.json"):
        with open(f) as fh:
            rois[f.stem] = list(json.load(fh).values())[0]
    return rois


def build_vertex_to_hdf5(subject):
    """For each surface vertex compute its HDF5 flat column index (-1 if outside mask).

    Uses the pycortex coord transform:
        epi_ijk = coord @ [pia_x, pia_y, pia_z, 1]^T   (float, then rounded)

    The HDF5 flat index = position of that voxel among the nonzero entries
    of mask_thick in C-ravel order.

    Returns
    -------
    vertex_to_hdf5 : (n_lh + n_rh,) int32 array
    n_lh           : number of LH vertices
    """
    surf = BASE / subject / "surfaces"

    # Load pia surface (actual cortical position, used by coord transform)
    if (surf / "pia_lh.gii").exists():
        lh_pia = nib.load(str(surf / "pia_lh.gii")).darrays[0].data
        rh_pia = nib.load(str(surf / "pia_rh.gii")).darrays[0].data
    else:
        lh_pia = load_npz_pts(surf / "pia_lh.npz")
        rh_pia = load_npz_pts(surf / "pia_rh.npz")

    n_lh      = len(lh_pia)
    all_pia   = np.vstack([lh_pia, rh_pia])
    n_total   = len(all_pia)

    coord_mat, _ = load_coord_and_ref(subject)
    mask_data    = load_mask(subject)
    ref_shape    = load_ref_shape(subject)

    # Map each pia vertex → EPI voxel [i,j,k]
    h         = np.column_stack([all_pia, np.ones(n_total)])
    epi_float = (coord_mat @ h.T).T[:, :3]
    epi_ijk   = np.round(epi_float).astype(int)

    # Build lookup: 3D-flat index → HDF5 flat index
    mask_ijk       = np.array(np.where(mask_data)).T
    s1, s2         = ref_shape[1], ref_shape[2]
    mask_3d_flat   = mask_ijk[:, 0] * s1 * s2 + mask_ijk[:, 1] * s2 + mask_ijk[:, 2]
    total_vox      = ref_shape[0] * s1 * s2
    mask_3d_to_hdf5 = np.full(total_vox, -1, dtype=np.int32)
    mask_3d_to_hdf5[mask_3d_flat] = np.arange(len(mask_ijk), dtype=np.int32)

    # Compute HDF5 index for every vertex
    in_bounds = (
        (epi_ijk[:, 0] >= 0) & (epi_ijk[:, 0] < ref_shape[0]) &
        (epi_ijk[:, 1] >= 0) & (epi_ijk[:, 1] < ref_shape[1]) &
        (epi_ijk[:, 2] >= 0) & (epi_ijk[:, 2] < ref_shape[2])
    )

    vertex_to_hdf5 = np.full(n_total, -1, dtype=np.int32)
    vi             = np.where(in_bounds)[0]
    flat_3d        = epi_ijk[vi, 0] * s1 * s2 + epi_ijk[vi, 1] * s2 + epi_ijk[vi, 2]
    vertex_to_hdf5[vi] = mask_3d_to_hdf5[flat_3d]

    return vertex_to_hdf5, n_lh


# ---------------------------------------
# PLOT  (inflated lateral + flatmap)
# ---------------------------------------

def plot(subject):
    PLOTDIR.mkdir(exist_ok=True)

    rois              = load_json_rois(subject)
    lh_inf, rh_inf    = load_surface_coords(subject, "inflated")
    lh_flat, rh_flat  = load_surface_coords(subject, "flat")

    # Map HDF5 voxel indices → surface vertices
    vertex_to_hdf5, n_lh = build_vertex_to_hdf5(subject)

    # Frontal mask on inflated surface (anterior vertices only)
    lh_front = lh_inf[:, 1] > -20
    rh_front = rh_inf[:, 1] > -20

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    def scatter_roi_inflated(ax, inf_pts, front_mask, is_lh, k, c):
        if k not in rois or len(rois[k]) == 0:
            return
        roi_set  = np.array(rois[k], dtype=np.int32)
        vt       = vertex_to_hdf5[:n_lh] if is_lh else vertex_to_hdf5[n_lh:]
        vert_idx = np.where(np.isin(vt, roi_set) & front_mask)[0]
        if len(vert_idx) == 0:
            return
        ax.scatter(inf_pts[vert_idx, 1], inf_pts[vert_idx, 2], s=3, color=c, label=k)

    def scatter_roi_flat(ax, flat_pts, is_lh, k, c):
        if k not in rois or len(rois[k]) == 0:
            return
        roi_set  = np.array(rois[k], dtype=np.int32)
        vt       = vertex_to_hdf5[:n_lh] if is_lh else vertex_to_hdf5[n_lh:]
        vert_idx = np.where(np.isin(vt, roi_set))[0]
        if len(vert_idx) == 0:
            return
        ax.scatter(flat_pts[vert_idx, 0], flat_pts[vert_idx, 1], s=3, color=c, label=k)

    # LH lateral
    ax = axes[0, 0]
    ax.scatter(lh_inf[lh_front, 1], lh_inf[lh_front, 2], s=1, alpha=0.05, color="gray")
    for k, c in COLORS.items():
        scatter_roi_inflated(ax, lh_inf, lh_front, True, k, c)
    ax.set_title("LH lateral")
    ax.axis("equal")
    ax.axis("off")

    # RH lateral
    ax = axes[0, 1]
    ax.scatter(rh_inf[rh_front, 1], rh_inf[rh_front, 2], s=1, alpha=0.05, color="gray")
    for k, c in COLORS.items():
        scatter_roi_inflated(ax, rh_inf, rh_front, False, k, c)
    ax.set_title("RH lateral")
    ax.axis("equal")
    ax.axis("off")

    # LH flatmap
    ax = axes[1, 0]
    ax.scatter(lh_flat[:, 0], lh_flat[:, 1], s=0.5, alpha=0.05, color="gray")
    for k, c in COLORS.items():
        scatter_roi_flat(ax, lh_flat, True, k, c)
    ax.set_title("LH flatmap")
    ax.axis("equal")
    ax.axis("off")

    # RH flatmap
    ax = axes[1, 1]
    ax.scatter(rh_flat[:, 0], rh_flat[:, 1], s=0.5, alpha=0.05, color="gray")
    for k, c in COLORS.items():
        scatter_roi_flat(ax, rh_flat, False, k, c)
    ax.set_title("RH flatmap")
    ax.axis("equal")
    ax.axis("off")

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=c, label=k, markersize=6)
               for k, c in COLORS.items()]
    fig.legend(handles=handles, loc="upper right")
    fig.suptitle(subject)

    path = PLOTDIR / f"{subject}_views.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", path)

# ---------------------------------------
# VALIDATE INDICES AGAINST HDF5
# ---------------------------------------

DATA_TEST_DIR = Path("data_test/test_response")

def validate_indices(subject):
    print(f"\n--- Index validation: {subject} ---")

    subj_dir   = DATA_TEST_DIR / subject / "perceived_speech"
    hdf5_files = sorted(subj_dir.glob("*.hf5"))
    if not hdf5_files:
        print(f"  WARNING: no .hf5 files found in {subj_dir}")
        return

    sample_file = hdf5_files[0]
    with h5py.File(sample_file, "r") as f:
        n_voxels = f["data"].shape[1]
    print(f"  Sample HDF5 : {sample_file.name}  ->  {n_voxels} voxel columns")

    rois   = load_json_rois(subject)
    all_ok = True

    for k in sorted(rois):
        idx      = np.array(rois[k], dtype=int)
        n_total  = len(idx)
        if n_total == 0:
            print(f"  {k:<20s}  EMPTY")
            continue
        n_valid   = int((idx < n_voxels).sum())
        n_invalid = n_total - n_valid
        status    = "OK" if n_invalid == 0 else "OUT-OF-RANGE"
        if n_invalid > 0:
            all_ok = False
        print(f"  {k:<20s}  n={n_total:6d}  in-range={n_valid:6d}  out-of-range={n_invalid:6d}  [{status}]")

    if all_ok:
        print(f"  PASS: all indices are valid HDF5 column indices.")
    else:
        print(f"  FAIL: {n_voxels} voxels in HDF5 but some JSON indices exceed this.")

# ---------------------------------------
# MAIN
# ---------------------------------------

def main():
    for s in SUBJECTS:
        rois = build_rois(s)
        rois = make_rois_exclusive(rois)
        save_json(s, rois)
        plot(s)
        validate_indices(s)

    print("\nDone.")

if __name__ == "__main__":
    main()
