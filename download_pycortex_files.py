#!/usr/bin/env python3
"""
Download minimal pycortex-db files needed for vertex mapping (all subjects).
"""

import urllib.request
from pathlib import Path

S3_BASE = "https://s3.amazonaws.com/openneuro.org"
DATASET = "ds003020"
SUBJECTS = ["UTS01", "UTS02", "UTS03"]

# Subject-specific files
FILES_BY_SUBJECT = {
    "UTS01": [
        "cache/flatpixel_identity_1024_nearest_l32.npz",
        "cache/flatverts_1024.npz",
        "cache/flatmask_1024.npz",
        "surfaces/pia_lh.gii",
        "surfaces/pia_rh.gii",
        "surfaces/flat_lh.gii",
        "surfaces/flat_rh.gii",
        "surfaces/inflated_lh.gii",                 
        "surfaces/inflated_rh.gii", 
        "transforms/UTS01_auto/matrices.xfm",
        "transforms/UTS01_auto/mask_cortical.nii.gz",
        "transforms/UTS01_auto/mask_thick.nii.gz",
        "transforms/UTS01_auto/reference.nii.gz"
    ],
    "UTS02": [
        "cache/identity_pointnn.npz",          
        "cache/flatverts_1024.npz",
        "cache/flatmask_1024.npz",
        "surfaces/pia_lh.npz",                 # different format!
        "surfaces/pia_rh.npz",
        "surfaces/flat_lh.npz",
        "surfaces/flat_rh.npz",
        "surfaces/inflated_lh.npz",
        "surfaces/inflated_rh.npz",
        "transforms/UTS02_auto/matrices.xfm",
        "transforms/UTS02_auto/mask_cortical.nii.gz",
        "transforms/UTS02_auto/mask_thick.nii.gz",
        "transforms/UTS02_auto/reference.nii.gz"  # needed for surface→MNI transform
    ],
    "UTS03": [
        "cache/flatverts_1024.npz",           
        "cache/flatmask_1024.npz",
        "surfaces/pia_lh.gii",
        "surfaces/pia_rh.gii",
        "surfaces/flat_lh.gii",
        "surfaces/flat_rh.gii",
        "surfaces/inflated_lh.gii",                 
        "surfaces/inflated_rh.gii",                 
        "transforms/UTS03_auto/matrices.xfm",
        "transforms/UTS03_auto/mask_cortical.nii.gz",
        "transforms/UTS03_auto/mask_thick.nii.gz",
        "transforms/UTS03_auto/reference.nii.gz"
    ],
}

def download_file(s3_key, dest):
    url = f"{S3_BASE}/{s3_key}"
    print(f"  Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, str(dest))

for subject in SUBJECTS:
    print(f"\n--- {subject} ---")
    files = FILES_BY_SUBJECT[subject]

    for rel in files:
        s3_key = f"{DATASET}/derivatives/pycortex-db/{subject}/{rel}"
        dest = Path("pycortex-db") / subject / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            print(f"  [exists] {rel}")
            continue

        try:
            download_file(s3_key, dest)
        except Exception as e:
            print(f"  [missing] {rel}")

print("\nAll done.")