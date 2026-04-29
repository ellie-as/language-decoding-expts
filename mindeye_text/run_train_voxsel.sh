#!/usr/bin/env bash
# End-to-end "language-selective top-K" training:
#
#   1. Compute per-voxel encoding-r ranking on training stories only
#      (skipped if the .npz file already exists for every subject).
#   2. Train the MindEye-text decoder, using only the top-K voxels per
#      subject as input. By default brain-PCA is OFF (set BRAIN_PCA>0 to
#      stack PCA on top of voxel selection).
#
# Override anything via env vars; pass extra train-script flags after
# the script name. Example:
#
#   TOPK=1000 LATENT=100 BLOCKS=2 DROPOUT=0.5 WD=0.1 \
#   bash mindeye_text/run_train_voxsel.sh \
#     --loss mse_cosine --cosine-weight 0.8 \
#     --lr 5e-4 --max-epochs 200 --patience 5
#
# This replaces a "BRAIN_PCA=1000" run with a "TOPK=1000 (no PCA)" run.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

# ---------------------------------------------------------------------------
# Selection params (encoding-rank step)
# ---------------------------------------------------------------------------
SUBJECTS=${SUBJECTS:-"S1 S2 S3"}
ROI=${ROI:-"full_frontal"}
FEATURE_MODEL=${FEATURE_MODEL:-"gtr-base"}
ENC_LAGS=${ENC_LAGS:-"1 2 3 4"}
ENC_FOLDS=${ENC_FOLDS:-5}
ENC_ALPHAS=${ENC_ALPHAS:-"1 10 100 1000 10000 100000"}
TOPK=${TOPK:-1000}
ENC_OUT=${ENC_OUT:-"mindeye_text/cache/encoding_corrs"}

# ---------------------------------------------------------------------------
# Decoder training defaults (selection-only mode: no input PCA)
# ---------------------------------------------------------------------------
BRAIN_PCA=${BRAIN_PCA:-0}
LATENT=${LATENT:-100}
BLOCKS=${BLOCKS:-2}
DROPOUT=${DROPOUT:-0.5}
WD=${WD:-0.1}

# ---------------------------------------------------------------------------
# Step 1: ensure encoding-r .npz exists for every subject
# ---------------------------------------------------------------------------
LAGS_TAG=$(echo "${ENC_LAGS}" | tr ' ' '-')
mkdir -p "${ENC_OUT}"

need_rank=0
for S in ${SUBJECTS}; do
  CORR_FILE="${ENC_OUT}/${S}__${FEATURE_MODEL}__${ROI}__lags${LAGS_TAG}.npz"
  if [ ! -f "${CORR_FILE}" ]; then
    need_rank=1
    break
  fi
done

if [ "${need_rank}" -eq 1 ]; then
  echo ">>> Computing encoding-r ranking (one-time per subject/ROI/lags)..."
  python -m mindeye_text.encoding_rank \
    --subjects ${SUBJECTS} \
    --roi "${ROI}" \
    --feature-model "${FEATURE_MODEL}" \
    --lags ${ENC_LAGS} \
    --n-folds "${ENC_FOLDS}" \
    --alphas ${ENC_ALPHAS} \
    --output-dir "${ENC_OUT}"
else
  echo ">>> Encoding-r ranking already cached for all subjects in ${ENC_OUT}"
fi

# ---------------------------------------------------------------------------
# Step 2: launch decoder training with top-K voxel selection
# ---------------------------------------------------------------------------
echo ">>> Training decoder on top-${TOPK} encoding-r voxels per subject"
echo ">>>   BRAIN_PCA=${BRAIN_PCA} LATENT=${LATENT} BLOCKS=${BLOCKS}"
echo ">>>   DROPOUT=${DROPOUT} WD=${WD}"

VOXEL_SELECT_CORRS="${ENC_OUT}/{subject}__${FEATURE_MODEL}__${ROI}__lags${LAGS_TAG}.npz" \
VOXEL_SELECT_TOPK="${TOPK}" \
BRAIN_PCA="${BRAIN_PCA}" \
LATENT="${LATENT}" \
BLOCKS="${BLOCKS}" \
DROPOUT="${DROPOUT}" \
WD="${WD}" \
SUBJECTS="${SUBJECTS}" \
ROI="${ROI}" \
FEATURE_MODEL="${FEATURE_MODEL}" \
  bash mindeye_text/run_train.sh "$@"
