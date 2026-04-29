#!/usr/bin/env bash
# Launch a multi-subject MindEye-style text decoding run on a single A100.
#
# Assumes you are running from the repo root on a node that has the project
# data laid out under <repo>/data_train (i.e. NOT in --local-compute-mode). If
# the cluster mount lives elsewhere, point CONFIG.DATA_TRAIN_DIR via a symlink
# or run from the mounted root.
#
# Override hyperparameters via env vars or extra positional args, e.g.
#     LOSS=mse_cosine BATCH=384 ./mindeye_text/run_train.sh
#     ./mindeye_text/run_train.sh --normalize-targets --max-epochs 300

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

SUBJECTS=${SUBJECTS:-"S1 S2 S3"}
ROI=${ROI:-"full_frontal"}
FEATURE_MODEL=${FEATURE_MODEL:-"gtr-base"}
LATENT=${LATENT:-4096}
BLOCKS=${BLOCKS:-4}
DROPOUT=${DROPOUT:-0.15}
BRAIN_PCA=${BRAIN_PCA:-0}
LOSS=${LOSS:-"mse_clip"}
CLIP_WEIGHT=${CLIP_WEIGHT:-0.5}
CLIP_TEMP=${CLIP_TEMP:-0.05}
LR=${LR:-3e-4}
WD=${WD:-1e-2}
BATCH=${BATCH:-256}
EPOCHS=${EPOCHS:-200}
PATIENCE=${PATIENCE:-30}
LAG=${LAG:-3}
CHUNK=${CHUNK:-5}
BRAIN_OFFSET=${BRAIN_OFFSET:-0}
SEED=${SEED:-0}
OUT=${OUT:-"mindeye_text/results"}

mkdir -p mindeye_text/logs

python -u mindeye_text/train_mindeye_text.py \
  --subjects ${SUBJECTS} \
  --roi "${ROI}" \
  --feature-model "${FEATURE_MODEL}" \
  --latent-dim "${LATENT}" \
  --n-blocks "${BLOCKS}" \
  --dropout "${DROPOUT}" \
  --brain-pca "${BRAIN_PCA}" \
  --loss "${LOSS}" \
  --clip-weight "${CLIP_WEIGHT}" \
  --clip-temp "${CLIP_TEMP}" \
  --lr "${LR}" \
  --weight-decay "${WD}" \
  --batch-size "${BATCH}" \
  --max-epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --lag-trs "${LAG}" \
  --chunk-trs "${CHUNK}" \
  --brain-offset "${BRAIN_OFFSET}" \
  --torch-device cuda \
  --output-dir "${OUT}" \
  --seed "${SEED}" \
  "$@"
