#!/usr/bin/env bash
# Train a 2D conv denoising autoencoder on per-TR pycortex flatmap images and
# compare against PCA at the same latent dims.
#
# Outputs (per subject):
#   ${OUT_ROOT}/<tag>/conv2d_autoencoder_vs_pca.csv
#   ${OUT_ROOT}/<tag>/sample_pngs/latent<dim>/{ae,pca}_sample_*.png
#   ${OUT_ROOT}/<tag>/checkpoints/conv2d_ae_latent<dim>.pt   (if --save-checkpoints)
#   ${OUT_ROOT}/<tag>/{config.json,zscore_stats.npz,pixel_mask.npy}
#
# A persistent cache of rendered flatmap images is written to
#   ${OUT_ROOT}/flatmap_image_cache/<subject>__<xfm>__<mask>__h<H>__{train,val}_images.npy
# so subsequent runs (different latent dims, hyperparams) skip re-rendering.
#
# Example:
#   cd /ceph/behrens/ellie/language-decoding-expts
#   bash 2D_representation_expt/run_flatmap_image_autoencoder_cluster.sh
#
# Useful overrides:
#   SUBJECTS="S1 S2" LATENT_DIMS="64 128 256" bash ...
#   IMAGE_HEIGHT=256 bash ...

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/ceph/behrens/ellie/language-decoding-expts}"
SUBJECTS="${SUBJECTS:-S1}"
LATENT_DIMS="${LATENT_DIMS:-64 128 256}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-192}"
PAD_TO_MULTIPLE="${PAD_TO_MULTIPLE:-8}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BASE_CHANNELS="${BASE_CHANNELS:-8}"
DROPOUT="${DROPOUT:-0.10}"
INPUT_NOISE_STD="${INPUT_NOISE_STD:-0.05}"
INPUT_MASK_PROB="${INPUT_MASK_PROB:-0.0}"
LR="${LR:-8e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
PATIENCE="${PATIENCE:-6}"
DEVICE="${DEVICE:-auto}"
SAMPLE_PNG_COUNT="${SAMPLE_PNG_COUNT:-8}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"
SEED="${SEED:-0}"

OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/2D_representation_expt/results/flatmap_image_autoencoder}"
PYCORTEX_FILESTORE="${PYCORTEX_FILESTORE:-${DATA_ROOT}/pycortex-db}"
SPLIT_RESULTS_DIR="${SPLIT_RESULTS_DIR:-${DATA_ROOT}/lag_preference_analysis/results/S1__embedding-summary-combo-h20-50-200__lags1-10__chunk1tr__seed0}"
IMAGE_CACHE_DIR="${IMAGE_CACHE_DIR:-${OUT_ROOT}/flatmap_image_cache}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

mkdir -p "${OUT_ROOT}" "${IMAGE_CACHE_DIR}"
cd "${DATA_ROOT}"

echo "DATA_ROOT=${DATA_ROOT}"
echo "SUBJECTS=${SUBJECTS}"
echo "LATENT_DIMS=${LATENT_DIMS}"
echo "IMAGE_HEIGHT=${IMAGE_HEIGHT}"
echo "DEVICE=${DEVICE}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "PYCORTEX_FILESTORE=${PYCORTEX_FILESTORE}"
echo "SPLIT_RESULTS_DIR=${SPLIT_RESULTS_DIR}"
echo "IMAGE_CACHE_DIR=${IMAGE_CACHE_DIR}"

CKPT_FLAG=""
if [[ "${SAVE_CHECKPOINTS}" == "1" || "${SAVE_CHECKPOINTS}" == "true" ]]; then
  CKPT_FLAG="--save-checkpoints"
fi

for SUB in ${SUBJECTS}; do
  TAG="${SUB}__flatmap_conv2d_ae_vs_pca__h${IMAGE_HEIGHT}__seed${SEED}"
  echo
  echo "=== ${SUB} -> ${OUT_ROOT}/${TAG} ==="

  python -u 2D_representation_expt/train_flatmap_image_autoencoder.py \
    --subject "${SUB}" \
    --data-root "${DATA_ROOT}" \
    --split-results-dir "${SPLIT_RESULTS_DIR}" \
    --pycortex-filestore "${PYCORTEX_FILESTORE}" \
    --image-cache-dir "${IMAGE_CACHE_DIR}" \
    --image-height "${IMAGE_HEIGHT}" \
    --pad-to-multiple "${PAD_TO_MULTIPLE}" \
    --latent-dims ${LATENT_DIMS} \
    --base-channels "${BASE_CHANNELS}" \
    --dropout "${DROPOUT}" \
    --input-noise-std "${INPUT_NOISE_STD}" \
    --input-mask-prob "${INPUT_MASK_PROB}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --device "${DEVICE}" \
    --sample-png-count "${SAMPLE_PNG_COUNT}" \
    --seed "${SEED}" \
    --output-dir "${OUT_ROOT}" \
    --tag "${TAG}" \
    ${CKPT_FLAG}
done
