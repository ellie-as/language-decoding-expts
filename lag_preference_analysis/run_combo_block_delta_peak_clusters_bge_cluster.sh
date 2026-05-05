#!/usr/bin/env bash
# Per-voxel block-ablation delta_r maps + clusters + ROI counts for the
# combo-bge ridge model.
#
# Outputs (per subject):
#   ${OUT_ROOT}/${SUBJECT}_bge/voxel_block_deltas.csv
#   ${OUT_ROOT}/${SUBJECT}_bge/block_high_cluster_summary.csv
#   ${OUT_ROOT}/${SUBJECT}_bge/roi_block_cluster_counts.csv
#   ${OUT_ROOT}/${SUBJECT}_bge/block_delta_peak_cluster_maps.npz
#   ${OUT_ROOT}/${SUBJECT}_bge/{full_model_r,delta_drop_*,best_block_*}.png
#
# Example:
#   cd /ceph/behrens/ellie/language-decoding-expts
#   bash lag_preference_analysis/run_combo_block_delta_peak_clusters_bge_cluster.sh
#
# Useful overrides:
#   SUBJECTS=S1 bash ...
#   DELTA_QUANTILE=0.85 bash ...

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/ceph/behrens/ellie/language-decoding-expts}"
SUBJECTS="${SUBJECTS:-S1}"
LAG="${LAG:-2}"
SUMMARY_HORIZONS="${SUMMARY_HORIZONS:-20 50 200 500}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-base-en-v1.5}"
CONTEXT_FEATURE_SOURCE="${CONTEXT_FEATURE_SOURCE:-summary}"
RIDGE_ALPHAS="${RIDGE_ALPHAS:-1000 10000 100000 300000 1000000 3000000 10000000}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-auto}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"
VOXEL_CHUNK_SIZE="${VOXEL_CHUNK_SIZE:-1000}"
RELIABLE_R_THRESHOLD="${RELIABLE_R_THRESHOLD:-0.05}"
DELTA_QUANTILE="${DELTA_QUANTILE:-0.90}"
DELTA_MIN="${DELTA_MIN:-0.0}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-3}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/lag_preference_analysis/results/combo_block_delta_peak_clusters}"

EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-${DATA_ROOT}/lag_preference_analysis/cache}"
ONE_TR_CACHE_DIR="${ONE_TR_CACHE_DIR:-${DATA_ROOT}/27-04-expts/cache}"
PYCORTEX_FILESTORE="${PYCORTEX_FILESTORE:-${DATA_ROOT}/pycortex-db}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

mkdir -p "${OUT_ROOT}"

cd "${DATA_ROOT}"

echo "DATA_ROOT=${DATA_ROOT}"
echo "SUBJECTS=${SUBJECTS}"
echo "LAG=${LAG}"
echo "EMBEDDING_MODEL=${EMBEDDING_MODEL}"
echo "CONTEXT_FEATURE_SOURCE=${CONTEXT_FEATURE_SOURCE}"
echo "RIDGE_ALPHAS=${RIDGE_ALPHAS}"
echo "DELTA_QUANTILE=${DELTA_QUANTILE}"
echo "OUT_ROOT=${OUT_ROOT}"

for SUB in ${SUBJECTS}; do
  if [[ "${CONTEXT_FEATURE_SOURCE}" == "summary" ]]; then
    OUT_DIR="${OUT_ROOT}/${SUB}_bge"
  else
    OUT_DIR="${OUT_ROOT}/${SUB}_bge_${CONTEXT_FEATURE_SOURCE}"
  fi
  echo
  echo "=== ${SUB} -> ${OUT_DIR} ==="

  python -u lag_preference_analysis/analyze_combo_block_delta_peak_clusters.py \
    --subject "${SUB}" \
    --data-root "${DATA_ROOT}" \
    --lag "${LAG}" \
    --summary-horizons ${SUMMARY_HORIZONS} \
    --embedding-model "${EMBEDDING_MODEL}" \
    --context-feature-source "${CONTEXT_FEATURE_SOURCE}" \
    --ridge-alphas ${RIDGE_ALPHAS} \
    --embedding-device "${EMBEDDING_DEVICE}" \
    --embed-batch-size "${EMBED_BATCH_SIZE}" \
    --embedding-cache-dir "${EMBEDDING_CACHE_DIR}" \
    --one-tr-cache-dir "${ONE_TR_CACHE_DIR}" \
    --voxel-chunk-size "${VOXEL_CHUNK_SIZE}" \
    --reliable-r-threshold "${RELIABLE_R_THRESHOLD}" \
    --delta-quantile "${DELTA_QUANTILE}" \
    --delta-min "${DELTA_MIN}" \
    --min-cluster-size "${MIN_CLUSTER_SIZE}" \
    --out-dir "${OUT_DIR}" \
    --pycortex-filestore "${PYCORTEX_FILESTORE}"

  echo "Wrote outputs to ${OUT_DIR}"
done
