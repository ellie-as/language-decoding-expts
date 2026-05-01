#!/usr/bin/env bash
# Cluster runner for standalone MiniLM combo encoding + optional decoding.
#
# Example:
#   cd /ceph/behrens/ellie/language-decoding-expts
#   bash gpt1_encoding_comparison/run_minilm_combo_cluster.sh
#
# Environment overrides:
#   SUBJECTS="S1 S2 S3"
#   DATA_ROOT=/ceph/behrens/ellie/language-decoding-expts
#   RUN_DECODE=1
#   TASKS=wheretheressmoke
#   N_NULL=10
#   DEVICE=cpu

set -euo pipefail

SUBJECTS="${SUBJECTS:-S1}"
DATA_ROOT="${DATA_ROOT:-/ceph/behrens/ellie/language-decoding-expts}"
TASKS="${TASKS:-wheretheressmoke}"
N_NULL="${N_NULL:-10}"
DEVICE="${DEVICE:-cpu}"
RUN_DECODE="${RUN_DECODE:-1}"
LAG="${LAG:-2}"
VOXEL_SET="${VOXEL_SET:-full_frontal}"
VOXEL_COUNT="${VOXEL_COUNT:-10000}"

echo "Repo: $(pwd)"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SUBJECTS=${SUBJECTS}"
echo "RUN_DECODE=${RUN_DECODE}"

python gpt1_encoding_comparison/train_minilm_combo_encoding.py \
  --data-root "${DATA_ROOT}" \
  --subjects ${SUBJECTS} \
  --conditions minilm_summary_combo minilm_window_combo \
  --lag "${LAG}" \
  --voxel-set "${VOXEL_SET}" \
  --voxel-count "${VOXEL_COUNT}" \
  --skip-existing

if [[ "${RUN_DECODE}" == "1" ]]; then
  for subject in ${SUBJECTS}; do
    python gpt1_encoding_comparison/decode_and_score.py \
      --subject "${subject}" \
      --experiment perceived_speech \
      --tasks ${TASKS} \
      --conditions paper finetuned pretrained minilm_summary_combo minilm_window_combo \
      --device "${DEVICE}" \
      --n-null "${N_NULL}" \
      --skip-existing
  done
fi
