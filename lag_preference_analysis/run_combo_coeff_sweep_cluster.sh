#!/usr/bin/env bash
# Compare coefficient block fractions and single-block predictive performance
# across semantic embedding models and linear estimators.
#
# Example:
#   cd /ceph/behrens/ellie/language-decoding-expts
#   bash lag_preference_analysis/run_combo_coeff_sweep_cluster.sh
#
# Defaults:
#   embeddings: MiniLM, all-mpnet-base-v2, bge-base-en-v1.5
#   regressors: ridge, linear, elasticnet
#   blocks: 1TR, h20, h50, h200, h500
#   extra table: full combined and block-only validation r by ROI

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/ceph/behrens/ellie/language-decoding-expts}"
SUBJECTS="${SUBJECTS:-S1 S2 S3}"
LAG="${LAG:-2}"
SUMMARY_HORIZONS="${SUMMARY_HORIZONS:-20 50 200 500}"
EMBEDDING_MODELS="${EMBEDDING_MODELS:-sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/all-mpnet-base-v2 BAAI/bge-base-en-v1.5}"
REGRESSORS="${REGRESSORS:-ridge linear elasticnet}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-auto}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"
VOXEL_CHUNK_SIZE="${VOXEL_CHUNK_SIZE:-1000}"
ELASTICNET_ALPHA="${ELASTICNET_ALPHA:-0.001}"
ELASTICNET_L1_RATIO="${ELASTICNET_L1_RATIO:-0.1}"
ELASTICNET_MAX_ITER="${ELASTICNET_MAX_ITER:-2000}"
OUT_ROOT="${OUT_ROOT:-lag_preference_analysis/results/combo_coeff_sweep}"

EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-${DATA_ROOT}/lag_preference_analysis/cache}"
ONE_TR_CACHE_DIR="${ONE_TR_CACHE_DIR:-${DATA_ROOT}/27-04-expts/cache}"

mkdir -p "${OUT_ROOT}"

echo "DATA_ROOT=${DATA_ROOT}"
echo "SUBJECTS=${SUBJECTS}"
echo "LAG=${LAG}"
echo "SUMMARY_HORIZONS=${SUMMARY_HORIZONS}"
echo "EMBEDDING_MODELS=${EMBEDDING_MODELS}"
echo "REGRESSORS=${REGRESSORS}"
echo "EMBEDDING_DEVICE=${EMBEDDING_DEVICE}"
echo "VOXEL_CHUNK_SIZE=${VOXEL_CHUNK_SIZE}"
echo "OUT_ROOT=${OUT_ROOT}"

for SUB in ${SUBJECTS}; do
  OUT_PREFIX="${OUT_ROOT}/${SUB}_combo_coeff_sweep_lag${LAG}"

  echo
  echo "=== ${SUB} ==="

  python lag_preference_analysis/analyze_combo_coeff_sweep.py \
    --subject "${SUB}" \
    --data-root "${DATA_ROOT}" \
    --lag "${LAG}" \
    --summary-horizons ${SUMMARY_HORIZONS} \
    --embedding-models ${EMBEDDING_MODELS} \
    --regressors ${REGRESSORS} \
    --embedding-device "${EMBEDDING_DEVICE}" \
    --embed-batch-size "${EMBED_BATCH_SIZE}" \
    --embedding-cache-dir "${EMBEDDING_CACHE_DIR}" \
    --one-tr-cache-dir "${ONE_TR_CACHE_DIR}" \
    --voxel-chunk-size "${VOXEL_CHUNK_SIZE}" \
    --elasticnet-alpha "${ELASTICNET_ALPHA}" \
    --elasticnet-l1-ratio "${ELASTICNET_L1_RATIO}" \
    --elasticnet-max-iter "${ELASTICNET_MAX_ITER}" \
    --out-prefix "${OUT_PREFIX}"
done

OUT_ROOT="${OUT_ROOT}" python - <<'PY'
import csv
import os
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])
for suffix in ("long", "wide", "model_r_summary"):
    paths = sorted(out_root.glob(f"S*_combo_coeff_sweep_lag*_{suffix}.csv"))
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    if rows:
        out = out_root / f"combo_coeff_sweep_all_subjects_{suffix}.csv"
        with open(out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out} ({len(rows)} rows)")
PY
