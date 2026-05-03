#!/usr/bin/env bash
# Run summary-combo ridge block ablations on the cluster.
#
# Example:
#   cd /ceph/behrens/ellie/language-decoding-expts
#   bash lag_preference_analysis/run_combo_block_ablation_cluster.sh
#
# Useful overrides:
#   SUBJECTS="S1 S2 S3" TOP_N=1000 bash lag_preference_analysis/run_combo_block_ablation_cluster.sh
#   TOP_N=0 bash lag_preference_analysis/run_combo_block_ablation_cluster.sh  # all full-frontal voxels

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/ceph/behrens/ellie/language-decoding-expts}"
SUBJECTS="${SUBJECTS:-S1 S2 S3}"
LAG="${LAG:-2}"
TOP_N="${TOP_N:-1000}"
VOXEL_CHUNK_SIZE="${VOXEL_CHUNK_SIZE:-1000}"
OUT_ROOT="${OUT_ROOT:-lag_preference_analysis/results/combo_block_ablation}"

EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-${DATA_ROOT}/lag_preference_analysis/cache}"
ONE_TR_CACHE_DIR="${ONE_TR_CACHE_DIR:-${DATA_ROOT}/27-04-expts/cache}"

mkdir -p "${OUT_ROOT}"

echo "DATA_ROOT=${DATA_ROOT}"
echo "SUBJECTS=${SUBJECTS}"
echo "LAG=${LAG}"
echo "TOP_N=${TOP_N}"
echo "VOXEL_CHUNK_SIZE=${VOXEL_CHUNK_SIZE}"
echo "OUT_ROOT=${OUT_ROOT}"

for SUB in ${SUBJECTS}; do
  TAG="${SUB}__embedding-summary-combo-h20-50-200__lags1-10__chunk1tr__seed0"
  RESULTS_DIR="${DATA_ROOT}/lag_preference_analysis/results/${TAG}"
  OUT_PREFIX="${OUT_ROOT}/${SUB}_combo_block_ablation_lag${LAG}_top${TOP_N}"

  echo
  echo "=== ${SUB}: ${RESULTS_DIR} ==="

  python lag_preference_analysis/analyze_combo_block_ablation.py \
    --results-dir "${RESULTS_DIR}" \
    --subject "${SUB}" \
    --data-root "${DATA_ROOT}" \
    --embedding-cache-dir "${EMBEDDING_CACHE_DIR}" \
    --one-tr-cache-dir "${ONE_TR_CACHE_DIR}" \
    --lag "${LAG}" \
    --top-n "${TOP_N}" \
    --voxel-chunk-size "${VOXEL_CHUNK_SIZE}" \
    --out-prefix "${OUT_PREFIX}"
done

python - <<'PY'
import csv
from pathlib import Path

out_root = Path("lag_preference_analysis/results/combo_block_ablation")
summary_paths = sorted(out_root.glob("*_summary.csv"))
combined = out_root / "combo_block_ablation_summary_all_subjects.csv"

rows = []
for path in summary_paths:
    with open(path, encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))

if rows:
    with open(combined, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {combined} ({len(rows)} rows)")
else:
    print(f"No summary CSVs found under {out_root}")
PY
