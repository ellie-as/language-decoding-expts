#!/bin/bash
#SBATCH --job-name=libribrain_gtr_ridge
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/ceph/behrens/ellie/language-decoding-expts/libribrain_gtr_decode/logs/%x_%j.out
#SBATCH --error=/ceph/behrens/ellie/language-decoding-expts/libribrain_gtr_decode/logs/%x_%j.err

set -euo pipefail

# Edit these two lines for your cluster if needed.
REPO_DIR="${REPO_DIR:-/ceph/behrens/ellie/language-decoding-expts/libribrain_gtr_decode}"
CONDA_ENV="${CONDA_ENV:-libribrain-gtr}"

source ~/.bashrc
conda activate "${CONDA_ENV}"

cd "${REPO_DIR}"
mkdir -p logs

python scripts/run_ridge_pipeline.py \
    --config configs/ridge_cluster.yaml
