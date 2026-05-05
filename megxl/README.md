# MEG-XL LibriBrain Text Embedding Probe

This directory contains a local checkout of `neural-processing-lab/MEG-XL` plus
a standalone script for decoding text-window embeddings from LibriBrain MEG with
a frozen MEG-XL backbone and one trained linear layer.

## Setup

```bash
cd megxl
conda create -n megxlenv python=3.12
conda activate megxlenv
pip install -r MEG-XL/requirements.txt
pip install huggingface_hub
```

The script downloads `pnpl/MEG-XL/meg-xl-med.ckpt` into `megxl/checkpoints/` on
first run. The BioCodec checkpoint is already included in the cloned MEG-XL repo.

## Smoke Test

```bash
python scripts/libribrain_text_embedding_linear_probe.py \
  --libribrain-root /path/to/LibriBrain \
  --subjects sub-0 \
  --tasks Sherlock1 \
  --max-segments 40 \
  --epochs 1 \
  --batch-size 1 \
  --loss mse
```

## Node Run

```bash
python scripts/libribrain_text_embedding_linear_probe.py \
  --libribrain-root /path/to/LibriBrain \
  --subjects sub-0 \
  --tasks Sherlock1 \
  --words-per-segment 5 \
  --batch-size 2 \
  --epochs 20 \
  --num-workers 8
```

Outputs are written to `outputs/libribrain_text_probe/`, including
`best_probe.pt`, `config.json`, and `metrics.json`.
