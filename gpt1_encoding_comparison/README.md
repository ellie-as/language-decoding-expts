# GPT-1 Encoding Comparison

Compare paper-style Huth/Tang finetuned GPT-1 encoding models against a
size-matched original pretrained GPT-1 (`openai-gpt`).

This experiment mirrors `decoding/train_EM.py`:

1. load the full perceived-speech training story set from `sess_to_story.json`
2. extract layer-9 contextual word features
3. interpolate word features to TRs and apply the standard FIR delays
4. fit bootstrap ridge encoding models
5. select the top language-responsive voxels
6. optionally estimate the paper-style noise model for decoder compatibility

The comparison is intentionally encoding-only for now. The finetuned condition
uses the local Huth/Tang checkpoint in `data_lm/perceived`. The pretrained
condition uses Hugging Face `openai-gpt` with the Hugging Face tokenizer, while
keeping the same model family, layer, context length, ridge setup, sessions, and
response data.

## Quick Start

Run S1 only and skip the expensive noise model:

```bash
python gpt1_encoding_comparison/compare_gpt1_encoding.py \
  --subjects S1 \
  --skip-noise-model
```

Run the full paper-style setup for S1-S3:

```bash
python gpt1_encoding_comparison/compare_gpt1_encoding.py \
  --subjects S1 S2 S3
```

Outputs are written to `gpt1_encoding_comparison/outputs/<subject>/` by
default. Each subject gets one `.npz` per condition plus
`comparison_summary.json`.

## Outputs

Each condition file includes:

- `weights`: ridge weights for all voxels
- `noise_model`: selected-voxel noise covariance, unless `--skip-noise-model`
- `alphas`: selected ridge alpha per voxel
- `bootstrap_corrs`: cross-validated encoding score per voxel
- `voxels`: top selected voxels
- `tr_stats` / `word_stats`: normalization statistics used by the stimulus model

The JSON summary reports mean and max bootstrap correlations, selected-voxel
overlap, and the correlation between the finetuned and pretrained per-voxel
score maps.
