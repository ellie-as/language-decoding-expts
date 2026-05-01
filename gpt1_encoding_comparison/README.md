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

If the data are on the mounted SWC share but outputs should stay in this repo,
use `--local-compute-mode`.

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

## Alternate Short-Window Features

To keep the Huth encoding method fixed but swap in GPT-2 or sentence-transformer
features over the same short word window, run:

```bash
python gpt1_encoding_comparison/compare_huth_style_features.py \
  --subjects S1 \
  --conditions gpt2 sentence \
  --skip-noise-model
```

If the Huth data live on the mounted SWC share but outputs should stay in this
local repo, use:

```bash
python gpt1_encoding_comparison/compare_huth_style_features.py \
  --local-compute-mode \
  --subjects S1 \
  --conditions gpt2 sentence
```

This uses:

- `gpt2`: GPT-2 small (`openai-community/gpt2`) layer 9, current word's final BPE token
- `sentence`: `sentence-transformers/all-MiniLM-L6-v2` over the current word plus the previous `config.GPT_WORDS` words

Both conditions use the same Huth pipeline as `decoding/train_EM.py`: `get_stim`
TR interpolation, standard FIR delays, `bootstrap_ridge(..., use_corr=False)`,
and top-voxel selection. This is deliberately different from earlier summary /
context experiments that used much larger text windows.

Outputs are written to
`gpt1_encoding_comparison/huth_style_feature_outputs/<subject>/`.

## Decoding Test

After training encoders **with** noise models, run the held-out perceived-speech
decoder comparison:

```bash
python gpt1_encoding_comparison/decode_and_score.py \
  --subject S1 \
  --experiment perceived_speech \
  --tasks wheretheressmoke \
  --conditions paper finetuned pretrained \
  --n-null 10
```

`paper` uses the downloaded Huth/Tang model from `models/S1/`; `finetuned` and
`pretrained` use this experiment's trained encoders from
`gpt1_encoding_comparison/outputs/S1/`.

To compare the standalone MiniLM combo ridge models, train the two GPT-1
comparison versions directly:

```bash
python gpt1_encoding_comparison/train_minilm_combo_encoding.py \
  --data-root /ceph/behrens/ellie/language-decoding-expts \
  --subjects S1 \
  --conditions minilm_summary_combo minilm_window_combo \
  --lag 2 \
  --voxel-set full_frontal
```

This writes:

- `gpt1_encoding_comparison/outputs/S1/encoding_model_minilm_summary_combo.npz`
- `gpt1_encoding_comparison/outputs/S1/encoding_model_minilm_window_combo.npz`
- `gpt1_encoding_comparison/outputs/S1/minilm_combo_encoding_summary.csv`

Then include the MiniLM conditions in the decoding comparison:

```bash
python gpt1_encoding_comparison/decode_and_score.py \
  --subject S1 \
  --experiment perceived_speech \
  --tasks wheretheressmoke \
  --conditions paper finetuned pretrained minilm_summary_combo minilm_window_combo \
  --n-null 10
```

`minilm_window_combo` is decoded with the same candidate text-window features it
was trained on. `minilm_summary_combo` is trained with true training-story
summaries, but decoding uses candidate text-window proxy features because oracle
summaries of the held-out test story would leak the answer.

On the cluster, the same train+decode comparison can be launched with:

```bash
cd /ceph/behrens/ellie/language-decoding-expts
SUBJECTS="S1" RUN_DECODE=1 bash gpt1_encoding_comparison/run_minilm_combo_cluster.sh
```

For all three subjects:

```bash
cd /ceph/behrens/ellie/language-decoding-expts
SUBJECTS="S1 S2 S3" RUN_DECODE=1 bash gpt1_encoding_comparison/run_minilm_combo_cluster.sh
```

The script writes decoded transcripts under
`gpt1_encoding_comparison/decoding_outputs/` plus a
`decoding_score_summary.csv` with WER, BLEU-1, METEOR, and BERTScore recall
story scores, null-normalized z-scores, and significant-window fractions.

For the pretrained encoder condition, the decoder keeps the Huth/Tang language
model as the proposal prior and scores candidate words using the pretrained
GPT-1 feature space. This isolates the effect of the trained encoding model
while keeping the beam-search vocabulary and language prior matched.

If you want the CSV to include deltas from paper headline values, pass a JSON
file mapping metric names to values, for example:

```json
{
  "BERT": 0.0,
  "WER": 0.0
}
```

and run with `--paper-headline-json /path/to/headlines.json`.
