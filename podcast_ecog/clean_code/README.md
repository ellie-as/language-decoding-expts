# Clean Podcast ECoG Word-Decoding Pipeline

This directory contains the cleaned version of the word-decoding analyses used in the local `podcast_ecog` work. The scripts read the mounted Podcast ECoG BIDS dataset and write local caches/results under `podcast_ecog/outputs/clean_code` by default.

## What Is Implemented

The pipeline tests whether ECoG activity at word `t` predicts word representations of nearby words:

- past targets: `t-1`, `t-2`, ...
- future-control targets: `t+1`, `t+2`, ...

By default, word vectors come from the dataset's bundled static `en_core_web_lg` vectors and are reduced to 5D with PCA. The same decoder code can also use contextual GPT-2 word features prepared with `prepare_gpt2_word_features.py`. Neural features are the mean high-gamma value in a `-0.5` to `+0.5` s window around each word onset.

Analyses are produced for:

- `ALL`: all channels in each subject
- paper ROIs: `EAC`, `STG`, `IFG`, `PRC`, `MFG`, `TMP`

Boundary analyses compare decoding when the decoded target word is within the same segment versus across a boundary. Implemented boundary types:

- sentence boundaries from `podcast_ecog/outputs/transcript_boundaries.json`
- event boundaries from the same JSON
- constituent boundaries generated with `benepar` constituency parsing

## Scripts

### `prepare_data.py`

Builds the reusable local cache.

It writes:

- `prepared/words.csv`: transcript words, times, sentence/event boundary flags
- `prepared/word_vectors_raw.npy`: raw static word vectors
- `prepared/word_vectors_pca5.npy`: 5D PCA word vectors
- `prepared/embedding_coverage.csv`: which words had static vectors
- `prepared/embedding_pca_explained_variance.csv`
- `prepared/roi_channels.csv`: subject/channel to paper ROI mapping
- `prepared/subjects/sub-XX/neural_word_features.npz`: word x channel neural features
- `prepared/subjects/sub-XX/channels.csv`

Default command:

```bash
python podcast_ecog/clean_code/prepare_data.py \
  --bids-root /Volumes/ellie/language-decoding-expts/podcast_ecog/data/ds005574 \
  --output-dir podcast_ecog/outputs/clean_code/prepared \
  --subjects 01 02 03 04 05 06 07 08 09
```

If a subject FIF is slow to random-access over the mount, copying that FIF locally and passing the local cache root to `--bids-root` also works. The script only needs the BIDS-like `derivatives/ecogprep/sub-XX/ieeg/...desc-highgamma_ieeg.fif` layout for neural feature extraction.

### `prepare_constituent_boundaries.py`

Adds constituency-parser boundary annotations to the prepared cache.

It uses `benepar_en3`, parses each sentence, and marks within-sentence endings of major multi-word constituents:

`NP`, `VP`, `PP`, `ADJP`, `ADVP`, `SBAR`, `S`, `SINV`, `SQ`

It excludes full-sentence constituents and sentence-final boundaries by default, so it is not just recreating the sentence-boundary analysis.

It writes:

- `prepared/constituent_spans.csv`
- `prepared/constituent_boundary_config.json`
- adds to `prepared/words.csv`:
  - `constituent_boundary_after`
  - `constituent_boundary_strength`
  - `constituent_boundary_labels`
  - `constituent_id`

Command:

```bash
python podcast_ecog/clean_code/prepare_constituent_boundaries.py \
  --prepared-dir podcast_ecog/outputs/clean_code/prepared \
  --bids-root /Volumes/ellie/language-decoding-expts/podcast_ecog/data/ds005574 \
  --overwrite
```

Current generated constituent annotations:

- parsed spans: `10,681`
- marked constituent boundaries: `1,044`

### `prepare_gpt2_word_features.py`

Adds contextual GPT-2 word targets to the prepared cache.

For each transcript word, it runs GPT-2 over a rolling token context and extracts the hidden state at the requested layer. If a word spans multiple GPT tokens, token features are averaged to one word feature. The raw GPT-2 features are then PCA-reduced with the same `compute_word_pca` helper used for the static vectors.

Command used for the GPT-2 layer-8, 32-token-context analysis:

```bash
python podcast_ecog/clean_code/prepare_gpt2_word_features.py \
  --bids-root /Volumes/ellie/language-decoding-expts/podcast_ecog/data/ds005574 \
  --prepared-dir podcast_ecog/outputs/clean_code/prepared \
  --model gpt2 \
  --context-token-length 32 \
  --layer 8 \
  --n-components 5 \
  --batch-size 32 \
  --device cpu \
  --local-files-only
```

It writes:

- `prepared/gpt2_ctx32_layer8_pca5_raw.npy`: raw 768D GPT-2 word features
- `prepared/gpt2_ctx32_layer8_pca5.npy`: 5D PCA GPT-2 word features
- `prepared/gpt2_ctx32_layer8_pca5_tokens.csv`
- `prepared/gpt2_ctx32_layer8_pca5_explained_variance.csv`
- `prepared/gpt2_ctx32_layer8_pca5_config.json`

### `train_decoders.py`

Trains subject-level ridge decoders.

For each subject, ROI, lag, and direction, it trains a cross-validated ridge model:

```text
neural activity at word t -> 5D word vector at t-N or t+N
```

The default uses the same valid current-word positions across all requested lags and both directions within a subject.

By default, targets are `prepared/word_vectors_pca5.npy`. To decode an alternative prepared representation, pass `--target-stem`; for example, `--target-stem gpt2_ctx32_layer8_pca5`.

It writes:

- `decoders/decoder_summary.csv`
- `decoders/predictions/sub-XX__ROI__past__lag-NN.npz`
- `decoders/predictions/sub-XX__ROI__future__lag-NN.npz`

Command:

```bash
python podcast_ecog/clean_code/train_decoders.py \
  --prepared-dir podcast_ecog/outputs/clean_code/prepared \
  --output-dir podcast_ecog/outputs/clean_code/decoders \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --rois ALL EAC STG IFG PRC MFG TMP \
  --lags 1 2 3 4 5 6 7 8 9 10 \
  --ridge-alpha 1000 \
  --outer-splits 5
```

GPT-2 target command:

```bash
python podcast_ecog/clean_code/train_decoders.py \
  --prepared-dir podcast_ecog/outputs/clean_code/prepared \
  --output-dir podcast_ecog/outputs/clean_code/decoders_gpt2_ctx32_layer8 \
  --target-stem gpt2_ctx32_layer8_pca5 \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --rois ALL EAC STG IFG PRC MFG TMP \
  --lags 1 2 3 4 5 6 7 8 9 10 \
  --ridge-alpha 1000 \
  --outer-splits 5
```

### `decoding_analysis.py`

Aggregates decoder outputs and makes heatmaps/tables.

It writes:

- raw directional prediction heatmaps
- future-controlled past-word prediction heatmap
- sentence/event/constituent boundary directional heatmaps
- sentence/event/constituent future-controlled boundary heatmaps
- combined PDF containing the main heatmaps in order

Command:

```bash
python podcast_ecog/clean_code/decoding_analysis.py \
  --prepared-dir podcast_ecog/outputs/clean_code/prepared \
  --decoder-dir podcast_ecog/outputs/clean_code/decoders \
  --output-dir podcast_ecog/outputs/clean_code/analysis \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --rois ALL EAC STG IFG PRC MFG TMP \
  --lags 1 2 3 4 5 6 7 8 9 10 \
  --boundary-levels sentence event constituent
```

GPT-2 target command:

```bash
python podcast_ecog/clean_code/decoding_analysis.py \
  --prepared-dir podcast_ecog/outputs/clean_code/prepared \
  --decoder-dir podcast_ecog/outputs/clean_code/decoders_gpt2_ctx32_layer8 \
  --output-dir podcast_ecog/outputs/clean_code/analysis_gpt2_ctx32_layer8 \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --rois ALL EAC STG IFG PRC MFG TMP \
  --lags 1 2 3 4 5 6 7 8 9 10 \
  --boundary-levels sentence event constituent
```

## Main Outputs

Analysis outputs are in:

```text
podcast_ecog/outputs/clean_code/analysis
```

Main figures:

- `raw_directional_prediction_heatmap.png`
- `raw_directional_prediction_weighted_pc_heatmap.png`
- `future_controlled_prediction_delta_heatmap.png`
- `constituent_boundary_directional_delta_heatmap.png`
- `constituent_boundary_future_controlled_delta_heatmap.png`
- `sentence_boundary_directional_delta_heatmap.png`
- `sentence_boundary_future_controlled_delta_heatmap.png`
- `event_boundary_directional_delta_heatmap.png`
- `event_boundary_future_controlled_delta_heatmap.png`
- `combined_decoding_heatmaps.pdf`

The equivalent GPT-2 layer-8, 32-token-context analysis is in:

```text
podcast_ecog/outputs/clean_code/analysis_gpt2_ctx32_layer8
```

Its combined PDF is:

```text
podcast_ecog/outputs/clean_code/analysis_gpt2_ctx32_layer8/combined_decoding_heatmaps.pdf
```

The combined PDF page order is:

1. `raw_directional_prediction_heatmap.png`
2. `future_controlled_prediction_delta_heatmap.png`
3. `constituent_boundary_directional_delta_heatmap.png`
4. `constituent_boundary_future_controlled_delta_heatmap.png`
5. `sentence_boundary_directional_delta_heatmap.png`
6. `sentence_boundary_future_controlled_delta_heatmap.png`
7. `event_boundary_directional_delta_heatmap.png`
8. `event_boundary_future_controlled_delta_heatmap.png`

## Metrics

### `point_r`

For each held-out word, compute:

```text
corr(true_5D_word_vector, predicted_5D_word_vector)
```

This gives one score per word. Boundary analyses use this metric because the within/across-boundary split is defined at the word level.

### `weighted_pc_r`

For each PCA component, compute correlation over time:

```text
corr(true_PC_k_over_words, predicted_PC_k_over_words)
```

Then average components weighted by the variance of the target components. This is a more standard decoder performance metric, but it does not directly give a pointwise score for each word.

### Future-Controlled Prediction

For each subject, ROI, and lag:

```text
mean point_r(past target t-N) - mean point_r(future target t+N)
```

This asks whether past-word information is decodable above the matched future-word control.

### Boundary Delta

For each boundary type, subject, ROI, lag, and direction:

```text
delta = mean point_r(within segment) - mean point_r(across boundary)
```

Positive delta means decoding is better when the target and current word are within the same segment.

### Future-Controlled Boundary Delta

For each subject, ROI, and lag:

```text
(past within-across) - (future within-across)
```

This controls for symmetric local effects around boundaries.

## Aggregation

Heatmap cells are averaged across subject-level scores.

For ROI rows, a cell can use fewer than 9 subjects if a subject has no channels in that ROI. The relevant `*_group_scores.csv` files include `n_subjects`.

## Notes

- The default raw directional prediction plot is `raw_directional_prediction_heatmap.png`, using `point_r`.
- `raw_directional_prediction_weighted_pc_heatmap.png` is a companion plot using the component-wise score.
- Significance stars are based on exact sign-flip tests across subjects, with FDR columns written in the group score tables.
- The scripts are designed to reuse existing outputs unless overwrite/force flags are supplied where available.
