# Podcast ECoG dataset analysis

`clean_code_v2` is the latest code as of 02/06/2026.

It writes results under:

- `podcast_ecog/outputs/clean_code_v2/prepared_midpoint`
- `podcast_ecog/outputs/clean_code_v2/preferred_lag_midpoint`
- `podcast_ecog/outputs/clean_code_v2/super_brain_midpoint`
- `podcast_ecog/outputs/clean_code_v2/sentence_boundary_channel_timecourses`

The original `podcast_ecog/clean_code` directory is preserved as legacy code and is not imported by v2. 

## What the code does

`run_all.py` builds the `super_brain_midpoint` results folder:

- midpoint high-gamma and word-feature preparation
- constituent-boundary annotation
- preferred-lag encoding cross-check inputs
- pooled super-brain decoders for `word_vectors_pca20`
- pooled super-brain decoders for `gpt2_ctx32_layer8_pca20`
- decoding heatmaps and combined PDFs
- present-word pooled decoding JSON summaries
- decoder-weight anatomy outputs
- lag-sensor intuition outputs and Nilearn lag plots
- GPT2 current-word confound plots
- GPT2 sentence/constituent boundary-locked plots

It also rebuilds the matched sentence-boundary channel panels:

- top 24 high-gamma decrease channels
- top 24 high-gamma increase channels

## Required Inputs

The full pipeline needs source data/resources:

- Podcast ECoG BIDS data, defaulting to:
  `podcast_ecog/data/ds005574`
- transcript boundaries:
  `podcast_ecog/outputs/transcript_boundaries.json`
- ROI/channel anatomy table:
  `podcast_ecog/outputs/gpt2_paper_roi_context_layer_exact/channel_paper_roi_metrics.csv`
- a usable GPT2 model cache or internet access for `transformers`
- the `benepar_en3` parser model for constituency boundaries

If the local BIDS dataset is a git-annex checkout, make sure the small static-vector files are present, for example:

```bash
git -C podcast_ecog/data/ds005574 annex get stimuli/en_core_web_lg/features.hdf5 stimuli/podcast_transcript.csv
```

## Full Run

From the repository root:

```bash
python podcast_ecog/clean_code_v2/run_all.py --overwrite
```

This prepares v2 caches from source data, refits the super-brain decoders, recomputes preferred-lag inputs, and recomputes the sentence-boundary channel screen from raw high-gamma before making the channel panels.

## Faster Development Run

If the v2 caches and decoder predictions already exist and you only want to refresh downstream analyses/plots:

```bash
python podcast_ecog/clean_code_v2/run_all.py --analyze-only --reuse-channel-tracking
```

`--reuse-channel-tracking` uses:

```text
podcast_ecog/outputs/boundary_channel_tracking/sentence
```

for the expensive channel-screen intermediate only. It is a speed option, not the full reproduction mode.

To use existing v2 preparation caches while refitting decoders/plots:

```bash
python podcast_ecog/clean_code_v2/run_all.py --overwrite --skip-prepare
```

## Verify

```bash
python podcast_ecog/clean_code_v2/verify_outputs.py
```

The runner also writes:

```text
podcast_ecog/outputs/clean_code_v2/manifest.json
```

## Boundary Anchor Notes

- In super-brain boundary-locked plots, `0` is the final word before the boundary and `+1` is the first word after the boundary.
- In sentence-boundary channel timecourse panels, `0` is `next_start`, the first word onset of the new sentence. This matches the original channel-screen convention.
