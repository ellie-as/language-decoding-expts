# fMRI Text MAE

This folder is a first-pass implementation of direct text decoding from the Huth fMRI dataset, inspired by the NEDS multi-modal masking setup.

The workflow is deliberately staged:

1. **Baseline retrieval**: predict a transcript-window sentence embedding from an aligned fMRI window, then retrieve the matching transcript window from distractors.
2. **Masked multimodal autoencoder**: train a small transformer over fMRI tokens and text tokens with NEDS-style masking modes: within-fMRI, within-text, fMRI-to-text, text-to-fMRI, and heavy multimodal masking.
3. **Generation/evaluation**: run fMRI-only masked-token decoding and retrieval controls. Autoregressive GPT-prefix decoding is scaffolded as a later extension.

The code reuses this repository's Huth conventions:

- subject IDs such as `S1`
- responses in `data_train/train_response/<subject>/<story>.hf5`
- word timings via `decoding/utils_stim.py::get_story_wordseqs`
- story-level train/validation/test splits
- train-only fMRI normalization
- a fixed haemodynamic lag, default 4 seconds

## Quick Start

Create aligned windows:

```bash
python fmri_text_mae/data/make_windows.py \
  --subject S1 \
  --sessions 2 3 4 5 6 7 8 9 10 11 12 14 15 18 20 \
  --output-dir fmri_text_mae/outputs/windows/S1
```

Run the first milestone retrieval baseline:

```bash
python fmri_text_mae/src/train_retrieval_baseline.py \
  --config fmri_text_mae/configs/baseline_retrieval.yaml
```

Train the masked multimodal model:

```bash
python fmri_text_mae/src/train_mae.py \
  --config fmri_text_mae/configs/mae_subject01.yaml
```

Evaluate fMRI-only masked text decoding and retrieval:

```bash
python fmri_text_mae/src/evaluate_generation.py \
  --config fmri_text_mae/configs/mae_subject01.yaml \
  --checkpoint fmri_text_mae/outputs/mae/S1/best.pt
```

## Controls

The retrieval and MAE scripts include switches for shuffled pairs and wrong-lag/window files. Keep the scientific claim tied to held-out stories and controls: the key question is whether correctly aligned fMRI improves text prediction or transcript retrieval above shuffled/no-fMRI baselines.

## NEDS-Inspired Details

NEDS alternates task masks for encoding, decoding, self-modality reconstruction, and random token masking. This implementation maps those ideas to Huth fMRI/text:

- `fmri_mask`: partial fMRI masked, text visible
- `text_mask`: partial text masked, fMRI visible
- `fmri_to_text`: all text masked, fMRI visible
- `text_to_fmri`: all fMRI masked, text visible
- `both_masked`: random fMRI and text spans masked together

The MAE computes cross entropy only on masked text positions and MSE only on masked fMRI positions.
