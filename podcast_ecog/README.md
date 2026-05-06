# Podcast ECoG Encoding Tutorial on a Cluster

This directory contains a Python-script version of the Hasson Lab Podcast ECoG encoding tutorial:

- Dataset paper: Zada et al., Scientific Data 12, 1135 (2025), DOI `10.1038/s41597-025-05462-2`
- Dataset: OpenNeuro `ds005574`
- Tutorial ported here: `04-encoding`, training ridge encoding models with GPT-2 XL embeddings and high-gamma ECoG.

## Files

- `run_encoding.py`: command-line Python port of the notebook.
- `download_minimal.py`: downloads the three files needed for the sub-03 tutorial run.
- `requirements.txt`: packages needed in a clean environment.

## 1. Check the Python Environment

Use the Python environment you normally use on the cluster. Check that the required packages are importable:

```bash
python - <<'PY'
import h5py
import himalaya
import matplotlib
import mne
import mne_bids
import nilearn
import numpy
import pandas
import sklearn
import torch
print("Podcast ECoG Python dependencies are importable.")
PY
```

If anything is missing, install the requirements into that environment using your cluster's normal package workflow:

```bash
python -m pip install -r podcast_ecog/requirements.txt
```

If your cluster provides PyTorch modules, use those instead of pip-installing `torch`. The script uses the Himalaya `torch_cuda` backend when CUDA is visible.

## 2. Set the Project Path

On the cluster, use the repo clone as the working root:

```bash
export SCRATCH=/ceph/behrens/ellie/language-decoding-expts
cd "$SCRATCH"
```

All commands below keep data and outputs inside that directory.

## 3. Put the Dataset Somewhere Sensible

The tutorial needs:

- `stimuli/gpt2-xl/features.hdf5` around 1.6 GB
- `stimuli/gpt2-xl/transcript.tsv`
- `derivatives/ecogprep/sub-03/ieeg/sub-03_task-podcast_desc-highgamma_ieeg.fif`

For the minimal tutorial data:

```bash
python podcast_ecog/download_minimal.py --bids-root "$SCRATCH/podcast_ecog/data/ds005574"
```

For the full OpenNeuro `ds005574` dataset:

```bash
python podcast_ecog/download_minimal.py --all --bids-root "$SCRATCH/podcast_ecog/data/ds005574"
```

To inspect the file list first:

```bash
python podcast_ecog/download_minimal.py --all --dry-run
```

## 4. Run a Smoke Test

Run a tiny job first. This catches environment and path issues without spending GPU hours:

```bash
python podcast_ecog/run_encoding.py \
  --bids-root "$SCRATCH/podcast_ecog/data/ds005574" \
  --output-dir podcast_ecog/outputs/smoke \
  --max-words 200 \
  --max-channels 8 \
  --backend auto
```

Expected shapes depend on the channel filter:

- token embeddings: `(5491, 1600)`
- word embeddings: `(5136, 1600)`
- all channels with `--picks-regex '.*'`: epochs `(5130, 235, 128)`, correlations `(2, 235, 128)`
- left-grid channels with `--picks-regex 'LG.*'`: epochs `(5130, 127, 128)`, correlations `(2, 127, 128)`

The rendered tutorial shows `(5130, 235, 128)`, which corresponds to all 235 ECoG channels in the current `task-podcast` high-gamma file. If you run with `LG.*`, you are intentionally plotting only the 127 channels whose names start with `LG`.

## 5. Run on an Allocated Node

After you have an interactive node via your usual `srun` command, run:

```bash
mkdir -p podcast_ecog/outputs

python podcast_ecog/run_encoding.py \
  --bids-root "$SCRATCH/podcast_ecog/data/ds005574" \
  --output-dir podcast_ecog/outputs \
  --backend auto \
  --subject 03 \
  --picks-regex '.*'
```

The script writes:

- `*_encoding_results.npz`: correlations, lags, channel names, coordinates, and max-by-channel values.
- `*_lag_profile.png`: temporal encoding profile.
- `*_brain_markers.png`: electrode marker plot.

## Notes

The full tutorial fit is memory-heavy because it predicts every selected electrode at every lag. If CUDA runs out of memory, retry with `--backend torch` or `--backend numpy`, reduce channels with `--picks-regex` or `--max-channels`, or reduce lags by lowering the epoch window or resampling more aggressively.

## Derived Analyses

If the cluster clone is mounted locally at `/Volumes/ellie/language-decoding-expts`, derived analyses can read the saved encoding results from Ceph and write local outputs here.

Preferred lag per channel:

```bash
python podcast_ecog/plot_preferred_lag.py
```

This writes a colorbar brain plot and a CSV table to `podcast_ecog/outputs/preferred_lag/`. Pass `--results-npz` if you want to use a specific encoding result file.

Text-window horizon encoding:

```bash
python podcast_ecog/run_text_window_encoding.py
```

By default this reads the podcast transcript and high-gamma FIF from `/Volumes/ellie/language-decoding-expts`, embeds trailing text windows of 1, 5, 10, 20, 50, 100, 200, and 500 words with MiniLM, and predicts each channel's response at its GPT-2-preferred lag. Outputs are local under `podcast_ecog/outputs/text_window_encoding/`.

After the run, write compact interpretation tables and plots:

```bash
python podcast_ecog/summarize_text_window_encoding.py
```

To inspect long-window channels and their temporal GPT-2 lag profiles:

```bash
python podcast_ecog/plot_long_window_lag_profiles.py
```

To compare a few encoding model classes on the same sub-03 text-window task:

```bash
python podcast_ecog/compare_encoding_models.py
```

To compare GPT-2 features against MiniLM sentence/text-window embeddings:

```bash
python podcast_ecog/compare_feature_spaces.py
```

To test whether concatenating GPT-2 embeddings from the last N words helps:

```bash
python podcast_ecog/compare_gpt2_context_windows.py
```

To recompute one-word GPT-2 features while varying the internal model context length:

```bash
python podcast_ecog/compare_gpt2_internal_context.py
```

To sweep several Hugging Face causal LMs over several internal context lengths
on an allocated server node:

```bash
python podcast_ecog/run_llm_context_sweep.py \
  --bids-root /ceph/behrens/ellie/language-decoding-expts/podcast_ecog/data/ds005574 \
  --mounted-root /ceph/behrens/ellie/language-decoding-expts \
  --output-dir podcast_ecog/outputs/llm_context_sweep \
  --subjects 03 \
  --models gpt2 gpt2-medium gpt2-large openai-community/gpt2-xl \
  --context-token-lengths 0 1 2 4 8 16 32 64 \
  --layers final \
  --device cuda \
  --batch-size 8
```

Use `--extract-only` if you only want to cache features first. The ridge
evaluation step needs a preferred-lag reference result for each subject; with
the current defaults that is expected at
`podcast_ecog/outputs_all_channels/sub-XX_gpt2-xl_layer-24_encoding_results.npz`
under `--mounted-root`.

To run the ridge text-window preference analysis across all nine podcast subjects:

```bash
python podcast_ecog/run_all_subject_window_preferences.py
```
