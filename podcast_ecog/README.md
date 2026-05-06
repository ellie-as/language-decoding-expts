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

For the full dataset, download OpenNeuro `ds005574` using your cluster's preferred OpenNeuro/DataLad workflow, then pass that directory as `--bids-root`.

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

Expected shapes for the full tutorial are roughly:

- token embeddings: `(5491, 1600)`
- word embeddings: `(5136, 1600)`
- epochs after resampling: `(5130, 235, 128)`
- correlation result: `(2, 235, 128)`

## 5. Run on an Allocated Node

After you have an interactive node via your usual `srun` command, run:

```bash
mkdir -p podcast_ecog/outputs

python podcast_ecog/run_encoding.py \
  --bids-root "$SCRATCH/podcast_ecog/data/ds005574" \
  --output-dir podcast_ecog/outputs \
  --backend auto \
  --subject 03 \
  --picks-regex 'LG[AB]*'
```

The script writes:

- `*_encoding_results.npz`: correlations, lags, channel names, coordinates, and max-by-channel values.
- `*_lag_profile.png`: temporal encoding profile.
- `*_brain_markers.png`: electrode marker plot.

## Notes

The full tutorial fit is memory-heavy because it predicts every selected electrode at every lag. If CUDA runs out of memory, retry with `--backend torch` or `--backend numpy`, reduce channels with `--picks-regex` or `--max-channels`, or reduce lags by lowering the epoch window or resampling more aggressively.
