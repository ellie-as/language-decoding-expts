# lag_preference_analysis

Train a separate MiniLM-to-fMRI encoding model at each lag and look at where
in the frontal cortex prediction is best.

For each lag `L` in 1..10 TRs:

```
X = MiniLM embedding of the words spoken during a single 1-TR window at chunk i
Y = brain response at TR (i + L) for every voxel in BA_full_frontal
```

Models are story-grouped train/val: a held-out story split is used to compute
per-voxel Pearson r at every lag. The point is to ask which voxels respond
to the *current* TR (small L) versus several TRs after it (large L) and how
that lag preference is organised anatomically.

## Files

| Script | Purpose |
| --- | --- |
| `train_lag_encoding.py` | Train one ridge encoder per lag, save per-voxel r per lag. |
| `analyze_lag_preference.py` | Read the saved r matrix, break down by Brodmann sub-ROI. |
| `plot_lag_flatmaps.py` | Render pycortex flatmaps of preferred lag, best-lag r, etc. |

## Outputs (per run)

```
results/<tag>/
├── lag_corrs.npz                # corrs[n_lags, n_voxels], best_alphas, voxels, lags, ...
├── lag_summary.csv              # per-lag aggregate stats over full_frontal
├── config.json                  # CLI args, train/val story lists, embedding cache path, ...
├── lag_preference_breakdown.csv # written by analyze_lag_preference.py
└── per_lag/lag01.npz, lag02.npz, ...  # per-lag corrs + best_alphas
```

`<tag>` defaults to e.g. `S1__embedding__lags1-10__chunk1tr__seed0`.

## Usage

### Cluster (data lives on `/ceph/...`)

```
python lag_preference_analysis/train_lag_encoding.py --subject S1
python lag_preference_analysis/analyze_lag_preference.py \
    --results-dir lag_preference_analysis/results/S1__embedding__lags1-10__chunk1tr__seed0
```

### Local with Ceph mounted at `/Volumes/ellie/language-decoding-expts`

```
python lag_preference_analysis/train_lag_encoding.py \
    --subject S1 --local-compute-mode
python lag_preference_analysis/analyze_lag_preference.py \
    --results-dir lag_preference_analysis/results/S1__embedding__lags1-10__chunk1tr__seed0 \
    --ba-dir /Volumes/ellie/language-decoding-expts/ba_indices
```

`--local-compute-mode` reads `data_train`, `data_lm`, `models`, and the
Brodmann index files from the mounted volume; outputs always land inside this
repo. Pass `--mounted-project-root` to point at a different mount.

To use an alternative data root explicitly (no smb):

```
python lag_preference_analysis/train_lag_encoding.py \
    --subject S1 --data-root /path/to/language-decoding-expts
```

### Useful flags

- `--lags 1 2 3 4 5 6 7 8 9 10` – set of lags (TRs).
- `--chunk-trs 1` – text window size (default 1 TR ≈ a few words).
- `--feature-model embedding` – `embedding` (MiniLM, 384-d) or `gtr-base` (GTR, 768-d).
- `--val-stories <s1> <s2> ...` – pin held-out stories (otherwise random subset of size `--val-story-count`).
- `--ridge-alphas 1 10 100 ... 1e5` – per-voxel LOO RidgeCV grid.
- `--voxel-chunk-size 5000` – fit voxels in slices to keep peak memory bounded.

## Method notes

- Text features come from `mindeye_text/_shared.load_or_build_chunk_embeddings`,
  which reuses the existing `27-04-expts/cache/<subject>/...` cache layout.
  Embeddings are computed once with `chunk_trs = 1` and `lag_trs = max(args.lags)`,
  then reused for every lag (the *text* at chunk `i` is the same regardless of
  the lag we apply on the brain side).
- Brain alignment matches the rest of the repo's chunk pipeline:
  `Y[i] = response[story][i + L]` after trimming nothing, with the text TR
  offset (`TRIM_START`) handled inside `_shared.text_for_tr_chunk`.
- Train/val are story-grouped: every voxel score is on entirely held-out
  stories, so per-voxel r is comparable across lags.
- Per-voxel ridge alphas are chosen with sklearn `RidgeCV(alpha_per_target=True)`
  in chunks of `--voxel-chunk-size` voxels for memory.
- Per-voxel scores are Pearson r between predicted and true held-out
  responses, returned to the original (un-z-scored) target scale.

## Analysis

`analyze_lag_preference.py` loads `lag_corrs.npz` and reports:

- Mean / median per-voxel r per lag, restricted to each frontal sub-ROI
  (`BA_10`, `BA_6`, `BA_8`, `BA_9_46`, `BROCA`).
- Within-lag z (a sub-ROI mean minus the full-frontal mean for that lag,
  divided by the full-frontal std). Positive ⇒ the sub-ROI is better than
  average for that lag.
- Distribution of preferred lags per ROI (`argmax_lag corrs[:, voxel]`).
- Mean tuning sharpness (`best_r - mean(other_lag_r)`) per ROI.
- Top-K (default 2000) most predicted voxels: which lag they prefer.

The CSV (`lag_preference_breakdown.csv`) is long-form (`metric, roi, lag, value`)
so it is easy to plot from a notebook.

## Cortical-surface flatmaps

`plot_lag_flatmaps.py` projects the saved per-voxel results back onto the
subject's surface using pycortex. Run it after `train_lag_encoding.py`:

```
python lag_preference_analysis/plot_lag_flatmaps.py \
    --results-dir lag_preference_analysis/results/S1__embedding__lags1-10__chunk1tr__seed0
```

It writes:

```
results/<tag>/flatmaps/
├── preferred_lag.png             # argmax_lag corrs per full_frontal voxel
├── preferred_lag_masked.png      # same, but voxels with best-lag r < threshold are NaN
├── com_lag.png                   # r-weighted centre-of-mass lag (smoother gradient)
├── com_lag_masked.png            # com_lag with the same mask applied
├── best_lag_r.png                # intensity map: r at each voxel's preferred lag
├── per_lag/lag<lag>.png          # (optional, --per-lag) per-lag r flatmaps
└── lag_preference_maps.npz       # arrays projected onto full-brain indices
```

Useful flags:

- `--mask-r-threshold 0.05` — voxels with best-lag r below this are masked
  (NaN ⇒ background curvature) in the `*_masked` figures.
- `--cmap-pref viridis` — colormap for lag preference.
- `--per-lag` — also dump one flatmap per individual lag (red/blue diverging).
- `--pycortex-filestore /path/to/pycortex-db` — point pycortex at a specific
  filestore. By default the script auto-uses `<repo>/pycortex-db` when present
  (run `python download_pycortex_files.py` if it's empty).
- `--n-total-voxels 81126` — full-volume voxel count; auto-detected from a
  response file when possible.

Voxels outside `BA_full_frontal` are NaN, so the rest of the cortex shows
curvature only and the gradient is easy to read.
