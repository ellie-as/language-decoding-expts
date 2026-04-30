# voxel_timescale

Per-voxel intrinsic timescale during story listening. The point: see which
cortical areas vary fast (small timescale) versus slowly (large timescale)
while a subject listens to natural speech.

For each voxel we compute the autocorrelation function (ACF) of the BOLD
timeseries and summarise its decay. Slow decay ⇒ slow voxel.

## Files

| Script | Purpose |
| --- | --- |
| `compute_voxel_timescale.py`           | Computes per-voxel ACF + timescale metrics whole-brain (or full_frontal). |
| `plot_voxel_timescale_flatmaps.py`     | Renders pycortex flatmaps for each metric. |

## Method

For each story:

1. Trim the silence: drop the first `5 + config.TRIM` TRs and the trailing
   `config.TRIM` TRs (default; `--trim none` disables).
2. Linear-detrend the timecourse per voxel (`--detrend linear`; `none` disables).
3. Compute the biased sample ACF up to `--max-lag-trs` via FFT.

ACFs are summed across stories (length-weighted) and divided by total weight.

Per-voxel metrics are then derived from the averaged ACF (lag 0 = 1 by
construction):

| Metric | Definition | Larger means |
| --- | --- | --- |
| `half_life` | First lag where ACF crosses 0.5, linear-interpolated. | slower |
| `exp_tau`   | Time-constant from a log-linear fit on lags 1..K (`--exp-fit-max-lag`). | slower |
| `integrated_ac` | Signed sum of ACF for lags 1..max_lag. | slower when positive |
| `positive_integrated_ac` | Sum of `max(ACF, 0)` for lags 1..max_lag. | slower |

All three are saved both in TR units (`*_trs`) and seconds (`*_seconds`,
default TR = 2 s, set via `--tr-seconds`). Voxels whose ACF never crosses 0.5
within `--max-lag-trs` get `half_life = NaN`; voxels whose ACF is non-positive
in the early lags get `exp_tau = NaN`.

## Usage

### Cluster (data on `/ceph/...`)

```
python voxel_timescale/compute_voxel_timescale.py --subject S1
python voxel_timescale/plot_voxel_timescale_flatmaps.py \
    --results-dir voxel_timescale/results/S1__all__lag30__detrend-linear__trim-huth
```

### Local with Ceph mounted at `/Volumes/ellie/language-decoding-expts`

```
python voxel_timescale/compute_voxel_timescale.py --subject S1 --local-compute-mode
python voxel_timescale/plot_voxel_timescale_flatmaps.py \
    --results-dir voxel_timescale/results/S1__all__lag30__detrend-linear__trim-huth \
    --pycortex-filestore /Users/eleanorspens/PycharmProjects/language-decoding-expts/pycortex-db
```

### Useful flags

- `--voxel-scope all|full_frontal` — whole brain (default) or BA_full_frontal.
- `--max-lag-trs 30` — ACF max lag (default 30 TRs ~ 60 s).
- `--exp-fit-max-lag 10` — lags 1..K used in the exponential fit.
- `--tr-seconds 2.0` — TR length, used to convert TR units → seconds.
- `--metrics half_life_seconds exp_tau_seconds positive_integrated_ac_seconds` — pick
  which flatmaps to render (default: those three).
- `--clip-quantiles 0.05 0.95` — auto-vmin/vmax from these quantiles of finite
  values (or pass `--vmin/--vmax` to override).

## Outputs

```
results/<tag>/
├── voxel_timescale.npz       # voxels, acf_avg, *_trs, *_seconds, ...
├── summary.json              # mean/median/p05/p95 per metric, story counts
├── roi_summary.csv           # only meaningful for --voxel-scope full_frontal
└── flatmaps/<metric>.png     # produced by plot_voxel_timescale_flatmaps.py
```

`<tag>` defaults to `<subject>__<scope>__lag<N>__detrend-<mode>__trim-<mode>`.

## Interpretation

- Primary auditory and early speech areas should be at the small end (fast,
  short timescales) — they track sub-second acoustic and word-level features
  that fluctuate every TR or two.
- Higher-association cortex (posterior temporal-parietal, lateral PFC, mPFC)
  is expected to sit at the large end — these regions integrate over many
  seconds of context and so their BOLD signal varies more slowly.
- The metrics are correlated but not identical:
  - `half_life` is the most robust to slow drifts (looks only at the initial
    decay).
  - `exp_tau` is most directly comparable to the Honey/Hasson "intrinsic
    timescale" measure but assumes a roughly exponential decay.
  - `positive_integrated_ac` is a smooth scalar useful for percentile-mapping
    when the ACF undershoots below zero.

If you see runaway / very large `exp_tau_seconds` values in a few voxels, that
usually means the early ACF is roughly flat (`slope ≈ 0`); falling back to
`half_life_seconds` or `integrated_ac_seconds` for those voxels is fine.
