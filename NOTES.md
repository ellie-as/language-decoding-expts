# Ellie / Svenja experiment notes

We are using the pretrained models from [Tang et al. (2023)](https://www.nature.com/articles/s41593-023-01304-9) to ask **which subregions of frontal cortex drive semantic decoding predictions**.

## Data preparation

1. **Install dependencies** (on the CUDA server):

```bash
pip install -r requirements.txt
```

2. **Download training data** (LM checkpoint, training stimuli + fMRI responses):

```bash
python test_pretrained.py download-train
```

This downloads the GPT checkpoint into `data_lm/`, training metadata from [Box](https://utexas.box.com/shared/static/3go1g4gcdar2cntjit2knz5jwr3mvxwe.zip) into `data_train/`, and training stimuli + fMRI responses from [OpenNeuro ds003020](https://openneuro.org/datasets/ds003020/).

3. **Download test data** (test stimuli + fMRI responses):

```bash
python test_pretrained.py download-test
```

This downloads test brain responses + transcripts from [OpenNeuro ds004510](https://openneuro.org/datasets/ds004510/) into `data_test/`.

4. **Download pretrained models** manually from [Box](https://utexas.app.box.com/s/ri13t06iwpkyk17h8tfk0dtyva7qtqlz). Unzip into `models/` so the layout is:

```
models/
  S1/
    encoding_model_perceived.npz
    word_rate_model_auditory.npz
    word_rate_model_speech.npz
  S2/...
  S3/...
```

5. **Verify** everything is in place:

```bash
python test_pretrained.py list
```

## Running

**Decode** (beam width 200 is full quality; 50 is a quick smoke test):

```bash
python test_pretrained.py run --subject S1 --experiment perceived_speech --task wheretheressmoke
```

**Attribution analysis** on saved decoder output:

```bash
python run_attribution.py --subject S1 --experiment perceived_speech --task wheretheressmoke --use-saved --rois frontal_rois_UTS01.json
```

The subject-to-ROI mapping is S1 = UTS01, S2 = UTS02, S3 = UTS03.

## Frontal ROI files

`frontal_rois_UTS0{1,2,3}.json` split frontal cortex into three posterior-to-anterior strips using Desikan-Killiany labels from each subject's FreeSurfer parcellation:

| ROI | Regions |
|-----|---------|
| **posterior_frontal** | precentral, paracentral, caudal middle frontal |
| **middle_frontal** | rostral middle frontal, pars opercularis, pars triangularis, caudal anterior cingulate |
| **anterior_frontal** | superior frontal, pars orbitalis, lateral/medial orbitofrontal, frontal pole, rostral anterior cingulate |

## Analysis plan

### Step 1: Attribution + lag decomposition

```bash
python run_attribution.py --subject S1 --experiment perceived_speech \
    --task wheretheressmoke --use-saved --rois frontal_rois_UTS01.json
```

This now saves per-lag attribution (`lag_attr`) and weight energy fractions (`lag_weights`) alongside the existing per-voxel and per-word arrays. The FIR delays are 1–4 TRs (2–8 seconds), decomposed exactly:

```
attr[j] = Σ_k attr_lag_k[j] + attr_resp[j]
```

where `attr_lag_k[j] = -0.5 Σ_t pred_lag_k[t,j] × (Σ⁻¹ diff)[t,j]`.

### Step 2: Lexical feature analysis

```bash
python run_analysis.py --subject S1 --experiment perceived_speech \
    --task wheretheressmoke --rois frontal_rois_UTS01.json --surprisal
```

Computes word length, log-frequency, and (optionally) GPT-2 surprisal for each decoded word, then regresses per-word regional attribution against these features. Key outputs:

- **Lag × region table**: Does anterior frontal peak at longer lags (higher-order, longer temporal context) vs posterior frontal at short lags (articulatory)?
- **Feature × region betas**: Does surprisal predict attribution more in anterior regions (predictive processing) while word length predicts posterior (articulatory)?
- **Peak lag by region**: Which temporal delay dominates each frontal subregion.

### Analysis ideas

1. **Frontal gradient of attribution** — Does the encoding model rely more on posterior frontal (motor/premotor) or anterior frontal (prefrontal) voxels?

2. **Temporal hierarchy** — The lag decomposition tests whether anterior frontal has disproportionate weight at long FIR delays (6–8s history) while posterior frontal peaks at short delays (2–4s). This would indicate a posterior-to-anterior gradient of temporal integration, consistent with hierarchical predictive processing.

3. **Word-level dissociations** — Do word properties (surprisal, frequency, length) predict which regions contribute most to decoding each word?

4. **Discriminability vs attribution** — Voxels can be informative (high discriminability) without being well-fit by the model (low attribution). Comparing these across regions could reveal frontal areas that *encode* language but are underexploited by the linear decoder.
