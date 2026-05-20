# Podcast ECoG Boundary Decoding Analyses

This note describes how to reproduce the sentence-boundary word-decoding result and the follow-up controls.

## Inputs

Dataset root on the cluster:

```bash
BIDS_ROOT=/ceph/behrens/ellie/language-decoding-expts/podcast_ecog/data/ds005574
```

Local mounted equivalent:

```bash
BIDS_ROOT=/Volumes/ellie/language-decoding-expts/podcast_ecog/data/ds005574
```

LLM-derived sentence and event boundaries:

```bash
podcast_ecog/outputs/transcript_boundaries.json
```

This JSON contains:

- `sentence_end_word_indices`
- `event_end_word_indices`

## 1. Build Neural Cache

This creates all-channel central-window neural caches and static word-vector PCA targets. The later N-back scripts reuse the neural cache.

```bash
python podcast_ecog/test_subject_word2vec_rank_decodability.py \
  --bids-root "$BIDS_ROOT" \
  --output-dir podcast_ecog/outputs/subject_static_rank_decodability \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --embedding-source dataset_static \
  --dataset-feature-space en_core_web_lg \
  --n-components 5 \
  --ranks 0 1 2 4 8 16 32 \
  --target-modes raw_pca \
  --n-permutations 100 \
  --min-shift 200 \
  --save-cache
```

## 2. N-Back Word Decoding

For each subject, we used all ECoG channels to decode the static word embedding of the word `N` words back.

The word embeddings were reduced to 5D with PCA. We trained simple ridge decoders with held-out cross-validation.

For each current word `t`, the decoder target was:

```text
N = 1: word t-1
N = 2: word t-2
...
N = 10: word t-10
```

Run:

```bash
python podcast_ecog/run_best_subject_nback_boundary_decoding.py \
  --bids-root "$BIDS_ROOT" \
  --source-output-dir podcast_ecog/outputs/subject_static_rank_decodability \
  --output-dir podcast_ecog/outputs/all_subject_nback_sentence_boundary_decoding \
  --boundary-json podcast_ecog/outputs/transcript_boundaries.json \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --n-back 0 1 2 3 4 5 6 7 8 9 10 \
  --n-components 5 \
  --ridge-alpha 1000 \
  --outer-splits 5
```

Held-out predictions are saved in:

```text
podcast_ecog/outputs/all_subject_nback_sentence_boundary_decoding/prediction_cache/
```

## Word-Lag Decoder Model Sweep

To check whether the N-back word-decoding results depend on using ridge, we ran a decoder-family sweep for the same basic target: dataset static `en_core_web_lg` word vectors reduced to 5D with PCA. The input was all channels for each subject in the cached central word window. The main score was `mean_point_r`, the per-word correlation between the true and predicted 5D vectors, averaged across held-out words.

Main run:

```bash
python podcast_ecog/sweep_word_lag_decoder_models.py \
  --bids-root "$BIDS_ROOT" \
  --source-output-dir /Volumes/ellie/language-decoding-expts/podcast_ecog/outputs/subject_static_rank_decodability \
  --output-dir podcast_ecog/outputs/word_lag_decoder_model_sweep \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --ranks 0 1 2 3 4 5 6 7 8 9 10 \
  --directions past \
  --models mean ridgecv ridge_1000 pls2 pls5 pcridge20 pcridge50 rff_ridge extra_trees \
  --outer-splits 5 \
  --trees 60 \
  --tree-max-depth 10 \
  --rff-components 128
```

Supplementary slow-model screen:

```bash
python podcast_ecog/sweep_word_lag_decoder_models.py \
  --bids-root "$BIDS_ROOT" \
  --source-output-dir /Volumes/ellie/language-decoding-expts/podcast_ecog/outputs/subject_static_rank_decodability \
  --output-dir podcast_ecog/outputs/word_lag_decoder_model_sweep_slow_screen \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --ranks 0 1 2 5 10 \
  --directions past \
  --models elasticnet mlp \
  --outer-splits 5
```

Main outputs:

```text
podcast_ecog/outputs/word_lag_decoder_model_sweep/word_lag_decoder_model_subject_summary.csv
podcast_ecog/outputs/word_lag_decoder_model_sweep/word_lag_decoder_model_group_summary.csv
podcast_ecog/outputs/word_lag_decoder_model_sweep/word_lag_decoder_model_best_by_rank.csv
podcast_ecog/outputs/word_lag_decoder_model_sweep/past_mean_point_r_model_rank_heatmap.png
podcast_ecog/outputs/word_lag_decoder_model_sweep/past_mean_point_r_model_rank_lines.png
```

Result: full ridge remained the best overall choice. `ridge_1000` and `ridgecv` were tied at near lags, but fixed `ridge_1000` was more stable for weak distant targets. `PLS5` was the closest competitor and slightly beat ridge at some distant lags, but not at the stronger near lags. PCA-before-ridge, random Fourier ridge, extra trees, and a small MLP did not improve the group result. ElasticNet was close to ridge on the screened ranks, but produced convergence warnings and did not beat ridge.

Group `mean_point_r` by lag:

```text
N   best model   mean_point_r
0   ridge_1000   0.1168
1   ridge_1000   0.1174
2   ridgecv      0.1006
3   ridge_1000   0.0813
4   ridge_1000   0.0598
5   ridge_1000   0.0429
6   ridge_1000   0.0350
7   PLS5         0.0292
8   PLS5         0.0225
9   PLS5         0.0178
10  PLS5         0.0149
```

The important interpretation is that the earlier ridge-based word analyses are not an artifact of using an obviously weak decoder. Ridge is at or near the top, and nonlinear models did not reveal hidden decodability. For long-lag effects, fixed regularization is preferable to the automatic `ridgecv` setting used here.

### GloVe Target Repeat

We repeated the same model sweep using `glove-wiki-gigaword-300` vectors from `gensim`, reduced to 5D with PCA. The GloVe cache is local:

```text
podcast_ecog/resources/gensim_data/
```

Run:

```bash
python podcast_ecog/sweep_word_lag_decoder_models.py \
  --bids-root "$BIDS_ROOT" \
  --source-output-dir /Volumes/ellie/language-decoding-expts/podcast_ecog/outputs/subject_static_rank_decodability \
  --output-dir podcast_ecog/outputs/word_lag_decoder_model_sweep_glove \
  --embedding-source gensim_glove \
  --gensim-model glove-wiki-gigaword-300 \
  --gensim-data-dir podcast_ecog/resources/gensim_data \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --ranks 0 1 2 3 4 5 6 7 8 9 10 \
  --directions past \
  --models mean ridgecv ridge_1000 pls2 pls5 pcridge20 pcridge50 rff_ridge extra_trees \
  --outer-splits 5 \
  --trees 60 \
  --tree-max-depth 10 \
  --rff-components 128
```

Slow-model screen:

```bash
python podcast_ecog/sweep_word_lag_decoder_models.py \
  --bids-root "$BIDS_ROOT" \
  --source-output-dir /Volumes/ellie/language-decoding-expts/podcast_ecog/outputs/subject_static_rank_decodability \
  --output-dir podcast_ecog/outputs/word_lag_decoder_model_sweep_glove_slow_screen \
  --embedding-source gensim_glove \
  --gensim-model glove-wiki-gigaword-300 \
  --gensim-data-dir podcast_ecog/resources/gensim_data \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --ranks 0 1 2 5 10 \
  --directions past \
  --models elasticnet mlp \
  --outer-splits 5
```

Main outputs:

```text
podcast_ecog/outputs/word_lag_decoder_model_sweep_glove/word_lag_decoder_model_group_summary.csv
podcast_ecog/outputs/word_lag_decoder_model_sweep_glove/word_lag_decoder_model_best_by_rank.csv
podcast_ecog/outputs/word_lag_decoder_model_sweep_glove/past_mean_point_r_model_rank_heatmap.png
podcast_ecog/outputs/word_lag_decoder_model_sweep_glove_slow_screen/word_lag_decoder_model_group_summary.csv
```

GloVe coverage was lower than the dataset static vectors:

```text
5025 / 5136 words = 97.8%
```

Best GloVe model by lag:

```text
N   best model   mean_point_r
0   ridge_1000   0.0867
1   ridge_1000   0.0910
2   ridgecv      0.0856
3   ridge_1000   0.0715
4   ridge_1000   0.0491
5   ridge_1000   0.0366
6   ridge_1000   0.0240
7   PLS5         0.0201
8   PLS5         0.0153
9   ridge_1000   0.0162
10  ridge_1000   0.0117
```

Compared with the dataset static vectors, GloVe was weaker in `mean_point_r` at near lags, but similar in `weighted_pc_r`. For `ridge_1000`, `N=1` was `0.0910` with GloVe versus `0.1174` with the dataset static vectors. The model-ranking conclusion was unchanged: ridge remains the safest default; `PLS5` is competitive at weaker distant lags; nonlinear models did not improve performance.

## Static Word-Vector ROI Past/Future Heatmap

This is the word2vec-style plot that tests static lexical representations over a wider temporal range, from 15 words in the past to 15 words in the future.

It uses the dataset's bundled static `en_core_web_lg` vectors, reduces them to 10D with PCA, builds pseudo-populations for the six paper ROIs across all 9 subjects, and trains held-out ridge decoders from central-window neural activity. This is not the 5D N-back boundary analysis above; it uses `--n-components 10`.

Important convention:

```text
positive rank  = past target,   e.g. rank 1 means word t-1
rank 0         = current word,  word t
negative rank  = future target, e.g. rank -1 means word t+1
```

Run:

```bash
python podcast_ecog/test_roi_rank_pca_decodability.py \
  --bids-root "$BIDS_ROOT" \
  --roi-metrics podcast_ecog/outputs/gpt2_paper_roi_context_layer_exact/channel_paper_roi_metrics.csv \
  --output-dir podcast_ecog/outputs/roi_rank_static_pca_decodability_future_past_m15_15 \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --rois EAC STG IFG PRC MFG TMP \
  --task podcast \
  --embedding-source dataset_static \
  --dataset-feature-space en_core_web_lg \
  --n-components 10 \
  --ranks -15 -14 -13 -12 -11 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --target-modes raw_pca residual_pca \
  --window-start -0.5 \
  --window-end 0.5 \
  --ridge-alpha 1000 \
  --outer-splits 5 \
  --random-seed 0 \
  --cache-dir podcast_ecog/outputs/roi_rank_pca_decodability/pseudo_cache \
  --reuse-cache
```

Main outputs:

```text
podcast_ecog/outputs/roi_rank_static_pca_decodability_future_past_m15_15/raw_pca_weighted_pc_r_heatmap.png
podcast_ecog/outputs/roi_rank_static_pca_decodability_future_past_m15_15/residual_pca_weighted_pc_r_heatmap.png
podcast_ecog/outputs/roi_rank_static_pca_decodability_future_past_m15_15/raw_pca_mean_pc_r_heatmap.png
podcast_ecog/outputs/roi_rank_static_pca_decodability_future_past_m15_15/residual_pca_mean_pc_r_heatmap.png
podcast_ecog/outputs/roi_rank_static_pca_decodability_future_past_m15_15/roi_rank_pca_decodability_summary.csv
```

Notes from the local run:

```text
embedding coverage = 5122 / 5136 words, 99.7%
valid samples per ROI = 4692
```

The clearest pattern is that static lexical information is strongest around the current and immediately previous words. In the raw PCA heatmap, the largest cells were:

```text
STG, t-1   weighted PC r = 0.168
IFG, t-1   weighted PC r = 0.163
STG, t     weighted PC r = 0.161
IFG, t     weighted PC r = 0.152
STG, t-2   weighted PC r = 0.142
IFG, t-2   weighted PC r = 0.136
```

Future-word decoding is mostly near zero, as expected for static non-contextual word vectors. The residual PCA plot is stricter: it asks what remains at each word position after removing linear information shared with the other tested positions.

## Word-Boundary Phoneme-Rank Decoding

This is the phoneme-level version of the past/future control, but it does not require phoneme timings. Instead, neural activity is anchored at each word onset and the target is defined by ordered phoneme position around that word boundary. For the main version, the neural feature is the existing `central_allch` cache from `test_subject_word2vec_rank_decodability.py`. That cache was built with the script defaults, `--window-start -0.5 --window-end 0.5`, so the high-gamma feature is symmetric around word onset.

For the boundary before word `t`:

```text
past rank N   = Nth phoneme before word t starts
future rank N = Nth phoneme after word t starts

N = 1 past    = final phoneme of the previous word
N = 1 future  = first phoneme of the current word
```

The script uses the dataset's word timings and phonetic transcript tokenization, then gets ordered pronunciations from CMUdict. This is different from the dataset's bundled `phonetic` feature matrix, which only contains word-level phoneme presence/absence and therefore cannot by itself tell us the Nth phoneme before or after a boundary.

Run on the cluster:

```bash
python podcast_ecog/run_word_boundary_phoneme_rank_decoding.py \
  --bids-root /ceph/behrens/ellie/language-decoding-expts/podcast_ecog/data/ds005574 \
  --source-output-dir /ceph/behrens/ellie/language-decoding-expts/podcast_ecog/outputs/subject_static_rank_decodability \
  --roi-metrics podcast_ecog/outputs/gpt2_paper_roi_context_layer_exact/channel_paper_roi_metrics.csv \
  --output-dir podcast_ecog/outputs/word_boundary_phoneme_rank_decoding \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --rois EAC STG IFG PRC MFG TMP \
  --phoneme-ranks 1 2 3 4 5 6 7 8 9 10 \
  --ridge-alpha 1000 \
  --outer-splits 5 \
  --download-cmudict
```

The script also has a `--use-windowed-neural` option to rebuild the neural window directly from the raw high-gamma FIF files, but we do not need that here because the existing `central_allch` cache already corresponds to the symmetric `-0.5..+0.5 s` word-onset window.

If the cluster cannot download NLTK data, download CMUdict once somewhere else and pass `--cmudict-path /path/to/cmudict`, or pre-populate:

```bash
python -m nltk.downloader -d podcast_ecog/resources/nltk_data cmudict
```

Main outputs:

```text
podcast_ecog/outputs/word_boundary_phoneme_rank_decoding/word_pronunciations.csv
podcast_ecog/outputs/word_boundary_phoneme_rank_decoding/phoneme_sequence.csv
podcast_ecog/outputs/word_boundary_phoneme_rank_decoding/phoneme_vocabulary.csv
podcast_ecog/outputs/word_boundary_phoneme_rank_decoding/phoneme_rank_decoding_subject_summary.csv
podcast_ecog/outputs/word_boundary_phoneme_rank_decoding/phoneme_rank_decoding_group_summary.csv
podcast_ecog/outputs/word_boundary_phoneme_rank_decoding/phoneme_rank_future_controlled_delta_summary.csv
podcast_ecog/outputs/word_boundary_phoneme_rank_decoding/roi_phoneme_signed_rank_mean_point_r_heatmap.png
podcast_ecog/outputs/word_boundary_phoneme_rank_decoding/roi_phoneme_signed_rank_accuracy_heatmap.png
podcast_ecog/outputs/word_boundary_phoneme_rank_decoding/roi_phoneme_rank_future_controlled_mean_point_r_delta_heatmap.png
podcast_ecog/outputs/word_boundary_phoneme_rank_decoding/roi_phoneme_rank_future_controlled_accuracy_delta_heatmap.png
```

Interpretation:

```text
negative signed ranks = future phonemes after the boundary
positive signed ranks = past phonemes before the boundary
future-controlled delta = past-rank decoding score - matched future-rank decoding score
```

A one-subject smoke test using `sub-01`, `ALL` channels, and rank 1 gave about 97.9% word pronunciation coverage, 39 phoneme types, and nonzero held-out phoneme decoding. This verifies the method and file plumbing, but the full result should be interpreted from the all-subject ROI run above.

Completed all-subject cache-based run and metric check:

```text
output dir = podcast_ecog/outputs/word_boundary_phoneme_rank_decoding
word pronunciation coverage = 5028 / 5136 = 97.9%
phoneme sequence length = 17705
phoneme types = 39
valid word-boundary anchors = 4504
```

Important correction: the first raw `mean_point_r` heatmaps were vertically stripy because the metric was dominated by phoneme-position priors. A fold-prior decoder with no neural data reproduced the raw past/future stripe pattern almost exactly.

```text
rank   raw ROI-mean delta   fold-prior baseline delta   raw - baseline
1      0.0355               0.0358                     -0.0003
2      0.0178               0.0182                     -0.0004
3     -0.0190              -0.0179                     -0.0011
4     -0.0072              -0.0068                     -0.0003
5     -0.0036              -0.0038                      0.0002
6      0.0054               0.0052                      0.0002
7     -0.0060              -0.0059                     -0.0001
8      0.0069               0.0060                      0.0009
9     -0.0071              -0.0072                      0.0001
10     0.0019               0.0017                      0.0002
```

The baseline-corrected outputs are:

```text
podcast_ecog/outputs/word_boundary_phoneme_rank_decoding/phoneme_rank_decoding_subject_summary_baseline_corrected.csv
podcast_ecog/outputs/word_boundary_phoneme_rank_decoding/phoneme_rank_decoding_group_summary_baseline_corrected.csv
podcast_ecog/outputs/word_boundary_phoneme_rank_decoding/phoneme_rank_future_controlled_delta_summary_baseline_corrected.csv
podcast_ecog/outputs/word_boundary_phoneme_rank_decoding/roi_phoneme_signed_rank_mean_point_r_adjusted_heatmap.png
podcast_ecog/outputs/word_boundary_phoneme_rank_decoding/roi_phoneme_rank_future_controlled_mean_point_r_adjusted_delta_heatmap.png
```

Interpretation: the raw phoneme-rank result should not be treated as neural evidence. After subtracting the fold-prior baseline, the past-vs-future deltas are near zero and no ROI/rank cell survives the subject sign-flip FDR test for adjusted `mean_point_r`. The right next version of this analysis should use a chance-corrected/classification metric from the start, or a target representation that is not so dominated by rank-specific phoneme base rates.

### Proper Phoneme Identity Classifier

The classifier version is:

```bash
python podcast_ecog/run_word_boundary_phoneme_identity_decoding.py \
  --bids-root /ceph/behrens/ellie/language-decoding-expts/podcast_ecog/data/ds005574 \
  --source-output-dir /ceph/behrens/ellie/language-decoding-expts/podcast_ecog/outputs/subject_static_rank_decodability \
  --roi-metrics podcast_ecog/outputs/gpt2_paper_roi_context_layer_exact/channel_paper_roi_metrics.csv \
  --output-dir podcast_ecog/outputs/word_boundary_phoneme_identity_decoding \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --rois EAC STG IFG PRC MFG TMP \
  --phoneme-ranks 1 2 3 4 5 6 7 8 9 10 \
  --outer-splits 5 \
  --logistic-c 1.0 \
  --max-iter 1000 \
  --top-k 2 5 \
  --download-cmudict
```

This trains a class-balanced multinomial logistic regression for each subject, ROI, direction, and rank. It scores against a fold-wise prior baseline:

```text
balanced_accuracy_adjusted = model balanced accuracy - prior balanced accuracy
log_loss_improvement       = prior log loss - model log loss
top-k adjusted             = model top-k accuracy - prior top-k accuracy
```

Main outputs:

```text
podcast_ecog/outputs/word_boundary_phoneme_identity_decoding/phoneme_identity_subject_summary.csv
podcast_ecog/outputs/word_boundary_phoneme_identity_decoding/phoneme_identity_group_summary.csv
podcast_ecog/outputs/word_boundary_phoneme_identity_decoding/phoneme_identity_future_controlled_delta_summary.csv
podcast_ecog/outputs/word_boundary_phoneme_identity_decoding/phoneme_identity_signed_rank_prior_adjusted_stats.csv
podcast_ecog/outputs/word_boundary_phoneme_identity_decoding/roi_phoneme_identity_signed_rank_balanced_accuracy_adjusted_heatmap.png
podcast_ecog/outputs/word_boundary_phoneme_identity_decoding/roi_phoneme_identity_signed_rank_log_loss_improvement_heatmap.png
podcast_ecog/outputs/word_boundary_phoneme_identity_decoding/roi_phoneme_identity_signed_rank_balanced_accuracy_adjusted_prior_adjusted_stats_heatmap.png
podcast_ecog/outputs/word_boundary_phoneme_identity_decoding/roi_phoneme_identity_future_controlled_balanced_accuracy_adjusted_delta_heatmap.png
```

Result from the local run against mounted Ceph data:

```text
ROI mean balanced_accuracy_adjusted across signed ranks
EAC    0.00158
IFG    0.00123
MFG   -0.00015
PRC   -0.00048
STG    0.00310
TMP    0.00001
```

The best individual group cells were small:

```text
ROI  signed rank  direction  mean balanced_accuracy_adjusted
STG      -1        future     0.01200
STG      -2        future     0.00928
STG       1        past       0.00696
MFG       1        past       0.00555
EAC       1        past       0.00503
```

No ROI/rank cell survived FDR correction for above-prior balanced accuracy, and no past-minus-future balanced-accuracy delta survived FDR. Log-loss improvement was negative, which means the class-balanced logistic model's probability estimates were worse than the fold-prior baseline. That does not prove phoneme information is absent, but it means this particular word-boundary-anchored classifier gives at most weak evidence for ordered phoneme identity decoding. A more sensitive next pass would tune the classifier/calibration, test narrower windows, or add actual phoneme timings.

## 3. Sentence-Boundary Split

We split held-out predictions into:

```text
no boundary:        word t-N and word t are in the same sentence
boundary in last N: a sentence boundary occurred between word t-N and word t
```

Decoding is better when the decoded word is still within the same sentence.

Example group means:

```text
N   all     no boundary   boundary
1   0.112   0.112         0.071
2   0.099   0.101         0.046
3   0.081   0.083         0.049
5   0.047   0.049         0.021
10  0.013   0.019        -0.008
```

Summary table:

```text
podcast_ecog/outputs/all_subject_nback_sentence_boundary_decoding/nback_sentence_boundary_decoding_summary.csv
```

## 4. Sentence-Boundary Permutation Test

Test statistic:

```text
delta_N = decoding r with no boundary - decoding r with boundary in last N
```

A positive value means decoding drops across sentence boundaries.

To build a null distribution, we kept the held-out predictions fixed and circularly shifted the sentence-boundary pattern along the transcript. This preserves the number of boundaries and the spacing between them, but destroys their true alignment with the words.

For each shifted boundary pattern, we recomputed `delta_N`. Then we asked whether the real sentence boundaries produced a larger delta than shifted boundaries.

Run:

```bash
python podcast_ecog/permutation_test_sentence_boundary_nback.py \
  --nback-output-dir podcast_ecog/outputs/all_subject_nback_sentence_boundary_decoding \
  --boundary-json podcast_ecog/outputs/transcript_boundaries.json \
  --boundary-level sentence \
  --output-dir podcast_ecog/outputs/sentence_boundary_nback_permutation_test \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --n-back 1 2 3 4 5 6 7 8 9 10 \
  --n-permutations 5000 \
  --metric weighted_pc_r
```

Quick 500-permutation result:

```text
observed mean delta = mean(delta_1, ..., delta_10) = 0.0318
shifted-null 95% interval = roughly [-0.0165, 0.0166]
two-sided p = 0.002
```

This suggests the sentence-boundary effect is unlikely to be explained by generic position in the transcript. Remaining possible confounds include pauses, lexical features near sentence starts/ends, word duration, surprisal, and acoustic/prosodic resets.

## 5. Coarse Event Boundaries

The same permutation analysis can be run with `event_end_word_indices`.

```bash
python podcast_ecog/permutation_test_sentence_boundary_nback.py \
  --nback-output-dir podcast_ecog/outputs/all_subject_nback_sentence_boundary_decoding \
  --boundary-json podcast_ecog/outputs/transcript_boundaries.json \
  --boundary-level event \
  --output-dir podcast_ecog/outputs/event_boundary_nback_permutation_test \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --n-back 1 2 3 4 5 6 7 8 9 10 \
  --n-permutations 5000 \
  --metric weighted_pc_r \
  --min-condition-samples 20
```

Event-boundary summary:

```text
N   no-event-boundary   event-boundary   delta   one-sided p
2   0.0986              0.0632           0.035   0.229
3   0.0815              0.0190           0.062   0.057
4   0.0630              0.0106           0.052   0.073
5   0.0473             -0.0027           0.050   0.064
6   0.0364             -0.0049           0.041   0.074
7   0.0297             -0.0114           0.041   0.058
8   0.0219             -0.0273           0.049   0.021
9   0.0150             -0.0187           0.034   0.067
10  0.0136             -0.0087           0.022   0.142
```

Omnibus over `N=2..10`:

```text
observed mean delta = 0.0431
shifted-null 95% interval = [-0.0457, 0.0430]
one-sided p = 0.0250
two-sided p = 0.0612
```

This is directionally similar to the sentence-boundary effect, but less decisive because there are only 13 usable event transitions.

## 6. Pointwise Decoding Score

For some follow-up analyses, we used a per-sample score:

```text
true vector      = actual 5D word embedding for word t-N
predicted vector = decoder's predicted 5D vector from neural activity at word t
point_r          = corr(true_5D_vector, predicted_5D_vector)
```

`point_r` is useful for sample-level regressions and boundary-wise plots. The main decoding tables use `weighted_pc_r`, which is more stable for overall model performance.

## 7. Semantic-Similarity Control

Possible confound: within-sentence word pairs are more semantically related than across-boundary word pairs. If neural activity mainly tracks semantic context, that could mimic a boundary effect.

First, generate the sample-level semantic-similarity table:

```bash
python podcast_ecog/plot_decoding_vs_semantic_similarity.py \
  --bids-root "$BIDS_ROOT" \
  --nback-output-dir podcast_ecog/outputs/all_subject_nback_sentence_boundary_decoding \
  --boundary-json podcast_ecog/outputs/transcript_boundaries.json \
  --output-dir podcast_ecog/outputs/decoding_vs_semantic_similarity \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --n-back 1 2 3 4 5 6 7 8 9 10
```

Then fit:

```text
point_r ~ boundary_crossed + semantic_similarity + C(N) + C(subject)
```

Reproduction snippet:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

out_dir = Path("podcast_ecog/outputs/semantic_similarity_boundary_regression")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("podcast_ecog/outputs/decoding_vs_semantic_similarity/decoding_vs_semantic_similarity_samples.csv")
df = df.rename(columns={"n_back": "N"})
df["boundary_crossed"] = df["crossed_sentence_boundary"].astype(int)
df["subject"] = df["subject"].astype(str)

model_df = df[["point_r", "boundary_crossed", "semantic_similarity", "N", "subject"]].dropna()
fit = smf.ols(
    "point_r ~ boundary_crossed + semantic_similarity + C(N) + C(subject)",
    data=model_df,
).fit()

coef = pd.DataFrame({
    "term": fit.params.index,
    "coef": fit.params.values,
    "std_err": fit.bse.values,
    "t": fit.tvalues.values,
    "p": fit.pvalues.values,
})
coef.to_csv(out_dir / "point_r_boundary_semantic_ols_coefficients.csv", index=False)
print(coef[coef["term"].isin(["boundary_crossed", "semantic_similarity"])])
PY
```

Result:

```text
boundary_crossed     beta = -0.0266
semantic_similarity  beta =  0.0320
```

If semantic similarity explained the boundary effect, the boundary coefficient should become much smaller, possibly near zero.

Comparison:

```text
without semantic similarity:
boundary beta = -0.0274

with semantic similarity:
boundary beta = -0.0266
```

So semantic relatedness does not explain away the boundary effect in this fixed-effects regression. However, OLS p-values are optimistic because samples are temporally autocorrelated and repeated across subjects/words. For final inference, use a circular boundary-shift permutation on the regression coefficient.

## 8. Sentence Surprise Quartiles

We also asked whether the boundary effect varies with GPT-2 surprise of sentence 2 given sentence 1.

```bash
python podcast_ecog/correlate_sentence_perplexity_boundary_delta.py \
  --nback-output-dir podcast_ecog/outputs/all_subject_nback_sentence_boundary_decoding \
  --word-table podcast_ecog/outputs/boundary_decoding_suite_no_phoneme/time_word/word_table.csv \
  --boundary-json podcast_ecog/outputs/transcript_boundaries.json \
  --output-dir podcast_ecog/outputs/sentence_perplexity_boundary_delta \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --n-back 1 2 3 4 5 6 7 8 9 10 \
  --model gpt2 \
  --device auto \
  --local-files-only
```

Continuous boundary-wise correlations were weak. A more interpretable analysis split sentence transitions into GPT-2 conditional-surprise quartiles:

```bash
python podcast_ecog/sentence_surprise_quartile_boundary_delta.py \
  --nback-output-dir podcast_ecog/outputs/all_subject_nback_sentence_boundary_decoding \
  --perplexity-dir podcast_ecog/outputs/sentence_perplexity_boundary_delta \
  --output-dir podcast_ecog/outputs/sentence_surprise_quartile_boundary_delta \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --n-back 1 2 3 4 5 6 7 8 9 10
```

Mean delta over `N=1..10`:

```text
Quartile   mean NLL   delta
Q1         3.57      -0.024
Q2         4.14       0.033
Q3         4.68       0.018
Q4         5.58       0.028
```

The main pattern is that the lowest-surprise sentence transitions show little or no boundary drop, whereas the other quartiles show a positive drop. This is exploratory.

## 9. Future-Controlled Boundary Drop

A further control asks whether the boundary drop is specific to decoding past words, or whether it also appears when decoding future words.

For each current word `t`, we trained two matched decoders from the same neural activity:

```text
past target:    word t-N
future target:  word t+N
```

Then we computed:

```text
past delta       = within-segment point_r - across-boundary point_r, for word t-N
future delta     = within-segment point_r - across-boundary point_r, for word t+N
controlled delta = past delta - future delta
```

This controls for symmetric effects of local semantic structure. If nearby words are just more similar within a sentence than across a boundary, a similar boundary drop should appear for future-word decoding.

Run the sentence-boundary control:

```bash
python podcast_ecog/run_time_reversal_boundary_control.py \
  --bids-root "$BIDS_ROOT" \
  --source-output-dir podcast_ecog/outputs/subject_static_rank_decodability \
  --boundary-json podcast_ecog/outputs/transcript_boundaries.json \
  --output-dir podcast_ecog/outputs/time_reversal_boundary_control \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --boundary-level sentence \
  --dataset-feature-space en_core_web_lg \
  --n-components 5 \
  --n-back 1 2 3 4 5 6 7 8 9 10 \
  --ridge-alpha 1000 \
  --outer-splits 5 \
  --save-predictions
```

When running locally with Ceph mounted, `--bids-root` and `--source-output-dir` can point into `/Volumes/ellie/...`, but keep `--output-dir` local.

Sentence-boundary result, averaged across subjects and `N=1..10`:

```text
past delta                 0.0287
future-control delta        0.0212
past - future controlled    0.0075
```

By `N`:

```text
N   past_delta   future_delta   controlled
1   0.0481       0.0224         0.0257
2   0.0641       0.0404         0.0237
3   0.0408       0.0291         0.0117
4   0.0138       0.0292        -0.0154
5   0.0038       0.0278        -0.0240
6   0.0201       0.0189         0.0012
7   0.0219       0.0121         0.0098
8   0.0196       0.0133         0.0063
9   0.0216       0.0131         0.0085
10  0.0332       0.0061         0.0271
```

Regression:

```text
point_r ~ direction_past * boundary_crossed + semantic_similarity + C(N) + direction_past:C(N) + C(subject)

direction_past:boundary_crossed beta = -0.00646
OLS p = 0.043
```

The negative interaction means crossing a boundary hurts past-word decoding slightly more than future-word decoding. This weakens the original unqualified sentence-boundary claim, because future decoding also drops across sentence boundaries, but leaves a small past-specific residual. The OLS p-value is a screening statistic; a final claim should use a boundary-shift permutation or block bootstrap.

Run the same future-control analysis for coarse event boundaries:

```bash
python podcast_ecog/run_time_reversal_boundary_control.py \
  --bids-root "$BIDS_ROOT" \
  --source-output-dir podcast_ecog/outputs/subject_static_rank_decodability \
  --boundary-json podcast_ecog/outputs/transcript_boundaries.json \
  --output-dir podcast_ecog/outputs/time_reversal_event_boundary_control \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --boundary-level event \
  --dataset-feature-space en_core_web_lg \
  --n-components 5 \
  --n-back 1 2 3 4 5 6 7 8 9 10 \
  --ridge-alpha 1000 \
  --outer-splits 5 \
  --save-predictions
```

Coarse event-boundary result:

```text
N   past_delta   future_delta   controlled
2   0.1005       0.0194         0.0811
3   0.1053      -0.0035         0.1087
4   0.0382       0.0026         0.0356
5   0.0182       0.0121         0.0060
6  -0.0203       0.0124        -0.0327
7   0.0168       0.0111         0.0057
8   0.0448      -0.0015         0.0463
9   0.0310       0.0079         0.0231
10  0.0173       0.0026         0.0147
```

Overall across `N=2..10`:

```text
past delta                 0.0391
future-control delta        0.0070
past - future controlled    0.0321
```

`N=1` is absent because there were fewer than 20 across-event samples per subject/direction. The event-boundary effect looks more past-specific than the sentence-boundary effect, but the estimate is noisy because there are only 14 coarse event boundaries.

Permutation test for the future-controlled effect:

```bash
python podcast_ecog/permutation_test_time_reversal_boundary_control.py \
  --time-reversal-output-dir podcast_ecog/outputs/time_reversal_boundary_control \
  --boundary-json podcast_ecog/outputs/transcript_boundaries.json \
  --boundary-level sentence \
  --output-dir podcast_ecog/outputs/time_reversal_sentence_boundary_permutation_test \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --n-back 1 2 3 4 5 6 7 8 9 10 \
  --n-permutations 5000 \
  --min-shift 200 \
  --min-condition-samples 20 \
  --random-seed 0

python podcast_ecog/permutation_test_time_reversal_boundary_control.py \
  --time-reversal-output-dir podcast_ecog/outputs/time_reversal_event_boundary_control \
  --boundary-json podcast_ecog/outputs/transcript_boundaries.json \
  --boundary-level event \
  --output-dir podcast_ecog/outputs/time_reversal_event_boundary_permutation_test \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --n-back 2 3 4 5 6 7 8 9 10 \
  --n-permutations 5000 \
  --min-shift 200 \
  --min-condition-samples 20 \
  --random-seed 0
```

The null keeps held-out decoder predictions fixed and circularly shifts the boundary vector. The same shifted boundaries are used for the past and future terms in each permutation.

Omnibus permutation result:

```text
sentence boundaries, mean over N=1..10:
observed controlled delta = 0.00746
shifted-null 95% interval = [-0.02105, 0.02059]
one-sided p = 0.255
two-sided p = 0.502

event boundaries, mean over N=2..10:
observed controlled delta = 0.03206
shifted-null 95% interval = [-0.05859, 0.05908]
one-sided p = 0.139
two-sided p = 0.281
```

For `N<=10`, the future-controlled sentence and event effects are directionally positive, but the omnibus circular-shift tests do not provide strong evidence that the true boundaries are special after subtracting the future control. Individual lags are suggestive, especially sentence `N=10` and event `N=3`, but these do not survive max-statistic correction across lags.

Extended-lag check:

```bash
python podcast_ecog/run_time_reversal_boundary_control.py \
  --bids-root "$BIDS_ROOT" \
  --source-output-dir podcast_ecog/outputs/subject_static_rank_decodability \
  --boundary-json podcast_ecog/outputs/transcript_boundaries.json \
  --output-dir podcast_ecog/outputs/time_reversal_boundary_control_n10_15 \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --boundary-level sentence \
  --dataset-feature-space en_core_web_lg \
  --n-components 5 \
  --n-back 10 11 12 13 14 15 \
  --ridge-alpha 1000 \
  --outer-splits 5 \
  --save-predictions

python podcast_ecog/permutation_test_time_reversal_boundary_control.py \
  --time-reversal-output-dir podcast_ecog/outputs/time_reversal_boundary_control_n10_15 \
  --boundary-json podcast_ecog/outputs/transcript_boundaries.json \
  --boundary-level sentence \
  --output-dir podcast_ecog/outputs/time_reversal_sentence_boundary_permutation_test_n10_15 \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --n-back 10 11 12 13 14 15 \
  --n-permutations 5000 \
  --min-shift 200 \
  --min-condition-samples 20 \
  --random-seed 0

python podcast_ecog/permutation_test_time_reversal_boundary_control.py \
  --time-reversal-output-dir podcast_ecog/outputs/time_reversal_boundary_control_n10_15 \
  --boundary-json podcast_ecog/outputs/transcript_boundaries.json \
  --boundary-level event \
  --output-dir podcast_ecog/outputs/time_reversal_event_boundary_permutation_test_n10_15 \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --n-back 10 11 12 13 14 15 \
  --n-permutations 5000 \
  --min-shift 200 \
  --min-condition-samples 20 \
  --random-seed 0
```

For sentence boundaries, the longer lags were much stronger:

```text
sentence boundaries, mean over N=10..15:
observed controlled delta = 0.03028
shifted-null 95% interval = [-0.01758, 0.01891]
one-sided p = 0.0006
two-sided p = 0.0020

N   controlled   one-sided p   two-sided p   max-corrected p
10  0.02924      0.00640       0.01080       0.02100
11  0.01673      0.07738       0.14557       0.19936
12  0.02733      0.00760       0.01660       0.03019
13  0.02440      0.01380       0.02779       0.05619
14  0.04239      0.00020       0.00080       0.00020
15  0.04155      0.00020       0.00020       0.00020
```

For event boundaries, the same long-lag window was directionally positive but not significant:

```text
event boundaries, mean over N=10..15:
observed controlled delta = 0.02485
shifted-null 95% interval = [-0.04912, 0.04595]
one-sided p = 0.151
two-sided p = 0.297
```

Because `N=10` appeared in both the `N=1..10` and `N=10..15` runs, and those runs used different common current-word samples, we also ran one unified `N=1..15` analysis:

```bash
python podcast_ecog/run_time_reversal_boundary_control.py \
  --bids-root "$BIDS_ROOT" \
  --source-output-dir podcast_ecog/outputs/subject_static_rank_decodability \
  --boundary-json podcast_ecog/outputs/transcript_boundaries.json \
  --output-dir podcast_ecog/outputs/time_reversal_boundary_control_n1_15 \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --boundary-level sentence \
  --dataset-feature-space en_core_web_lg \
  --n-components 5 \
  --n-back 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --ridge-alpha 1000 \
  --outer-splits 5 \
  --save-predictions
```

Unified `N=1..15` omnibus:

```text
sentence boundaries, mean over N=1..15:
observed controlled delta = 0.01086
shifted-null 95% interval = [-0.01684, 0.01655]
one-sided p = 0.105
two-sided p = 0.206

event boundaries, mean over N=1..15:
observed controlled delta = 0.01513
shifted-null 95% interval = [-0.04717, 0.04832]
one-sided p = 0.269
two-sided p = 0.539
```

Interpretation: the long-lag sentence result is encouraging but needs care. Larger `N` gives many more across-boundary samples, so the boundary contrast is estimated with more data. Also, this test is not asking whether absolute decoding at `N=10..15` is high; it asks whether whatever decodable information remains is disproportionately disrupted by true sentence boundaries, after subtracting the future-word control. It is therefore possible for absolute long-lag decoding to be weak while the boundary-specific contrast is statistically reliable.

### ROI Future-Controlled Heatmaps

We also asked where, anatomically, past-word prediction and boundary disruption appear after subtracting the matched future-word control.

The first ROI heatmap uses the same ROI-specific held-out decoders and asks whether past-word prediction is greater than future-word prediction:

```text
future-controlled prediction = mean point_r(t-N) - mean point_r(t+N)
```

Run/plot script:

```bash
python podcast_ecog/plot_roi_future_controlled_prediction_heatmap.py \
  --roi-boundary-output-dir podcast_ecog/outputs/roi_time_reversal_boundary_permutation_heatmap \
  --output-dir podcast_ecog/outputs/roi_future_controlled_prediction_heatmap
```

Main outputs:

```text
podcast_ecog/outputs/roi_future_controlled_prediction_heatmap/roi_lag_future_controlled_prediction_delta_heatmap.png
podcast_ecog/outputs/roi_future_controlled_prediction_heatmap/roi_lag_future_controlled_prediction_minuslog10p_heatmap.png
podcast_ecog/outputs/roi_future_controlled_prediction_heatmap/roi_lag_future_controlled_prediction_delta_heatmap_n1_10.png
```

The second ROI heatmap asks where event boundaries specifically reduce past-word decoding more than future-word decoding:

```text
controlled event-boundary delta =
  (past within-event point_r - past across-event point_r)
  -
  (future within-event point_r - future across-event point_r)
```

Main outputs:

```text
podcast_ecog/outputs/roi_time_reversal_boundary_permutation_heatmap/event_roi_lag_controlled_delta_heatmap.png
podcast_ecog/outputs/roi_time_reversal_boundary_permutation_heatmap/event_roi_lag_minuslog10p_heatmap.png
podcast_ecog/outputs/roi_time_reversal_boundary_permutation_heatmap/event_roi_lag_controlled_delta_heatmap_n1_10.png
```

The cropped `N=1..10` versions were made for easier comparison:

```text
podcast_ecog/outputs/roi_future_controlled_prediction_heatmap/roi_lag_future_controlled_prediction_delta_heatmap_n1_10.png
podcast_ecog/outputs/roi_time_reversal_boundary_permutation_heatmap/event_roi_lag_controlled_delta_heatmap_n1_10.png
```

Interpretation: the future-controlled prediction heatmap shows where the neural population carries more information about past than future words. The event-boundary heatmap is a different quantity: it shows where that past-specific information is disproportionately disrupted when the remembered word lies across a coarse event boundary. Comparing the two is useful because MFG does not simply look like the strongest raw past-word decoding region, but the event-boundary effect is broadest and most consistently positive there, especially across `N=2..10`.

## 10. Future-Controlled Sentence Surprise

We repeated the sentence-surprise quartile analysis using the same future-control idea.

```bash
python podcast_ecog/sentence_surprise_time_reversal_control.py \
  --time-reversal-output-dir podcast_ecog/outputs/time_reversal_boundary_control \
  --perplexity-dir podcast_ecog/outputs/sentence_perplexity_boundary_delta \
  --output-dir podcast_ecog/outputs/sentence_surprise_time_reversal_control \
  --subjects 01 02 03 04 05 06 07 08 09 \
  --n-back 1 2 3 4 5 6 7 8 9 10 \
  --n-quartiles 4
```

Main outputs:

```text
podcast_ecog/outputs/sentence_surprise_time_reversal_control/sentence_surprise_time_reversal_delta_mean_over_n.png
podcast_ecog/outputs/sentence_surprise_time_reversal_control/sentence_surprise_time_reversal_controlled_delta_heatmap_by_n.png
podcast_ecog/outputs/sentence_surprise_time_reversal_control/sentence_surprise_time_reversal_group_mean_over_n.csv
podcast_ecog/outputs/sentence_surprise_time_reversal_control/sentence_surprise_time_reversal_correlations.csv
```

Mean over `N=1..10`:

```text
Quartile   mean NLL   past_delta   future_delta   controlled
Q1         3.57      -0.0246       0.0237        -0.0496
Q2         4.14       0.0309      -0.0194         0.0502
Q3         4.68       0.0179       0.0025         0.0158
Q4         5.58       0.0264       0.0006         0.0243
```

Boundary-wise continuous correlations with GPT-2 conditional NLL were weak after future control:

```text
conditional NLL vs controlled delta:
Pearson r  = 0.051, p = 0.569
Spearman r = 0.044, p = 0.621
```

Interpretation: the lowest-surprise sentence transitions still behave differently, showing no past-specific boundary drop. Less predictable transitions generally show a positive controlled drop, especially Q2, but the effect is not monotonic with perplexity. This supports a categorical-looking distinction between very predictable transitions and the rest, rather than a simple linear relationship where more perplexing sentence transitions produce larger memory drops.
