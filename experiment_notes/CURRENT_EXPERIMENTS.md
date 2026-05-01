# Current Experiments

This note tracks the two main active threads.

## 1. Better Encoding Scores: Do They Improve Decoding?

The summary-combo MiniLM encoding model appears to beat the original Huth-style
encoding scores in the frontal lag experiments. The key question now is whether
that better encoding performance leads to better held-out decoding.

Currently running the GPT-2 decoding process with the MiniLM/summary-combo
encoding model:

```bash
cd /ceph/behrens/ellie/language-decoding-expts

SUBJECTS="S1" RUN_DECODE=1 bash gpt1_encoding_comparison/run_minilm_combo_cluster.sh
```

What to check next:

- Whether decoded story similarity improves over the Huth GPT-1/ridge baseline.
- Whether improvements are visible in actual decoded text samples, not only in
  aggregate metrics.
- Whether the null / language-model-only decoding scores remain well separated
  from the brain-conditioned runs.
- Whether S2/S3 show the same pattern once S1 finishes.

## 2. Spatial Autoencoder: Can Conv3D Beat PCA?

The dimensionality-reduction experiments were disappointing: PCA outperformed
the plain and denoising MLP autoencoders for reconstructing held-out
full-frontal responses. That suggests the reliable response covariance is very
PCA-like, but the MLP autoencoder treats voxels as an unordered list.

The current idea is to preserve spatial structure by mapping full-frontal voxel
responses back into a 3D functional volume using pycortex. A 3D convolutional
autoencoder can then exploit local spatial smoothness and voxel neighbourhoods.

The first Conv3D denoising autoencoder run was:

```bash
python lag_preference_analysis/train_fullfrontal_volume_autoencoder.py \
  --subject S1 \
  --data-root /ceph/behrens/ellie/language-decoding-expts \
  --split-results-dir lag_preference_analysis/results/S1__embedding-summary-combo-h20-50-200__lags1-10__chunk1tr__seed0 \
  --pycortex-filestore /ceph/behrens/ellie/language-decoding-expts/pycortex-db \
  --latent-dims 64 128 256 \
  --base-channels 8 \
  --input-noise-std 0.05 \
  --input-mask-prob 0.0 \
  --epochs 40 \
  --batch-size 16
```

What to check next:

- Does Conv3D reconstruction beat PCA at matched latent dimensions?
- If it beats PCA, are the learned latents predictable from text features?
- If the latents are predictable, does text-to-latent-to-volume encoding improve
  voxel prediction or decoding?
- If `base_channels=8` is stable but underpowered, try `--base-channels 16`.

Interpretation rule of thumb: beating the MLP autoencoder would show that
spatial structure helps reconstruction; beating PCA would be much stronger
evidence that the response manifold has useful nonlinear/spatial structure that
the linear PCA baseline misses.

## Other Threads

`mindeye_text` contains the shared-backbone neural decoding/encoding experiments:
brain/text latents with subject-specific heads, plus the later ridge-residual
encoding variant. These were useful architectural probes, but the neural
encoding runs tended to overfit and did not clearly beat the like-for-like ridge
baselines.

`libribrain_gtr_decode` / `brainlm_method` contain the direct semantic-decoding
experiments: predicting reduced text embeddings or generated text from frontal
brain responses. The useful lesson so far is that dense semantic targets can be
decoded above chance, but ridge/PCA-style baselines remain hard to beat.
