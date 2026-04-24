# Cross-subject self-supervised brain encoder

Small MAE-style Transformer trained across LeBel subjects (S1, S2, S3...) on
their native surface-voxel timeseries. The goal is a learned feature extractor
that replaces raw voxels in the downstream decoder pipeline, without the
problems that rule out existing fMRI foundation models for this task:

- No parcellation / atlas resampling (keeps voxel-level resolution).
- No MNI volume pipeline (works in each subject's native surface space).
- Trained on the actual task-fMRI (narrative listening) rather than resting
  state.
- Pools supervision across subjects via per-subject input/output projections
  while sharing one Transformer encoder.

## Architecture (MAE on TR tokens)

```
per-subject input Linear(V_s -> d)    shared Transformer encoder
                                    \      |
              voxels [T, V_s] ----->  tokens [T, d]  -- mask 50% of TRs -->
                                                     (visible tokens only)
                                                               |
shared Transformer decoder (small) + [mask] tokens at masked positions
                                                               |
per-subject output Linear(d -> V_s) ---> predict voxels at masked TRs
loss: MSE at masked positions only, averaged over V_s per subject
```

Only the per-subject projections depend on `V_s`; all other weights are shared
across subjects. `d_model = 256`, 4 encoder layers + 2 decoder layers by
default. The model is small enough to train on a single GPU.

## Files

- `dataset.py` - multi-subject TR-chunk sampler. Per-subject per-story
  z-scoring; sampled windows live strictly within a single story.
- `model.py` - `BrainEncoderMAE` (per-subject projections + shared
  encoder/decoder) and `masked_mse_loss`.
- `train.py` - training CLI (AdamW, cosine LR, MAE loss).
- `extract_features.py` - runs the trained encoder over full stories and saves
  per-TR features `[T, d_model]` as `.npz`.
- `README.md` - this file.

## Data hygiene

The training loop uses the same `split_story_list(...)` helper as the
downstream decoders, so the `--holdout-count` test stories are **never** seen
during pretraining. Extracted features for those stories are still valid to
feed the decoder since the encoder never saw their responses or any related
signal.

## Example training run

```
python -m brain_encoder_pretrain.train \
    --subjects S1 S2 S3 \
    --sessions 1 2 3 4 5 \
    --chunk-len 64 --batch-size 32 \
    --d-model 256 --n-enc-layers 4 --n-dec-layers 2 \
    --steps 20000 --lr 3e-4 --weight-decay 0.05 \
    --eval-every 500 --save-every 2000 \
    --output-dir brain_encoder_pretrain/runs/run1
```

On the server add `--device cuda`; locally it auto-selects CUDA / MPS / CPU.
On a mounted Ceph + local cache setup you can pass `--local-compute-mode` (the
same pattern used by the encoding / decoding scripts).

## Feature extraction

After training, extract features for the subject you want to decode:

```
python -m brain_encoder_pretrain.extract_features \
    --ckpt brain_encoder_pretrain/runs/run1/ckpt_best.pt \
    --subject S1 --stories-mode all \
    --output-dir brain_encoder_pretrain/features/run1
```

Each `<subject>/<story>.npz` holds a single key `"X"` of shape `[T, d_model]`.

## Wiring into the decoder (manual, phase 2)

The existing decoder scripts load voxel responses via `get_resp(...)` from
`data_train/train_response/<subject>/<story>.hf5`. To switch them onto the
learned features, load each story's `X` from the `.npz` produced above and
replace the `X_train_raw`/`X_test_raw` construction. A small wrapper adding a
`--features-dir` argument to `run_h20_decoder_sweep.py` (and
`run_summary_decoding.py`) is the cleanest integration; not done in this
changeset to keep the current behaviour intact.

## Design notes / knobs

- `chunk_len`: length of TR windows fed to the Transformer. 64 TRs ~ 2 minutes.
  Longer windows let the model learn slower temporal structure but cost more
  memory and make random windowing less effective for small stories.
- `mask_ratio`: MAE-style 0.5 is a good default. Lower ratios (0.25) make the
  task easier and can help if the loss plateaus high.
- `d_model`: 256 is plenty given the data size. Bigger models overfit quickly.
- No delays on the input - the encoder sees native BOLD; the downstream
  decoder can still add delays on the encoder output if helpful.
- Cross-subject weight sharing is the main "foundation-model" ingredient;
  with only ~80 stories per subject, pooling across 3 subjects triples the
  effective training data for the shared weights.
