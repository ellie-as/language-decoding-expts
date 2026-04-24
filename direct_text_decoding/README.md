# Direct raw-text decoding

This folder tests whether brain responses decode the actual recently heard
story text, rather than GPT-generated summaries.

For each response TR, `run_text_window_decoding.py`:

1. Finds the corresponding stimulus TR in the original story.
2. Builds a raw-text target from words in the previous `N` stimulus TRs.
3. Optionally shifts that text window earlier by `target_lag` TRs to match the
   BOLD lag.
4. Embeds the raw text window with `all-MiniLM-L6-v2` by default.
5. Trains brain-to-text-embedding decoders.
6. Reports TR retrieval, story retrieval, and paragraph-window retrieval.

The main scientific readout should be paragraph/window retrieval, not exact
TR-level retrieval. Exact TR targets are still highly autocorrelated.

## Recommended first server run

```bash
python -m direct_text_decoding.run_text_window_decoding \
  --subject S1 \
  --roi full_frontal \
  --window-trs 1 2 3 5 8 10 \
  --target-lags 1 2 3 4 \
  --brain-pca 512 \
  --target-pca 50 \
  --skip-sklearn \
  --loss infonce \
  --torch-device cuda
```

This writes a CSV under `direct_text_decoding/results/S1/`.

## Faster smoke test

```bash
python -m direct_text_decoding.run_text_window_decoding \
  --subject S1 \
  --roi BA_10 \
  --window-trs 3 5 \
  --target-lags 2 3 \
  --brain-pca 256 \
  --target-pca 32 \
  --skip-sklearn \
  --loss infonce \
  --torch-device cuda
```

## Interpreting results

- `dim_r_test`: mean per-dimension Pearson correlation for held-out story TRs.
- `tr_top1` / `tr_mrr`: exact TR retrieval among all held-out TRs. This is very
  hard and not the primary metric.
- `story_top1`: ranks the five held-out stories by mean similarity for each TR.
- `paragraph_top1`: averages predicted/true embeddings over sliding paragraph
  windows, then retrieves the exact paragraph window.
- `paragraph_story_top1`: same paragraph windows, but asks only whether the
  correct held-out story is ranked first. Chance is `1 / n_test_stories`
  (usually 0.2).

If this works, I would expect `paragraph_story_top1` and maybe
`paragraph_top1` to move before exact `tr_top1` does.
