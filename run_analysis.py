#!/usr/bin/env python3
"""
Post-hoc analysis of attribution results.

Loads the .npz output from run_attribution.py, computes lexical features for
each decoded word, and examines how per-word regional attribution varies with
lag and word properties.

Usage:
    python run_analysis.py --subject S1 --experiment perceived_speech \
        --task wheretheressmoke --rois frontal_rois_UTS01.json

    # With a GPT model for surprisal (requires data_lm/ to be populated)
    python run_analysis.py --subject S1 --experiment perceived_speech \
        --task wheretheressmoke --rois frontal_rois_UTS01.json --surprisal
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "decoding"))


# ---------------------------------------------------------------------------
# Lexical feature computation
# ---------------------------------------------------------------------------

def word_lengths(words):
    return np.array([len(w.strip()) for w in words], dtype=float)


def log_word_frequency(words, vocab_counts=None):
    """Approximate log-frequency using the decoded corpus itself as the
    frequency source (since we don't have an external frequency table).
    log(count + 1) is used so all words get a non-zero value.
    """
    if vocab_counts is None:
        vocab_counts = Counter(w.lower().strip() for w in words)
    return np.array([np.log(vocab_counts[w.lower().strip()] + 1) for w in words])


def compute_surprisal(words, device="cpu"):
    """Compute per-word surprisal using GPT-2 (from transformers).

    Surprisal_i = -log P(word_i | word_1 ... word_{i-1}).
    We compute this in chunks to keep memory reasonable.
    """
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    text = " ".join(words)
    token_ids = tokenizer.encode(text)
    surprisals_per_token = np.zeros(len(token_ids))

    chunk_size = 512
    stride = 256

    with torch.no_grad():
        for start in range(0, len(token_ids), stride):
            end = min(start + chunk_size, len(token_ids))
            input_ids = torch.tensor([token_ids[start:end]], device=device)
            outputs = model(input_ids, labels=input_ids)
            logits = outputs.logits[0]  # (seq_len, vocab)
            log_probs = torch.log_softmax(logits, dim=-1)
            for t in range(1, end - start):
                global_t = start + t
                if global_t < len(token_ids) and surprisals_per_token[global_t] == 0:
                    surprisals_per_token[global_t] = -log_probs[t - 1, token_ids[global_t]].item()
            if end >= len(token_ids):
                break

    word_surprisals = np.zeros(len(words))
    token_offset = 0
    for wi, word in enumerate(words):
        word_tokens = tokenizer.encode(" " + word if wi > 0 else word)
        n_tok = len(word_tokens)
        if token_offset + n_tok <= len(surprisals_per_token):
            word_surprisals[wi] = surprisals_per_token[token_offset:token_offset + n_tok].sum()
        token_offset += n_tok

    return word_surprisals


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def ols_regression(X, y):
    """Ordinary least squares: y = X @ beta + residual.
    Returns beta, R^2, and per-predictor t-statistics.
    """
    n, p = X.shape
    XtX = X.T @ X
    try:
        beta = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    dof = max(n - p, 1)
    mse = ss_res / dof
    try:
        cov_beta = mse * np.linalg.inv(XtX)
        se = np.sqrt(np.clip(np.diag(cov_beta), 0, None))
        t_stats = beta / np.where(se > 0, se, 1)
    except np.linalg.LinAlgError:
        t_stats = np.zeros(p)

    return beta, r2, t_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyse attribution results with lexical features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--rois", required=True,
                        help="JSON file mapping region names to global voxel indices")
    parser.add_argument("--surprisal", action="store_true",
                        help="Compute GPT-2 surprisal (needs transformers, ~1 min)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print("=" * 65)
    print("  Attribution Analysis — Lexical Features × Region × Lag")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load attribution results
    # ------------------------------------------------------------------
    attr_path = REPO_DIR / "attribution" / args.subject / args.experiment / f"{args.task}.npz"
    if not attr_path.exists():
        print(f"\n  No attribution file at {attr_path}")
        print(f"  Run run_attribution.py first.\n")
        sys.exit(1)

    data = np.load(str(attr_path), allow_pickle=True)
    words = list(data["decoded_words"])
    per_word_per_voxel = data["per_word_per_voxel"]
    lag_attr = data["lag_attr"]
    lag_weights = data["lag_weights"]
    voxels = data["voxels"]
    stim_delays = list(data["stim_delays"])
    n_words = len(words)
    n_voxels = len(voxels)

    print(f"\n  Loaded {n_words} words, {n_voxels} voxels, {len(stim_delays)} lags")
    print(f"  Subject: {args.subject}  Task: {args.task}")

    # ------------------------------------------------------------------
    # 2. Load ROIs
    # ------------------------------------------------------------------
    with open(args.rois) as f:
        rois_global = json.load(f)

    global_to_local = {int(v): i for i, v in enumerate(voxels)}
    local_rois = {}
    for name, idxs in rois_global.items():
        local = [global_to_local[v] for v in idxs if v in global_to_local]
        local_rois[name] = np.array(local, dtype=int)

    region_names = sorted(local_rois)
    print(f"  ROIs: {', '.join(f'{n} ({len(local_rois[n])})' for n in region_names)}")

    # ------------------------------------------------------------------
    # 3. Compute per-word, per-region attribution (and per-lag variant)
    # ------------------------------------------------------------------
    per_word_per_region = np.zeros((n_words, len(region_names)))
    for ri, name in enumerate(region_names):
        idx = local_rois[name]
        if len(idx) > 0:
            per_word_per_region[:, ri] = per_word_per_voxel[:, idx].sum(axis=1)

    # Per-region lag weight profile (aggregated across voxels)
    lag_profile_by_region = np.zeros((len(stim_delays), len(region_names)))
    for ri, name in enumerate(region_names):
        idx = local_rois[name]
        if len(idx) > 0:
            lag_profile_by_region[:, ri] = lag_attr[:, idx].mean(axis=1)

    # ------------------------------------------------------------------
    # 4. Compute lexical features
    # ------------------------------------------------------------------
    print(f"\n--- Computing lexical features ---\n")
    wlen = word_lengths(words)
    wfreq = log_word_frequency(words)

    feature_names = ["word_length", "log_frequency"]
    features = np.column_stack([wlen, wfreq])

    if args.surprisal:
        print("    Computing GPT-2 surprisal ...")
        surp = compute_surprisal(words, device=device)
        feature_names.append("surprisal")
        features = np.column_stack([features, surp])

    # z-score features for regression
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0)
    feat_std[feat_std == 0] = 1
    features_z = (features - feat_mean) / feat_std

    for fi, fname in enumerate(feature_names):
        vals = features[:, fi]
        print(f"    {fname:<18s}  mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"min={vals.min():.3f}  max={vals.max():.3f}")

    # ------------------------------------------------------------------
    # 5. Regression: per-word regional attribution ~ lexical features
    # ------------------------------------------------------------------
    print(f"\n--- Regression: word-level regional attribution ~ lexical features ---\n")

    X = np.column_stack([np.ones(n_words), features_z])
    predictor_names = ["intercept"] + feature_names

    for ri, name in enumerate(region_names):
        y = per_word_per_region[:, ri]
        beta, r2, t_stats = ols_regression(X, y)

        print(f"  {name}  (R² = {r2:.4f})")
        for pi, pname in enumerate(predictor_names):
            sig = " *" if abs(t_stats[pi]) > 1.96 else ""
            print(f"    {pname:<18s}  β={beta[pi]:+.6f}  t={t_stats[pi]:+.2f}{sig}")
        print()

    # ------------------------------------------------------------------
    # 6. Regression: per-word TOTAL attribution ~ lexical features
    # ------------------------------------------------------------------
    print(f"--- Regression: total word attribution ~ lexical features ---\n")
    y_total = per_word_per_voxel.sum(axis=1)
    beta, r2, t_stats = ols_regression(X, y_total)
    print(f"  Total  (R² = {r2:.4f})")
    for pi, pname in enumerate(predictor_names):
        sig = " *" if abs(t_stats[pi]) > 1.96 else ""
        print(f"    {pname:<18s}  β={beta[pi]:+.6f}  t={t_stats[pi]:+.2f}{sig}")
    print()

    # ------------------------------------------------------------------
    # 7. Lag profile by region (summary from saved attribution)
    # ------------------------------------------------------------------
    print(f"--- Lag profile by region (mean attribution per lag) ---\n")
    hdr = f"    {'Lag':>4s}  {'Delay':>6s}"
    for name in region_names:
        hdr += f"  {name:>20s}"
    print(hdr)
    print(f"    {'─'*4}  {'─'*6}" + "".join(f"  {'─'*20}" for _ in region_names))
    for ki, d in enumerate(stim_delays):
        row = f"    {d:4d}  {d*2:5d}s"
        for ri, name in enumerate(region_names):
            row += f"  {lag_profile_by_region[ki, ri]:20.6f}"
        print(row)

    # weight energy profile
    lag_wt_by_region = np.zeros((len(stim_delays), len(region_names)))
    for ri, name in enumerate(region_names):
        idx = local_rois[name]
        if len(idx) > 0:
            lag_wt_by_region[:, ri] = lag_weights[:, idx].mean(axis=1)

    print(f"\n    Weight energy fraction:")
    print(f"    {'Lag':>4s}  {'Delay':>6s}" + "".join(f"  {name:>20s}" for name in region_names))
    print(f"    {'─'*4}  {'─'*6}" + "".join(f"  {'─'*20}" for _ in region_names))
    for ki, d in enumerate(stim_delays):
        row = f"    {d:4d}  {d*2:5d}s"
        for ri in range(len(region_names)):
            row += f"  {lag_wt_by_region[ki, ri]:20.4f}"
        print(row)

    # ------------------------------------------------------------------
    # 8. Region × lag × feature interaction
    # ------------------------------------------------------------------
    print(f"\n--- Region × lag interaction: which features predict lag-specific weight? ---\n")
    print(f"  For each region, we regress the voxel-level weight fraction at each")
    print(f"  lag against the region-mean attribution of each lexical feature.\n")

    # Per-word attribution at each lag within each region
    # We approximate this by distributing the lag-level voxel attribution
    # back to words using the same word_to_tr mapping.
    # But lag_attr is (n_delays, n_voxels) not (n_delays, n_trs, n_voxels),
    # so we report the region × lag table and the feature regressions separately.

    # A simpler analysis: for each region, compute the peak lag
    print(f"  Peak lag by region:")
    for ri, name in enumerate(region_names):
        peak_lag_idx = np.argmax(lag_profile_by_region[:, ri])
        peak_delay = stim_delays[peak_lag_idx]
        print(f"    {name:<25s}  peak at lag {peak_delay} TR ({peak_delay*2}s)")

    # ------------------------------------------------------------------
    # 9. Save analysis results
    # ------------------------------------------------------------------
    save_dir = REPO_DIR / "analysis" / args.subject / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{args.task}.npz"

    np.savez(
        str(save_path),
        per_word_per_region=per_word_per_region,
        region_names=np.array(region_names),
        feature_names=np.array(feature_names),
        features=features,
        lag_profile_by_region=lag_profile_by_region,
        lag_wt_by_region=lag_wt_by_region,
    )
    print(f"\n--- Saved analysis to {save_path} ---\n")

    print("=" * 65)
    print("  Done.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
