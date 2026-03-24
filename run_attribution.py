#!/usr/bin/env python3
"""
Voxel-level attribution analysis for semantic decoding.

Decomposes the encoding model's log-likelihood by voxel to determine which
brain regions most influence word-by-word decoding predictions.

The log-likelihood P(R|S) = -½ (SW−R)ᵀ Σ⁻¹ (SW−R) decomposes exactly as:

    attribution_j = -½ Σ_t  diff[t,j] × (Σ⁻¹ diff[t,:])_j

where diff = SW − R (predicted minus observed response), and the sum over
all voxels j recovers the total log-likelihood.

For word-level attribution, TR-level contributions are distributed back to
words using the Lanczos interpolation weights and FIR delays that link each
word to the TRs it influences.

Usage:
    # Decode + attribute in one step
    python run_attribution.py --subject S1 --experiment perceived_speech --task wheretheressmoke

    # Quick test with smaller beam
    python run_attribution.py --subject S1 --experiment perceived_speech --task wheretheressmoke --beam-width 50

    # Attribute from already-saved decoder output (skips decoding)
    python run_attribution.py --subject S1 --experiment perceived_speech --task wheretheressmoke --use-saved

    # With custom frontal ROI file
    python run_attribution.py --subject S1 --experiment perceived_speech --task wheretheressmoke --use-saved --rois frontal_rois.json
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "decoding"))


# ---------------------------------------------------------------------------
# Attribution math
# ---------------------------------------------------------------------------

def reconstruct_stimulus(decoded_words, features, lanczos_mat, tr_stats, stim_delays):
    """Build the full delayed-and-normalised stimulus matrix from decoded words.

    Returns:
        del_stim: (n_trs, n_delays * n_features) stimulus matrix ready for
                  multiplication with encoding model weights.
        word_embs: (n_words, n_features) raw GPT embeddings per word.
    """
    from utils_ridge.util import make_delayed

    word_embs = features.make_stim(decoded_words)  # (n_words, n_features)

    tr_embs = lanczos_mat @ word_embs  # (n_trs, n_features)

    tr_mean, tr_std = tr_stats[0], tr_stats[1]
    tr_std_safe = tr_std.copy()
    tr_std_safe[tr_std_safe == 0] = 1
    tr_embs_norm = (tr_embs - tr_mean) / tr_std_safe

    del_stim = make_delayed(tr_embs_norm, stim_delays)
    return del_stim, word_embs


def compute_attribution(del_stim, resp, weights, voxels, precision):
    """Per-voxel, per-TR attribution of the encoding model likelihood.

    Returns:
        per_voxel_per_tr: (n_trs, n_model_voxels)
        per_voxel_total:  (n_model_voxels,)
        total_ll:         scalar
    """
    pred = del_stim @ weights[:, voxels]
    diff = pred - resp[:, voxels]
    prec_diff = diff @ precision

    per_voxel_per_tr = -0.5 * diff * prec_diff
    per_voxel_total = per_voxel_per_tr.sum(axis=0)
    total_ll = per_voxel_total.sum()
    return per_voxel_per_tr, per_voxel_total, total_ll


def compute_lag_attribution(del_stim, resp, weights, voxels, precision, stim_delays):
    """Decompose attribution by FIR lag.

    The prediction pred = Σ_k pred_lag_k, so the residual diff = pred - resp
    can be split as diff = Σ_k pred_lag_k - resp.  The per-voxel attribution
    then decomposes exactly as:

        attr[j] = Σ_k attr_lag_k[j] + attr_resp[j]

    where attr_lag_k[j] = -0.5 Σ_t pred_lag_k[t,j] * (Σ⁻¹ diff)[t,j]
    and   attr_resp[j]  = +0.5 Σ_t resp[t,j]        * (Σ⁻¹ diff)[t,j].

    Returns:
        lag_attr:     (n_delays, n_model_voxels) — attribution from each lag
        lag_weights:  (n_delays, n_model_voxels) — fraction of weight energy per lag
    """
    n_trs, total_feats = del_stim.shape
    n_delays = len(stim_delays)
    nf = total_feats // n_delays
    W = weights[:, voxels]
    R = resp[:, voxels]

    pred = del_stim @ W
    diff = pred - R
    prec_diff = diff @ precision

    lag_attr = np.zeros((n_delays, len(voxels)))
    lag_weight_energy = np.zeros((n_delays, len(voxels)))

    for ki, d in enumerate(stim_delays):
        block = slice(ki * nf, (ki + 1) * nf)
        pred_k = del_stim[:, block] @ W[block, :]
        lag_attr[ki] = -0.5 * (pred_k * prec_diff).sum(axis=0)
        lag_weight_energy[ki] = (W[block, :] ** 2).sum(axis=0)

    total_energy = lag_weight_energy.sum(axis=0)
    total_energy[total_energy == 0] = 1
    lag_weights = lag_weight_energy / total_energy

    return lag_attr, lag_weights


def compute_word_attribution(per_voxel_per_tr, lanczos_mat, stim_delays):
    """Distribute TR-level attribution back to individual words.

    Each word influences a set of TRs through the Lanczos interpolation and
    FIR delays.  We compute a (n_trs, n_words) weight matrix encoding this
    influence, normalise rows, and project TR-level attribution onto words.

    Returns:
        per_word_per_voxel: (n_words, n_model_voxels)
    """
    n_trs, n_words = lanczos_mat.shape

    word_to_tr = np.zeros((n_trs, n_words))
    for d in stim_delays:
        if d < n_trs:
            word_to_tr[d:, :] += np.abs(lanczos_mat[:n_trs - d, :])

    row_sums = word_to_tr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    word_to_tr_norm = word_to_tr / row_sums

    return word_to_tr_norm.T @ per_voxel_per_tr  # (n_words, n_voxels)


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------

def load_rois(roi_path, model_voxels):
    """Load ROI definitions and map global voxel indices to model-local indices.

    Args:
        roi_path: path to JSON file {"region_name": [global_voxel_idx, ...], ...}
        model_voxels: array of global voxel indices used by the encoding model.

    Returns:
        dict  {region_name: np.array of local indices into the model voxel set}
    """
    with open(roi_path) as f:
        rois = json.load(f)

    global_to_local = {int(v): i for i, v in enumerate(model_voxels)}

    local_rois = {}
    for name, voxels in rois.items():
        local = [global_to_local[v] for v in voxels if v in global_to_local]
        local_rois[name] = np.array(local, dtype=int)
    return local_rois


def print_roi_table(local_rois, per_voxel_total, discriminability, total_ll):
    """Print a summary table of attribution by ROI."""
    hdr = f"    {'Region':<25s}  {'Voxels':>7s}  {'Mean attr':>10s}  {'Sum attr':>10s}  {'% of total':>11s}  {'Mean discrim':>13s}"
    sep = f"    {'─'*25}  {'─'*7}  {'─'*10}  {'─'*10}  {'─'*11}  {'─'*13}"
    print(hdr)
    print(sep)

    for name in sorted(local_rois):
        idx = local_rois[name]
        if len(idx) == 0:
            print(f"    {name:<25s}  {'0':>7s}  {'—':>10s}  {'—':>10s}  {'—':>11s}  {'—':>13s}")
            continue
        mean_a = per_voxel_total[idx].mean()
        sum_a = per_voxel_total[idx].sum()
        pct = sum_a / total_ll * 100 if total_ll != 0 else 0
        mean_d = discriminability[idx].mean()
        print(f"    {name:<25s}  {len(idx):7d}  {mean_a:10.4f}  {sum_a:10.2f}  {pct:10.1f}%  {mean_d:13.4f}")


# ---------------------------------------------------------------------------
# Decoder wrapper
# ---------------------------------------------------------------------------

def run_decoder_for_attribution(subject, experiment, task, device, beam_width):
    """Run beam-search decoding, returning everything needed for attribution."""
    import torch
    import h5py
    import config

    config.GPT_DEVICE = device
    config.EM_DEVICE = device
    config.SM_DEVICE = device

    from GPT import GPT
    from Decoder import Decoder, Hypothesis
    from LanguageModel import LanguageModel
    from EncodingModel import EncodingModel
    from StimulusModel import StimulusModel, get_lanczos_mat, affected_trs, LMFeatures
    from utils_stim import predict_word_rate, predict_word_times

    gpt_ckpt = "imagined" if experiment == "imagined_speech" else "perceived"
    wr_type = "speech" if experiment in ("imagined_speech", "perceived_movies") else "auditory"

    resp_path = REPO_DIR / "data_test" / "test_response" / subject / experiment / f"{task}.hf5"
    with h5py.File(str(resp_path), "r") as hf:
        resp = np.nan_to_num(hf["data"][:])
    print(f"    Response shape: {resp.shape}")

    print("    Loading GPT ...")
    with open(str(REPO_DIR / "data_lm" / gpt_ckpt / "vocab.json")) as f:
        gpt_vocab = json.load(f)
    with open(str(REPO_DIR / "data_lm" / "decoder_vocab.json")) as f:
        decoder_vocab = json.load(f)
    gpt = GPT(path=str(REPO_DIR / "data_lm" / gpt_ckpt / "model"), vocab=gpt_vocab, device=device)
    features = LMFeatures(model=gpt, layer=config.GPT_LAYER, context_words=config.GPT_WORDS)
    lm = LanguageModel(gpt, decoder_vocab, nuc_mass=config.LM_MASS, nuc_ratio=config.LM_RATIO)

    print("    Loading encoding model ...")
    model_dir = REPO_DIR / "models" / subject
    em_data = np.load(str(model_dir / f"encoding_model_{gpt_ckpt}.npz"))
    em = EncodingModel(resp, em_data["weights"], em_data["voxels"], em_data["noise_model"], device=device)
    em.set_shrinkage(config.NM_ALPHA)

    print("    Loading word-rate model ...")
    wr_data = np.load(str(model_dir / f"word_rate_model_{wr_type}.npz"), allow_pickle=True)

    word_rate = predict_word_rate(resp, wr_data["weights"], wr_data["voxels"], wr_data["mean_rate"])
    starttime = -10 if experiment == "perceived_speech" else 0
    word_times, tr_times = predict_word_times(word_rate, resp, starttime=starttime)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)
    n_words = len(word_times)
    print(f"    Predicted {n_words} words over {len(tr_times)} TRs")

    width = beam_width or config.WIDTH
    print(f"    Beam search (width={width}) ...")
    decoder = Decoder(word_times, width)
    sm = StimulusModel(lanczos_mat, em_data["tr_stats"], em_data["word_stats"][0], device=device)

    t0 = time.time()
    for si in range(n_words):
        if si % 25 == 0 or si == n_words - 1:
            pct = (si + 1) / n_words * 100
            elapsed = time.time() - t0
            eta = elapsed / (si + 1) * (n_words - si - 1) if si > 0 else 0
            print(f"\r    {pct:5.1f}%  word {si+1}/{n_words}  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s    ", end="", flush=True)

        trs = affected_trs(decoder.first_difference(), si, lanczos_mat)
        ncontext = decoder.time_window(si, config.LM_TIME, floor=5)
        beam_nucs = lm.beam_propose(decoder.beam, ncontext)
        for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
            nuc, logprobs = beam_nucs[c]
            if len(nuc) < 1:
                continue
            extend_words = [hyp.words + [x] for x in nuc]
            extend_embs = list(features.extend(extend_words))
            stim = sm.make_variants(si, hyp.embs, extend_embs, trs)
            likelihoods = em.prs(stim, trs)
            local_extensions = [Hypothesis(parent=hyp, extension=x)
                                for x in zip(nuc, logprobs, extend_embs)]
            decoder.add_extensions(local_extensions, likelihoods, nextensions)
        decoder.extend(verbose=False)

    elapsed = time.time() - t0
    print(f"\n    Decoding finished in {elapsed:.1f}s")

    if experiment in ("perceived_movie", "perceived_multispeaker"):
        decoder.word_times += 10

    save_dir = REPO_DIR / "results" / subject / experiment
    save_dir.mkdir(parents=True, exist_ok=True)
    decoder.save(str(save_dir / task))

    return {
        "words": decoder.beam[0].words,
        "word_times": np.array(decoder.word_times) if isinstance(decoder.word_times, list) else decoder.word_times,
        "tr_times": tr_times,
        "resp": resp,
        "em_data": em_data,
        "features": features,
        "lanczos_mat": lanczos_mat,
    }


def load_for_attribution(subject, experiment, task, device):
    """Load a saved prediction and re-initialise models needed for attribution."""
    import h5py
    import config
    config.GPT_DEVICE = device
    config.EM_DEVICE = device
    config.SM_DEVICE = device

    from GPT import GPT
    from StimulusModel import get_lanczos_mat, LMFeatures
    from utils_stim import predict_word_rate, predict_word_times

    gpt_ckpt = "imagined" if experiment == "imagined_speech" else "perceived"
    wr_type = "speech" if experiment in ("imagined_speech", "perceived_movies") else "auditory"

    pred_path = REPO_DIR / "results" / subject / experiment / f"{task}.npz"
    if not pred_path.exists():
        print(f"  No saved prediction at {pred_path}")
        print(f"  Run without --use-saved to decode first.")
        sys.exit(1)

    pred = np.load(str(pred_path), allow_pickle=True)
    decoded_words = list(pred["words"])
    print(f"    Loaded {len(decoded_words)} decoded words from {pred_path.name}")

    resp_path = REPO_DIR / "data_test" / "test_response" / subject / experiment / f"{task}.hf5"
    with h5py.File(str(resp_path), "r") as hf:
        resp = np.nan_to_num(hf["data"][:])
    print(f"    Response shape: {resp.shape}")

    print("    Loading GPT for feature extraction ...")
    with open(str(REPO_DIR / "data_lm" / gpt_ckpt / "vocab.json")) as f:
        gpt_vocab = json.load(f)
    gpt = GPT(path=str(REPO_DIR / "data_lm" / gpt_ckpt / "model"), vocab=gpt_vocab, device=device)
    features = LMFeatures(model=gpt, layer=config.GPT_LAYER, context_words=config.GPT_WORDS)

    print("    Loading encoding model ...")
    model_dir = REPO_DIR / "models" / subject
    em_data = np.load(str(model_dir / f"encoding_model_{gpt_ckpt}.npz"))

    print("    Reconstructing word times ...")
    wr_data = np.load(str(model_dir / f"word_rate_model_{wr_type}.npz"), allow_pickle=True)
    word_rate = predict_word_rate(resp, wr_data["weights"], wr_data["voxels"], wr_data["mean_rate"])
    starttime = -10 if experiment == "perceived_speech" else 0
    word_times, tr_times = predict_word_times(word_rate, resp, starttime=starttime)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)
    print(f"    {len(word_times)} words, {len(tr_times)} TRs")

    return {
        "words": decoded_words,
        "word_times": word_times,
        "tr_times": tr_times,
        "resp": resp,
        "em_data": em_data,
        "features": features,
        "lanczos_mat": lanczos_mat,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Voxel-level attribution analysis for semantic decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default=None, help="cpu | cuda (auto-detected)")
    parser.add_argument("--beam-width", type=int, default=None,
                        help="Beam width (default 200; 20-50 for quick tests)")
    parser.add_argument("--use-saved", action="store_true",
                        help="Use existing decoder output instead of re-decoding")
    parser.add_argument("--rois", type=str, default=None,
                        help="JSON file mapping region names to global voxel indices")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print("=" * 65)
    print("  Voxel-Level Attribution Analysis")
    print("  Semantic Decoding — Tang et al. (2023)")
    print("=" * 65)
    print(f"\n  {args.subject} / {args.experiment} / {args.task}   device={device}")

    # ------------------------------------------------------------------
    # 1. Decode (or load saved)
    # ------------------------------------------------------------------
    if args.use_saved:
        print(f"\n--- Loading saved prediction ---\n")
        data = load_for_attribution(args.subject, args.experiment, args.task, device)
    else:
        print(f"\n--- Decoding ---\n")
        data = run_decoder_for_attribution(
            args.subject, args.experiment, args.task, device, args.beam_width)

    decoded_words = data["words"]
    print(f"\n  Decoded text preview: {' '.join(decoded_words[:25])} ...")

    import config

    # ------------------------------------------------------------------
    # 2. Reconstruct full stimulus from decoded words
    # ------------------------------------------------------------------
    print(f"\n--- Reconstructing stimulus features ---\n")
    del_stim, word_embs = reconstruct_stimulus(
        decoded_words, data["features"],
        data["lanczos_mat"], data["em_data"]["tr_stats"], config.STIM_DELAYS,
    )
    print(f"    Stimulus matrix: {del_stim.shape}")

    # ------------------------------------------------------------------
    # 3. Compute per-voxel, per-TR attribution
    # ------------------------------------------------------------------
    print(f"\n--- Computing voxel-level attribution ---\n")
    noise_model = data["em_data"]["noise_model"]
    precision = np.linalg.inv(
        noise_model * (1 - config.NM_ALPHA) + np.eye(len(noise_model)) * config.NM_ALPHA
    )
    voxels = data["em_data"]["voxels"]

    per_voxel_per_tr, per_voxel_total, total_ll = compute_attribution(
        del_stim, data["resp"], data["em_data"]["weights"], voxels, precision,
    )
    # discriminability: how variable each voxel's contribution is across TRs
    # high discriminability = voxel response changes meaningfully with stimulus
    discriminability = per_voxel_per_tr.std(axis=0)

    print(f"    Total log-likelihood:  {total_ll:.2f}")
    print(f"    Voxels in model:       {len(voxels)}")
    print(f"    Per-voxel mean attr:   {per_voxel_total.mean():.4f}")
    print(f"    Per-voxel std attr:    {per_voxel_total.std():.4f}")
    print(f"    Mean discriminability: {discriminability.mean():.4f}")

    # ------------------------------------------------------------------
    # 3b. Lag decomposition
    # ------------------------------------------------------------------
    print(f"\n--- Lag decomposition (delays {config.STIM_DELAYS} TRs = {[d*2 for d in config.STIM_DELAYS]}s) ---\n")
    lag_attr, lag_weights = compute_lag_attribution(
        del_stim, data["resp"], data["em_data"]["weights"], voxels, precision,
        config.STIM_DELAYS,
    )

    print(f"    {'Lag (TRs)':<12s}  {'Delay (s)':<10s}  {'Mean attr':>10s}  {'% of pred attr':>15s}")
    print(f"    {'─'*12}  {'─'*10}  {'─'*10}  {'─'*15}")
    pred_attr_total = lag_attr.sum()
    for ki, d in enumerate(config.STIM_DELAYS):
        la_mean = lag_attr[ki].mean()
        la_pct = lag_attr[ki].sum() / pred_attr_total * 100 if pred_attr_total != 0 else 0
        print(f"    {d:<12d}  {d*2:<10d}  {la_mean:10.4f}  {la_pct:14.1f}%")

    # ------------------------------------------------------------------
    # 4. Word-level attribution
    # ------------------------------------------------------------------
    print(f"\n--- Word-level attribution ---\n")
    per_word_per_voxel = compute_word_attribution(
        per_voxel_per_tr, data["lanczos_mat"], config.STIM_DELAYS,
    )
    per_word_total = per_word_per_voxel.sum(axis=1)

    # best-supported words (least negative = best fit to observed responses)
    top_idx = np.argsort(per_word_total)[-10:][::-1]
    bot_idx = np.argsort(per_word_total)[:10]

    print("    Best-supported words (highest log-likelihood contribution):")
    for i in top_idx:
        print(f"      [{i:4d}] {decoded_words[i]:>15s}  attr = {per_word_total[i]:+.4f}")

    print("\n    Worst-supported words (lowest log-likelihood contribution):")
    for i in bot_idx:
        print(f"      [{i:4d}] {decoded_words[i]:>15s}  attr = {per_word_total[i]:+.4f}")

    # ------------------------------------------------------------------
    # 5. ROI analysis
    # ------------------------------------------------------------------
    local_rois = None

    if args.rois:
        print(f"\n--- ROI analysis: {args.rois} ---\n")
        local_rois = load_rois(args.rois, voxels)
        print_roi_table(local_rois, per_voxel_total, discriminability, total_ll)
    else:
        roi_path = REPO_DIR / "data_train" / "ROIs" / f"{args.subject}.json"
        if roi_path.exists():
            print(f"\n--- ROI analysis (built-in speech / auditory ROIs) ---\n")
            local_rois = load_rois(str(roi_path), voxels)
            print_roi_table(local_rois, per_voxel_total, discriminability, total_ll)

    if local_rois is not None and len(local_rois) >= 2:
        # lag × region table
        print(f"\n--- Lag × Region attribution ---\n")
        region_names = sorted(local_rois)
        hdr = f"    {'Lag':>4s}  {'Delay':>6s}"
        for name in region_names:
            hdr += f"  {name:>20s}"
        print(hdr)
        print(f"    {'─'*4}  {'─'*6}" + "".join(f"  {'─'*20}" for _ in region_names))
        for ki, d in enumerate(config.STIM_DELAYS):
            row = f"    {d:4d}  {d*2:5d}s"
            for name in region_names:
                idx = local_rois[name]
                val = lag_attr[ki, idx].mean() if len(idx) > 0 else 0
                row += f"  {val:20.6f}"
            print(row)
        # weight profile
        print(f"\n    Weight energy fraction per lag:")
        print(f"    {'Lag':>4s}  {'Delay':>6s}" + "".join(f"  {name:>20s}" for name in region_names))
        print(f"    {'─'*4}  {'─'*6}" + "".join(f"  {'─'*20}" for _ in region_names))
        for ki, d in enumerate(config.STIM_DELAYS):
            row = f"    {d:4d}  {d*2:5d}s"
            for name in region_names:
                idx = local_rois[name]
                val = lag_weights[ki, idx].mean() if len(idx) > 0 else 0
                row += f"  {val:20.4f}"
            print(row)

        # per-word breakdown by region
        print(f"\n--- Per-word attribution by region (top 15 words) ---\n")
        hdr_parts = [f"    {'word':>15s}"]
        for name in sorted(local_rois):
            hdr_parts.append(f"{name:>15s}")
        hdr_parts.append(f"{'total':>10s}")
        print("  ".join(hdr_parts))
        print("    " + "─" * (18 + 17 * len(local_rois) + 12))

        for i in top_idx[:15]:
            parts = [f"    {decoded_words[i]:>15s}"]
            for name in sorted(local_rois):
                idx = local_rois[name]
                val = per_word_per_voxel[i, idx].sum() if len(idx) > 0 else 0
                parts.append(f"{val:15.4f}")
            parts.append(f"{per_word_total[i]:10.4f}")
            print("  ".join(parts))

    if local_rois is None:
        print(f"\n  To analyse frontal subregions, create a JSON ROI file:")
        print(f"  {{")
        print(f'    "posterior_frontal": [voxel_idx, ...],')
        print(f'    "middle_frontal":   [voxel_idx, ...],')
        print(f'    "anterior_frontal": [voxel_idx, ...]')
        print(f"  }}")
        print(f"  where voxel indices are into the full brain volume,")
        print(f"  then pass --rois <file>.json")

    # ------------------------------------------------------------------
    # 6. Save everything
    # ------------------------------------------------------------------
    save_dir = REPO_DIR / "attribution" / args.subject / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{args.task}.npz"

    save_dict = dict(
        per_voxel_total=per_voxel_total,
        per_voxel_per_tr=per_voxel_per_tr,
        per_word_per_voxel=per_word_per_voxel,
        per_word_total=per_word_total,
        discriminability=discriminability,
        decoded_words=np.array(decoded_words),
        word_embs=word_embs,
        word_times=data["word_times"],
        tr_times=data["tr_times"],
        voxels=voxels,
        total_likelihood=total_ll,
        lag_attr=lag_attr,
        lag_weights=lag_weights,
        stim_delays=np.array(config.STIM_DELAYS),
    )
    np.savez(str(save_path), **save_dict)

    print(f"\n--- Saved to {save_path} ---")
    print(f"\n  Arrays in file:")
    print(f"    per_voxel_total      ({len(voxels)},)           sum over TRs per voxel")
    print(f"    per_voxel_per_tr     {per_voxel_per_tr.shape}       full TR × voxel matrix")
    print(f"    per_word_per_voxel   {per_word_per_voxel.shape}      word × voxel attribution")
    print(f"    per_word_total       ({len(decoded_words)},)           sum over voxels per word")
    print(f"    discriminability     ({len(voxels)},)           std of per-TR attr per voxel")
    print(f"    decoded_words        ({len(decoded_words)},)           decoded word strings")
    print(f"    word_embs            {word_embs.shape}     GPT embeddings per word")
    print(f"    voxels               ({len(voxels)},)           global voxel indices")
    print(f"    lag_attr             {lag_attr.shape}      per-lag, per-voxel attribution")
    print(f"    lag_weights          {lag_weights.shape}      weight energy fraction per lag")
    print(f"    stim_delays          ({len(config.STIM_DELAYS)},)              FIR delays in TRs")

    print("\n" + "=" * 65)
    print("  Done.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
