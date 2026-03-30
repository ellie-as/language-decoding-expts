#!/usr/bin/env python3
"""
Context-length encoding analysis.

Tests whether more temporally abstracted (longer-context) LLM representations
are preferentially encoded by more frontal voxels.

For each (model, context_length) pair, extracts word-level features using a
sliding window of the last N words, interpolates to TR times, and trains a
ridge encoding model (features -> voxels).  Per-voxel prediction correlations
are saved and optionally summarised by ROI.

Models
------
  gpt1       : OpenAI GPT (openai-gpt), layer 9 hidden states
  gpt2       : GPT-2 small, last hidden layer
  embedding  : Sentence-transformer (all-MiniLM-L6-v2)

Usage
-----
  python run_context_encoding.py --subject S1
  python run_context_encoding.py --subject S1 --models gpt1 gpt2 \\
      --context-lengths 20 60 200 --rois frontal_rois_UTS01.json
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "decoding"))
import config
from GPT import GPT
from utils_stim import get_story_wordseqs
from utils_resp import get_resp
from utils_ridge.ridge import bootstrap_ridge
from utils_ridge.interpdata import lanczosinterp2D
from utils_ridge.util import make_delayed

CONTEXT_LENGTHS = [20, 40, 60, 100, 200]
MODEL_CHOICES = ["gpt1", "gpt2", "embedding"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("context_encoding")
np.random.seed(42)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_gpt1_features(stories, word_seqs, context_words, device,
                           batch_size=128):
    """GPT-1 hidden states (layer 9) with a sliding context window.

    Parameters
    ----------
    context_words : int
        Number of preceding words.  Total window = context_words + 1.
    """
    vocab_path = os.path.join(config.DATA_LM_DIR, "perceived", "vocab.json")
    with open(vocab_path) as f:
        vocab = json.load(f)
    gpt = GPT(
        path=os.path.join(config.DATA_LM_DIR, "perceived", "model"),
        vocab=vocab,
        device=device,
    )
    layer = config.GPT_LAYER

    word_vecs = {}
    for story in stories:
        words = word_seqs[story].data
        ctx_array = gpt.get_story_array(words, context_words)
        n = ctx_array.shape[0]

        all_embs = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            embs = gpt.get_hidden(ctx_array[start:end], layer=layer)
            all_embs.append(embs)
        all_embs = np.concatenate(all_embs, axis=0)

        word_vecs[story] = np.vstack([
            all_embs[0, :context_words],
            all_embs[:n - context_words, context_words],
        ])
        log.info("  %s: %d words -> %s", story, len(words),
                 word_vecs[story].shape)

    del gpt
    torch.cuda.empty_cache()
    return word_vecs


def extract_gpt2_features(stories, word_seqs, context_length, device):
    """GPT-2 (small) last-layer hidden states with a sliding word window.

    For each word position, the preceding *context_length - 1* words plus the
    current word are concatenated, BPE-tokenised, and fed through GPT-2.
    The hidden state of the last BPE token at the final layer is returned.
    """
    from transformers import GPT2Tokenizer, GPT2Model

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2").eval().to(device)
    hidden_dim = model.config.n_embd

    word_vecs = {}
    for story in stories:
        words = word_seqs[story].data
        n = len(words)
        vecs = np.empty((n, hidden_dim), dtype=np.float32)

        for i in range(n):
            start = max(0, i - context_length + 1)
            text = " ".join(words[start:i + 1])
            ids = tokenizer.encode(text)
            if len(ids) > 1024:
                ids = ids[-1024:]
            ids_t = torch.tensor([ids], device=device)
            with torch.no_grad():
                out = model(ids_t, output_hidden_states=True)
            vecs[i] = out.hidden_states[-1][0, -1].cpu().numpy()

            if (i + 1) % 500 == 0:
                log.info("  %s: %d / %d words", story, i + 1, n)

        word_vecs[story] = vecs
        log.info("  %s: %d words -> %s", story, n, vecs.shape)

    del model
    torch.cuda.empty_cache()
    return word_vecs


def extract_embedding_features(stories, word_seqs, context_length, device):
    """Sentence-transformer embeddings of sliding word windows.

    For each word position, the last *context_length* words (including the
    current one) are joined into a string and encoded.
    """
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    word_vecs = {}
    for story in stories:
        words = word_seqs[story].data
        n = len(words)
        texts = []
        for i in range(n):
            start = max(0, i - context_length + 1)
            texts.append(" ".join(words[start:i + 1]))

        vecs = st_model.encode(texts, batch_size=64, show_progress_bar=True,
                               convert_to_numpy=True).astype(np.float32)
        word_vecs[story] = vecs
        log.info("  %s: %d words -> %s", story, n, vecs.shape)

    del st_model
    torch.cuda.empty_cache()
    return word_vecs


EXTRACTORS = {
    "gpt1": extract_gpt1_features,
    "gpt2": extract_gpt2_features,
    "embedding": extract_embedding_features,
}


# ---------------------------------------------------------------------------
# Feature -> TR conversion
# ---------------------------------------------------------------------------

def features_to_tr(word_vecs, word_seqs, stories):
    """Lanczos-interpolate word vectors to TRs, z-score, add FIR delays."""
    ds_vecs = {
        story: lanczosinterp2D(
            word_vecs[story],
            word_seqs[story].data_times,
            word_seqs[story].tr_times,
        )
        for story in stories
    }
    ds_mat = np.vstack([
        ds_vecs[story][5 + config.TRIM:-config.TRIM] for story in stories
    ])
    r_mean, r_std = ds_mat.mean(0), ds_mat.std(0)
    r_std[r_std == 0] = 1
    ds_mat = np.nan_to_num((ds_mat - r_mean) / r_std)
    del_mat = make_delayed(ds_mat, config.STIM_DELAYS)
    return del_mat


# ---------------------------------------------------------------------------
# ROI summary
# ---------------------------------------------------------------------------

def print_roi_summary(all_corrs, rois_path, vox, context_lengths, model_types):
    """Print per-ROI mean encoding correlation for every condition.

    *vox* maps local indices (into corrs) to global voxel indices.
    ROI files use global indices, so we build a reverse mapping.
    """
    with open(rois_path) as f:
        rois = json.load(f)
    region_names = sorted(rois.keys())

    global_to_local = {int(g): i for i, g in enumerate(vox)}

    local_rois = {}
    for rn in region_names:
        local_rois[rn] = np.array(
            [global_to_local[v] for v in rois[rn] if v in global_to_local],
            dtype=int,
        )

    hdr = f"  {'condition':<25s}"
    for rn in region_names:
        hdr += f"  {rn + f' ({len(local_rois[rn])})':>25s}"
    hdr += f"  {'all':>10s}"

    print("\n" + "=" * len(hdr))
    print("  Per-ROI mean encoding correlation")
    print("=" * len(hdr))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for model_type in model_types:
        for ctx_len in context_lengths:
            label = f"{model_type}_ctx{ctx_len}"
            if label not in all_corrs:
                continue
            corrs = all_corrs[label]
            row = f"  {label:<25s}"
            for rn in region_names:
                idx = local_rois[rn]
                mean_r = corrs[idx].mean() if len(idx) > 0 else float("nan")
                row += f"  {mean_r:25.4f}"
            row += f"  {corrs.mean():10.4f}"
            print(row)
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_voxel_set(subject, all_voxels):
    """Load the language-responsive voxel indices from the pretrained model.

    Falls back to all voxels if no pretrained model is found.
    Returns (voxel_indices, is_subset) where voxel_indices index into the
    full response matrix.
    """
    em_path = os.path.join(config.MODEL_DIR, subject, "encoding_model_perceived.npz")
    if os.path.exists(em_path):
        em = np.load(em_path)
        vox = em["voxels"]
        log.info("Loaded %d language-responsive voxels from pretrained model",
                 len(vox))
        return vox, True
    log.warning("No pretrained model found at %s — using all %d voxels "
                "(may require a lot of RAM)", em_path, all_voxels)
    return np.arange(all_voxels), False


def chunked_bootstrap_ridge(rstim, rresp, chunk_size=10000, **kwargs):
    """Run bootstrap_ridge in voxel chunks to stay within memory limits."""
    n_voxels = rresp.shape[1]
    if n_voxels <= chunk_size:
        wt, valphas, bscorrs = bootstrap_ridge(rstim, rresp, **kwargs)
        corrs = bscorrs.mean(2).max(0)
        del bscorrs
        return corrs, wt

    all_corrs = np.zeros(n_voxels)
    all_wt = None
    for start in range(0, n_voxels, chunk_size):
        end = min(start + chunk_size, n_voxels)
        log.info("  Voxel chunk [%d:%d] / %d", start, end, n_voxels)
        wt, valphas, bscorrs = bootstrap_ridge(
            rstim, rresp[:, start:end], **kwargs)
        all_corrs[start:end] = bscorrs.mean(2).max(0)
        del wt, bscorrs
    return all_corrs, None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--sessions", nargs="+", type=int,
                        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--context-lengths", nargs="+", type=int,
                        default=CONTEXT_LENGTHS)
    parser.add_argument("--models", nargs="+", default=MODEL_CHOICES,
                        choices=MODEL_CHOICES)
    parser.add_argument("--rois", default=None,
                        help="Frontal ROI JSON for per-region summary "
                             "(e.g. frontal_rois_UTS01.json)")
    parser.add_argument("--nboots", type=int, default=config.NBOOTS)
    parser.add_argument("--output-dir", default="context_results")
    parser.add_argument("--voxels-from-rois", action="store_true",
                        help="Use only voxels inside the ROIs (requires --rois)")
    parser.add_argument("--all-voxels", action="store_true",
                        help="Use all voxels instead of pretrained language-"
                             "responsive set (needs more RAM, uses chunking)")
    parser.add_argument("--voxel-chunk-size", type=int, default=10000,
                        help="Voxel chunk size when using --all-voxels "
                             "(default 10000)")
    parser.add_argument("--save-weights", action="store_true",
                        help="Also save full regression weights (large files)")
    args = parser.parse_args()

    device = config.GPT_DEVICE
    log.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Load stories
    # ------------------------------------------------------------------
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json")) as f:
        sess_to_story = json.load(f)
    stories = []
    for s in args.sessions:
        stories.extend(sess_to_story[str(s)])
    log.info("Stories (%d): %s", len(stories), stories)

    # ------------------------------------------------------------------
    # Word sequences & fMRI responses
    # ------------------------------------------------------------------
    word_seqs = get_story_wordseqs(stories)
    rresp_full = get_resp(args.subject, stories, stack=True)
    log.info("Full response matrix: %s (TRs x voxels)", rresp_full.shape)

    # ------------------------------------------------------------------
    # Voxel selection
    # ------------------------------------------------------------------
    if args.voxels_from_rois:
        if not args.rois or not os.path.exists(args.rois):
            log.error("--voxels-from-rois requires --rois <file>")
            sys.exit(1)
        with open(args.rois) as f:
            rois = json.load(f)
        roi_voxels = set()
        for idx_list in rois.values():
            roi_voxels.update(idx_list)
        vox = np.sort(np.array(list(roi_voxels), dtype=int))
        vox = vox[vox < rresp_full.shape[1]]
        log.info("Using %d voxels from ROIs (%s)", len(vox), args.rois)
    elif args.all_voxels:
        vox = np.arange(rresp_full.shape[1])
        log.info("Using ALL %d voxels (chunked processing)", len(vox))
    else:
        vox, _ = load_voxel_set(args.subject, rresp_full.shape[1])

    rresp = rresp_full[:, vox]
    del rresp_full
    log.info("Working response matrix: %s (TRs x voxels)", rresp.shape)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    out_dir = os.path.join(config.REPO_DIR, args.output_dir, args.subject)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Run each (model, context_length) condition
    # ------------------------------------------------------------------
    all_corrs = {}

    for model_type in args.models:
        for ctx_len in args.context_lengths:
            label = f"{model_type}_ctx{ctx_len}"
            log.info("=" * 60)
            log.info("Condition: %s", label)
            log.info("=" * 60)

            # 1. Extract word-level features
            log.info("Extracting features...")
            if model_type == "gpt1":
                word_vecs = extract_gpt1_features(
                    stories, word_seqs,
                    context_words=ctx_len - 1,
                    device=device,
                )
            elif model_type == "gpt2":
                word_vecs = extract_gpt2_features(
                    stories, word_seqs,
                    context_length=ctx_len,
                    device=device,
                )
            elif model_type == "embedding":
                word_vecs = extract_embedding_features(
                    stories, word_seqs,
                    context_length=ctx_len,
                    device=device,
                )

            # 2. Word-level -> TR-level design matrix
            log.info("Interpolating to TRs and adding FIR delays...")
            rstim = features_to_tr(word_vecs, word_seqs, stories)
            log.info("Design matrix: %s (TRs x delayed features)", rstim.shape)
            del word_vecs

            # 3. Ridge encoding model (features -> voxels)
            nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))
            log.info("Bootstrap ridge regression (%d boots, chunklen=%d, "
                     "nchunks=%d)...", args.nboots, config.CHUNKLEN, nchunks)

            corrs, wt = chunked_bootstrap_ridge(
                rstim, rresp,
                chunk_size=args.voxel_chunk_size,
                alphas=config.ALPHAS,
                nboots=args.nboots,
                chunklen=config.CHUNKLEN,
                nchunks=nchunks,
                use_corr=True,
            )
            del rstim

            all_corrs[label] = corrs
            log.info("  mean r=%.4f | max r=%.4f | n(r>0.1)=%d",
                     corrs.mean(), corrs.max(), (corrs > 0.1).sum())

            # 4. Save per-condition results
            save_dict = dict(
                corrs=corrs,
                voxels=vox,
                model_type=np.array(model_type),
                context_length=np.array(ctx_len),
                stories=np.array(stories),
            )
            if args.save_weights and wt is not None:
                save_dict["weights"] = wt

            np.savez(os.path.join(out_dir, f"{label}.npz"), **save_dict)
            del wt
            log.info("  -> saved %s/%s.npz", out_dir, label)

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    summary = {label: c for label, c in all_corrs.items()}
    summary["context_lengths"] = np.array(args.context_lengths)
    summary["model_types"] = np.array(args.models)
    summary["voxels"] = vox
    np.savez(os.path.join(out_dir, "summary.npz"), **summary)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY — per-voxel encoding correlation (%d voxels)", len(vox))
    log.info("=" * 60)
    for label, corrs in all_corrs.items():
        log.info("  %-25s  mean=%.4f  max=%.4f  n(r>0.1)=%d",
                 label, corrs.mean(), corrs.max(), (corrs > 0.1).sum())

    # ------------------------------------------------------------------
    # Optional ROI breakdown
    # ------------------------------------------------------------------
    if args.rois and os.path.exists(args.rois):
        print_roi_summary(all_corrs, args.rois, vox, args.context_lengths,
                          args.models)

    log.info("All results saved to %s", out_dir)


if __name__ == "__main__":
    main()
