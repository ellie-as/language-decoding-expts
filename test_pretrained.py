#!/usr/bin/env python3
"""
Test pretrained semantic decoding models from Tang et al. (2023).

This script runs the full pipeline using pre-fit encoding and word rate models:
  1. Downloads required data (language model, test data, pretrained models)
  2. Runs the decoder on held-out test brain responses
  3. Evaluates predictions against reference transcripts
  4. Prints decoded text and similarity scores

Usage:
  # First time: download all required data
  python test_pretrained.py download

  # See what subjects, experiments, and tasks are available
  python test_pretrained.py list

  # Run decoding + evaluation for a subject/experiment/task
  python test_pretrained.py run --subject S1 --experiment perceived_speech --task wheretheressmoke

  # Only decode (skip evaluation)
  python test_pretrained.py run --subject S1 --experiment perceived_speech --task wheretheressmoke --decode-only

  # Only evaluate existing predictions
  python test_pretrained.py run --subject S1 --experiment perceived_speech --task wheretheressmoke --eval-only

  # Use a smaller beam width for faster (but less accurate) testing
  python test_pretrained.py run --subject S1 --experiment perceived_speech --task wheretheressmoke --beam-width 50
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent
DECODING_DIR = REPO_DIR / "decoding"
sys.path.insert(0, str(DECODING_DIR))

DATA_LM_URL = "https://utexas.box.com/shared/static/7ab8qm5e3i0vfsku0ee4dc6hzgeg7nyh.zip"
DATA_TEST_URL = "https://utexas.box.com/shared/static/ae5u0t3sh4f46nvmrd3skniq0kk2t5uh.zip"
MODELS_FOLDER_URL = "https://utexas.box.com/s/ri13t06iwpkyk17h8tfk0dtyva7qtqlz"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_with_progress(url, dest):
    """Download a file from url to dest with a progress indicator."""
    print(f"  Downloading {dest.name} ...")

    def _hook(block, block_size, total):
        done = block * block_size
        if total > 0:
            pct = min(100.0, done * 100.0 / total)
            mb = done / 1048576
            mb_total = total / 1048576
            print(f"\r  {pct:5.1f}%  ({mb:.1f} / {mb_total:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_hook)
    print()


def _extract_zip(zip_path, target_dir):
    """Extract zip_path into target_dir, handling nested root folders."""
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        members = zf.namelist()
        prefixes = {m.split("/")[0] for m in members if "/" in m}
        # If everything lives under a single root folder, strip it
        if len(prefixes) == 1:
            root_prefix = prefixes.pop() + "/"
            for member in members:
                if member == root_prefix or member.endswith("/"):
                    continue
                rel = member[len(root_prefix):]
                out = target_dir / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(str(out), "wb") as dst:
                    dst.write(src.read())
        else:
            zf.extractall(str(target_dir))


def download_data():
    """Download language-model data, test data, and (if possible) pretrained models."""
    downloads = [
        ("data_lm", DATA_LM_URL, "Language model data (GPT checkpoints + vocab)"),
        ("data_test", DATA_TEST_URL, "Test data (brain responses + transcripts)"),
    ]

    for dirname, url, desc in downloads:
        target = REPO_DIR / dirname
        if target.exists() and any(target.iterdir()):
            print(f"  [exists]  {dirname}/  — {desc}")
            continue

        print(f"\n  Downloading {desc} ...")
        zip_path = REPO_DIR / f"{dirname}.zip"
        try:
            _download_with_progress(url, zip_path)
            target.mkdir(parents=True, exist_ok=True)
            print(f"  Extracting to {dirname}/ ...")
            _extract_zip(zip_path, target)
            zip_path.unlink()
            print(f"  [ok]  {dirname}/")
        except Exception as exc:
            print(f"  [error]  {exc}")
            if zip_path.exists():
                zip_path.unlink()

    models_dir = REPO_DIR / "models"
    if models_dir.exists() and any(models_dir.iterdir()):
        print(f"  [exists]  models/  — Pretrained encoding + word-rate models")
    else:
        print(f"\n  Pretrained models must be downloaded manually.")
        print(f"  Open this link in a browser and download the folder contents:")
        print(f"    {MODELS_FOLDER_URL}")
        print(f"  Then extract into:  {models_dir}/")
        print(f"  Expected layout:")
        print(f"    models/S1/encoding_model_perceived.npz")
        print(f"    models/S1/word_rate_model_auditory.npz")
        print(f"    models/S1/word_rate_model_speech.npz")
        print(f"    models/S2/...   models/S3/...")


# ---------------------------------------------------------------------------
# Listing / verification
# ---------------------------------------------------------------------------

def list_available():
    """Print available subjects, experiments, and tasks."""
    print("\n--- Pretrained models ---\n")
    models_dir = REPO_DIR / "models"
    if models_dir.exists():
        for subj in sorted(models_dir.iterdir()):
            if not subj.is_dir():
                continue
            files = sorted(f.name for f in subj.glob("*.npz"))
            print(f"  {subj.name}:  {', '.join(files)}")
    else:
        print("  (models/ not found — run 'download' first)")

    print("\n--- Test data ---\n")
    test_resp = REPO_DIR / "data_test" / "test_response"
    if test_resp.exists():
        for subj in sorted(test_resp.iterdir()):
            if not subj.is_dir():
                continue
            for exp in sorted(subj.iterdir()):
                if not exp.is_dir():
                    continue
                tasks = sorted(f.stem for f in exp.glob("*.hf5"))
                print(f"  {subj.name} / {exp.name}")
                for t in tasks:
                    print(f"      task: {t}")
    else:
        print("  (data_test/ not found — run 'download' first)")


def _resolve_gpt_checkpoint(experiment):
    return "imagined" if experiment == "imagined_speech" else "perceived"


def _resolve_wr_type(experiment):
    return "speech" if experiment in ("imagined_speech", "perceived_movies") else "auditory"


def verify_files(subject, experiment, task):
    """Return True if every file needed for decoding+evaluation is present."""
    gpt_ckpt = _resolve_gpt_checkpoint(experiment)
    wr_type = _resolve_wr_type(experiment)

    checks = {
        "Test response": REPO_DIR / "data_test" / "test_response" / subject / experiment / f"{task}.hf5",
        "GPT vocab": REPO_DIR / "data_lm" / gpt_ckpt / "vocab.json",
        "Decoder vocab": REPO_DIR / "data_lm" / "decoder_vocab.json",
        "GPT model dir": REPO_DIR / "data_lm" / gpt_ckpt / "model",
        "Encoding model": REPO_DIR / "models" / subject / f"encoding_model_{gpt_ckpt}.npz",
        "Word-rate model": REPO_DIR / "models" / subject / f"word_rate_model_{wr_type}.npz",
        "Eval segments": REPO_DIR / "data_test" / "eval_segments.json",
    }

    ok = True
    for label, path in checks.items():
        if path.exists():
            print(f"  [ok]      {label}")
        else:
            print(f"  [missing] {label}  ({path.relative_to(REPO_DIR)})")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def run_decode(subject, experiment, task, device, beam_width):
    """Load pretrained models and run beam-search decoding on test responses."""
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

    gpt_ckpt = _resolve_gpt_checkpoint(experiment)
    wr_type = _resolve_wr_type(experiment)

    # --- load test brain responses ---
    print("  Loading test responses ...")
    resp_path = REPO_DIR / "data_test" / "test_response" / subject / experiment / f"{task}.hf5"
    with h5py.File(str(resp_path), "r") as hf:
        resp = np.nan_to_num(hf["data"][:])
    print(f"    shape = {resp.shape}  (TRs x voxels)")

    # --- load GPT ---
    print("  Loading GPT language model ...")
    with open(str(REPO_DIR / "data_lm" / gpt_ckpt / "vocab.json")) as f:
        gpt_vocab = json.load(f)
    with open(str(REPO_DIR / "data_lm" / "decoder_vocab.json")) as f:
        decoder_vocab = json.load(f)

    gpt = GPT(
        path=str(REPO_DIR / "data_lm" / gpt_ckpt / "model"),
        vocab=gpt_vocab,
        device=device,
    )
    features = LMFeatures(model=gpt, layer=config.GPT_LAYER, context_words=config.GPT_WORDS)
    lm = LanguageModel(gpt, decoder_vocab, nuc_mass=config.LM_MASS, nuc_ratio=config.LM_RATIO)

    # --- load pretrained encoding model ---
    print("  Loading pretrained encoding model ...")
    model_dir = REPO_DIR / "models" / subject
    em_data = np.load(str(model_dir / f"encoding_model_{gpt_ckpt}.npz"))
    em = EncodingModel(resp, em_data["weights"], em_data["voxels"], em_data["noise_model"], device=device)
    em.set_shrinkage(config.NM_ALPHA)
    tr_stats = em_data["tr_stats"]
    word_stats = em_data["word_stats"]
    training_stories = em_data["stories"]
    if task in training_stories:
        print(f"  WARNING: task '{task}' was used during model training — results will be inflated.")

    # --- load pretrained word-rate model ---
    print("  Loading pretrained word-rate model ...")
    wr_data = np.load(str(model_dir / f"word_rate_model_{wr_type}.npz"), allow_pickle=True)

    # --- predict word times ---
    print("  Predicting word times ...")
    word_rate = predict_word_rate(resp, wr_data["weights"], wr_data["voxels"], wr_data["mean_rate"])
    starttime = -10 if experiment == "perceived_speech" else 0
    word_times, tr_times = predict_word_times(word_rate, resp, starttime=starttime)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)
    n_words = len(word_times)
    print(f"    predicted {n_words} words over {len(tr_times)} TRs")

    # --- beam-search decoding ---
    width = beam_width if beam_width is not None else config.WIDTH
    print(f"  Running beam-search decoder  (beam_width={width}, device={device}) ...")
    decoder = Decoder(word_times, width)
    sm = StimulusModel(lanczos_mat, tr_stats, word_stats[0], device=device)

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
            local_extensions = [
                Hypothesis(parent=hyp, extension=x)
                for x in zip(nuc, logprobs, extend_embs)
            ]
            decoder.add_extensions(local_extensions, likelihoods, nextensions)
        decoder.extend(verbose=False)

    elapsed = time.time() - t0
    print(f"\n    Decoding finished in {elapsed:.1f}s")

    if experiment in ("perceived_movie", "perceived_multispeaker"):
        decoder.word_times += 10

    # --- save ---
    save_dir = REPO_DIR / "results" / subject / experiment
    save_dir.mkdir(parents=True, exist_ok=True)
    decoder.save(str(save_dir / task))
    print(f"    Saved to results/{subject}/{experiment}/{task}.npz")

    # --- print decoded text ---
    decoded = decoder.beam[0].words
    print(f"\n  === Decoded text ({len(decoded)} words) ===\n")
    text = " ".join(decoded)
    for i in range(0, len(text), 90):
        print(f"    {text[i:i+90]}")
    print()

    return decoded


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluate(subject, experiment, task, device, n_null=5):
    """Score decoder predictions against reference transcripts."""
    import config
    config.GPT_DEVICE = device

    from utils_eval import (
        load_transcript, windows, segment_data, generate_null,
        WER,
    )

    # --- load prediction ---
    pred_path = REPO_DIR / "results" / subject / experiment / f"{task}.npz"
    if not pred_path.exists():
        print(f"  Prediction file not found: {pred_path}")
        print(f"  Run decoding first.")
        return None
    pred_data = np.load(str(pred_path))
    pred_words, pred_times = pred_data["words"], pred_data["times"]
    print(f"  Loaded prediction: {len(pred_words)} words")

    # --- eval segments ---
    seg_path = REPO_DIR / "data_test" / "eval_segments.json"
    with open(str(seg_path)) as f:
        eval_segments = json.load(f)
    if task not in eval_segments:
        print(f"  No eval-segment entry for task '{task}'.")
        print(f"  Available: {sorted(eval_segments.keys())}")
        return None

    # --- reference transcript ---
    print("  Loading reference transcript ...")
    try:
        ref_data = load_transcript(experiment, task)
    except FileNotFoundError as exc:
        print(f"  Reference transcript not found: {exc}")
        return None
    ref_words, ref_times = ref_data["words"], ref_data["times"]
    print(f"    reference: {len(ref_words)} words")

    # --- segment into windows ---
    window_cutoffs = windows(*eval_segments[task], config.WINDOW)
    ref_windows = segment_data(ref_words, ref_times, window_cutoffs)
    pred_windows = segment_data(pred_words, pred_times, window_cutoffs)

    # --- metrics ---
    metrics = {}
    metrics["WER (1−error)"] = WER(use_score=True)

    try:
        from utils_eval import BLEU
        metrics["BLEU-1"] = BLEU(n=1)
    except Exception as exc:
        print(f"  [skip] BLEU: {exc}")

    try:
        from utils_eval import METEOR
        metrics["METEOR"] = METEOR()
    except Exception as exc:
        print(f"  [skip] METEOR: {exc}")

    try:
        from utils_eval import BERTSCORE
        idf_path = REPO_DIR / "data_test" / "idf_segments.npy"
        idf_sents = np.load(str(idf_path)) if idf_path.exists() else None
        metrics["BERTScore"] = BERTSCORE(idf_sents=idf_sents, rescale=False, score="recall")
    except Exception as exc:
        print(f"  [skip] BERTScore: {exc}")

    if not metrics:
        print("  No metrics could be loaded.")
        return None

    # --- null baselines ---
    print(f"  Generating {n_null} null sequences for z-score normalization ...")
    gpt_ckpt = _resolve_gpt_checkpoint(experiment)
    null_word_list = generate_null(pred_times, gpt_ckpt, n_null)
    null_window_list = [segment_data(nw, pred_times, window_cutoffs) for nw in null_word_list]

    # --- score ---
    print()
    print(f"  {'Metric':<16s}  {'Score':>8s}  {'Null mean':>10s}  {'z-score':>8s}  {'% sig windows':>14s}")
    print(f"  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*14}")

    results = {}
    for mname, metric in metrics.items():
        try:
            w_null = np.array([metric.score(ref=ref_windows, pred=nw) for nw in null_window_list])
            s_null = w_null.mean(axis=1)

            w_scores = metric.score(ref=ref_windows, pred=pred_windows)
            story_score = w_scores.mean()
            null_mean = s_null.mean()
            null_std = s_null.std()
            z = (story_score - null_mean) / null_std if null_std > 0 else 0.0

            sig_frac = np.mean(
                w_scores > w_null.mean(axis=0) + 1.96 * w_null.std(axis=0)
            )

            results[mname] = dict(score=story_score, zscore=z, null_mean=null_mean, sig_frac=sig_frac)
            print(f"  {mname:<16s}  {story_score:8.4f}  {null_mean:10.4f}  {z:8.2f}  {sig_frac:13.1%}")
        except Exception as exc:
            print(f"  {mname:<16s}  [error] {exc}")

    # --- save ---
    save_dir = REPO_DIR / "scores" / subject / experiment
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(save_dir / task), **{k: v for k, v in results.items()})
    print(f"\n  Scores saved to scores/{subject}/{experiment}/{task}.npz")

    return results


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def pick_device(requested):
    """Choose compute device; fall back to CPU on macOS."""
    import torch
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    top = argparse.ArgumentParser(
        description="Test pretrained semantic decoding models (Tang et al. 2023)",
    )
    sub = top.add_subparsers(dest="command")

    sub.add_parser("download", help="Download required data (LM, test, models)")
    sub.add_parser("list", help="List available subjects / experiments / tasks")

    run_p = sub.add_parser("run", help="Run decoding and/or evaluation")
    run_p.add_argument("--subject", required=True, help="Subject ID, e.g. S1")
    run_p.add_argument("--experiment", required=True,
                       help="Experiment name: perceived_speech | imagined_speech | perceived_movies | perceived_multispeaker")
    run_p.add_argument("--task", required=True, help="Task / story name, e.g. wheretheressmoke")
    run_p.add_argument("--device", default=None, help="cpu | cuda  (auto-detected if omitted)")
    run_p.add_argument("--beam-width", type=int, default=None,
                       help="Override beam width (default 200; use 20-50 for quick tests)")
    run_p.add_argument("--decode-only", action="store_true", help="Only decode, skip evaluation")
    run_p.add_argument("--eval-only", action="store_true", help="Only evaluate existing predictions")
    run_p.add_argument("--n-null", type=int, default=5,
                       help="Number of null sequences for z-score normalization (default 5)")

    args = top.parse_args()

    print()
    print("=" * 62)
    print("  Semantic Decoding — Pretrained Model Test")
    print("  Tang et al. (2023)  Nature Neuroscience 26, 858–866")
    print("=" * 62)

    if args.command == "download":
        print("\n--- Downloading data ---\n")
        download_data()
        return

    if args.command == "list":
        list_available()
        return

    if args.command == "run":
        device = pick_device(args.device)
        print(f"\n  Device: {device}")

        print(f"\n--- Verifying files: {args.subject} / {args.experiment} / {args.task} ---\n")
        if not verify_files(args.subject, args.experiment, args.task):
            print("\n  Some files are missing. Run:  python test_pretrained.py download")
            print(f"  For pretrained models, download from:\n    {MODELS_FOLDER_URL}")
            sys.exit(1)

        if not args.eval_only:
            print(f"\n--- Decoding ---\n")
            run_decode(args.subject, args.experiment, args.task, device, args.beam_width)

        if not args.decode_only:
            print(f"\n--- Evaluation ---\n")
            run_evaluate(args.subject, args.experiment, args.task, device, n_null=args.n_null)

        print("\n" + "=" * 62)
        print("  Done.")
        print("=" * 62 + "\n")
        return

    top.print_help()


if __name__ == "__main__":
    main()
