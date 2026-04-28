#!/usr/bin/env python3
"""Decode test stories with GPT-1 comparison encoders and score them.

This script compares:

* ``paper``: downloaded Huth/Tang model in ``models/<subject>/encoding_model_perceived.npz``
* ``finetuned``: retrained local finetuned GPT-1 encoder from this experiment
* ``pretrained``: retrained original ``openai-gpt`` encoder from this experiment

For the pretrained encoder, beam proposals still come from the Huth/Tang
decoder language model so that the search prior and candidate vocabulary stay
fixed. The brain likelihood is scored with the pretrained GPT-1 feature space.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "decoding"))
sys.path.insert(0, str(THIS_DIR))

import config  # noqa: E402
from Decoder import Decoder, Hypothesis  # noqa: E402
from EncodingModel import EncodingModel  # noqa: E402
from GPT import GPT  # noqa: E402
from LanguageModel import LanguageModel  # noqa: E402
from StimulusModel import StimulusModel, affected_trs, get_lanczos_mat  # noqa: E402
from compare_gpt1_encoding import (  # noqa: E402
    CONDITION_FINETUNED,
    CONDITION_PRETRAINED,
    HFOpenAIGPTFeatures,
    HuthFinetunedGPT1Features,
)
from utils_stim import predict_word_rate, predict_word_times  # noqa: E402


CONDITION_PAPER = "paper"
DEFAULT_TASKS = ["wheretheressmoke"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="S1")
    parser.add_argument("--experiment", default="perceived_speech")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[CONDITION_PAPER, CONDITION_FINETUNED, CONDITION_PRETRAINED],
        choices=[CONDITION_PAPER, CONDITION_FINETUNED, CONDITION_PRETRAINED],
    )
    parser.add_argument(
        "--trained-model-dir",
        default=str(THIS_DIR / "outputs"),
        help="Root containing <subject>/encoding_model_finetuned.npz and pretrained.npz.",
    )
    parser.add_argument("--pretrained-model", default="openai-gpt")
    parser.add_argument("--finetuned-checkpoint", default="perceived")
    parser.add_argument("--device", default=config.GPT_DEVICE)
    parser.add_argument("--beam-width", type=int, default=config.WIDTH)
    parser.add_argument("--pretrained-batch-size", type=int, default=64)
    parser.add_argument("--n-null", type=int, default=10)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["WER", "BLEU", "METEOR", "BERT"],
        choices=["WER", "BLEU", "METEOR", "BERT"],
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(THIS_DIR / "decoding_outputs"),
        help="Where decoded transcripts, scores, and CSV summaries are written.",
    )
    parser.add_argument(
        "--paper-headline-json",
        default=None,
        help=(
            "Optional JSON mapping metric names to paper headline values. "
            "If supplied, output includes decoded-minus-paper deltas."
        ),
    )
    return parser.parse_args()


def gpt_checkpoint_for_experiment(experiment):
    return "imagined" if experiment == "imagined_speech" else "perceived"


def word_rate_voxels_for_experiment(experiment):
    return "speech" if experiment in ("imagined_speech", "perceived_movies") else "auditory"


def load_huth_lm(device, checkpoint="perceived"):
    with open(Path(config.DATA_LM_DIR) / checkpoint / "vocab.json", encoding="utf-8") as f:
        gpt_vocab = json.load(f)
    with open(Path(config.DATA_LM_DIR) / "decoder_vocab.json", encoding="utf-8") as f:
        decoder_vocab = json.load(f)
    gpt = GPT(path=str(Path(config.DATA_LM_DIR) / checkpoint / "model"), vocab=gpt_vocab, device=device)
    lm = LanguageModel(gpt, decoder_vocab, nuc_mass=config.LM_MASS, nuc_ratio=config.LM_RATIO)
    return gpt, lm


def make_features(condition, args):
    if condition in {CONDITION_PAPER, CONDITION_FINETUNED}:
        return HuthFinetunedGPT1Features(args.finetuned_checkpoint, args.device)
    if condition == CONDITION_PRETRAINED:
        return HFOpenAIGPTFeatures(args.pretrained_model, args.device, args.pretrained_batch_size)
    raise ValueError(f"Unknown condition: {condition}")


def encoding_model_path(condition, subject, args):
    if condition == CONDITION_PAPER:
        return Path(config.MODEL_DIR) / subject / "encoding_model_perceived.npz"
    return Path(args.trained_model_dir).expanduser().resolve() / subject / f"encoding_model_{condition}.npz"


def load_test_response(subject, experiment, task):
    path = Path(config.DATA_TEST_DIR) / "test_response" / subject / experiment / f"{task}.hf5"
    with h5py.File(path, "r") as hf:
        return np.nan_to_num(hf["data"][:])


def load_word_times(subject, experiment, task, resp):
    wr_type = word_rate_voxels_for_experiment(experiment)
    model_dir = Path(config.MODEL_DIR) / subject
    wr_data = np.load(model_dir / f"word_rate_model_{wr_type}.npz", allow_pickle=True)
    word_rate = predict_word_rate(
        resp,
        wr_data["weights"],
        wr_data["voxels"],
        wr_data["mean_rate"],
    )
    starttime = -10 if experiment == "perceived_speech" else 0
    return predict_word_times(word_rate, resp, starttime=starttime)


def decode_one(condition, subject, experiment, task, args, huth_lm):
    output_dir = Path(args.output_dir).expanduser().resolve()
    pred_dir = output_dir / subject / experiment / condition
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{task}.npz"

    if args.skip_existing and pred_path.exists():
        print(f"[{condition}] reusing {pred_path}")
        return pred_path

    resp = load_test_response(subject, experiment, task)
    word_times, tr_times = load_word_times(subject, experiment, task, resp)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)

    em_path = encoding_model_path(condition, subject, args)
    if not em_path.exists():
        raise FileNotFoundError(f"Encoding model not found: {em_path}")
    em_data = np.load(em_path, allow_pickle=True)
    if em_data["noise_model"].size == 0:
        raise ValueError(
            f"{em_path} has an empty noise_model. Rerun compare_gpt1_encoding.py "
            "without --skip-noise-model before decoding."
        )
    if task in set(str(x) for x in em_data["stories"]):
        raise ValueError(f"{task!r} appears in training stories for {em_path}")

    features = make_features(condition, args)
    try:
        em = EncodingModel(
            resp,
            em_data["weights"],
            em_data["voxels"],
            em_data["noise_model"],
            device=config.EM_DEVICE,
        )
        em.set_shrinkage(config.NM_ALPHA)
        sm = StimulusModel(
            lanczos_mat,
            em_data["tr_stats"],
            em_data["word_stats"][0],
            device=config.SM_DEVICE,
        )

        _huth_gpt, lm = huth_lm
        decoder = Decoder(word_times, args.beam_width)
        t0 = time.time()
        for sample_index in range(len(word_times)):
            if sample_index % 25 == 0 or sample_index == len(word_times) - 1:
                elapsed = time.time() - t0
                print(
                    f"\r[{condition}] {task}: word {sample_index + 1}/{len(word_times)} "
                    f"elapsed={elapsed:.0f}s",
                    end="",
                    flush=True,
                )

            trs = affected_trs(decoder.first_difference(), sample_index, lanczos_mat)
            ncontext = decoder.time_window(sample_index, config.LM_TIME, floor=5)
            beam_nucs = lm.beam_propose(decoder.beam, ncontext)

            for beam_index, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
                nuc, logprobs = beam_nucs[beam_index]
                if len(nuc) < 1:
                    continue
                extend_words = [hyp.words + [word] for word in nuc]
                extend_embs = list(features.extend(extend_words))
                stim = sm.make_variants(sample_index, hyp.embs, extend_embs, trs)
                likelihoods = em.prs(stim, trs)
                local_extensions = [
                    Hypothesis(parent=hyp, extension=item)
                    for item in zip(nuc, logprobs, extend_embs)
                ]
                decoder.add_extensions(local_extensions, likelihoods, nextensions)
            decoder.extend(verbose=False)

        if experiment in ("perceived_movie", "perceived_multispeaker"):
            decoder.word_times += 10
        decoder.save(str(pred_path.with_suffix("")))
        print(f"\n[{condition}] saved {pred_path}")
        return pred_path
    finally:
        features.close()


def load_metrics(args):
    from utils_eval import BERTSCORE, BLEU, METEOR, WER

    metrics = {}
    if "WER" in args.metrics:
        metrics["WER"] = WER(use_score=True)
    if "BLEU" in args.metrics:
        metrics["BLEU"] = BLEU(n=1)
    if "METEOR" in args.metrics:
        metrics["METEOR"] = METEOR()
    if "BERT" in args.metrics:
        idf_path = Path(config.DATA_TEST_DIR) / "idf_segments.npy"
        idf_sents = np.load(idf_path) if idf_path.exists() else None
        metrics["BERT"] = BERTSCORE(idf_sents=idf_sents, rescale=False, score="recall")
    return metrics


def score_one(pred_path, subject, experiment, task, metrics, null_word_list):
    from utils_eval import load_transcript, segment_data, windows

    with open(Path(config.DATA_TEST_DIR) / "eval_segments.json", encoding="utf-8") as f:
        eval_segments = json.load(f)
    pred_data = np.load(pred_path, allow_pickle=True)
    pred_words, pred_times = pred_data["words"], pred_data["times"]

    ref_data = load_transcript(experiment, task)
    window_cutoffs = windows(*eval_segments[task], config.WINDOW)
    ref_windows = segment_data(ref_data["words"], ref_data["times"], window_cutoffs)
    pred_windows = segment_data(pred_words, pred_times, window_cutoffs)
    null_window_list = [
        segment_data(null_words, pred_times, window_cutoffs)
        for null_words in null_word_list
    ]

    rows = []
    for metric_name, metric in metrics.items():
        window_null_scores = np.array(
            [metric.score(ref=ref_windows, pred=null_windows) for null_windows in null_window_list]
        )
        story_null_scores = window_null_scores.mean(axis=1)
        window_scores = metric.score(ref=ref_windows, pred=pred_windows)
        story_score = float(window_scores.mean())
        null_mean = float(story_null_scores.mean())
        null_std = float(story_null_scores.std())
        zscore = float((story_score - null_mean) / null_std) if null_std > 0 else 0.0
        sig_window_fraction = float(
            np.mean(window_scores > window_null_scores.mean(axis=0) + 1.96 * window_null_scores.std(axis=0))
        )
        rows.append(
            {
                "subject": subject,
                "experiment": experiment,
                "task": task,
                "metric": metric_name,
                "story_score": story_score,
                "null_mean": null_mean,
                "null_std": null_std,
                "story_zscore": zscore,
                "sig_window_fraction": sig_window_fraction,
                "n_windows": len(window_scores),
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config.GPT_DEVICE = args.device
    config.EM_DEVICE = args.device
    config.SM_DEVICE = args.device

    paper_headlines = {}
    if args.paper_headline_json:
        with open(Path(args.paper_headline_json).expanduser(), encoding="utf-8") as f:
            paper_headlines = json.load(f)

    huth_lm = load_huth_lm(args.device, checkpoint=gpt_checkpoint_for_experiment(args.experiment))
    metrics = load_metrics(args)
    all_rows = []

    for task in args.tasks:
        print(f"\n=== {args.subject} / {args.experiment} / {task} ===")
        print(f"Generating {args.n_null} null decodes for shared score normalization")
        from utils_eval import generate_null

        # Use paper condition timing for null generation; all conditions share the same word times.
        resp = load_test_response(args.subject, args.experiment, task)
        word_times, _tr_times = load_word_times(args.subject, args.experiment, task, resp)
        null_words = generate_null(
            word_times,
            gpt_checkpoint_for_experiment(args.experiment),
            args.n_null,
        )

        for condition in args.conditions:
            pred_path = decode_one(condition, args.subject, args.experiment, task, args, huth_lm)
            rows = score_one(pred_path, args.subject, args.experiment, task, metrics, null_words)
            for row in rows:
                row["condition"] = condition
                if row["metric"] in paper_headlines:
                    row["paper_headline"] = float(paper_headlines[row["metric"]])
                    row["delta_vs_paper_headline"] = row["story_score"] - row["paper_headline"]
                else:
                    row["paper_headline"] = ""
                    row["delta_vs_paper_headline"] = ""
            all_rows.extend(rows)

    out_root = Path(args.output_dir).expanduser().resolve()
    csv_path = out_root / args.subject / args.experiment / "decoding_score_summary.csv"
    write_csv(csv_path, all_rows)
    print(f"\nWrote {csv_path}")
    for row in all_rows:
        print(
            f"{row['condition']:<10s} {row['task']:<18s} {row['metric']:<6s} "
            f"score={row['story_score']:.4f} z={row['story_zscore']:.2f} "
            f"sig_windows={row['sig_window_fraction']:.1%}"
        )


if __name__ == "__main__":
    main()
