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
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "decoding"))
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_DIR / "mindeye_text"))

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
from model import MindEyeEncoding  # noqa: E402
from utils_stim import predict_word_rate, predict_word_times  # noqa: E402
from utils_ridge.textgrid import TextGrid  # noqa: E402


CONDITION_PAPER = "paper"
CONDITION_MINDEYE_ENCODING = "mindeye_encoding"
CONDITION_LM_ONLY = "lm_only"
DEFAULT_TASKS = ["wheretheressmoke"]
BAD_WORDS_PERCEIVED_SPEECH = frozenset(["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp"])
BAD_WORDS_OTHER_TASKS = frozenset(["", "sp", "uh"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="S1")
    parser.add_argument("--experiment", default="perceived_speech")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[CONDITION_PAPER, CONDITION_FINETUNED, CONDITION_PRETRAINED],
        choices=[
            CONDITION_PAPER,
            CONDITION_FINETUNED,
            CONDITION_PRETRAINED,
            CONDITION_MINDEYE_ENCODING,
            CONDITION_LM_ONLY,
        ],
    )
    parser.add_argument(
        "--trained-model-dir",
        default=str(THIS_DIR / "outputs"),
        help="Root containing <subject>/encoding_model_finetuned.npz and pretrained.npz.",
    )
    parser.add_argument("--pretrained-model", default="openai-gpt")
    parser.add_argument("--finetuned-checkpoint", default="perceived")
    parser.add_argument(
        "--mindeye-encoding-checkpoint",
        default=None,
        help="Path to mindeye_text/train_mindeye_encoding.py model.pt.",
    )
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


def load_transcript(experiment, task):
    skip_words = (
        BAD_WORDS_PERCEIVED_SPEECH
        if experiment in {"perceived_speech", "perceived_multispeaker"}
        else BAD_WORDS_OTHER_TASKS
    )
    grid_path = Path(config.DATA_TEST_DIR) / "test_stimulus" / experiment / f"{task.split('_')[0]}.TextGrid"
    with open(grid_path, encoding="utf-8") as f:
        grid = TextGrid(f.read())
    transcript = (
        grid.tiers[1].make_simple_transcript()
        if experiment == "perceived_speech"
        else grid.tiers[0].make_simple_transcript()
    )
    transcript = [
        (float(start), float(end), word.lower())
        for start, end, word in transcript
        if word.lower().strip("{}").strip() not in skip_words
    ]
    return {
        "words": np.array([item[2] for item in transcript]),
        "times": np.array([(item[0] + item[1]) / 2 for item in transcript]),
    }


def windows(start_time, end_time, duration, step=1):
    start_time, end_time = int(start_time), int(end_time)
    half = int(duration / 2)
    return [
        (center - half, center + half)
        for center in range(start_time + half, end_time - half + 1)
        if center % step == 0
    ]


def segment_data(data, times, cutoffs):
    return [
        [item for time, item in zip(times, data) if time >= start and time < end]
        for start, end in cutoffs
    ]


def generate_null(pred_times, gpt_checkpoint, n, device):
    """Generate null sequences with the same word times as the prediction."""
    _gpt, lm = load_huth_lm(device, checkpoint=gpt_checkpoint)
    null_words = []
    for _count in range(n):
        decoder = Decoder(pred_times, 2 * config.EXTENSIONS)
        for sample_index in range(len(pred_times)):
            ncontext = decoder.time_window(sample_index, config.LM_TIME, floor=5)
            beam_nucs = lm.beam_propose(decoder.beam, ncontext)
            for beam_index, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
                nuc, logprobs = beam_nucs[beam_index]
                if len(nuc) < 1:
                    continue
                likelihoods = np.random.random(len(nuc))
                local_extensions = [
                    Hypothesis(parent=hyp, extension=item)
                    for item in zip(nuc, logprobs, [np.zeros(1) for _ in nuc])
                ]
                decoder.add_extensions(local_extensions, likelihoods, nextensions)
            decoder.extend(verbose=False)
        null_words.append(decoder.beam[0].words)
    return null_words


class WERMetric:
    def __init__(self, use_score=True):
        from jiwer import wer

        self.wer = wer
        self.use_score = use_score

    def score(self, ref, pred):
        scores = []
        for ref_seg, pred_seg in zip(ref, pred):
            # jiwer treats list[str] as sentence batches, not token lists, so
            # score each window as a single whitespace-joined utterance.
            error = 1.0 if len(ref_seg) == 0 else self.wer(" ".join(ref_seg), " ".join(pred_seg))
            scores.append(1 - error if self.use_score else error)
        return np.array(scores)


def load_metric_compat(name):
    try:
        from evaluate import load

        return load(name)
    except ImportError:
        from datasets import load_metric

        return load_metric(name, keep_in_memory=True)


class BLEUMetric:
    def __init__(self, n=1):
        self.metric = load_metric_compat("bleu")
        self.n = n

    def score(self, ref, pred):
        results = []
        for ref_seg, pred_seg in zip(ref, pred):
            if len(ref_seg) == 0:
                results.append(0.0)
                continue
            ref_string = " ".join(ref_seg)
            pred_string = " ".join(pred_seg)
            computed = self.metric.compute(
                predictions=[pred_string],
                references=[[ref_string]],
                max_order=self.n,
            )
            results.append(computed["bleu"])
        return np.array(results)


class METEORMetric:
    def __init__(self):
        self.metric = load_metric_compat("meteor")

    def score(self, ref, pred):
        ref_strings = [" ".join(seg) for seg in ref]
        pred_strings = [" ".join(seg) for seg in pred]
        results = []
        for ref_string, pred_string in zip(ref_strings, pred_strings):
            computed = self.metric.compute(predictions=[pred_string], references=[ref_string])
            results.append(computed["meteor"])
        return np.array(results)


class BERTScoreMetric:
    def __init__(self, idf_sents=None, rescale=False, score="recall"):
        from bert_score import BERTScorer

        self.metric = BERTScorer(
            lang="en",
            rescale_with_baseline=rescale,
            idf=idf_sents is not None,
            idf_sents=idf_sents,
        )
        if score == "precision":
            self.score_id = 0
        elif score == "recall":
            self.score_id = 1
        else:
            self.score_id = 2

    def score(self, ref, pred):
        ref_strings = [" ".join(seg) for seg in ref]
        pred_strings = [" ".join(seg) for seg in pred]
        return self.metric.score(cands=pred_strings, refs=ref_strings)[self.score_id].numpy()


def torch_load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class NeuralEncodingModel:
    """Decoder likelihood wrapper for MindEyeEncoding checkpoints."""

    def __init__(self, checkpoint, subject, resp, device="cpu"):
        if subject not in checkpoint["subjects"]:
            raise KeyError(f"Subject {subject!r} not in neural checkpoint subjects={checkpoint['subjects']}")
        self.subject = subject
        self.device = device
        self.voxels = np.asarray(checkpoint["voxels"][subject], dtype=np.int64)
        self.resp = torch.from_numpy(resp[:, self.voxels]).float().to(device)
        self.sigma = np.asarray(checkpoint["noise_model"][subject], dtype=np.float32)
        self.model = MindEyeEncoding(
            output_dims={str(k): int(v) for k, v in checkpoint["output_dims"].items()},
            input_dim=int(checkpoint["input_dim"]),
            latent_dim=int(checkpoint["latent_dim"]),
            n_blocks=int(checkpoint["n_blocks"]),
            dropout=float(checkpoint["dropout"]),
            input_norm=bool(checkpoint.get("input_norm", True)),
            head_norm=bool(checkpoint.get("head_norm", True)),
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def set_shrinkage(self, alpha):
        precision = np.linalg.inv(self.sigma * (1 - alpha) + np.eye(len(self.sigma)) * alpha)
        self.precision = torch.from_numpy(precision).float().to(self.device)

    def prs(self, stim, trs):
        with torch.no_grad():
            stim_t = stim.float().to(self.device)
            n_variants, n_trs, n_feats = stim_t.shape
            flat = stim_t.reshape(n_variants * n_trs, n_feats)
            pred = self.model(flat, self.subject).reshape(n_variants, n_trs, -1)
            diff = pred - self.resp[trs]
            multi = torch.matmul(torch.matmul(diff, self.precision), diff.permute(0, 2, 1))
            return -0.5 * multi.diagonal(dim1=-2, dim2=-1).sum(dim=1).detach().cpu().numpy()


def make_features(condition, args):
    if condition in {CONDITION_PAPER, CONDITION_FINETUNED, CONDITION_MINDEYE_ENCODING}:
        return HuthFinetunedGPT1Features(args.finetuned_checkpoint, args.device)
    if condition == CONDITION_PRETRAINED:
        return HFOpenAIGPTFeatures(args.pretrained_model, args.device, args.pretrained_batch_size)
    raise ValueError(f"Unknown condition: {condition}")


def encoding_model_path(condition, subject, args):
    if condition == CONDITION_PAPER:
        return Path(config.MODEL_DIR) / subject / "encoding_model_perceived.npz"
    return Path(args.trained_model_dir).expanduser().resolve() / subject / f"encoding_model_{condition}.npz"


def load_mindeye_encoding_checkpoint(args):
    if not args.mindeye_encoding_checkpoint:
        raise ValueError(
            "--mindeye-encoding-checkpoint is required when using "
            f"--conditions {CONDITION_MINDEYE_ENCODING}"
        )
    path = Path(args.mindeye_encoding_checkpoint).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"MindEye encoding checkpoint not found: {path}")
    return torch_load_checkpoint(path, map_location="cpu")


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


def as_numeric_array(value):
    """Convert npz object-array payloads back into numeric ndarrays."""
    arr = np.asarray(value)
    while arr.dtype == object and arr.shape == ():
        arr = np.asarray(arr.item())
    if arr.dtype == object:
        arr = np.asarray(arr.tolist())
    return arr.astype(np.float32, copy=False)


def load_stimulus_stats(em_data):
    """Return numeric stats in the form expected by decoding.StimulusModel."""
    tr_stats = em_data["tr_stats"]
    word_stats = em_data["word_stats"]
    return (
        (as_numeric_array(tr_stats[0]), as_numeric_array(tr_stats[1])),
        as_numeric_array(word_stats[0]),
    )


def decode_lm_only(pred_path, word_times, huth_lm, args, task):
    """Decode using only the GPT proposal model, without brain likelihoods."""
    _huth_gpt, lm = huth_lm
    decoder = Decoder(word_times, args.beam_width)
    t0 = time.time()
    for sample_index in range(len(word_times)):
        if sample_index % 25 == 0 or sample_index == len(word_times) - 1:
            elapsed = time.time() - t0
            print(
                f"\r[{CONDITION_LM_ONLY}] {task}: word {sample_index + 1}/{len(word_times)} "
                f"elapsed={elapsed:.0f}s",
                end="",
                flush=True,
            )

        ncontext = decoder.time_window(sample_index, config.LM_TIME, floor=5)
        beam_nucs = lm.beam_propose(decoder.beam, ncontext)
        for beam_index, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
            nuc, logprobs = beam_nucs[beam_index]
            if len(nuc) < 1:
                continue
            # Decoder ranks extensions by "likelihoods"; here that is just the
            # language model log-probability, with dummy embeddings unused later.
            local_extensions = [
                Hypothesis(parent=hyp, extension=item)
                for item in zip(nuc, logprobs, [np.zeros(1, dtype=np.float32) for _ in nuc])
            ]
            decoder.add_extensions(local_extensions, np.asarray(logprobs), nextensions)
        decoder.extend(verbose=False)

    decoder.save(str(pred_path.with_suffix("")))
    print(f"\n[{CONDITION_LM_ONLY}] saved {pred_path}")
    return pred_path


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

    if condition == CONDITION_LM_ONLY:
        return decode_lm_only(pred_path, word_times, huth_lm, args, task)

    if condition == CONDITION_MINDEYE_ENCODING:
        em_data = load_mindeye_encoding_checkpoint(args)
        if task in set(str(x) for x in em_data["stories"]):
            raise ValueError(f"{task!r} appears in training stories for {args.mindeye_encoding_checkpoint}")
    else:
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
        tr_stats, word_mean = load_stimulus_stats(em_data)
        if condition == CONDITION_MINDEYE_ENCODING:
            em = NeuralEncodingModel(em_data, subject, resp, device=config.EM_DEVICE)
        else:
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
            tr_stats,
            word_mean,
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
    metrics = {}
    if "WER" in args.metrics:
        metrics["WER"] = WERMetric(use_score=True)
    if "BLEU" in args.metrics:
        metrics["BLEU"] = BLEUMetric(n=1)
    if "METEOR" in args.metrics:
        metrics["METEOR"] = METEORMetric()
    if "BERT" in args.metrics:
        idf_path = Path(config.DATA_TEST_DIR) / "idf_segments.npy"
        idf_sents = np.load(idf_path) if idf_path.exists() else None
        metrics["BERT"] = BERTScoreMetric(idf_sents=idf_sents, rescale=False, score="recall")
    return metrics


def score_one(pred_path, subject, experiment, task, metrics, null_word_list):
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

        # Use paper condition timing for null generation; all conditions share the same word times.
        resp = load_test_response(args.subject, args.experiment, task)
        word_times, _tr_times = load_word_times(args.subject, args.experiment, task, resp)
        null_words = generate_null(
            word_times,
            gpt_checkpoint_for_experiment(args.experiment),
            args.n_null,
            args.device,
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
