#!/usr/bin/env python3
"""
Summary-horizon encoding analysis.

Uses precomputed TR-aligned story summaries as the stimulus representation.
For each summary horizon, this script:

1. Loads one summary per TR for every training story
2. Extracts one feature vector per summary using a chosen text model
3. Applies the same TR trimming and FIR delays as the context-length pipeline
4. Trains a ridge encoding model (summary features -> voxels)
5. Saves per-voxel prediction correlations for each feature-model / horizon pair

This is analogous to ``run_context_encoding.py``, except the experimental
manipulation is the summary horizon used to generate the summaries rather than
the raw word-context length used to extract token features.

Usage
-----
  python run_summaries_encoding.py \
      --subject S1 \
      --summaries-dir /path/to/summaries \
      --summary-model gpt-4o-mini \
      --models gpt1 gpt2 gpt2-pool embedding \
      --voxels-from-rois

  python run_summaries_encoding.py \
      --subject S1 \
      --stories wildwomenanddancingqueens \
      --summary-horizons 20 50 200 500 \
      --summaries-dir /path/to/summaries \
      --voxels-from-rois
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "decoding"))

import config  # noqa: E402
from GPT import GPT  # noqa: E402
from utils_resp import get_resp  # noqa: E402
from utils_ridge.ridge import bootstrap_ridge  # noqa: E402
from utils_ridge.util import make_delayed  # noqa: E402

SUBJECT_TO_UTS = {"S1": "UTS01", "S2": "UTS02", "S3": "UTS03"}
MODEL_CHOICES = ["gpt1", "gpt2", "gpt2-pool", "embedding"]
SUMMARY_FILE_RE = re.compile(
    r"^(?P<story>[^.]+)\.(?P<model>.+)\.ctx(?P<horizon>\d+)\.jsonl$"
)
TRIM_START = 5 + config.TRIM
TRIM_END = config.TRIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("summaries_encoding")
np.random.seed(42)


def _configure_huggingface_downloads():
    """Raise HF Hub timeouts for clusters / remote servers."""
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")


def _load_gpt2_with_retry(model_name_or_path, device, n_retries=4):
    """Load GPT-2 tokenizer + model with retries for transient network issues."""
    import time
    from transformers import GPT2Model, GPT2Tokenizer

    last_err = None
    for attempt in range(n_retries):
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = GPT2Model.from_pretrained(model_name_or_path).eval().to(device)
            return tokenizer, model
        except Exception as err:
            last_err = err
            wait = 30 * (attempt + 1)
            log.warning(
                "GPT-2 load failed (attempt %d/%d): %s — retrying in %ds",
                attempt + 1,
                n_retries,
                err,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        "Could not download/load GPT-2 from Hugging Face. "
        "Either warm the Hugging Face cache on a machine with internet access, "
        "copy the cache to the cluster, or pass --gpt2-model /path/to/local/gpt2.\n"
        f"Original error: {last_err}"
    ) from last_err


def load_ba_rois(ba_subject_dir):
    """Load subject-specific Brodmann area ROI definitions."""
    import glob as globmod

    rois = {}
    for path in sorted(globmod.glob(os.path.join(ba_subject_dir, "*.json"))):
        fname = os.path.basename(path)
        if fname == "BA_full_frontal.json":
            continue
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for key, indices in d.items():
            rois[key] = indices
    return rois


def load_voxel_set(subject, all_voxels):
    """Load pretrained language-responsive voxels when available."""
    em_path = os.path.join(config.MODEL_DIR, subject, "encoding_model_perceived.npz")
    if os.path.exists(em_path):
        em = np.load(em_path)
        vox = em["voxels"]
        log.info("Loaded %d language-responsive voxels from pretrained model", len(vox))
        return vox, True
    log.warning(
        "No pretrained model found at %s — using all %d voxels (may require a lot of RAM)",
        em_path,
        all_voxels,
    )
    return np.arange(all_voxels), False


def chunked_bootstrap_ridge(rstim, rresp, chunk_size=10000, **kwargs):
    """Run bootstrap_ridge in voxel chunks to reduce memory pressure."""
    n_voxels = rresp.shape[1]
    if n_voxels <= chunk_size:
        wt, valphas, bscorrs = bootstrap_ridge(rstim, rresp, **kwargs)
        corrs = bscorrs.mean(2).max(0)
        del valphas, bscorrs
        return corrs, wt

    all_corrs = np.zeros(n_voxels)
    all_wt = None
    for start in range(0, n_voxels, chunk_size):
        end = min(start + chunk_size, n_voxels)
        log.info("  Voxel chunk [%d:%d] / %d", start, end, n_voxels)
        wt, valphas, bscorrs = bootstrap_ridge(rstim, rresp[:, start:end], **kwargs)
        all_corrs[start:end] = bscorrs.mean(2).max(0)
        del wt, valphas, bscorrs
    return all_corrs, all_wt


def sanitize_name(value):
    """Make strings safe for filenames while keeping them readable."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value)


def load_story_list(args):
    """Resolve the ordered story list from --stories or --sessions."""
    if args.stories:
        return list(args.stories)

    sess_to_story_path = Path(config.DATA_TRAIN_DIR) / "sess_to_story.json"
    with open(sess_to_story_path, encoding="utf-8") as f:
        sess_to_story = json.load(f)

    stories = []
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])
    return stories


def build_summary_index(summaries_dir):
    """Scan summary JSONL files and index them by (story, model, horizon)."""
    index = {}
    for path in sorted(summaries_dir.glob("*.jsonl")):
        match = SUMMARY_FILE_RE.match(path.name)
        if not match:
            continue
        key = (
            match.group("story"),
            match.group("model"),
            int(match.group("horizon")),
        )
        if key in index:
            raise ValueError(
                f"Duplicate summary file for story/model/horizon {key}: "
                f"{index[key]} and {path}"
            )
        index[key] = path

    if not index:
        raise FileNotFoundError(
            f"No summary JSONL files matching '<story>.<model>.ctx<h>.jsonl' found in "
            f"{summaries_dir}"
        )
    return index


def resolve_summary_model(index, stories, requested_model=None):
    """Choose the summary model to analyze."""
    models_by_story = {}
    for story in stories:
        models = {model for (s, model, _h) in index if s == story}
        if not models:
            raise FileNotFoundError(f"No summary files found for story '{story}'")
        models_by_story[story] = models

    if requested_model:
        missing = [story for story, models in models_by_story.items() if requested_model not in models]
        if missing:
            raise FileNotFoundError(
                f"Requested --summary-model '{requested_model}' is missing for stories: "
                + ", ".join(missing)
            )
        return requested_model

    common_models = set.intersection(*(models for models in models_by_story.values()))
    if len(common_models) == 1:
        return next(iter(common_models))

    details = ", ".join(
        f"{story}: {sorted(models)}" for story, models in sorted(models_by_story.items())
    )
    raise ValueError(
        "Could not infer a unique summary model across the selected stories. "
        "Pass --summary-model explicitly. Available models by story: "
        f"{details}"
    )


def resolve_summary_horizons(index, stories, summary_model, requested_horizons=None):
    """Choose which horizons to analyze, requiring availability for all stories."""
    horizons_by_story = {}
    for story in stories:
        horizons = {h for (s, model, h) in index if s == story and model == summary_model}
        if not horizons:
            raise FileNotFoundError(
                f"No summary files found for story '{story}' and model '{summary_model}'"
            )
        horizons_by_story[story] = horizons

    common_horizons = sorted(set.intersection(*(horizons for horizons in horizons_by_story.values())))
    if not common_horizons:
        details = ", ".join(
            f"{story}: {sorted(horizons)}"
            for story, horizons in sorted(horizons_by_story.items())
        )
        raise FileNotFoundError(
            f"No summary horizons are shared across all stories for model '{summary_model}'. "
            f"Found: {details}"
        )

    if requested_horizons:
        requested = sorted(set(requested_horizons))
        missing = [h for h in requested if h not in common_horizons]
        if missing:
            details = ", ".join(
                f"{story}: {sorted(horizons)}"
                for story, horizons in sorted(horizons_by_story.items())
            )
            raise FileNotFoundError(
                "Some requested --summary-horizons are not available for every selected story. "
                f"Missing horizons: {missing}. Available by story: {details}"
            )
        return requested

    return common_horizons


def load_resp_lengths(subject, stories):
    """Read per-story response lengths without loading the full response matrices."""
    resp_dir = Path(config.DATA_TRAIN_DIR) / "train_response" / subject
    lengths = {}
    for story in stories:
        resp_path = resp_dir / f"{story}.hf5"
        if not resp_path.exists():
            raise FileNotFoundError(f"Missing response file: {resp_path}")
        with h5py.File(resp_path, "r") as hf:
            lengths[story] = int(hf["data"].shape[0])
    return lengths


def load_summary_texts(path, expected_story, expected_model, expected_horizon):
    """Load and validate one summary JSONL file."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"Summary file is empty: {path}")

    rows.sort(key=lambda row: int(row["tr_index"]))
    tr_indices = [int(row["tr_index"]) for row in rows]
    expected_indices = list(range(len(rows)))
    if tr_indices != expected_indices:
        raise ValueError(
            f"Summary file has non-consecutive tr_index values: {path} "
            f"(found {tr_indices[:10]}...)"
        )

    stories_in_file = {str(row.get("story", "")) for row in rows}
    if stories_in_file != {expected_story}:
        raise ValueError(
            f"Summary file story mismatch in {path}: expected '{expected_story}', "
            f"found {sorted(stories_in_file)}"
        )

    models_in_file = {str(row.get("model", "")) for row in rows}
    if models_in_file != {expected_model}:
        raise ValueError(
            f"Summary file model mismatch in {path}: expected '{expected_model}', "
            f"found {sorted(models_in_file)}"
        )

    horizons_in_file = {int(row.get("context_window_words", -1)) for row in rows}
    if horizons_in_file != {expected_horizon}:
        raise ValueError(
            f"Summary file horizon mismatch in {path}: expected {expected_horizon}, "
            f"found {sorted(horizons_in_file)}"
        )

    summary_word_values = {
        int(row["summary_words"])
        for row in rows
        if row.get("summary_words") is not None
    }
    if len(summary_word_values) > 1:
        raise ValueError(
            f"Inconsistent summary_words values in {path}: {sorted(summary_word_values)}"
        )

    return {
        "texts": [str(row.get("summary", "")) for row in rows],
        "summary_words": next(iter(summary_word_values)) if summary_word_values else None,
    }


class SummaryEmbeddingEncoder:
    """Sentence-transformer wrapper with stable handling of empty summaries."""

    def __init__(self, model_name, device, batch_size):
        _configure_huggingface_downloads()
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        log.info("Loading summary embedding model %r on %s", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts):
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        nonempty_idx = [i for i, text in enumerate(texts) if text.strip()]
        if not nonempty_idx:
            return vecs

        enc = self.model.encode(
            [texts[i] for i in nonempty_idx],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        vecs[nonempty_idx] = enc
        return vecs

    def close(self):
        del self.model
        torch.cuda.empty_cache()


class GPT1SummaryEncoder:
    """Encode each summary with GPT-1 layer-9 hidden state of the final word."""

    def __init__(self, device, batch_size):
        vocab_path = os.path.join(config.DATA_LM_DIR, "perceived", "vocab.json")
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)

        self.device = device
        self.batch_size = batch_size
        self.backend = "perceived"
        self.gpt = GPT(
            path=os.path.join(config.DATA_LM_DIR, "perceived", "model"),
            vocab=vocab,
            device=device,
        )
        hidden_dim = getattr(self.gpt.model.config, "n_embd", None)
        if hidden_dim is None:
            hidden_dim = getattr(self.gpt.model.config, "hidden_size")
        self.dim = int(hidden_dim)

    def encode(self, texts):
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        token_lists = [
            [token for token in text.split() if token.strip()]
            for text in texts
        ]
        nonempty = [(i, tokens) for i, tokens in enumerate(token_lists) if tokens]
        if not nonempty:
            return vecs

        for start in range(0, len(nonempty), self.batch_size):
            batch = nonempty[start:start + self.batch_size]
            lengths = [len(tokens) for _idx, tokens in batch]
            max_len = max(lengths)

            ids = np.full((len(batch), max_len), self.gpt.UNK_ID, dtype=np.int64)
            mask = np.zeros((len(batch), max_len), dtype=np.int64)
            row_to_original = []

            for row_index, (original_index, tokens) in enumerate(batch):
                token_ids = self.gpt.encode(tokens)
                ids[row_index, :len(token_ids)] = token_ids
                mask[row_index, :len(token_ids)] = 1
                row_to_original.append(original_index)

            ids_t = torch.tensor(ids, device=self.device)
            mask_t = torch.tensor(mask, device=self.device)
            with torch.no_grad():
                outputs = self.gpt.model(
                    input_ids=ids_t,
                    attention_mask=mask_t,
                    output_hidden_states=True,
                )
            hidden = outputs.hidden_states[config.GPT_LAYER]
            last_idx = mask_t.sum(dim=1) - 1
            batch_vecs = hidden[
                torch.arange(hidden.shape[0], device=self.device),
                last_idx,
            ].detach().cpu().numpy().astype(np.float32)
            vecs[row_to_original] = batch_vecs

        return vecs

    def close(self):
        del self.gpt
        torch.cuda.empty_cache()


class GPT2SummaryEncoder:
    """Encode each summary with GPT-2 layer-9 last-token or mean-pooled state."""

    def __init__(self, model_name_or_path, device, batch_size, pool):
        _configure_huggingface_downloads()
        self.device = device
        self.batch_size = batch_size
        self.pool = pool
        self.backend = model_name_or_path
        pool_tag = " (mean-pool)" if pool else " (last-token)"
        log.info("Loading GPT-2 from %r%s", model_name_or_path, pool_tag)
        self.tokenizer, self.model = _load_gpt2_with_retry(model_name_or_path, device)
        self.dim = int(self.model.config.n_embd)

    def encode(self, texts):
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        nonempty_idx = [i for i, text in enumerate(texts) if text.strip()]
        if not nonempty_idx:
            return vecs

        for start in range(0, len(nonempty_idx), self.batch_size):
            batch_idx = nonempty_idx[start:start + self.batch_size]
            batch_texts = [texts[i] for i in batch_idx]
            tok = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            tok = {key: value.to(self.device) for key, value in tok.items()}
            with torch.no_grad():
                outputs = self.model(**tok, output_hidden_states=True)
            hidden = outputs.hidden_states[config.GPT_LAYER]
            attention_mask = tok["attention_mask"]
            if self.pool:
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                batch_vecs = pooled.detach().cpu().numpy().astype(np.float32)
            else:
                last_idx = attention_mask.sum(dim=1) - 1
                batch_vecs = hidden[
                    torch.arange(hidden.shape[0], device=self.device),
                    last_idx,
                ].detach().cpu().numpy().astype(np.float32)
            vecs[batch_idx] = batch_vecs

        return vecs

    def close(self):
        del self.model
        torch.cuda.empty_cache()


def make_summary_encoder(model_type, args, device):
    """Construct the requested summary feature encoder."""
    if model_type == "gpt1":
        encoder = GPT1SummaryEncoder(device=device, batch_size=args.embed_batch_size)
    elif model_type == "gpt2":
        encoder = GPT2SummaryEncoder(
            model_name_or_path=args.gpt2_model,
            device=device,
            batch_size=args.embed_batch_size,
            pool=False,
        )
    elif model_type == "gpt2-pool":
        encoder = GPT2SummaryEncoder(
            model_name_or_path=args.gpt2_model,
            device=device,
            batch_size=args.embed_batch_size,
            pool=True,
        )
    elif model_type == "embedding":
        encoder = SummaryEmbeddingEncoder(
            model_name=args.embedding_model,
            device=device,
            batch_size=args.embed_batch_size,
        )
        encoder.backend = args.embedding_model
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    encoder.feature_model = model_type
    return encoder


def build_design_matrix(stories, texts_by_story, resp_lengths, encoder):
    """Embed per-TR summaries, trim to response-aligned TRs, z-score, add delays."""
    trimmed_story_vecs = []

    for story in stories:
        texts = texts_by_story[story]
        if len(texts) <= TRIM_START + TRIM_END:
            raise ValueError(
                f"Story '{story}' has only {len(texts)} summary TRs; "
                f"need more than {TRIM_START + TRIM_END} to apply trimming."
            )

        story_vecs = encoder.encode(texts)
        trimmed = story_vecs[TRIM_START:-TRIM_END]
        expected_resp_trs = resp_lengths[story]
        if trimmed.shape[0] != expected_resp_trs:
            raise ValueError(
                f"Story '{story}' has {len(texts)} summary TRs, which trims to "
                f"{trimmed.shape[0]} TRs, but the response file has {expected_resp_trs} TRs. "
                "This usually means the summary JSONL is incomplete or was generated against "
                "different story timing."
            )

        trimmed_story_vecs.append(trimmed)
        log.info(
            "  %s: %d summaries -> %s after trim -> response %d TRs",
            story,
            len(texts),
            trimmed.shape,
            expected_resp_trs,
        )

    ds_mat = np.vstack(trimmed_story_vecs)
    r_mean, r_std = ds_mat.mean(0), ds_mat.std(0)
    r_std[r_std == 0] = 1
    ds_mat = np.nan_to_num((ds_mat - r_mean) / r_std)
    return make_delayed(ds_mat, config.STIM_DELAYS)


def format_condition_row(meta):
    """Compact human-readable label for summaries ROI tables."""
    return f"{meta['feature_model']}_h{meta['summary_horizon']}"


def print_roi_summary(all_corrs, ba_subject_dir, vox, label_to_meta):
    """Print mean encoding correlation per ROI for each summary horizon."""
    rois = load_ba_rois(ba_subject_dir)
    region_names = sorted(rois.keys())
    global_to_local = {int(g): i for i, g in enumerate(vox)}

    local_rois = {}
    for region_name in region_names:
        local_rois[region_name] = np.array(
            [global_to_local[v] for v in rois[region_name] if v in global_to_local],
            dtype=int,
        )

    hdr = f"  {'condition':<22s}"
    for region_name in region_names:
        hdr += f"  {region_name + f' ({len(local_rois[region_name])})':>25s}"
    hdr += f"  {'all':>10s}"

    print("\n" + "=" * len(hdr))
    print("  Per-ROI mean encoding correlation")
    print("=" * len(hdr))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for label in sorted(
        all_corrs,
        key=lambda item: (
            label_to_meta[item]["feature_model"],
            label_to_meta[item]["summary_horizon"],
        ),
    ):
        corrs = all_corrs[label]
        display_label = format_condition_row(label_to_meta[label])
        row = f"  {display_label:<22s}"
        for region_name in region_names:
            idx = local_rois[region_name]
            mean_r = corrs[idx].mean() if len(idx) > 0 else float("nan")
            row += f"  {mean_r:25.4f}"
        row += f"  {corrs.mean():10.4f}"
        print(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--stories",
        nargs="+",
        default=None,
        help="Explicit story list. If omitted, stories are derived from --sessions.",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20],
        help="Training sessions to include when --stories is not provided.",
    )
    parser.add_argument(
        "--summaries-dir",
        default=str(REPO_DIR / "generate_summaries" / "outputs"),
        help="Directory containing summary JSONL files.",
    )
    parser.add_argument(
        "--summary-model",
        default=None,
        help="Summary source model to use, e.g. gpt-4o-mini. If omitted, inferred.",
    )
    parser.add_argument(
        "--summary-horizons",
        nargs="+",
        type=int,
        default=None,
        help="Summary horizons to analyze. If omitted, uses all horizons shared across stories.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_CHOICES,
        choices=MODEL_CHOICES,
        help="Feature models to use for encoding the summary text.",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model used to embed each summary.",
    )
    parser.add_argument(
        "--gpt2-model",
        default="openai-community/gpt2",
        help="Hugging Face GPT-2 id or local directory for gpt2 / gpt2-pool features.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Batch size for summary feature extraction.",
    )
    parser.add_argument(
        "--ba-dir",
        default=str(REPO_DIR / "ba_indices"),
        help="Directory containing per-subject Brodmann area indices.",
    )
    parser.add_argument("--nboots", type=int, default=config.NBOOTS)
    parser.add_argument(
        "--single-alpha",
        type=float,
        default=None,
        help="Use a single fixed ridge alpha instead of cross-validated search.",
    )
    parser.add_argument(
        "--output-dir",
        default="summaries_encoding_results",
        help="Results directory root relative to the repo root.",
    )
    parser.add_argument(
        "--voxels-from-rois",
        action="store_true",
        help="Restrict to frontal voxels from ba_indices/ using BA_full_frontal.json.",
    )
    parser.add_argument(
        "--all-voxels",
        action="store_true",
        help="Use all voxels instead of the pretrained language-responsive set.",
    )
    parser.add_argument(
        "--voxel-chunk-size",
        type=int,
        default=10000,
        help="Voxel chunk size when using large voxel sets.",
    )
    parser.add_argument(
        "--save-weights",
        action="store_true",
        help="Also save full regression weights (large files).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse any existing per-horizon .npz file instead of recomputing it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = config.EM_DEVICE
    log.info("Device: %s", device)

    stories = load_story_list(args)
    log.info("Stories (%d): %s", len(stories), stories)

    summaries_dir = Path(args.summaries_dir).expanduser().resolve()
    if not summaries_dir.is_dir():
        raise FileNotFoundError(f"No summaries directory found at {summaries_dir}")

    summary_index = build_summary_index(summaries_dir)
    summary_model = resolve_summary_model(summary_index, stories, args.summary_model)
    summary_horizons = resolve_summary_horizons(
        summary_index, stories, summary_model, args.summary_horizons
    )
    log.info("Summary model: %s", summary_model)
    log.info("Summary horizons: %s", summary_horizons)
    log.info("Feature models: %s", args.models)

    resp_lengths = load_resp_lengths(args.subject, stories)
    rresp_full = get_resp(args.subject, stories, stack=True)
    log.info("Full response matrix: %s (TRs x voxels)", rresp_full.shape)

    uts_id = SUBJECT_TO_UTS.get(args.subject)
    ba_subject_dir = os.path.join(args.ba_dir, uts_id) if uts_id else None

    if args.voxels_from_rois:
        if not ba_subject_dir or not os.path.isdir(ba_subject_dir):
            log.error(
                "--voxels-from-rois: no BA directory found at %s (subject %s -> %s)",
                ba_subject_dir,
                args.subject,
                uts_id,
            )
            sys.exit(1)
        frontal_path = os.path.join(ba_subject_dir, "BA_full_frontal.json")
        with open(frontal_path, encoding="utf-8") as f:
            frontal = json.load(f)
        frontal_voxels = list(frontal.values())[0]
        vox = np.sort(np.array(frontal_voxels, dtype=int))
        vox = vox[vox < rresp_full.shape[1]]
        log.info("Using %d frontal voxels from %s", len(vox), frontal_path)
    elif args.all_voxels:
        vox = np.arange(rresp_full.shape[1])
        log.info("Using ALL %d voxels (chunked processing)", len(vox))
    else:
        vox, _ = load_voxel_set(args.subject, rresp_full.shape[1])

    rresp = rresp_full[:, vox]
    del rresp_full
    log.info("Working response matrix: %s (TRs x voxels)", rresp.shape)

    out_dir = Path(config.REPO_DIR) / args.output_dir / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)

    all_corrs = {}
    label_to_meta = {}
    safe_summary_model = sanitize_name(summary_model)
    summary_cache = {}
    nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))
    alphas = np.array([args.single_alpha]) if args.single_alpha else config.ALPHAS

    def get_cached_summaries(horizon):
        if horizon in summary_cache:
            return summary_cache[horizon]

        texts_by_story = {}
        summary_word_values = set()
        for story in stories:
            path = summary_index[(story, summary_model, horizon)]
            loaded = load_summary_texts(
                path=path,
                expected_story=story,
                expected_model=summary_model,
                expected_horizon=horizon,
            )
            texts_by_story[story] = loaded["texts"]
            if loaded["summary_words"] is not None:
                summary_word_values.add(loaded["summary_words"])

        if len(summary_word_values) > 1:
            raise ValueError(
                f"Summary horizon {horizon} has inconsistent summary_words values: "
                f"{sorted(summary_word_values)}"
            )

        summary_cache[horizon] = (
            texts_by_story,
            next(iter(summary_word_values)) if summary_word_values else -1,
        )
        return summary_cache[horizon]

    for model_type in args.models:
        encoder = make_summary_encoder(model_type=model_type, args=args, device=device)
        safe_feature_model = sanitize_name(model_type)
        safe_feature_backend = sanitize_name(encoder.backend)

        try:
            for horizon in summary_horizons:
                label = (
                    f"{safe_feature_model}__{safe_feature_backend}__"
                    f"{safe_summary_model}__h{horizon}"
                )
                out_path = out_dir / f"{label}.npz"
                meta = {
                    "feature_model": model_type,
                    "feature_backend": encoder.backend,
                    "summary_horizon": horizon,
                    "summary_model": summary_model,
                }
                label_to_meta[label] = meta

                if args.skip_existing and out_path.exists():
                    existing = np.load(out_path, allow_pickle=True)
                    all_corrs[label] = existing["corrs"]
                    log.info("Skipping existing condition %s", label)
                    continue

                log.info("=" * 60)
                log.info("Condition: %s", label)
                log.info("=" * 60)

                texts_by_story, summary_words = get_cached_summaries(horizon)

                log.info("Extracting %s summary features and building design matrix...", model_type)
                rstim = build_design_matrix(stories, texts_by_story, resp_lengths, encoder)
                if rstim.shape[0] != rresp.shape[0]:
                    raise ValueError(
                        f"Design matrix rows ({rstim.shape[0]}) do not match response rows "
                        f"({rresp.shape[0]})."
                    )
                log.info("Design matrix: %s (TRs x delayed features)", rstim.shape)

                log.info(
                    "Bootstrap ridge regression (%d boots, chunklen=%d, nchunks=%d, alphas=%s)...",
                    args.nboots,
                    config.CHUNKLEN,
                    nchunks,
                    alphas,
                )

                corrs, wt = chunked_bootstrap_ridge(
                    rstim,
                    rresp,
                    chunk_size=args.voxel_chunk_size,
                    alphas=alphas,
                    nboots=args.nboots,
                    chunklen=config.CHUNKLEN,
                    nchunks=nchunks,
                    use_corr=True,
                )
                del rstim

                all_corrs[label] = corrs
                log.info(
                    "  mean r=%.4f | max r=%.4f | n(r>0.1)=%d",
                    corrs.mean(),
                    corrs.max(),
                    (corrs > 0.1).sum(),
                )

                save_dict = dict(
                    corrs=corrs,
                    voxels=vox,
                    feature_model=np.array(model_type),
                    feature_backend=np.array(encoder.backend),
                    summary_horizon=np.array(horizon),
                    summary_model=np.array(summary_model),
                    embedding_model=np.array(args.embedding_model),
                    summary_words=np.array(summary_words),
                    stories=np.array(stories),
                    condition_label=np.array(label),
                )
                if args.save_weights and wt is not None:
                    save_dict["weights"] = wt

                np.savez(out_path, **save_dict)
                del wt
                log.info("  -> saved %s", out_path)
        finally:
            encoder.close()

    summary = {label: corr for label, corr in all_corrs.items()}
    summary["summary_horizons"] = np.array(summary_horizons)
    summary["summary_model"] = np.array(summary_model)
    summary["feature_models"] = np.array(args.models)
    summary["embedding_model"] = np.array(args.embedding_model)
    summary["gpt2_model"] = np.array(args.gpt2_model)
    summary["voxels"] = vox
    np.savez(out_dir / "summary.npz", **summary)

    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY — per-voxel encoding correlation (%d voxels)", len(vox))
    log.info("=" * 60)
    for label in sorted(
        all_corrs,
        key=lambda item: (
            label_to_meta[item]["feature_model"],
            label_to_meta[item]["summary_horizon"],
        ),
    ):
        corrs = all_corrs[label]
        log.info(
            "  %-18s h=%-5d mean=%.4f  max=%.4f  n(r>0.1)=%d",
            label_to_meta[label]["feature_model"],
            label_to_meta[label]["summary_horizon"],
            corrs.mean(),
            corrs.max(),
            (corrs > 0.1).sum(),
        )

    if ba_subject_dir and os.path.isdir(ba_subject_dir):
        print_roi_summary(all_corrs, ba_subject_dir, vox, label_to_meta)

    log.info("All results saved to %s", out_dir)


if __name__ == "__main__":
    main()
