#!/usr/bin/env python3
"""
Generate story continuations from the repo's GPT-1 checkpoint.

This script reconstructs a prompt from the first paragraph of each training
story using the story TextGrid word tier, feeds that prompt to a local GPT-1
checkpoint from ``data_lm/<checkpoint>/``, and saves the generated continuation
alongside the prompt and the true next words from the story.

The "first paragraph" is approximated from the first substantial break in the
word tier:

1. explicit break markers such as ``br`` / ``{BR}`` / ``lg`` / ``{LG}``
2. otherwise, the first silent interval at least ``--break-pause-s`` long
3. otherwise, the first ``--fallback-words`` lexical tokens

Usage
-----
  python continue_stories_with_gpt1.py --checkpoint perceived

  python continue_stories_with_gpt1.py \
      --checkpoint perceived \
      --stories alternateithicatom avatar \
      --max-new-words 80 \
      --num-return-sequences 3
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "decoding"))

import config  # noqa: E402
from GPT import GPT  # noqa: E402
from utils_ridge.textgrid import TextGrid  # noqa: E402

PAUSE_LABELS = {
    "",
    "sp",
    "br",
    "lg",
    "ls",
    "ns",
    "cg",
    "{br}",
    "{lg}",
    "{ls}",
    "{ns}",
    "{cg}",
}
EXPLICIT_BREAK_LABELS = {"br", "lg", "{br}", "{lg}"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default="perceived",
        help="Checkpoint name under data_lm/, e.g. perceived or imagined.",
    )
    parser.add_argument(
        "--stories",
        nargs="+",
        default=None,
        help="Explicit story list. If omitted, uses stories from --sessions.",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20],
        help="Sessions used to resolve stories when --stories is not provided.",
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=None,
        help="Optional cap on the number of stories to run.",
    )
    parser.add_argument(
        "--list-stories",
        action="store_true",
        help="Print available training stories and exit.",
    )
    parser.add_argument(
        "--max-new-words",
        type=int,
        default=80,
        help="Maximum number of new words to generate.",
    )
    parser.add_argument(
        "--reference-words",
        type=int,
        default=80,
        help="Number of true continuation words to save for comparison.",
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=1,
        help="Number of sampled continuations per story.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (ignored with --greedy).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling threshold (ignored with --greedy).",
    )
    parser.add_argument(
        "--break-pause-s",
        type=float,
        default=1.5,
        help="Pause length treated as a paragraph break when no explicit break marker appears.",
    )
    parser.add_argument(
        "--min-prompt-words",
        type=int,
        default=40,
        help="Require at least this many lexical words before allowing a break to end the prompt.",
    )
    parser.add_argument(
        "--fallback-words",
        type=int,
        default=120,
        help="Prompt length if no paragraph-like break is found.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_DIR / "gpt1_story_continuations"),
        help="Directory for JSONL and readable text output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def list_available_stories():
    sess_to_story_path = Path(config.DATA_TRAIN_DIR) / "sess_to_story.json"
    if not sess_to_story_path.exists():
        return []

    with open(sess_to_story_path, encoding="utf-8") as f:
        sess_to_story = json.load(f)

    stories = set()
    for value in sess_to_story.values():
        if isinstance(value, list):
            stories.update(value)
    return sorted(stories)


def resolve_stories(args):
    available = list_available_stories()
    if args.list_stories:
        print("\n".join(available))
        return None

    if args.stories:
        stories = list(args.stories)
    else:
        sess_to_story_path = Path(config.DATA_TRAIN_DIR) / "sess_to_story.json"
        with open(sess_to_story_path, encoding="utf-8") as f:
            sess_to_story = json.load(f)
        stories = []
        for sess in args.sessions:
            stories.extend(sess_to_story[str(sess)])

    if args.max_stories is not None:
        stories = stories[:args.max_stories]

    return stories


def normalize_token(text):
    return text.strip().lower()


def is_pause_label(label):
    return normalize_token(label) in PAUSE_LABELS


def load_word_intervals(story):
    tg_path = Path(config.DATA_TRAIN_DIR) / "train_stimulus" / f"{story}.TextGrid"
    if not tg_path.exists():
        raise FileNotFoundError(f"Missing TextGrid: {tg_path}")

    grid = TextGrid.load(str(tg_path))
    for tier in grid.tiers:
        if "word" in tier.nameid.lower():
            intervals = []
            for start, end, text in tier.simple_transcript:
                intervals.append((float(start), float(end), str(text)))
            return intervals

    raise ValueError(f"No word tier found in {tg_path}")


def lexical_story_words(intervals):
    words = []
    for _start, _end, label in intervals:
        token = normalize_token(label)
        if not token or is_pause_label(token):
            continue
        if token.startswith("{") and token.endswith("}"):
            continue
        words.append(token)
    return words


def extract_first_paragraph(intervals, min_prompt_words, break_pause_s, fallback_words, reference_words):
    prompt_words = []
    stop_reason = None

    for start, end, label in intervals:
        token = normalize_token(label)
        duration = float(end) - float(start)

        if not token:
            if len(prompt_words) >= min_prompt_words and duration >= break_pause_s:
                stop_reason = f"blank_pause>={break_pause_s:.2f}s"
                break
            continue

        if is_pause_label(token):
            if len(prompt_words) >= min_prompt_words and (
                token in EXPLICIT_BREAK_LABELS or duration >= break_pause_s
            ):
                stop_reason = f"{token or 'pause'}@{duration:.2f}s"
                break
            continue

        if token.startswith("{") and token.endswith("}"):
            continue

        prompt_words.append(token)

    all_words = lexical_story_words(intervals)
    if not all_words:
        raise ValueError("No lexical words found in story transcript.")

    if stop_reason is None:
        prompt_words = all_words[:fallback_words]
        stop_reason = f"fallback_first_{len(prompt_words)}_words"

    prompt_len = len(prompt_words)
    if prompt_len == 0:
        raise ValueError("Prompt extraction produced zero words.")

    reference = all_words[prompt_len:prompt_len + reference_words]
    return {
        "prompt_words": prompt_words,
        "reference_words": reference,
        "prompt_method": stop_reason,
        "prompt_word_count": prompt_len,
        "story_word_count": len(all_words),
    }


def load_checkpoint(checkpoint_name):
    vocab_path = Path(config.DATA_LM_DIR) / checkpoint_name / "vocab.json"
    model_path = Path(config.DATA_LM_DIR) / checkpoint_name / "model"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocab file: {vocab_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model directory: {model_path}")

    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)

    return GPT(path=str(model_path), vocab=vocab, device=config.GPT_DEVICE)


def get_model_context_limit(model):
    return int(
        getattr(
            model.config,
            "n_positions",
            getattr(model.config, "max_position_embeddings", 512),
        )
    )


def generate_continuations(gpt, prompt_words, args):
    if args.greedy and args.num_return_sequences != 1:
        raise ValueError("--greedy only supports --num-return-sequences 1.")

    prompt_ids_full = gpt.encode(prompt_words)
    model_limit = get_model_context_limit(gpt.model)
    truncated = len(prompt_ids_full) > model_limit
    prompt_ids = prompt_ids_full[-model_limit:]

    prompt_unk_count = sum(token_id == gpt.UNK_ID for token_id in prompt_ids_full)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=config.GPT_DEVICE)

    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=args.max_new_words,
        num_return_sequences=args.num_return_sequences,
        pad_token_id=getattr(gpt.model.config, "pad_token_id", None) or gpt.UNK_ID,
    )

    eos_token_id = getattr(gpt.model.config, "eos_token_id", None)
    if eos_token_id is not None:
        generate_kwargs["eos_token_id"] = eos_token_id

    if args.greedy:
        generate_kwargs["do_sample"] = False
    else:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        outputs = gpt.model.generate(**generate_kwargs)

    results = []
    prompt_len_used = input_ids.shape[1]
    for output in outputs:
        output_ids = output.detach().cpu().tolist()
        continuation_ids = output_ids[prompt_len_used:]
        continuation_words = [gpt.vocab[token_id] for token_id in continuation_ids]
        results.append(
            {
                "continuation_words": continuation_words,
                "continuation_text": " ".join(continuation_words),
            }
        )

    return {
        "results": results,
        "prompt_unk_count": prompt_unk_count,
        "prompt_unk_fraction": prompt_unk_count / max(1, len(prompt_ids_full)),
        "prompt_truncated_to_model_limit": truncated,
        "model_context_limit": model_limit,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    stories = resolve_stories(args)
    if stories is None:
        return
    if not stories:
        raise ValueError("No stories resolved. Use --list-stories to inspect availability.")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = "greedy" if args.greedy else "sample"
    base_name = f"{args.checkpoint}.{mode_tag}.new{args.max_new_words}.n{args.num_return_sequences}"
    jsonl_path = out_dir / f"{base_name}.jsonl"
    txt_path = out_dir / f"{base_name}.txt"

    if not args.overwrite and (jsonl_path.exists() or txt_path.exists()):
        raise FileExistsError(
            f"Output already exists: {jsonl_path} or {txt_path}. Use --overwrite to replace."
        )

    print(f"Loading GPT checkpoint: {args.checkpoint}")
    gpt = load_checkpoint(args.checkpoint)
    print(f"Running {len(stories)} stories on device {config.GPT_DEVICE}")

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_f, open(txt_path, "w", encoding="utf-8") as txt_f:
        for idx, story in enumerate(stories, start=1):
            print(f"[{idx}/{len(stories)}] {story}")
            intervals = load_word_intervals(story)
            extracted = extract_first_paragraph(
                intervals=intervals,
                min_prompt_words=args.min_prompt_words,
                break_pause_s=args.break_pause_s,
                fallback_words=args.fallback_words,
                reference_words=args.reference_words,
            )
            generated = generate_continuations(gpt, extracted["prompt_words"], args)

            prompt_text = " ".join(extracted["prompt_words"])
            reference_text = " ".join(extracted["reference_words"])

            txt_f.write("=" * 100 + "\n")
            txt_f.write(f"STORY: {story}\n")
            txt_f.write(
                f"PROMPT WORDS: {extracted['prompt_word_count']} / {extracted['story_word_count']}  "
                f"METHOD: {extracted['prompt_method']}  "
                f"UNK: {generated['prompt_unk_count']} ({generated['prompt_unk_fraction']:.3f})\n"
            )
            if generated["prompt_truncated_to_model_limit"]:
                txt_f.write(f"PROMPT TRUNCATED TO LAST {generated['model_context_limit']} WORD IDS FOR GENERATION\n")
            txt_f.write("\nPROMPT:\n")
            txt_f.write(prompt_text + "\n")
            txt_f.write("\nREFERENCE CONTINUATION:\n")
            txt_f.write((reference_text or "[no held-out reference words]") + "\n")

            for sample_index, sample in enumerate(generated["results"], start=1):
                row = {
                    "story": story,
                    "checkpoint": args.checkpoint,
                    "sample_index": sample_index,
                    "greedy": args.greedy,
                    "temperature": args.temperature if not args.greedy else None,
                    "top_p": args.top_p if not args.greedy else None,
                    "max_new_words": args.max_new_words,
                    "prompt_word_count": extracted["prompt_word_count"],
                    "story_word_count": extracted["story_word_count"],
                    "prompt_method": extracted["prompt_method"],
                    "prompt_unk_count": generated["prompt_unk_count"],
                    "prompt_unk_fraction": generated["prompt_unk_fraction"],
                    "prompt_truncated_to_model_limit": generated["prompt_truncated_to_model_limit"],
                    "model_context_limit": generated["model_context_limit"],
                    "prompt_text": prompt_text,
                    "reference_text": reference_text,
                    "generated_text": sample["continuation_text"],
                }
                jsonl_f.write(json.dumps(row, ensure_ascii=True) + "\n")

                txt_f.write(f"\nGENERATED #{sample_index}:\n")
                txt_f.write((sample["continuation_text"] or "[empty generation]") + "\n")

            txt_f.write("\n")

    print(f"Saved {jsonl_path}")
    print(f"Saved {txt_path}")


if __name__ == "__main__":
    main()
