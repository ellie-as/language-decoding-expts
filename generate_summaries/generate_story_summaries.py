#!/usr/bin/env python3
"""
Generate rolling context summaries at each TR for one or more stories.

For each TR, this script summarizes the last N words
(default: 20, 50, 200, 500)
using the OpenAI API and enforces a fixed summary length (default: 50 words).
It saves separate JSONL + metadata files per context length.

Usage:
  python generate_summaries/generate_story_summaries.py --story alternateithicatom
  python generate_summaries/generate_story_summaries.py
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "decoding"))

import config  # noqa: E402
from openai import APITimeoutError, APIError, OpenAI, RateLimitError  # noqa: E402
from utils_stim import get_story_wordseqs  # noqa: E402


def list_available_stories() -> List[str]:
    sess_to_story_path = Path(config.DATA_TRAIN_DIR) / "sess_to_story.json"
    if not sess_to_story_path.exists():
        return []

    with open(sess_to_story_path, "r", encoding="utf-8") as f:
        sess_to_story = json.load(f)

    stories = set()
    for value in sess_to_story.values():
        if isinstance(value, list):
            stories.update(value)
    return sorted(stories)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rolling summaries at each TR for one or all stories."
    )
    parser.add_argument(
        "--story",
        type=str,
        default=None,
        help="Story name from data_train textgrids. If omitted, runs all stories.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[20, 50, 200, 500],
        help="Word-window sizes to summarize at each TR.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--summary-words",
        type=int,
        default=50,
        help="Exact number of words each summary should contain.",
    )
    parser.add_argument(
        "--max-summary-tokens",
        type=int,
        default=140,
        help="Max completion tokens for each API call.",
    )
    parser.add_argument(
        "--max-trs",
        type=int,
        default=None,
        help="Optional cap on number of TRs (useful for quick tests).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_DIR / "generate_summaries" / "outputs"),
        help="Directory for JSONL + metadata output.",
    )
    parser.add_argument(
        "--list-stories",
        action="store_true",
        help="Print available story names and exit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files for this story/model.",
    )
    return parser.parse_args()


def build_user_prompt(
    words: List[str],
    end_exclusive: int,
    context_window: int,
    summary_words: int,
) -> str:
    start = max(0, end_exclusive - context_window)
    snippet = " ".join(words[start:end_exclusive]).strip()
    snippet = snippet if snippet else "[NO WORDS YET]"

    return (
        f"Summarize the context below in exactly {summary_words} words.\n"
        "Preserve story content faithfully. Do not add unsupported details.\n"
        "Return only the summary text, with no preamble, labels, quotes, or JSON.\n\n"
        f"TEXT:\n{snippet}"
    )


def enforce_max_word_count(text: str, target_words: int) -> str:
    tokens = text.strip().split()
    if len(tokens) <= target_words:
        return " ".join(tokens)
    return " ".join(tokens[:target_words])


def request_summary(
    client: OpenAI,
    model: str,
    words: List[str],
    end_exclusive: int,
    context_window: int,
    summary_words: int,
    max_summary_tokens: int,
    n_retries: int = 5,
) -> Dict[str, object]:
    if end_exclusive <= 0:
        return {"summary": "", "summary_word_count": 0}

    prompt = build_user_prompt(
        words=words,
        end_exclusive=end_exclusive,
        context_window=context_window,
        summary_words=summary_words,
    )
    system_prompt = (
        "You summarize stories faithfully. "
        "Do not add speculation. "
        "Output only summary text."
    )

    for attempt in range(n_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                max_tokens=max_summary_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            raw_summary = (response.choices[0].message.content or "").strip()
            fixed_summary = enforce_max_word_count(raw_summary, target_words=summary_words)
            return {
                "summary": fixed_summary,
                "summary_word_count": len(fixed_summary.split()) if fixed_summary else 0,
            }
        except (RateLimitError, APITimeoutError, APIError, ValueError) as err:
            if attempt == n_retries - 1:
                raise
            wait_s = min(2 ** attempt, 30)
            print(f"  API issue ({type(err).__name__}), retrying in {wait_s}s...")
            time.sleep(wait_s)

    raise RuntimeError("Unreachable retry loop state.")


def main() -> None:
    args = parse_args()
    available_stories = list_available_stories()

    if args.list_stories:
        if not available_stories:
            print("No stories found. Is data_train/sess_to_story.json available?")
            return
        print("\n".join(available_stories))
        return

    if args.story:
        if args.story not in available_stories and available_stories:
            raise ValueError(
                f"Story '{args.story}' not found in sess_to_story.json. "
                f"Example stories: {available_stories[:10]}"
            )
        stories_to_run = [args.story]
    else:
        if not available_stories:
            raise ValueError(
                "No stories found in data_train/sess_to_story.json. "
                "Use --story if you want to run a specific story path."
            )
        stories_to_run = available_stories

    windows = sorted(set(args.windows))
    if windows[0] <= 0:
        raise ValueError("--windows must be positive integers.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export it before running this script."
        )

    client = OpenAI()

    safe_model = args.model.replace("/", "_")
    print(f"Running {len(stories_to_run)} story(s): {', '.join(stories_to_run)}")

    for story_name in stories_to_run:
        output_paths = {
            w: {
                "jsonl": out_dir / f"{story_name}.{safe_model}.ctx{w}.jsonl",
                "meta": out_dir / f"{story_name}.{safe_model}.ctx{w}.meta.json",
            }
            for w in windows
        }

        if not args.overwrite:
            existing = []
            for w in windows:
                if output_paths[w]["jsonl"].exists() or output_paths[w]["meta"].exists():
                    existing.append(w)
            if existing:
                raise FileExistsError(
                    f"Output exists for story '{story_name}' context windows: "
                    + ", ".join(str(w) for w in existing)
                    + ". Use --overwrite to replace."
                )

        print(f"\nLoading story word sequence: {story_name}")
        word_seq = get_story_wordseqs([story_name])[story_name]
        words = list(word_seq.data)

        tr_count = len(word_seq.split_inds) + 1
        tr_times = list(word_seq.tr_times)
        split_inds = list(word_seq.split_inds)
        end_exclusive_by_tr = split_inds + [len(words)]

        if args.max_trs is not None:
            tr_count = min(tr_count, args.max_trs)
            tr_times = tr_times[:tr_count]
            end_exclusive_by_tr = end_exclusive_by_tr[:tr_count]

        print(f"Story has {len(words)} words, {tr_count} TRs to summarize.")

        file_handles = {
            w: open(output_paths[w]["jsonl"], "w", encoding="utf-8")
            for w in windows
        }
        try:
            for tr_idx in range(tr_count):
                end_exclusive = int(end_exclusive_by_tr[tr_idx])
                for w in windows:
                    summary_data = request_summary(
                        client=client,
                        model=args.model,
                        words=words,
                        end_exclusive=end_exclusive,
                        context_window=w,
                        summary_words=args.summary_words,
                        max_summary_tokens=args.max_summary_tokens,
                    )
                    row = {
                        "story": story_name,
                        "model": args.model,
                        "context_window_words": w,
                        "summary_words": args.summary_words,
                        "tr_index": tr_idx,
                        "tr_time_s": (
                            float(tr_times[tr_idx]) if tr_idx < len(tr_times) else None
                        ),
                        "n_words_seen": end_exclusive,
                        "context_words_used": int(min(w, end_exclusive)),
                        "summary": summary_data["summary"],
                        "summary_word_count": summary_data["summary_word_count"],
                    }
                    file_handles[w].write(json.dumps(row, ensure_ascii=True) + "\n")

                if (tr_idx + 1) % 10 == 0 or tr_idx == tr_count - 1:
                    print(f"  summarized TR {tr_idx + 1}/{tr_count}")
        finally:
            for handle in file_handles.values():
                handle.close()

        for w in windows:
            metadata = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "story": story_name,
                "model": args.model,
                "context_window_words": w,
                "summary_words": args.summary_words,
                "tr_count": tr_count,
                "total_words": len(words),
                "max_summary_tokens": args.max_summary_tokens,
                "jsonl_path": str(output_paths[w]["jsonl"]),
                "notes": (
                    "One JSON object per TR. Summary text is truncated to at most summary_words."
                ),
            }
            with open(output_paths[w]["meta"], "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        print("Saved files:")
        for w in windows:
            print(f"  ctx={w}: {output_paths[w]['jsonl']}")
            print(f"           {output_paths[w]['meta']}")


if __name__ == "__main__":
    main()
