# Generate TR-Aligned Story Summaries

This directory contains a test-case pipeline for generating rolling story
summaries at each TR using the OpenAI API.

## Script

- `generate_story_summaries.py`

For each TR in one story, it summarizes the last `N` words (default windows:
50, 200, 500), enforces a fixed summary length (default: 50 words), and saves:

- one JSONL per context window
- one metadata JSON per context window

## Setup

1. Install dependencies:

```bash
pip install openai
```

2. Set API key:

```bash
export OPENAI_API_KEY="your_key_here"
```

## Usage

List available stories:

```bash
python generate_summaries/generate_story_summaries.py --list-stories
```

Run a quick test on one story (first 20 TRs):

```bash
python generate_summaries/generate_story_summaries.py \
  --story alternateithicatom \
  --max-trs 20
```

Run full story:

```bash
python generate_summaries/generate_story_summaries.py \
  --story alternateithicatom \
  --summary-words 50 \
  --overwrite
```

## Output format

Default output directory:

- `generate_summaries/outputs/`

Generated files:

- `<story>.<model>.ctx50.jsonl`
- `<story>.<model>.ctx50.meta.json`
- `<story>.<model>.ctx200.jsonl`
- `<story>.<model>.ctx200.meta.json`
- `<story>.<model>.ctx500.jsonl`
- `<story>.<model>.ctx500.meta.json`

Each JSONL line has:

- `story`, `model`
- `context_window_words`, `summary_words`
- `tr_index`, `tr_time_s`
- `n_words_seen`
- `context_words_used` (actual available words in that context)
- `summary` (plain text from model, truncated to at most `summary_words`)
- `summary_word_count`
