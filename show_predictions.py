#!/usr/bin/env python3
"""Print aligned windows of reference and decoded text side by side.

Usage:
    python show_predictions.py --subject S1 --experiment perceived_speech --task wheretheressmoke
    python show_predictions.py --subject S1 --experiment perceived_speech --task wheretheressmoke --window 30
"""
import os, sys, re, argparse
import numpy as np
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "decoding"))
import config

def load_textgrid_words(path):
    with open(path) as f:
        text = f.read()
    tier_pat = r'item \[(\d+)\]:\s+class = "IntervalTier"\s+name = "([^"]+)"'
    tiers = list(re.finditer(tier_pat, text))
    for i, m in enumerate(tiers):
        if "word" in m.group(2).lower():
            start = m.end()
            end = tiers[i+1].start() if i+1 < len(tiers) else len(text)
            chunk = text[start:end]
            break
    else:
        raise ValueError("No word tier found")

    iv_pat = r'intervals \[\d+\]:\s+xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"'
    skip = {"sp", "br", "lg", "ls", "ns", "cg", "{LG}", "{BR}", "{LS}", "{NS}", "{CG}", ""}
    matches = re.findall(iv_pat, chunk)
    return [(float(a), float(b), w) for a, b, w in matches if w.strip() and w.strip() not in skip]

def segment(words, times, cutoffs):
    segs = []
    for t0, t1 in cutoffs:
        seg = [w for tw, _, w in zip(times, times, words) if t0 <= tw < t1]
        # filter by matching word times
        seg = []
        for j, w in enumerate(words):
            if j < len(times) and t0 <= times[j] < t1:
                seg.append(w)
        segs.append(" ".join(seg))
    return segs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--window", type=float, default=20, help="Window size in seconds (default 20)")
    p.add_argument("--max-windows", type=int, default=None, help="Limit number of windows shown")
    args = p.parse_args()

    # Load prediction
    pred_path = REPO_DIR / "results" / args.subject / args.experiment / f"{args.task}.npz"
    if not pred_path.exists():
        print(f"No prediction file at {pred_path}")
        sys.exit(1)
    pred = np.load(str(pred_path), allow_pickle=True)
    pred_words = list(pred["words"])
    pred_times = pred["times"]
    print(f"Prediction: {len(pred_words)} words")

    # Load reference transcript
    tg_path = REPO_DIR / "data_test" / "test_stimulus" / args.experiment / f"{args.task.split('_')[0]}.TextGrid"
    if not tg_path.exists():
        # try without split
        tg_path = REPO_DIR / "data_test" / "test_stimulus" / args.experiment / f"{args.task}.TextGrid"
    if not tg_path.exists():
        print(f"No TextGrid at {tg_path}")
        print("Available:")
        stim_dir = REPO_DIR / "data_test" / "test_stimulus" / args.experiment
        if stim_dir.exists():
            for f in sorted(stim_dir.iterdir()):
                print(f"  {f.name}")
        sys.exit(1)

    ref_data = load_textgrid_words(str(tg_path))
    ref_words = [w.lower() for _, _, w in ref_data]
    ref_times = np.array([(t0 + t1) / 2 for t0, t1, _ in ref_data])
    print(f"Reference:  {len(ref_words)} words")

    # Time range
    t_start = max(pred_times[0], ref_times[0])
    t_end = min(pred_times[-1], ref_times[-1])
    win = args.window

    cutoffs = []
    t = t_start
    while t + win <= t_end:
        cutoffs.append((t, t + win))
        t += win

    # Segment
    ref_segs = []
    pred_segs = []
    for t0, t1 in cutoffs:
        r = [w for w, tm in zip(ref_words, ref_times) if t0 <= tm < t1]
        ref_segs.append(" ".join(r))
        p_ = [w for w, tm in zip(pred_words, pred_times) if t0 <= tm < t1]
        pred_segs.append(" ".join(p_))

    n = len(cutoffs)
    if args.max_windows:
        n = min(n, args.max_windows)

    w = 80
    print(f"\n{'='*w}")
    print(f"  {args.subject} / {args.experiment} / {args.task}")
    print(f"  {len(cutoffs)} windows of {win:.0f}s each")
    print(f"{'='*w}\n")

    for i in range(n):
        t0, t1 = cutoffs[i]
        print(f"--- {t0:.0f}–{t1:.0f}s ---\n")
        print(f"  REF:  {ref_segs[i][:200]}")
        print(f"  PRED: {pred_segs[i][:200]}")
        print()

if __name__ == "__main__":
    main()
