#!/usr/bin/env python3
"""Verify the curated clean_code_v2 output tree."""

from __future__ import annotations

import argparse
from pathlib import Path

PODCAST_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PODCAST_DIR / "outputs" / "clean_code_v2"

EXPECTED_COUNTS = {
    "super_brain_midpoint": 565,
    "sentence_boundary_channel_timecourses": 8,
}

EXPECTED_FILES = [
    "prepared_midpoint/words.csv",
    "prepared_midpoint/boundaries.csv",
    "prepared_midpoint/constituent_spans.csv",
    "prepared_midpoint/constituent_boundary_config.json",
    "prepared_midpoint/word_vectors_pca20.npy",
    "prepared_midpoint/gpt2_ctx32_layer8_pca20.npy",
    "prepared_midpoint/subjects/sub-01/neural_word_features.npz",
    "prepared_midpoint/subjects/sub-09/neural_word_features.npz",
    "preferred_lag_midpoint/preferred_lag__word_vectors_pca20.csv",
    "preferred_lag_midpoint/preferred_lag__gpt2_ctx32_layer8_pca20.csv",
    "preferred_lag_midpoint/preferred_lag_dist_by_roi__word_vectors_pca20_sig.png",
    "preferred_lag_midpoint/preferred_lag_dist_by_roi__gpt2_ctx32_layer8_pca20_sig.png",
    "super_brain_midpoint/super_brain_word_vectors_pca20.json",
    "super_brain_midpoint/super_brain_gpt2_ctx32_layer8_pca20.json",
    "super_brain_midpoint/word_vectors_pca20/analysis/combined_decoding_heatmaps.pdf",
    "super_brain_midpoint/word_vectors_pca20/analysis/combined_decoding_heatmaps_point_r.pdf",
    "super_brain_midpoint/word_vectors_pca20/analysis/combined_decoding_heatmaps_rank_identification.pdf",
    "super_brain_midpoint/gpt2_ctx32_layer8_pca20/analysis/combined_decoding_heatmaps.pdf",
    "super_brain_midpoint/gpt2_ctx32_layer8_pca20/analysis/combined_decoding_heatmaps_point_r.pdf",
    "super_brain_midpoint/gpt2_ctx32_layer8_pca20/analysis/combined_decoding_heatmaps_rank_identification.pdf",
    "super_brain_midpoint/gpt2_ctx32_layer8_pca20/boundary_locked/super_brain_n1-5_sentence_boundary_locked_roi_panel_past-future_point_r.png",
    "super_brain_midpoint/gpt2_ctx32_layer8_pca20/boundary_locked/super_brain_n1-5_constituent_boundary_locked_roi_panel_past-future_point_r.png",
    "super_brain_midpoint/gpt2_ctx32_layer8_pca20/current_word_confound/super_brain_prediction_target_minus_current_word.png",
    "super_brain_midpoint/lag_sensor_intuition/cross_target_summary.json",
    "super_brain_midpoint/lag_sensor_intuition/word_vectors_pca20/encoding_crosscheck.csv",
    "super_brain_midpoint/lag_sensor_intuition/gpt2_ctx32_layer8_pca20/encoding_crosscheck.csv",
    "sentence_boundary_channel_timecourses/config_decrease.json",
    "sentence_boundary_channel_timecourses/config_increase.json",
    "sentence_boundary_channel_timecourses/top24_sentence_boundary_decrease_channel_timecourses.pdf",
    "sentence_boundary_channel_timecourses/top24_sentence_boundary_decrease_channel_timecourses.png",
    "sentence_boundary_channel_timecourses/top24_sentence_boundary_decrease_channels.csv",
    "sentence_boundary_channel_timecourses/top24_sentence_boundary_increase_channel_timecourses.pdf",
    "sentence_boundary_channel_timecourses/top24_sentence_boundary_increase_channel_timecourses.png",
    "sentence_boundary_channel_timecourses/top24_sentence_boundary_increase_channels.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = []
    for rel in EXPECTED_FILES:
        path = args.output_root / rel
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(rel)

    count_mismatches = []
    for tree in ("prepared_midpoint", "preferred_lag_midpoint", "super_brain_midpoint", "sentence_boundary_channel_timecourses"):
        base = args.output_root / tree
        files = [p for p in base.rglob("*") if p.is_file() and p.name != ".DS_Store"] if base.exists() else []
        size = sum(p.stat().st_size for p in files)
        print(f"{tree}: {len(files)} files, {size / (1024 ** 2):.1f} MiB", flush=True)
        if tree in EXPECTED_COUNTS:
            expected_count = EXPECTED_COUNTS[tree]
            if len(files) != expected_count:
                count_mismatches.append((tree, expected_count, len(files)))

    if missing:
        print("\nMissing or empty expected files:", flush=True)
        for rel in missing:
            print(f"  {rel}", flush=True)
        return 1
    if count_mismatches:
        print("\nUnexpected file counts:", flush=True)
        for tree, expected, observed in count_mismatches:
            print(f"  {tree}: expected {expected}, observed {observed}", flush=True)
        return 1

    print("\nVerification passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
