#!/usr/bin/env python3
"""Regenerate the curated clean_code_v2 outputs.

This runner writes/rebuilds:

  - outputs/clean_code_v2/prepared_midpoint
  - outputs/clean_code_v2/preferred_lag_midpoint
  - outputs/clean_code_v2/super_brain_midpoint
  - outputs/clean_code_v2/sentence_boundary_channel_timecourses

It does not import clean_code modules or copy legacy result trees.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PODCAST_DIR = THIS_DIR.parent
REPO_ROOT = PODCAST_DIR.parent

DEFAULT_OUTPUT_ROOT = PODCAST_DIR / "outputs" / "clean_code_v2"
DEFAULT_PREPARED_DIR = DEFAULT_OUTPUT_ROOT / "prepared_midpoint"
DEFAULT_PREFERRED_LAG_DIR = DEFAULT_OUTPUT_ROOT / "preferred_lag_midpoint"
LOCAL_BIDS_ROOT = PODCAST_DIR / "data" / "ds005574"
DEFAULT_BIDS_ROOT = LOCAL_BIDS_ROOT if LOCAL_BIDS_ROOT.exists() else Path("/Volumes/ellie/language-decoding-expts/podcast_ecog/data/ds005574")
DEFAULT_BOUNDARY_JSON = PODCAST_DIR / "outputs" / "transcript_boundaries.json"
DEFAULT_ROI_METRICS = PODCAST_DIR / "outputs" / "gpt2_paper_roi_context_layer_exact" / "channel_paper_roi_metrics.csv"
DEFAULT_CHANNEL_TRACKING_DIR = PODCAST_DIR / "outputs" / "boundary_channel_tracking" / "sentence"

TARGET_STEMS = ("word_vectors_pca20", "gpt2_ctx32_layer8_pca20")
TARGET_TREES = ("prepared_midpoint", "preferred_lag_midpoint", "super_brain_midpoint", "sentence_boundary_channel_timecourses")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--preferred-lag-dir", type=Path, default=DEFAULT_PREFERRED_LAG_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT)
    parser.add_argument("--boundary-json", type=Path, default=DEFAULT_BOUNDARY_JSON)
    parser.add_argument("--roi-metrics", type=Path, default=DEFAULT_ROI_METRICS)
    parser.add_argument("--target-stems", nargs="+", default=list(TARGET_STEMS))
    parser.add_argument("--subjects", nargs="+", default=[f"{i:02d}" for i in range(1, 10)])
    parser.add_argument("--overwrite", action="store_true", help="Remove v2 target trees before regenerating.")
    parser.add_argument("--skip-prepare", action="store_true", help="Use existing v2 prepared/preferred-lag caches.")
    parser.add_argument("--prepare-only", action="store_true", help="Build v2 prepared/preferred-lag caches and stop.")
    parser.add_argument("--prepare-epochs", action="store_true", help="Also cache word-locked high-gamma epochs.")
    parser.add_argument("--gpt2-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--gpt2-local-files-only", action="store_true")
    parser.add_argument("--preferred-lag-null-iters", type=int, default=200)
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Recompute analysis/plots from existing v2 decoder predictions instead of refitting decoders.",
    )
    parser.add_argument(
        "--reuse-channel-tracking",
        action="store_true",
        help="Use an existing boundary-channel screen CSV/NPZ instead of recomputing it from raw high-gamma.",
    )
    parser.add_argument("--channel-tracking-dir", type=Path, default=DEFAULT_CHANNEL_TRACKING_DIR)
    parser.add_argument("--channel-screen-permutations", type=int, default=1000)
    parser.add_argument("--channel-screen-seed", type=int, default=0)
    return parser.parse_args()


def is_inside(path: Path, parent: Path) -> bool:
    path = path.resolve()
    parent = parent.resolve()
    return path == parent or parent in path.parents


def remove_tree(path: Path, safe_root: Path) -> None:
    if not path.exists():
        return
    if not is_inside(path, safe_root):
        raise ValueError(f"Refusing to remove path outside {safe_root}: {path}")
    shutil.rmtree(path)


def run_command(cmd: list[str | Path]) -> None:
    rendered = " ".join(str(part) for part in cmd)
    print(f"+ {rendered}", flush=True)
    subprocess.run([str(part) for part in cmd], cwd=REPO_ROOT, check=True)


def run_prepare(args: argparse.Namespace) -> None:
    if args.skip_prepare:
        return

    force_prepare = bool(args.overwrite and not args.analyze_only)
    prepare_cmd: list[str | Path] = [
        sys.executable,
        THIS_DIR / "prepare_data.py",
        "--bids-root",
        args.bids_root,
        "--output-dir",
        args.prepared_dir,
        "--boundary-json",
        args.boundary_json,
        "--roi-metrics",
        args.roi_metrics,
        "--subjects",
        *args.subjects,
        "--n-components",
        "20",
        "--window-mode",
        "midpoint",
        "--window-start",
        "-0.5",
        "--window-end",
        "0.5",
    ]
    if args.prepare_epochs:
        prepare_cmd.append("--prepare-epochs")
    if force_prepare:
        prepare_cmd.append("--force")
        if args.prepare_epochs:
            prepare_cmd.append("--force-epochs")
    run_command(prepare_cmd)

    gpt2_cmd: list[str | Path] = [
        sys.executable,
        THIS_DIR / "prepare_gpt2_word_features.py",
        "--bids-root",
        args.bids_root,
        "--prepared-dir",
        args.prepared_dir,
        "--model",
        "gpt2",
        "--context-token-length",
        "32",
        "--layer",
        "8",
        "--n-components",
        "20",
        "--batch-size",
        "32",
        "--device",
        args.gpt2_device,
    ]
    if args.gpt2_local_files_only:
        gpt2_cmd.append("--local-files-only")
    if force_prepare:
        gpt2_cmd.append("--force")
    run_command(gpt2_cmd)

    constituent_cmd: list[str | Path] = [
        sys.executable,
        THIS_DIR / "prepare_constituent_boundaries.py",
        "--prepared-dir",
        args.prepared_dir,
        "--bids-root",
        args.bids_root,
    ]
    if force_prepare:
        constituent_cmd.append("--overwrite")
    run_command(constituent_cmd)

    preferred_csvs = []
    for target_stem in args.target_stems:
        preferred_cmd: list[str | Path] = [
            sys.executable,
            THIS_DIR / "preferred_lag_analysis.py",
            "--prepared-dir",
            args.prepared_dir,
            "--output-dir",
            args.preferred_lag_dir,
            "--target-stem",
            target_stem,
            "--subjects",
            *args.subjects,
            "--null-iters",
            str(int(args.preferred_lag_null_iters)),
        ]
        if force_prepare:
            preferred_cmd.append("--overwrite")
        run_command(preferred_cmd)
        preferred_csvs.append(args.preferred_lag_dir / f"preferred_lag__{target_stem}.csv")

    run_command(
        [
            sys.executable,
            THIS_DIR / "plot_preferred_lag_distributions.py",
            "--csv",
            *preferred_csvs,
            "--output-dir",
            args.preferred_lag_dir,
        ]
    )
    run_command(
        [
            sys.executable,
            THIS_DIR / "plot_preferred_lag_distributions.py",
            "--csv",
            *preferred_csvs,
            "--output-dir",
            args.preferred_lag_dir,
            "--significant-only",
            "--suffix",
            "_sig",
        ]
    )


def run_super_brain(args: argparse.Namespace, super_brain_root: Path) -> None:
    for target_stem in args.target_stems:
        cmd: list[str | Path] = [
            sys.executable,
            THIS_DIR / "super_brain_pipeline.py",
            "--prepared-dir",
            args.prepared_dir,
            "--output-root",
            super_brain_root,
            "--target-stem",
            target_stem,
            "--subjects",
            *args.subjects,
        ]
        if args.overwrite and not args.analyze_only:
            cmd.append("--overwrite")
        if args.analyze_only:
            cmd.append("--analyze-only")
        run_command(cmd)


def run_present_word_summary(args: argparse.Namespace, super_brain_root: Path) -> None:
    run_command(
        [
            sys.executable,
            THIS_DIR / "super_brain_decode.py",
            "--prepared-dir",
            args.prepared_dir,
            "--output-dir",
            super_brain_root,
            "--target-stems",
            *args.target_stems,
            "--subjects",
            *args.subjects,
        ]
    )


def run_anatomy(args: argparse.Namespace, super_brain_root: Path) -> None:
    for target_stem in args.target_stems:
        run_command(
            [
                sys.executable,
                THIS_DIR / "super_brain_channel_anatomy.py",
                "--prepared-dir",
                args.prepared_dir,
                "--output-root",
                super_brain_root,
                "--target-stem",
                target_stem,
                "--roi-metrics",
                args.roi_metrics,
                "--subjects",
                *args.subjects,
            ]
        )


def run_lag_sensor_intuition(args: argparse.Namespace, super_brain_root: Path) -> None:
    out_dir = super_brain_root / "lag_sensor_intuition"
    cmd: list[str | Path] = [
        sys.executable,
        THIS_DIR / "lag_sensor_intuition.py",
        "--prepared-dir",
        args.prepared_dir,
        "--super-brain-root",
        super_brain_root,
        "--preferred-lag-dir",
        args.preferred_lag_dir,
        "--output-dir",
        out_dir,
        "--target-stems",
        *args.target_stems,
        "--subjects",
        *args.subjects,
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    run_command(cmd)
    run_command(
        [
            sys.executable,
            THIS_DIR / "plot_lag_sensor_nilearn.py",
            "--input-dir",
            out_dir,
            "--target-stems",
            *args.target_stems,
        ]
    )


def run_gpt2_auxiliary(args: argparse.Namespace, super_brain_root: Path) -> None:
    target_stem = "gpt2_ctx32_layer8_pca20"
    target_root = super_brain_root / target_stem
    decoder_dir = target_root / "decoders"
    boundary_dir = target_root / "boundary_locked"

    run_command(
        [
            sys.executable,
            THIS_DIR / "super_brain_current_word_confound.py",
            "--prepared-dir",
            args.prepared_dir,
            "--decoder-dir",
            decoder_dir,
            "--output-dir",
            target_root / "current_word_confound",
            "--target-stem",
            target_stem,
            "--directions",
            "past",
            "future",
        ]
    )
    for direction in ("past", "future"):
        run_command(
            [
                sys.executable,
                THIS_DIR / "super_brain_boundary_locked_performance.py",
                "--prepared-dir",
                args.prepared_dir,
                "--decoder-dir",
                decoder_dir,
                "--output-dir",
                boundary_dir,
                "--direction",
                direction,
            ]
        )
    for level in ("sentence", "constituent"):
        run_command(
            [
                sys.executable,
                THIS_DIR / "super_brain_boundary_locked_roi_panel.py",
                "--prepared-dir",
                args.prepared_dir,
                "--decoder-dir",
                decoder_dir,
                "--output-dir",
                boundary_dir,
                "--boundary-level",
                level,
            ]
        )


def channel_tracking_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.reuse_channel_tracking:
        return (
            args.channel_tracking_dir / "sentence_boundary_channel_tracking.csv",
            args.channel_tracking_dir / "sentence_boundary_channel_timecourses.npz",
        )

    tmp_obj = tempfile.TemporaryDirectory(prefix="clean_code_v2_channel_tracking_")
    tmp_dir = Path(tmp_obj.name)
    # Keep the TemporaryDirectory object alive by attaching it to the function.
    channel_tracking_inputs._tmp_obj = tmp_obj  # type: ignore[attr-defined]
    run_command(
        [
            sys.executable,
            THIS_DIR / "sentence_boundary_channel_tracking.py",
            "--bids-root",
            args.bids_root,
            "--boundary-json",
            args.boundary_json,
            "--roi-metrics",
            args.roi_metrics,
            "--output-dir",
            tmp_dir,
            "--subjects",
            *args.subjects,
            "--boundary-level",
            "sentence",
            "--boundary-anchor",
            "next_start",
            "--n-permutations",
            str(int(args.channel_screen_permutations)),
            "--random-seed",
            str(int(args.channel_screen_seed)),
        ]
    )
    return (
        tmp_dir / "sentence_boundary_channel_tracking.csv",
        tmp_dir / "sentence_boundary_channel_timecourses.npz",
    )


def run_sentence_boundary_panels(args: argparse.Namespace, output_root: Path) -> None:
    output_dir = output_root / "sentence_boundary_channel_timecourses"
    output_dir.mkdir(parents=True, exist_ok=True)
    tracking_csv, timecourses_npz = channel_tracking_inputs(args)
    for direction in ("decrease", "increase"):
        run_command(
            [
                sys.executable,
                THIS_DIR / "plot_sentence_boundary_direction_timecourses.py",
                "--direction",
                direction,
                "--top-n",
                "24",
                "--n-cols",
                "4",
                "--tracking-csv",
                tracking_csv,
                "--timecourses-npz",
                timecourses_npz,
                "--output-dir",
                output_dir,
            ]
        )


def write_point_r_compatibility_aliases(super_brain_root: Path, target_stems: list[str]) -> None:
    """Create legacy unsuffixed aliases from regenerated point-r analysis files."""
    for target_stem in target_stems:
        analysis_dir = super_brain_root / target_stem / "analysis"
        if not analysis_dir.exists():
            continue
        for source in sorted(analysis_dir.glob("*_point_r.*")):
            dest = source.with_name(source.name.replace("_point_r", ""))
            shutil.copy2(source, dest)


def write_manifest(output_root: Path) -> None:
    rows = []
    for tree in TARGET_TREES:
        base = output_root / tree
        files = [p for p in base.rglob("*") if p.is_file() and p.name != ".DS_Store"] if base.exists() else []
        rows.append(
            {
                "tree": tree,
                "n_files": len(files),
                "size_bytes": int(sum(p.stat().st_size for p in files)),
            }
        )
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"target_trees": rows}, handle, indent=2)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    super_brain_root = args.output_root / "super_brain_midpoint"

    if args.overwrite and not args.analyze_only:
        remove_tree(super_brain_root, args.output_root)
        remove_tree(args.output_root / "sentence_boundary_channel_timecourses", args.output_root)

    run_prepare(args)
    if args.prepare_only:
        write_manifest(args.output_root)
        print(f"\nclean_code_v2 preparation caches regenerated at: {args.output_root}", flush=True)
        return 0

    run_super_brain(args, super_brain_root)
    run_present_word_summary(args, super_brain_root)
    run_anatomy(args, super_brain_root)
    run_lag_sensor_intuition(args, super_brain_root)
    run_gpt2_auxiliary(args, super_brain_root)
    run_sentence_boundary_panels(args, args.output_root)
    write_point_r_compatibility_aliases(super_brain_root, args.target_stems)
    write_manifest(args.output_root)

    print(f"\nclean_code_v2 outputs regenerated at: {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
