#!/usr/bin/env python3
"""Per-voxel intrinsic timescale during story listening.

For each voxel, compute the autocorrelation function (ACF) of its BOLD
timecourse averaged across stories, and summarise how fast it decays. This
gives a per-voxel timescale that maps cleanly onto cortex:

    small ⇒ signal varies fast (fast-changing local features),
    large ⇒ signal varies slowly (longer-context integration).

For each story we:
  - trim the silence (``5 + config.TRIM`` TRs at the start and ``config.TRIM``
    at the end - matches ``utils_stim.get_stim``),
  - linear-detrend the timecourse,
  - compute the (biased) sample ACF up to ``--max-lag-trs`` via FFT,
  - accumulate the per-voxel ACF (length-weighted across stories).

Per-voxel metrics derived from the averaged ACF:
  - ``half_life``         smallest lag where ACF crosses 0.5 (linear-interpolated).
  - ``exp_tau``           time-constant of a log-linear fit on lags 1..K
                          (``--exp-fit-max-lag``); positive lags only.
  - ``integrated_ac``     signed area under ACF for lags 1..max_lag (TR or s).
  - ``positive_integrated_ac`` area under max(ACF, 0), which is often more
                          interpretable when BOLD ACF undershoots below zero.

All metrics are saved both in TR units and seconds (using ``--tr-seconds``).
Use ``plot_voxel_timescale_flatmaps.py`` to project them onto cortex.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))

import config  # noqa: E402
import run_summaries_encoding as rse  # noqa: E402
from utils_resp import get_resp  # noqa: E402

SUBJECT_TO_UTS = rse.SUBJECT_TO_UTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voxel_timescale")

SUB_ROIS = ["BA_10", "BA_6", "BA_8", "BA_9_46", "BROCA", "BA_full_frontal"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", default="S1", choices=sorted(SUBJECT_TO_UTS.keys()))
    p.add_argument(
        "--sessions",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20],
    )
    p.add_argument("--stories", nargs="+", default=None,
                   help="Explicit story list (overrides --sessions).")
    p.add_argument(
        "--voxel-scope",
        default="all",
        choices=["all", "full_frontal"],
        help="all = whole brain (default); full_frontal restricts to BA_full_frontal.",
    )
    p.add_argument("--max-lag-trs", type=int, default=30,
                   help="Max lag in TRs for ACF (default 30 TRs ~ 60 s).")
    p.add_argument("--exp-fit-max-lag", type=int, default=10,
                   help="Lags 1..K used for the log-linear exponential fit.")
    p.add_argument("--tr-seconds", type=float, default=2.0)

    p.add_argument(
        "--detrend",
        default="linear",
        choices=["none", "linear"],
        help="Per-story detrending applied before ACF.",
    )
    p.add_argument(
        "--trim",
        default="huth",
        choices=["huth", "none"],
        help="huth = drop first 5+TRIM and last TRIM TRs; none = use raw response.",
    )

    p.add_argument("--ba-dir", default=str(rse.LOCAL_DEFAULT_BA_DIR))
    p.add_argument("--data-root", default=None)
    p.add_argument("--local-compute-mode", action="store_true")
    p.add_argument(
        "--mounted-project-root",
        default="/Volumes/ellie/language-decoding-expts",
    )
    p.add_argument(
        "--summaries-dir",
        default=str(rse.LOCAL_DEFAULT_SUMMARIES_DIR),
        help="Unused but required by configure_local_compute_mode.",
    )

    p.add_argument("--output-dir", default=str(THIS_DIR / "results"))
    p.add_argument("--tag", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data prep helpers
# ---------------------------------------------------------------------------


def configure_data_root(args: argparse.Namespace) -> Path | None:
    if args.data_root:
        root = Path(args.data_root).expanduser().resolve()
        for sub in ("data_train", "data_lm", "models"):
            target = root / sub
            if not target.is_dir():
                raise FileNotFoundError(f"--data-root missing required dir: {target}")
        config.DATA_TRAIN_DIR = str(root / "data_train")
        config.DATA_LM_DIR = str(root / "data_lm")
        config.MODEL_DIR = str(root / "models")
        config.DATA_TEST_DIR = str(root / "data_test")
        cand = root / "ba_indices"
        if not Path(args.ba_dir).is_dir() and cand.is_dir():
            args.ba_dir = str(cand)
        log.info("Using --data-root: %s", root)
        return root
    if args.local_compute_mode:
        return rse.configure_local_compute_mode(args)
    return None


def load_stories(args: argparse.Namespace) -> List[str]:
    if args.stories:
        return list(args.stories)
    sess_to_story_path = Path(config.DATA_TRAIN_DIR) / "sess_to_story.json"
    with open(sess_to_story_path, encoding="utf-8") as f:
        sess_to_story = json.load(f)
    stories: List[str] = []
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])
    return stories


def select_voxels(args: argparse.Namespace, total_voxels: int) -> Tuple[np.ndarray, str]:
    if args.voxel_scope == "all":
        return np.arange(total_voxels, dtype=int), "all"
    uts = SUBJECT_TO_UTS[args.subject]
    path = Path(args.ba_dir).expanduser().resolve() / uts / "BA_full_frontal.json"
    if not path.is_file():
        raise FileNotFoundError(f"Full-frontal ROI file not found: {path}")
    with open(path, encoding="utf-8") as f:
        vox = np.asarray(next(iter(json.load(f).values())), dtype=int)
    vox = np.sort(np.unique(vox))
    vox = vox[(vox >= 0) & (vox < int(total_voxels))]
    if vox.size == 0:
        raise ValueError(f"Full-frontal ROI is empty for subject {args.subject!r}.")
    return vox, "full_frontal"


def load_sub_roi_masks(ba_dir: Path, uts_id: str, voxels: np.ndarray) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    for name in SUB_ROIS:
        path = ba_dir / uts_id / f"{name}.json"
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            ids = np.asarray(next(iter(json.load(f).values())), dtype=int)
        mask = np.isin(voxels, ids)
        if mask.any():
            masks[name] = mask
    return masks


# ---------------------------------------------------------------------------
# ACF + timescale metrics
# ---------------------------------------------------------------------------


def trim_response(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return X
    start = 5 + config.TRIM
    end = config.TRIM
    if X.shape[0] <= start + end + 5:
        return X
    return X[start : X.shape[0] - end]


def detrend_linear_inplace(X: np.ndarray) -> None:
    T = X.shape[0]
    if T < 2:
        return
    t = np.arange(T, dtype=np.float64)
    t_mean = t.mean()
    t_centered = t - t_mean
    t_var = float((t_centered * t_centered).sum())
    if t_var == 0:
        return
    x_mean = X.mean(axis=0)
    cov = (t_centered[:, None] * (X - x_mean[None, :])).sum(axis=0)
    slope = cov / t_var
    intercept = x_mean - slope * t_mean
    X -= (intercept[None, :] + slope[None, :] * t[:, None])


def compute_acf_matrix(X: np.ndarray, max_lag: int) -> np.ndarray:
    """ACF for each column of X (T, V) up to ``max_lag`` (lag 0 included).

    Uses biased estimator (divide by N-k) followed by per-column normalisation
    to ACF[0]=1 (since columns are mean-removed first).
    """
    T = X.shape[0]
    X = X - X.mean(axis=0, keepdims=True)
    var = (X * X).mean(axis=0)
    var = np.where(var > 0, var, 1.0)
    n_fft = 1 << int(np.ceil(np.log2(2 * T)))
    F = np.fft.rfft(X, n=n_fft, axis=0)
    psd = (F * np.conj(F)).real
    ac = np.fft.irfft(psd, n=n_fft, axis=0)[: max_lag + 1]
    norm = np.arange(T, T - max_lag - 1, -1, dtype=np.float64)
    ac = ac / (norm[:, None] * var[None, :])
    return ac.astype(np.float32)


def compute_metrics(
    acf_avg: np.ndarray,
    tr_seconds: float,
    exp_fit_max_lag: int,
) -> Dict[str, np.ndarray]:
    """Per-voxel timescale metrics from averaged ACF (lag 0 = 1)."""
    n_lags, V = acf_avg.shape

    # --- Half-life: first lag where ACF <= 0.5 (linear-interpolated). ---------
    drops_below = acf_avg <= 0.5
    crossed = drops_below.any(axis=0)
    first_below = np.argmax(drops_below, axis=0)  # 0 if never crossed
    half_life_trs = np.full(V, np.nan, dtype=np.float32)
    valid = crossed & (first_below > 0)
    if valid.any():
        idx_above = first_below[valid] - 1
        vox_idx = np.where(valid)[0]
        a = acf_avg[idx_above, vox_idx]
        b = acf_avg[idx_above + 1, vox_idx]
        denom = a - b
        denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
        frac = (a - 0.5) / denom
        half_life_trs[vox_idx] = (idx_above + frac).astype(np.float32)
    half_life_trs[crossed & (first_below == 0)] = 0.0

    # --- Exponential tau via log-linear fit on lags 1..K ---------------------
    K = max(2, min(int(exp_fit_max_lag), n_lags - 1))
    lags_fit = np.arange(1, K + 1, dtype=np.float64)
    acf_fit = acf_avg[1 : K + 1].astype(np.float64)
    pos_mask = acf_fit > 1e-6
    n_pos = pos_mask.sum(axis=0)
    exp_tau_trs = np.full(V, np.nan, dtype=np.float32)
    fit_mask = n_pos >= 2
    if fit_mask.any():
        log_acf = np.where(pos_mask, np.log(np.maximum(acf_fit, 1e-12)), 0.0)
        L = lags_fit[:, None] * np.ones((1, V))  # (K, V)
        L_masked = np.where(pos_mask, L, 0.0)
        n = n_pos.astype(np.float64).clip(min=1.0)
        l_mean = L_masked.sum(axis=0) / n
        y_mean = log_acf.sum(axis=0) / n
        l_dev = np.where(pos_mask, lags_fit[:, None] - l_mean[None, :], 0.0)
        y_dev = np.where(pos_mask, log_acf - y_mean[None, :], 0.0)
        num = (l_dev * y_dev).sum(axis=0)
        den = (l_dev * l_dev).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
            tau_vals = -1.0 / slope
        tau_vals = np.where(np.isfinite(tau_vals) & (tau_vals > 0), tau_vals, np.nan)
        exp_tau_trs[fit_mask] = tau_vals[fit_mask].astype(np.float32)

    # --- Integrated ACF (lags 1..max_lag, TR units). -------------------------
    integrated_ac_trs = acf_avg[1:].sum(axis=0).astype(np.float32)
    positive_integrated_ac_trs = np.clip(acf_avg[1:], 0.0, None).sum(axis=0).astype(np.float32)

    return {
        "half_life_trs": half_life_trs,
        "half_life_seconds": (half_life_trs * tr_seconds).astype(np.float32),
        "exp_tau_trs": exp_tau_trs,
        "exp_tau_seconds": (exp_tau_trs * tr_seconds).astype(np.float32),
        "integrated_ac_trs": integrated_ac_trs,
        "integrated_ac_seconds": (integrated_ac_trs * tr_seconds).astype(np.float32),
        "positive_integrated_ac_trs": positive_integrated_ac_trs,
        "positive_integrated_ac_seconds": (positive_integrated_ac_trs * tr_seconds).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def build_tag(args: argparse.Namespace, scope: str) -> str:
    if args.tag:
        return args.tag
    return (
        f"{args.subject}__{scope}"
        f"__lag{args.max_lag_trs}__detrend-{args.detrend}__trim-{args.trim}"
    )


def write_roi_csv(
    csv_path: Path,
    metrics: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    voxels: np.ndarray,
) -> None:
    rows: List[Dict[str, object]] = []
    full_mask = np.ones_like(voxels, dtype=bool)
    rois = {"all": full_mask, **masks}
    for roi_name, mask in rois.items():
        if mask.sum() == 0:
            continue
        for metric_name, arr in metrics.items():
            sub = arr[mask]
            sub = sub[np.isfinite(sub)]
            if sub.size == 0:
                continue
            rows.append({
                "roi": roi_name,
                "metric": metric_name,
                "n_voxels": int(mask.sum()),
                "n_finite": int(sub.size),
                "mean": float(sub.mean()),
                "median": float(np.median(sub)),
                "p05": float(np.quantile(sub, 0.05)),
                "p95": float(np.quantile(sub, 0.95)),
            })
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["roi", "metric", "n_voxels", "n_finite", "mean", "median", "p05", "p95"]
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    mounted = configure_data_root(args)
    if mounted is not None:
        log.info("Mounted project root: %s", mounted)

    stories = load_stories(args)
    log.info("Stories: %d", len(stories))

    log.info("Probing total voxel count from a sample story")
    sample = get_resp(args.subject, [stories[0]], stack=True, vox=None)
    total_voxels = int(sample.shape[1])
    voxels, scope = select_voxels(args, total_voxels)
    log.info("Voxel scope: %s -> %d / %d voxels", scope, len(voxels), total_voxels)

    log.info("Loading per-story responses (vox=%s)", scope)
    t0 = time.time()
    resps = get_resp(args.subject, stories, stack=False, vox=voxels)
    log.info("Loaded responses in %.1fs", time.time() - t0)

    n_lag = int(args.max_lag_trs)
    n_voxels = int(len(voxels))
    acf_sum = np.zeros((n_lag + 1, n_voxels), dtype=np.float64)
    weight_sum = 0.0
    skipped: List[str] = []

    for i, story in enumerate(stories, start=1):
        X = np.asarray(resps[story], dtype=np.float32, order="C").copy()
        X = trim_response(X, args.trim)
        if X.shape[0] <= n_lag + 5:
            skipped.append(story)
            continue
        if args.detrend == "linear":
            detrend_linear_inplace(X)
        acf = compute_acf_matrix(X, n_lag)
        w = float(X.shape[0])
        acf_sum += acf.astype(np.float64) * w
        weight_sum += w
        if i % 10 == 0 or i == len(stories):
            log.info("  ACF: %d/%d stories done", i, len(stories))

    if weight_sum == 0:
        raise RuntimeError("No usable stories.")
    acf_avg = (acf_sum / weight_sum).astype(np.float32)
    if skipped:
        log.warning("Skipped %d stories (too short): %s", len(skipped), ", ".join(skipped))

    metrics = compute_metrics(acf_avg, args.tr_seconds, args.exp_fit_max_lag)

    out_dir = Path(args.output_dir).expanduser().resolve() / build_tag(args, scope)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "voxel_timescale.npz",
        voxels=voxels,
        n_total_voxels=int(total_voxels),
        acf_avg=acf_avg,
        lags_trs=np.arange(n_lag + 1, dtype=int),
        tr_seconds=float(args.tr_seconds),
        subject=args.subject,
        voxel_scope=scope,
        stories=np.array([s for s in stories if s not in set(skipped)]),
        **{name: arr for name, arr in metrics.items()},
    )

    summary = {
        "subject": args.subject,
        "voxel_scope": scope,
        "n_voxels": n_voxels,
        "n_total_voxels": total_voxels,
        "n_stories": int(len(stories) - len(skipped)),
        "skipped_stories": skipped,
        "max_lag_trs": n_lag,
        "tr_seconds": float(args.tr_seconds),
        "detrend": args.detrend,
        "trim": args.trim,
        "exp_fit_max_lag": int(args.exp_fit_max_lag),
        "metrics": {
            name: {
                "mean": float(np.nanmean(arr)),
                "median": float(np.nanmedian(arr)),
                "p05": float(np.nanquantile(arr, 0.05)),
                "p95": float(np.nanquantile(arr, 0.95)),
                "n_nan": int(np.isnan(arr).sum()),
            }
            for name, arr in metrics.items()
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    masks: Dict[str, np.ndarray] = {}
    if scope == "full_frontal":
        masks = load_sub_roi_masks(
            Path(args.ba_dir).expanduser().resolve(), SUBJECT_TO_UTS[args.subject], voxels
        )
    write_roi_csv(out_dir / "roi_summary.csv", metrics, masks, voxels)

    log.info("Wrote %s", out_dir / "voxel_timescale.npz")
    log.info("Wrote %s", out_dir / "summary.json")
    log.info("Wrote %s", out_dir / "roi_summary.csv")
    for name, st in summary["metrics"].items():
        log.info(
            "  %-25s mean=%.2f  median=%.2f  p05=%.2f  p95=%.2f  n_nan=%d",
            name, st["mean"], st["median"], st["p05"], st["p95"], st["n_nan"],
        )
    log.info("Run plot_voxel_timescale_flatmaps.py --results-dir %s", out_dir)


if __name__ == "__main__":
    main()
