from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import resolve_path
from .utils import parse_optional_list

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSpec:
    subject: str
    session: str
    run: str
    transcript_path: str | None = None
    meg_path: str | None = None

    @property
    def group_id(self) -> str:
        return f"{self.subject}/{self.session}/{self.run}"


def discover_runs(config: dict[str, Any]) -> list[RunSpec]:
    data_cfg = config.get("data", {})
    debug_cfg = config.get("debug", {})
    if data_cfg.get("mock_data") or debug_cfg.get("enabled"):
        return [
            RunSpec(
                subject=str(data_cfg.get("subject") or "debug_subject"),
                session="debug_session",
                run=f"debug_run_{i:02d}",
            )
            for i in range(int(debug_cfg.get("max_runs", 1)))
        ]

    root = resolve_path(data_cfg.get("libribrain_root"), Path.cwd())
    if root is None or not root.exists():
        raise FileNotFoundError("Set data.libribrain_root to an existing LibriBrain directory")

    subjects = parse_optional_list(data_cfg.get("subject"))
    sessions = parse_optional_list(data_cfg.get("sessions"))
    runs = parse_optional_list(data_cfg.get("runs"))
    transcript_files = _indexed_files(root, ["*events*.tsv", "*transcript*.tsv", "*words*.tsv", "*events*.csv", "*transcript*.csv"])
    meg_files = _indexed_files(root, ["*.fif", "*.fif.gz", "*.npy", "*.npz", "*.h5"])

    specs: list[RunSpec] = []
    keys = sorted(set(transcript_files) | set(meg_files))
    for key in keys:
        subject, session, run = key
        if subjects and subject not in subjects:
            continue
        if sessions and session not in sessions:
            continue
        if runs and run not in runs:
            continue
        specs.append(
            RunSpec(
                subject=subject,
                session=session,
                run=run,
                transcript_path=str(transcript_files.get(key)) if transcript_files.get(key) else None,
                meg_path=str(meg_files.get(key)) if meg_files.get(key) else None,
            )
        )
    if not specs:
        raise RuntimeError(f"No runs discovered under {root}. Check paths, filters, and file naming.")
    return specs


def _indexed_files(root: Path, patterns: list[str]) -> dict[tuple[str, str, str], Path]:
    files: dict[tuple[str, str, str], Path] = {}
    for pattern in patterns:
        for path in root.rglob(pattern):
            key = _infer_key(path)
            files.setdefault(key, path)
    return files


def _infer_key(path: Path) -> tuple[str, str, str]:
    parts = path.parts
    subject = next((p for p in parts if p.startswith("sub-")), "sub-unknown")
    session = next((p for p in parts if p.startswith("ses-")), "ses-unknown")
    stem = path.name
    run = next((token for token in stem.replace(".", "_").split("_") if token.startswith("run-")), path.stem)
    return subject, session, run


def inspect_libribrain(config: dict[str, Any]) -> dict[str, Any]:
    runs = discover_runs(config)
    summary: dict[str, Any] = {
        "n_runs": len(runs),
        "subjects": sorted({r.subject for r in runs}),
        "sessions": sorted({r.session for r in runs}),
        "runs": [r.__dict__ for r in runs[:20]],
        "has_more_runs": len(runs) > 20,
    }
    if runs:
        first = runs[0]
        summary["example_run"] = first.__dict__
        try:
            events = load_transcript_like(first)
            summary["word_level_timing_available"] = {"word", "onset"}.issubset(events.columns)
            summary["example_events_rows"] = events.head(5).to_dict(orient="records")
        except Exception as exc:
            summary["transcript_error"] = repr(exc)
        try:
            meg = load_meg_array(first, config, preload_seconds=20)
            summary["meg_sampling_rate_hz"] = float(meg["sfreq"])
            summary["n_meg_channels"] = int(meg["data"].shape[0])
            summary["example_meg_shape_channels_time"] = list(meg["data"].shape)
        except Exception as exc:
            summary["meg_error"] = repr(exc)
    return summary


def load_transcript_like(run: RunSpec) -> pd.DataFrame:
    if run.transcript_path is None:
        return mock_transcript(run, duration_sec=240)
    path = Path(run.transcript_path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def mock_transcript(run: RunSpec, duration_sec: float = 240.0) -> pd.DataFrame:
    words = "the quick brown fox tells a story about memory language and sound in the brain".split()
    onsets = np.arange(0.25, duration_sec, 0.45)
    return pd.DataFrame(
        {
            "subject": run.subject,
            "session": run.session,
            "run": run.run,
            "onset": onsets,
            "offset": onsets + 0.32,
            "word": [words[i % len(words)] for i in range(len(onsets))],
        }
    )


def load_meg_array(run: RunSpec, config: dict[str, Any], preload_seconds: float | None = None) -> dict[str, Any]:
    features_cfg = config.get("features", {})
    if run.meg_path is None or config.get("data", {}).get("mock_data") or config.get("debug", {}).get("enabled"):
        sfreq = float(features_cfg.get("mock_sampling_hz", 200))
        n_channels = int(features_cfg.get("mock_channels", 32))
        duration = float(features_cfg.get("mock_duration_sec", 240))
        if preload_seconds is not None:
            duration = min(duration, preload_seconds)
        rng = np.random.default_rng(abs(hash(run.group_id)) % (2**32))
        n_times = int(round(duration * sfreq))
        data = rng.normal(0, 1, size=(n_channels, n_times)).astype(np.float32)
        times = np.arange(n_times, dtype=np.float64) / sfreq
        return {"data": data, "sfreq": sfreq, "times": times, "ch_names": [f"MEG{i:03d}" for i in range(n_channels)]}

    path = Path(run.meg_path)
    if path.suffix == ".h5":
        import h5py

        with h5py.File(path, "r") as f:
            data = f["data"][:]
            sfreq = float(f.attrs.get("sample_frequency", f["data"].attrs.get("sample_frequency", features_cfg.get("raw_sampling_hz", 250))))
        if data.shape[0] > data.shape[1]:
            LOGGER.warning("Loaded H5 MEG array looks time x channels; transposing to channels x time")
            data = data.T
        if preload_seconds is not None:
            data = data[:, : int(round(preload_seconds * sfreq))]
        return {"data": data.astype(np.float32), "sfreq": sfreq, "times": np.arange(data.shape[1]) / sfreq, "ch_names": None}

    if path.suffix in {".npy", ".npz"}:
        arr = np.load(path)
        data = arr["data"] if isinstance(arr, np.lib.npyio.NpzFile) and "data" in arr else np.asarray(arr)
        sfreq = float(arr["sfreq"]) if isinstance(arr, np.lib.npyio.NpzFile) and "sfreq" in arr else float(features_cfg.get("raw_sampling_hz", 1000))
        if data.shape[0] > data.shape[1]:
            LOGGER.warning("Loaded MEG array looks time x channels; transposing to channels x time")
            data = data.T
        return {"data": data.astype(np.float32), "sfreq": sfreq, "times": np.arange(data.shape[1]) / sfreq, "ch_names": None}

    try:
        import mne

        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        if preload_seconds is not None:
            raw.crop(tmin=0, tmax=min(preload_seconds, raw.times[-1]))
        picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False, exclude="bads")
        data = raw.get_data(picks=picks).astype(np.float32)
        return {"data": data, "sfreq": float(raw.info["sfreq"]), "times": raw.times, "ch_names": [raw.ch_names[i] for i in picks]}
    except Exception as exc:
        raise RuntimeError(f"Could not load MEG file {path}. Add a loader for this format.") from exc


def save_summary_text(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
