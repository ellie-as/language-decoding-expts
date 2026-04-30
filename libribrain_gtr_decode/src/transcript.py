from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import RunSpec, load_transcript_like, mock_transcript
from .utils import stable_id


WORD_COLUMNS = ["word", "text", "token", "segment", "trial_type", "value"]
ONSET_COLUMNS = ["onset", "start", "start_time", "time", "sample_time", "timemeg"]
OFFSET_COLUMNS = ["offset", "end", "end_time"]


def normalize_word_events(df: pd.DataFrame, run: RunSpec) -> pd.DataFrame:
    df = df.copy()
    kind_col = _first_existing(df, ["kind", "type", "annotation_type"])
    if kind_col is not None:
        df = df[df[kind_col].astype(str).str.lower().eq("word")].copy()
    word_col = _first_existing(df, WORD_COLUMNS)
    onset_col = _first_existing(df, ONSET_COLUMNS)
    offset_col = _first_existing(df, OFFSET_COLUMNS)
    duration_col = _first_existing(df, ["duration", "duration_sec"])
    if word_col is None or onset_col is None:
        raise ValueError(f"Could not find word/text and onset columns in transcript columns: {list(df.columns)}")
    out = pd.DataFrame(
        {
            "subject": df.get("subject", run.subject),
            "session": df.get("session", run.session),
            "run": df.get("run", run.run),
            "onset": pd.to_numeric(df[onset_col], errors="coerce"),
            "word": df[word_col].astype(str),
        }
    )
    if offset_col is not None:
        out["offset"] = pd.to_numeric(df[offset_col], errors="coerce")
    elif duration_col is not None:
        out["offset"] = out["onset"] + pd.to_numeric(df[duration_col], errors="coerce").fillna(0.0)
    else:
        out["offset"] = out["onset"]
    out = out.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)
    out["word"] = out["word"].str.strip()
    out = out[out["word"].ne("") & out["word"].ne("nan")]
    return out


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_col = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_to_col:
            return lower_to_col[c]
    return None


def load_word_events(run: RunSpec, config: dict[str, Any]) -> pd.DataFrame:
    duration = float(config.get("features", {}).get("mock_duration_sec", 240))
    raw = mock_transcript(run, duration) if config.get("data", {}).get("mock_data") else load_transcript_like(run)
    return normalize_word_events(raw, run)


def build_text_windows_for_run(run: RunSpec, config: dict[str, Any]) -> pd.DataFrame:
    win_cfg = config["windows"]
    events = load_word_events(run, config)
    if events.empty:
        return pd.DataFrame()
    text_window = float(win_cfg["text_window_sec"])
    sample_interval = float(win_cfg["sample_interval_sec"])
    min_words = int(win_cfg["min_words"])
    start_t = max(text_window, float(events["onset"].min()))
    end_t = float(events["offset"].max())
    sample_times = np.arange(start_t, end_t + 1e-6, sample_interval)
    rows = []
    for t in sample_times:
        left = t - text_window
        mask = (events["onset"] >= left) & (events["offset"] <= t)
        words = events.loc[mask]
        if len(words) < min_words:
            continue
        text = " ".join(words["word"].astype(str).tolist())
        example_id = stable_id([run.subject, run.session, run.run, f"{t:.3f}"], prefix="lb")
        rows.append(
            {
                "example_id": example_id,
                "subject": run.subject,
                "session": run.session,
                "run": run.run,
                "run_group": run.group_id,
                "t": float(t),
                "text_start": float(left),
                "text_end": float(t),
                "text": text,
                "n_words": int(len(words)),
                "first_word_time": float(words["onset"].iloc[0]),
                "last_word_time": float(words["offset"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def build_text_windows(runs: list[RunSpec], config: dict[str, Any]) -> pd.DataFrame:
    dfs = [build_text_windows_for_run(run, config) for run in runs]
    windows = pd.concat([df for df in dfs if not df.empty], ignore_index=True) if dfs else pd.DataFrame()
    max_examples = config.get("debug", {}).get("max_examples") if config.get("debug", {}).get("enabled") else None
    if max_examples:
        windows = windows.head(int(max_examples)).copy()
    return windows


def write_examples(windows: pd.DataFrame, path: str | Path, n: int = 10) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for _, row in windows.head(n).iterrows():
        lines.append(f"{row.example_id} | {row.subject}/{row.session}/{row.run} | t={row.t:.1f}s | {row.text}\n")
    p.write_text("".join(lines), encoding="utf-8")
