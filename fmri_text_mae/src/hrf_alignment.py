from __future__ import annotations

import numpy as np


def response_tr_times(wordseq, n_response_trs: int, tr_sec: float, trim_start: int, trim_end: int) -> np.ndarray:
    """Return TR times aligned to response rows after Huth-style trimming."""
    tr_times = np.asarray(getattr(wordseq, "tr_times", []), dtype=np.float32)
    if tr_times.size:
        stop = -trim_end if trim_end else None
        trimmed = tr_times[trim_start:stop]
        if trimmed.shape[0] == n_response_trs:
            return trimmed.astype(np.float32)
    return (np.arange(n_response_trs, dtype=np.float32) * float(tr_sec)).astype(np.float32)


def words_in_lagged_window(wordseq, start_sec: float, end_sec: float, hrf_lag_sec: float) -> str:
    """Collect words whose onsets fall in the stimulus window that drives an fMRI window."""
    words = np.asarray(getattr(wordseq, "data", []), dtype=object)
    onsets = np.asarray(getattr(wordseq, "data_times", []), dtype=np.float32)
    if len(words) == 0 or len(onsets) == 0:
        return ""
    lo = float(start_sec) - float(hrf_lag_sec)
    hi = float(end_sec) - float(hrf_lag_sec)
    mask = (onsets >= lo) & (onsets < hi)
    return " ".join(str(w) for w in words[mask] if str(w).strip())
