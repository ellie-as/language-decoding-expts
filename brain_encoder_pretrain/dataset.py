"""Multi-subject TR-chunk dataset for self-supervised pretraining.

Each subject has its own voxel count `V_s`. We keep the data in its native
surface-voxel space (no parcellation / no MNI resampling) and let the model
handle variable width via per-subject input/output projections.

Per-subject per-story z-scoring is applied up front so that the Transformer
never sees raw scan-scale values. Stories that are held out for downstream
evaluation are excluded.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import h5py


log = logging.getLogger(__name__)


SUBJECT_TO_UTS = {"S1": "UTS01", "S2": "UTS02", "S3": "UTS03"}


@dataclass
class SubjectData:
    """All training-story responses for a single subject, z-scored per story."""
    subject: str
    stories: List[str]
    # Concatenated responses [T_total, V_s] and per-story (start, end) indices.
    X: np.ndarray
    story_spans: List[Tuple[str, int, int]]

    @property
    def n_voxels(self) -> int:
        return int(self.X.shape[1])

    def sample_chunk(self, chunk_len: int, rng: random.Random) -> np.ndarray:
        """Sample a random `chunk_len`-TR window from a random training story.

        Windows are drawn strictly within one story so temporal context is real.
        Returns an array of shape [chunk_len, V_s] (float32).
        """
        eligible = [(s, a, b) for (s, a, b) in self.story_spans if (b - a) >= chunk_len]
        if not eligible:
            raise ValueError(
                f"No story for subject {self.subject} is long enough for chunk_len={chunk_len}"
            )
        _, a, b = eligible[rng.randrange(len(eligible))]
        start = rng.randrange(a, b - chunk_len + 1)
        return self.X[start : start + chunk_len].astype(np.float32, copy=False)


def _zscore_per_story(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd == 0.0, 1.0, sd)
    out = np.nan_to_num((X - mu) / sd).astype(np.float32)
    return out


def load_subject_data(
    subject: str,
    stories: Sequence[str],
    data_train_dir: str | Path,
) -> SubjectData:
    """Load all training-story responses for one subject, z-scored per story."""
    data_train_dir = Path(data_train_dir)
    subj_dir = data_train_dir / "train_response" / subject

    parts: List[np.ndarray] = []
    spans: List[Tuple[str, int, int]] = []
    cursor = 0
    n_vox: int | None = None

    for story in stories:
        resp_path = subj_dir / f"{story}.hf5"
        if not resp_path.is_file():
            log.warning("Missing response file, skipping: %s", resp_path)
            continue
        with h5py.File(resp_path, "r") as hf:
            x = np.nan_to_num(hf["data"][:]).astype(np.float32)
        if n_vox is None:
            n_vox = int(x.shape[1])
        elif int(x.shape[1]) != n_vox:
            raise ValueError(
                f"Inconsistent voxel count for subject {subject}: "
                f"{resp_path} has {x.shape[1]}, expected {n_vox}"
            )
        x = _zscore_per_story(x)
        parts.append(x)
        spans.append((story, cursor, cursor + x.shape[0]))
        cursor += x.shape[0]

    if not parts:
        raise FileNotFoundError(f"No responses found for subject {subject} in {subj_dir}")

    X = np.concatenate(parts, axis=0)
    used_stories = [s for s, _, _ in spans]
    log.info(
        "Loaded subject %s: %d stories, %d TRs, %d voxels",
        subject, len(used_stories), X.shape[0], X.shape[1],
    )
    return SubjectData(subject=subject, stories=used_stories, X=X, story_spans=spans)


class MultiSubjectChunkSampler:
    """Iterable that yields per-step minibatches: {subject -> [B_s, L, V_s]}.

    Sampling weight per subject is proportional to the total TR count so every
    subject contributes its data. We randomly choose subjects with replacement
    within a batch; stories within a subject are sampled uniformly.
    """

    def __init__(
        self,
        subjects_data: Dict[str, SubjectData],
        chunk_len: int,
        batch_size: int,
        seed: int = 0,
    ):
        if not subjects_data:
            raise ValueError("At least one subject is required")
        self.subjects = list(subjects_data.keys())
        self.data = subjects_data
        self.chunk_len = int(chunk_len)
        self.batch_size = int(batch_size)
        self.rng = random.Random(seed)

        # TR counts for sampling weights
        tr_counts = np.array(
            [self.data[s].X.shape[0] for s in self.subjects], dtype=np.float64
        )
        self.weights = tr_counts / tr_counts.sum()

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Sample one training batch, grouped per subject.

        Returns dict subject -> array [B_s, L, V_s] where sum_s(B_s) == batch_size.
        """
        picks: Dict[str, List[np.ndarray]] = {s: [] for s in self.subjects}
        for _ in range(self.batch_size):
            s = self.rng.choices(self.subjects, weights=self.weights, k=1)[0]
            chunk = self.data[s].sample_chunk(self.chunk_len, self.rng)
            picks[s].append(chunk)
        out: Dict[str, np.ndarray] = {}
        for s, chunks in picks.items():
            if chunks:
                out[s] = np.stack(chunks, axis=0).astype(np.float32)
        return out
