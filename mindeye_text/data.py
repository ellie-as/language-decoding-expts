"""Data loading for the multi-subject MindEye-style text decoder.

Each subject contributes:
- An ROI voxel selection (default ``full_frontal``).
- Per-voxel z-score statistics computed on the subject's training stories.
- A 5-TR-window GTR text embedding target per chunk index ``i``, covering
  text TRs ``[i, i+chunk_trs)``. We re-use the cache produced by
  ``27-04-expts/train_5tr_chunk_nn.py`` so multiple pipelines share the same
  embeddings on disk.
- A single-TR brain input ``brain[i + lag + brain_offset]``. The chunk count
  per story (``n_chunks = resp_len - lag - chunk_trs + 1``) follows the
  cached convention so we can directly index into the cached embeddings.
"""
from __future__ import annotations

import argparse
import logging
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))
sys.path.insert(0, str(REPO_DIR / "27-04-expts"))

import run_summaries_encoding as rse  # noqa: E402
from train_5tr_chunk_nn import (  # noqa: E402
    load_or_build_chunk_embeddings,
    load_responses_by_story,
)
from train_lagged_text_pca_mlp import (  # noqa: E402
    load_stories,
    resolve_response_root,
    resolve_roi,
)


log = logging.getLogger("mindeye_text.data")


@dataclass
class SubjectData:
    """Everything we need to build train/val/test datasets for one subject."""

    subject: str
    roi_name: str
    voxels: np.ndarray  # (n_vox,) ROI voxel indices into the full response matrix
    voxel_mean: np.ndarray  # (n_vox,) train z-score mean
    voxel_std: np.ndarray  # (n_vox,) train z-score std
    train_stories: List[str]
    test_stories: List[str]
    responses_train: Dict[str, np.ndarray]  # per-voxel z-scored, per story
    responses_test: Dict[str, np.ndarray]
    embeddings_train: Dict[str, np.ndarray]  # raw text embeddings per chunk
    embeddings_test: Dict[str, np.ndarray]
    response_root: str
    embedding_dim: int
    embedding_cache: str

    @property
    def n_voxels(self) -> int:
        return int(self.voxels.shape[0])


def _voxel_zscore(
    train_responses: Mapping[str, np.ndarray],
    test_responses: Mapping[str, np.ndarray],
    train_stories: List[str],
    test_stories: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    train_concat = np.vstack([train_responses[s] for s in train_stories]).astype(np.float32)
    mean = train_concat.mean(axis=0).astype(np.float32)
    std = train_concat.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    del train_concat

    train_z = {
        s: np.nan_to_num((train_responses[s] - mean) / std).astype(np.float32)
        for s in train_stories
    }
    test_z = {
        s: np.nan_to_num((test_responses[s] - mean) / std).astype(np.float32)
        for s in test_stories
    }
    return train_z, test_z, mean, std


def load_subject_data(args: argparse.Namespace, subject: str) -> SubjectData:
    """Load voxel selection, z-scored responses, and chunk text embeddings for one subject."""
    args = argparse.Namespace(**deepcopy(vars(args)))
    args.subject = subject

    response_root, mounted_root = resolve_response_root(args)
    if args.local_compute_mode and mounted_root is not None:
        stories_for_cache = load_stories(args, response_root)
        response_root = rse.stage_local_response_cache(
            subject,
            stories_for_cache,
            mounted_data_train_dir=str(mounted_root / "data_train"),
            cache_root=Path(args.local_cache_root).expanduser().resolve(),
        )

    stories = load_stories(args, response_root)
    train_stories, test_stories = rse.split_story_list(stories, args)
    if not test_stories:
        raise ValueError(
            f"Subject {subject}: need held-out stories; do not pass --no-story-holdout."
        )

    train_resp_lengths, total_voxels = rse.load_resp_info(
        subject, train_stories, data_train_dir=response_root
    )
    test_resp_lengths, _ = rse.load_resp_info(
        subject, test_stories, data_train_dir=response_root
    )
    resp_lengths = {**train_resp_lengths, **test_resp_lengths}

    roi_name, vox = resolve_roi(args, total_voxels)
    log.info("[%s] ROI %s: %d voxels", subject, roi_name, len(vox))

    raw_train = load_responses_by_story(train_stories, subject, vox, response_root)
    raw_test = load_responses_by_story(test_stories, subject, vox, response_root)
    train_z, test_z, vox_mean, vox_std = _voxel_zscore(
        raw_train, raw_test, train_stories, test_stories
    )

    embeddings_by_story, emb_dim, cache_path = load_or_build_chunk_embeddings(
        args, stories, resp_lengths, response_root=response_root
    )
    embeddings_train = {s: embeddings_by_story[s] for s in train_stories}
    embeddings_test = {s: embeddings_by_story[s] for s in test_stories}

    return SubjectData(
        subject=subject,
        roi_name=str(roi_name),
        voxels=np.asarray(vox, dtype=np.int64),
        voxel_mean=vox_mean,
        voxel_std=vox_std,
        train_stories=list(train_stories),
        test_stories=list(test_stories),
        responses_train=train_z,
        responses_test=test_z,
        embeddings_train=embeddings_train,
        embeddings_test=embeddings_test,
        response_root=str(response_root),
        embedding_dim=int(emb_dim),
        embedding_cache=str(cache_path),
    )


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


def compute_target_stats(
    embeddings_train_per_subject: Mapping[str, Mapping[str, np.ndarray]],
    train_stories_per_subject: Mapping[str, List[str]],
    normalize_targets: bool,
    shared: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Per-dim z-score statistics for the text-embedding targets.

    ``shared=True`` pools all subjects' training embeddings into a single
    mean/std (since they are derived from largely overlapping text content,
    this gives a single, well-conditioned standardization).
    """
    if shared:
        parts = []
        for subj, embeddings in embeddings_train_per_subject.items():
            stories = train_stories_per_subject[subj]
            parts.append(np.vstack([embeddings[s] for s in stories]).astype(np.float32))
        y = np.vstack(parts)
        if normalize_targets:
            y = _l2_normalize_rows(y)
        mean = y.mean(axis=0).astype(np.float32)
        std = y.std(axis=0).astype(np.float32)
        std[std == 0] = 1.0
        return (
            {subj: mean for subj in embeddings_train_per_subject},
            {subj: std for subj in embeddings_train_per_subject},
        )

    means: Dict[str, np.ndarray] = {}
    stds: Dict[str, np.ndarray] = {}
    for subj, embeddings in embeddings_train_per_subject.items():
        stories = train_stories_per_subject[subj]
        y = np.vstack([embeddings[s] for s in stories]).astype(np.float32)
        if normalize_targets:
            y = _l2_normalize_rows(y)
        m = y.mean(axis=0).astype(np.float32)
        s = y.std(axis=0).astype(np.float32)
        s[s == 0] = 1.0
        means[subj] = m
        stds[subj] = s
    return means, stds


class SingleTRChunkDataset(Dataset):
    """Maps chunk index ``i`` to a single-TR brain input and a 5-TR text target.

    For each chunk:
      ``input  = responses[i + lag_trs + brain_offset]`` (shape ``(n_vox,)``)
      ``target = (embeddings[i] - target_mean) / target_std`` (shape ``(emb_dim,)``)

    Embeddings are L2-normalized first when ``normalize_targets`` is True.
    """

    def __init__(
        self,
        responses_by_story: Mapping[str, np.ndarray],
        embeddings_by_story: Mapping[str, np.ndarray],
        stories: List[str],
        chunk_trs: int,
        lag_trs: int,
        brain_offset: int,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        normalize_targets: bool,
    ) -> None:
        self.stories = list(stories)
        self.responses = [responses_by_story[s] for s in self.stories]
        self.embeddings = [embeddings_by_story[s] for s in self.stories]
        self.chunk_trs = int(chunk_trs)
        self.lag_trs = int(lag_trs)
        self.brain_offset = int(brain_offset)
        if not (0 <= self.brain_offset < self.chunk_trs):
            raise ValueError(
                f"brain_offset {self.brain_offset} must lie in [0, chunk_trs={self.chunk_trs})"
            )
        self.target_mean = target_mean.astype(np.float32)
        self.target_std = target_std.astype(np.float32)
        self.normalize_targets = bool(normalize_targets)

        story_ids: List[int] = []
        starts: List[int] = []
        for story_idx, (y_story, resp) in enumerate(zip(self.embeddings, self.responses)):
            n_chunks = int(y_story.shape[0])
            need_rows = n_chunks + self.lag_trs + self.chunk_trs - 1
            if resp.shape[0] < need_rows:
                raise ValueError(
                    f"{self.stories[story_idx]}: response has {resp.shape[0]} TRs, "
                    f"need at least {need_rows} for chunk_trs={self.chunk_trs}, "
                    f"lag_trs={self.lag_trs}"
                )
            story_ids.extend([story_idx] * n_chunks)
            starts.extend(range(n_chunks))
        self.story_ids = np.asarray(story_ids, dtype=np.int32)
        self.starts = np.asarray(starts, dtype=np.int32)

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        story_idx = int(self.story_ids[idx])
        i = int(self.starts[idx])
        x = self.responses[story_idx][i + self.lag_trs + self.brain_offset]
        y = self.embeddings[story_idx][i].astype(np.float32)
        if self.normalize_targets:
            n = float(np.linalg.norm(y))
            if n > 0.0:
                y = y / n
        y = np.nan_to_num((y - self.target_mean) / self.target_std).astype(np.float32)
        return (
            torch.from_numpy(np.ascontiguousarray(x).astype(np.float32)),
            torch.from_numpy(np.ascontiguousarray(y)),
        )

    def chunk_story_groups(self) -> np.ndarray:
        """Story-name array aligned with ``__getitem__`` indices (for grouped splits)."""
        labels = np.empty(len(self), dtype=object)
        cursor = 0
        for story_idx, emb in enumerate(self.embeddings):
            n = int(emb.shape[0])
            labels[cursor : cursor + n] = self.stories[story_idx]
            cursor += n
        return labels

    def stack_targets_raw(self) -> np.ndarray:
        """Return the raw (un-normalized, un-zscored) targets in dataset order."""
        return np.vstack(self.embeddings).astype(np.float32)


def make_subject_dataset(
    subject_data: SubjectData,
    chunk_trs: int,
    lag_trs: int,
    brain_offset: int,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    normalize_targets: bool,
    split: str,
) -> SingleTRChunkDataset:
    """Build a :class:`SingleTRChunkDataset` for a subject's train or test split."""
    if split == "train":
        responses = subject_data.responses_train
        embeddings = subject_data.embeddings_train
        stories = subject_data.train_stories
    elif split == "test":
        responses = subject_data.responses_test
        embeddings = subject_data.embeddings_test
        stories = subject_data.test_stories
    else:
        raise ValueError(f"Unknown split: {split!r}")
    return SingleTRChunkDataset(
        responses_by_story=responses,
        embeddings_by_story=embeddings,
        stories=stories,
        chunk_trs=chunk_trs,
        lag_trs=lag_trs,
        brain_offset=brain_offset,
        target_mean=target_mean,
        target_std=target_std,
        normalize_targets=normalize_targets,
    )
