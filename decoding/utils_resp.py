import os
import numpy as np
import h5py

import config


def _load_resp_dataset(dataset, vox=None):
    """Load either the full response matrix or only requested voxel columns."""
    if vox is None:
        return np.nan_to_num(dataset[:])

    vox = np.asarray(vox, dtype=int)
    if vox.ndim != 1:
        raise ValueError("vox must be a 1D index array")
    if vox.size == 0:
        return np.zeros((dataset.shape[0], 0), dtype=dataset.dtype)

    # h5py fancy indexing requires monotonically increasing indices.
    order = np.argsort(vox)
    sorted_vox = vox[order]
    data = np.nan_to_num(dataset[:, sorted_vox])
    if np.all(order == np.arange(len(order))):
        return data

    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return data[:, inverse]


def get_resp(subject, stories, stack = True, vox = None, response_root = None):
    """loads response data
    """
    base_dir = response_root if response_root is not None else config.DATA_TRAIN_DIR
    subject_dir = os.path.join(base_dir, "train_response", subject)
    resp = {}
    for story in stories:
        resp_path = os.path.join(subject_dir, "%s.hf5" % story)
        hf = h5py.File(resp_path, "r")
        resp[story] = _load_resp_dataset(hf["data"], vox=vox)
        hf.close()
    if stack: return np.vstack([resp[story] for story in stories]) 
    else: return resp
