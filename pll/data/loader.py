"""Unified dataset loading from .mat files."""

import numpy as np
import scipy.io as sio
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class Dataset:
    X: np.ndarray                       # (n_samples, n_features)
    partial_target: np.ndarray          # (n_classes, n_samples)
    target: Optional[np.ndarray]        # (n_classes, n_samples) or None
    name: str
    n_classes: int
    n_samples: int
    n_features: int


def load_dataset(name: str, path=None, data_dir='datasets') -> Dataset:
    """Load a PLL dataset from .mat file.

    Handles sparse matrices automatically. Returns dense arrays.
    """
    if path is None:
        path = Path(data_dir) / f'{name}.mat'
    else:
        path = Path(path)

    mat = sio.loadmat(str(path))

    X = np.asarray(mat['data'], dtype=float)

    partial_target = mat['partial_target']
    if hasattr(partial_target, 'toarray'):
        partial_target = partial_target.toarray()
    partial_target = np.asarray(partial_target, dtype=float)

    target = mat.get('target', None)
    if target is not None:
        if hasattr(target, 'toarray'):
            target = target.toarray()
        target = np.asarray(target, dtype=float)

    return Dataset(
        X=X,
        partial_target=partial_target,
        target=target,
        name=name,
        n_classes=partial_target.shape[0],
        n_samples=X.shape[0],
        n_features=X.shape[1],
    )
