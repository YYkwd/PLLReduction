"""Cross-validation splitter and Many/Medium/Few class grouping."""

import numpy as np
from sklearn.model_selection import StratifiedKFold


def get_many_medium_few_splits(y, n_classes):
    """Split classes into Many/Medium/Few by sample count (each ~1/3 of classes)."""
    counts = np.bincount(y, minlength=n_classes)
    sorted_idx = np.argsort(-counts)
    n = len(sorted_idx)
    m = (n + 2) // 3
    f = (2 * n + 2) // 3
    many = set(sorted_idx[:m].tolist())
    medium = set(sorted_idx[m:f].tolist())
    few = set(sorted_idx[f:].tolist())
    return many, medium, few


def create_cv_splitter(n_splits=5, shuffle=True, random_state=42):
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
