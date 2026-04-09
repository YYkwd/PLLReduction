"""Repeated holdout splitter with stratified-first, auto-fallback logic."""

import logging
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

log = logging.getLogger(__name__)


def get_many_medium_few_splits(y, n_classes):
    """Split classes into Many / Medium / Few by sample count (each ~1/3 of classes).

    Based on *training set* frequencies.  Returns three sets of class indices.
    """
    counts = np.bincount(y, minlength=n_classes)
    sorted_idx = np.argsort(-counts)
    n = len(sorted_idx)
    m = (n + 2) // 3
    f = (2 * n + 2) // 3
    many = set(sorted_idx[:m].tolist())
    medium = set(sorted_idx[m:f].tolist())
    few = set(sorted_idx[f:].tolist())
    return many, medium, few


class RepeatedHoldout:
    """Repeated stratified (or plain) train/test split.

    Parameters
    ----------
    n_repeats : int
        Number of random splits to generate.
    test_ratio : float
        Fraction of samples used for testing in each split.
    stratified : bool
        If True, attempt StratifiedShuffleSplit first.  Falls back to
        plain ShuffleSplit automatically when stratification is impossible
        (e.g. a class has only 1 sample).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(self, n_repeats=10, test_ratio=0.2,
                 stratified=True, random_state=42):
        self.n_repeats = n_repeats
        self.test_ratio = test_ratio
        self.stratified = stratified
        self.random_state = random_state
        self._fell_back = False

    def split(self, X, y):
        """Yield (train_idx, test_idx) for each repeat."""
        if self.stratified:
            try:
                splitter = StratifiedShuffleSplit(
                    n_splits=self.n_repeats,
                    test_size=self.test_ratio,
                    random_state=self.random_state,
                )
                # Dry-run the first split to catch errors early
                splits = list(splitter.split(X, y))
                yield from splits
                return
            except ValueError as e:
                counts = np.bincount(y)
                min_count = int(np.min(counts[counts > 0]))
                log.warning(
                    "Stratified split failed (min class count=%d): %s. "
                    "Falling back to random ShuffleSplit.",
                    min_count, e,
                )
                self._fell_back = True

        splitter = ShuffleSplit(
            n_splits=self.n_repeats,
            test_size=self.test_ratio,
            random_state=self.random_state,
        )
        yield from splitter.split(X, y)

    @property
    def fell_back(self):
        """True if stratified split was requested but failed."""
        return self._fell_back
