"""Abstract base class for disambiguation strategies."""

from abc import ABC, abstractmethod
import numpy as np


class BaseDisambiguator(ABC):
    """All disambiguators must implement this interface."""

    def __init__(self, params: dict = None):
        self.params = params or {}

    @abstractmethod
    def disambiguate(
        self,
        Y: np.ndarray,
        nn_indices: np.ndarray,
        nn_dists: np.ndarray,
        k: int,
        candidate_mask: np.ndarray,
        iteration: int = 0,
    ) -> tuple:
        """Update label confidence matrix Y.

        Parameters
        ----------
        Y : (n_classes, m) current label confidences
        nn_indices : (m, k) neighbor indices (-1 = filtered out)
        nn_dists : (m, k) neighbor distances (0 = filtered out)
        k : number of neighbors
        candidate_mask : (n_classes, m) binary mask
        iteration : current iteration number

        Returns
        -------
        Y_new : (n_classes, m)
        D_new : (m, m) semantic dissimilarity
        """
        ...
