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
        E_dist: np.ndarray,
        k: int,
        candidate_mask: np.ndarray,
        iteration: int = 0,
    ) -> tuple:
        """Update label confidence matrix Y.

        Parameters
        ----------
        Y : (n_classes, n_samples) current label confidences
        E_dist : (n_samples, n_samples) neighbor distance matrix
        k : number of neighbors
        candidate_mask : (n_classes, n_samples) binary mask

        Returns
        -------
        Y_new : (n_classes, n_samples)
        D_new : (n_samples, n_samples) semantic dissimilarity
        """
        ...
