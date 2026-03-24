"""Abstract base class for dimensionality reduction models."""

from abc import ABC, abstractmethod
import numpy as np


class BaseReducer(ABC):
    """All reducers must implement fit/transform with this interface."""

    def __init__(self, params: dict):
        self.params = params
        self.P_ = None

    @abstractmethod
    def fit(self, X: np.ndarray, partial_target: np.ndarray, disambiguator=None):
        """Learn projection matrix P from training data.

        Parameters
        ----------
        X : (n_samples, n_features)
        partial_target : (n_classes, n_samples)
        disambiguator : BaseDisambiguator or None
        """
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data using learned P.

        Parameters
        ----------
        X : (n_samples, n_features)

        Returns
        -------
        X_proj : (n_samples, target_d)
        """
        ...

    def fit_transform(self, X, partial_target, disambiguator=None):
        self.fit(X, partial_target, disambiguator)
        return self.transform(X)
