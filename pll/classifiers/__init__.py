"""Classifier registry."""

from .knn import KNNClassifier
from .ipal import IPALClassifier

CLASSIFIERS = {
    'knn': KNNClassifier,
    'ipal': IPALClassifier,
}
