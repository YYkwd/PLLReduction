"""Classifier registry."""

from .knn import KNNClassifier
from .ipal import IPALClassifier
from .pl_svm import PLSVMClassifier
from .sure import SUREClassifier

CLASSIFIERS = {
    'knn': KNNClassifier,
    'ipal': IPALClassifier,
    'plsvm': PLSVMClassifier,
    'sure': SUREClassifier,
}
