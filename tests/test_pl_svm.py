"""Tests for Nguyen & Caruana PL-SVM (PL-Pegasos)."""

import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pll.classifiers.pl_svm import (
    DEFAULT_LAMBDA_REG,
    DEFAULT_T,
    PLSVMClassifier,
)
class TestPLSVMClassifier(unittest.TestCase):
    def test_default_params(self):
        c = PLSVMClassifier()
        self.assertEqual(c.lambda_reg, DEFAULT_LAMBDA_REG)
        self.assertEqual(c.T, DEFAULT_T)

    def test_fit_predict_shape(self):
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 3, size=20)
        pt = np.zeros((3, 20))
        for i in range(20):
            pt[y[i], i] = 1.0
        clf = PLSVMClassifier({'lambda_reg': 0.1, 'T': 100})
        clf.fit(X, y, partial_target=pt)
        pred = clf.predict(X)
        self.assertEqual(pred.shape, (20,))

    def test_partial_label_uses_candidate_sets(self):
        # Two blobs; true 0/1 known only via candidates (not passed as supervision signal in loss paths)
        X = np.array([[2.0, 0], [2.1, 0.1], [-2.0, 0], [-2.1, -0.1]], dtype=float)
        pt = np.array(
            [
                [1, 1, 0, 0],
                [0, 0, 1, 1],
            ],
            dtype=float,
        )
        y = np.array([0, 0, 1, 1])
        clf = PLSVMClassifier({'lambda_reg': 0.01, 'T': 800})
        clf.fit(X, y=None, partial_target=pt)
        pred = clf.predict(X)
        self.assertEqual(pred[0], pred[1])
        self.assertEqual(pred[2], pred[3])
        self.assertNotEqual(pred[0], pred[2])

    def test_evaluator_registry(self):
        import pll.eval.evaluator as ev
        ev.REDUCERS.clear()
        ev.DISAMBIGUATORS.clear()
        ev.CLASSIFIERS.clear()
        ev._register_defaults()
        self.assertIn('plsvm', ev.CLASSIFIERS)


if __name__ == '__main__':
    unittest.main()
