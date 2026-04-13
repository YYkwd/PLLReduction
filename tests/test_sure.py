"""Tests for SURE (Feng & An, AAAI'19)."""

import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pll.classifiers.sure import DEFAULT_BETA, DEFAULT_LAMBDA, SUREClassifier


class TestSUREClassifier(unittest.TestCase):
    def test_default_params(self):
        c = SUREClassifier()
        self.assertEqual(c.lam, DEFAULT_LAMBDA)
        self.assertEqual(c.beta, DEFAULT_BETA)

    def test_fit_predict_shape_linear(self):
        X = np.random.randn(30, 4)
        y = np.random.randint(0, 3, size=30)
        pt = np.zeros((3, 30))
        for i in range(30):
            pt[y[i], i] = 1.0
        clf = SUREClassifier({'use_kernel': False, 'max_iter': 20, 'beta': 0.5})
        clf.fit(X, y, partial_target=pt)
        pred = clf.predict(X)
        self.assertEqual(pred.shape, (30,))

    def test_partial_label_separable(self):
        X = np.array([[2.0, 0], [2.1, 0.1], [-2.0, 0], [-2.1, -0.1]], dtype=float)
        pt = np.array(
            [
                [1, 1, 0, 0],
                [0, 0, 1, 1],
            ],
            dtype=float,
        )
        clf = SUREClassifier({'use_kernel': False, 'lambda': 0.05, 'beta': 0.01, 'max_iter': 80})
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
        self.assertIn('sure', ev.CLASSIFIERS)


if __name__ == '__main__':
    unittest.main()
