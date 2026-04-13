"""Partial-label multiclass SVM via PL-Pegasos (Nguyen & Caruana, KDD'08).

Linear margin on the Crammer–Singer feature map Φ(x, y): block y equals x, others 0.
Training minimizes λ‖w‖²/2 plus average hinge over training points; each point uses
either the full-label margin (|Y_i| = 1) or the partial-label margin from the paper.

Reference: Nguyen & Caruana, "Classification with Partial Labels", KDD 2008,
Optimization Problem II + Algorithm 1 (PL-Pegasos). Polynomial kernel (d>1) is not
implemented; λ here corresponds to paper's λ, T to the number of iterations.
"""

from __future__ import annotations

import numpy as np

# Defaults aligned with Nguyen & Caruana (KDD'08): λ from {10^i}_{i=-3}^3 (here 10^-2);
# T is outer Pegasos iterations (larger → closer to QP solution, linear in time per iter).
DEFAULT_LAMBDA_REG = 0.01
DEFAULT_T = 2000


class PLSVMClassifier:
    """PL-SVM with primal PL-Pegasos updates (linear kernel, degree d = 1)."""

    def __init__(self, params=None):
        params = params or {}
        self.lambda_reg = float(params.get('lambda_reg', DEFAULT_LAMBDA_REG))
        self.T = int(params.get('T', DEFAULT_T))
        if self.lambda_reg <= 0:
            raise ValueError('lambda_reg must be positive')
        if self.T < 1:
            raise ValueError('T must be >= 1')

    def fit(self, X, y=None, partial_target=None):
        """Fit PL-SVM.

        Parameters
        ----------
        X : (n_samples, n_features)
        y : unused (kept for Evaluator API); supervision comes from ``partial_target``
        partial_target : (n_classes, n_samples), nonzero entries mark candidate labels Y_i
        """
        X = np.asarray(X, dtype=np.float64, order='C')
        if X.ndim != 2:
            raise ValueError('X must be 2-dimensional')
        n_samples = X.shape[0]

        if partial_target is not None:
            if hasattr(partial_target, 'toarray'):
                partial_target = partial_target.toarray()
            pt = np.asarray(partial_target, dtype=np.float64)
            if pt.ndim != 2:
                raise ValueError('partial_target must be 2-dimensional')
            n_classes, n_pt_samples = pt.shape
            if n_pt_samples != n_samples:
                raise ValueError('partial_target second dim must match len(X)')
        else:
            if y is None:
                raise ValueError('y is required when partial_target is None')
            y = np.asarray(y, dtype=int).ravel()
            if y.shape[0] != n_samples:
                raise ValueError('y and X must have the same number of rows')
            n_classes = int(y.max()) + 1 if y.size else 0
            pt = np.zeros((n_classes, n_samples), dtype=np.float64)
            pt[y, np.arange(n_samples)] = 1.0

        # Weight matrix W[c] is the c-th class block of w; score c is W[c] @ x
        W = np.zeros((n_classes, X.shape[1]), dtype=np.float64)
        lam = self.lambda_reg
        inv_sqrt_lam = 1.0 / np.sqrt(lam)
        # (n_samples, n_classes): mask_in[i,c] iff class c is a candidate for x_i
        mask_in = (pt > 0).T.astype(bool)
        row_no_cand = ~mask_in.any(axis=1)
        row_all_cand = mask_in.all(axis=1)
        row_valid = ~(row_no_cand | row_all_cand)
        neg_inf = np.array(-np.inf, dtype=np.float64)

        for t in range(1, self.T + 1):
            eta = 1.0 / (lam * t)
            # One GEMM replaces n_samples matvecs; numerically same as W @ x_i per row.
            S = X @ W.T
            s_in = np.where(mask_in, S, neg_inf)
            s_out = np.where(mask_in, neg_inf, S)
            y_in = np.argmax(s_in, axis=1).astype(np.intp)
            y_out = np.argmax(s_out, axis=1).astype(np.intp)
            margin = S[np.arange(n_samples, dtype=np.intp), y_in] - S[
                np.arange(n_samples, dtype=np.intp), y_out
            ]
            violated = row_valid & (margin < 1.0)
            dW = np.zeros_like(W)
            if violated.any():
                np.add.at(dW, y_in[violated], X[violated])
                np.add.at(dW, y_out[violated], -X[violated])

            W *= 1.0 - eta * lam
            W += (eta / n_samples) * dW

            norm = np.linalg.norm(W.ravel())
            if norm > 1e-18:
                scale = min(1.0, inv_sqrt_lam / norm)
                W *= scale

        self._W = W
        self._n_classes = n_classes
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64, order='C')
        if X.ndim != 2:
            raise ValueError('X must be 2-dimensional')
        scores = X @ self._W.T
        return np.argmax(scores, axis=1).astype(int)
