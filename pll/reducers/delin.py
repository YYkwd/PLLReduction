"""DELIN: Disambiguation Enabled Linear Discriminant Analysis.

Reference: Wu & Zhang, KDD'19.
Ported from MATLAB: DELIN.m, LDA_calculation.m, KNN_disambiguation.m,
Initialization_labelConfidence.m.
"""

import numpy as np
from scipy.linalg import eig
from sklearn.neighbors import KDTree

from .base import BaseReducer


def _init_label_confidence(partial_target):
    """Uniform init: Y[i,j] = 1/|S_i| for candidate j, else 0.

    Input:  partial_target (n_classes, n_samples)
    Output: Y (n_samples, n_classes)  -- NOTE: transposed vs SDLPP convention
    """
    pt = partial_target.T  # (n_samples, n_classes)
    row_sum = pt.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    Y = pt / row_sum
    return Y.astype(float)


def _lda_projection(X, Y, ratio):
    """Weighted LDA projection.

    X: (n_features, n_samples)  -- columns are samples
    Y: (n_samples, n_classes)   -- label confidence
    ratio: fraction of min(Q, N) to retain

    Returns P: (n_features, reduced_dim)
    """
    N, M = X.shape
    Q = Y.shape[1]

    Wk = Y.sum(axis=0)            # (Q,) per-class weight sum
    Wt = Wk.sum()

    m = (X @ (Y.sum(axis=1))) / Wt   # (N,)
    Xm = X - m[:, np.newaxis]         # centered

    sumL = Y.sum(axis=1)              # (M,)

    W1 = np.zeros(Q)
    nz = Wk > 0
    W1[nz] = 1.0 / Wk[nz]

    # Avoid constructing large M*M diagonal matrices
    St = (Xm * sumL[np.newaxis, :]) @ Xm.T        # (N, N)
    Sb = Xm @ ((Y * W1) @ Y.T) @ Xm.T             # (N, N)
    Sw = St - Sb

    Sw = (Sw + Sw.T) / 2.0
    Sb = (Sb + Sb.T) / 2.0

    eigvals, eigvecs = eig(Sb, Sw)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    bad = ~(np.isreal(eigvals) & (eigvals > 0) & np.isfinite(eigvals))
    eigvals[bad] = 0.0

    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    reduced_dim = max(1, int(np.ceil(ratio * min(Q, N))))
    P = eigvecs[:, :reduced_dim]
    return P


def _knn_disambiguate_delin(X_low, k, Q, candidate_mask, Y):
    """KNN disambiguation (DELIN style), vectorized sumY/count.

    X_low: (n_samples, reduced_dim)
    Y: (n_samples, n_classes)
    candidate_mask: (n_samples, n_classes) binary

    Returns updated Y (n_samples, n_classes)
    """
    M = X_low.shape[0]
    tree = KDTree(X_low)
    dists, indices = tree.query(X_low, k=k + 1)
    neighbors = indices[:, 1:]  # (M, k)

    wr = np.arange(k, 0, -1, dtype=float)  # [k, k-1, ..., 1]

    # Vectorized: sumY[i,q] = sum_j wr[j] * Y[neighbor[i,j], q]
    Y_nb = Y[neighbors]  # (M, k, Q)
    sumY = np.einsum('ijk,j->ik', Y_nb, wr)        # (M, Q)
    sumY *= candidate_mask
    count = (Y_nb > 0).sum(axis=1).astype(float)    # (M, Q)
    count *= candidate_mask

    Y_new = Y.copy()

    for i in range(M):
        cand_indices = np.where(candidate_mask[i] > 0)[0]
        if len(cand_indices) == 0:
            continue

        cand_scores = sumY[i, cand_indices]
        max_score = cand_scores.max()
        if max_score == 0:
            continue

        best_labels = cand_indices[cand_scores == max_score]
        if len(best_labels) == 0:
            continue

        if len(best_labels) > 1:
            best_counts = count[i, best_labels]
            max_count = best_counts.max()
            best_labels = best_labels[best_counts == max_count]
            if len(best_labels) > 1:
                n_votes = int(max_count) * len(best_labels)
                if n_votes > k:
                    pick = np.random.randint(len(best_labels))
                    best_labels = best_labels[pick:pick + 1]

        neighbor_votes = count[i, best_labels[0]]
        estimated = best_labels
        n_est = len(estimated)

        if np.array_equal(np.sort(estimated), np.sort(cand_indices)):
            continue

        diff = np.setdiff1d(cand_indices, estimated)
        n_diff = len(diff)

        Y_new[i, estimated] = neighbor_votes / k
        if n_diff > 0:
            remainder = max(1.0 - (neighbor_votes * n_est) / k, 0.0)
            Y_new[i, diff] = remainder / n_diff

    # Vectorized L1 normalize
    s = Y_new.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    Y_new /= s

    return Y_new


class DELINReducer(BaseReducer):
    """DELIN dimensionality reduction via iterative LDA + KNN disambiguation.

    Params: T, ratio, k
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.T = params.get('T', 75)
        self.ratio = params.get('ratio', 0.6)
        self.k = params.get('k', 8)

    def fit(self, X, partial_target, disambiguator=None):
        if hasattr(partial_target, 'toarray'):
            partial_target = partial_target.toarray()

        M, N = X.shape
        Q = partial_target.shape[0]

        # Y: (M, Q) -- DELIN uses sample-major layout
        Y = _init_label_confidence(partial_target)
        candidate_mask = (partial_target.T > 0).astype(float)

        X_t = X.T  # (N, M) for LDA

        for t in range(self.T):
            Y_old = Y.copy()

            P = _lda_projection(X_t, Y, self.ratio)
            X_low = (P.T @ X_t).T  # (M, reduced_dim)

            Y = _knn_disambiguate_delin(X_low, self.k, Q, candidate_mask, Y)

            if np.allclose(Y_old, Y):
                break

        self.P_ = _lda_projection(X_t, Y, self.ratio)
        return self

    def transform(self, X):
        return X @ self.P_
