"""SDLPP: Semi-supervised Dimensionality Learning with Partial Labels.

Ported from original sdlpp.py. The disambiguation step is delegated
to a pluggable BaseDisambiguator instance.
"""

import numpy as np
from scipy.linalg import eig

from .base import BaseReducer


# ---------------------------------------------------------------------------
# Mathematical utilities
# ---------------------------------------------------------------------------

def eudist2(fea_a, fea_b=None, b_sqrt=True):
    if fea_b is None:
        aa = np.sum(fea_a * fea_a, axis=1, keepdims=True)
        ab = fea_a @ fea_a.T
        D = aa + aa.T - 2 * ab
        D = np.maximum(D, 0)
        if b_sqrt:
            D = np.sqrt(D)
        D = np.maximum(D, D.T)
    else:
        aa = np.sum(fea_a * fea_a, axis=1, keepdims=True)
        bb = np.sum(fea_b * fea_b, axis=1, keepdims=True).T
        ab = fea_a @ fea_b.T
        D = aa + bb - 2 * ab
        D = np.maximum(D, 0)
        if b_sqrt:
            D = np.sqrt(D)
    return D


def get_proper_dim(lambda_vals, dim_para):
    if dim_para == 0:
        return len(lambda_vals)
    if dim_para < 1:
        thr = dim_para
        sum_lambda = np.sum(lambda_vals)
        tmp_lambda = 0
        for lind in range(len(lambda_vals)):
            tmp_lambda += lambda_vals[lind]
            if tmp_lambda >= thr * sum_lambda:
                return lind + 1
        return len(lambda_vals)
    return int(dim_para)


def construct_s(X, D, k):
    """Build similarity S and compact neighbor arrays.

    Uses eudist2 for exact distance computation. Returns compact (m, k)
    neighbor arrays (replacing the old m x m E_dist) and a dense S matrix.
    Temporary m x m distance matrix is freed before return.

    Returns
    -------
    nn_indices : (m, k) int — neighbor sample indices, -1 if filtered out
    nn_dists   : (m, k) float — euclidean distances, 0.0 if filtered out
    S          : (m, m) float — heat-kernel similarity (dense)
    """
    m = X.shape[0]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    X_norm = X / norms

    dist = eudist2(X_norm)
    sorted_indices = np.argsort(dist, axis=1)
    neighbor = sorted_indices[:, 1:k+1]       # (m, k)
    allDist = np.sort(dist, axis=1)[:, 1:k+1]  # (m, k)
    del dist, sorted_indices

    sigma = np.mean(allDist[:, -1])
    if sigma < 1e-10:
        sigma = 1.0

    nn_indices = np.full((m, k), -1, dtype=int)
    nn_dists = np.zeros((m, k))
    S = np.zeros((m, m))

    for i in range(m):
        neighbor_idx = np.where(D[i, neighbor[i, :]] < 1)[0]
        if len(neighbor_idx) > 0:
            for j in neighbor_idx:
                neighbor_j = neighbor[i, j]
                distance = allDist[i, j]
                nn_indices[i, j] = neighbor_j
                nn_dists[i, j] = distance
                S[i, neighbor_j] = np.exp(
                    -distance * distance / (sigma * sigma))

    return nn_indices, nn_dists, S


def solve_projection(X, D, S, thr, miu):
    """Generalized eigenvalue problem: maximize Rayleigh quotient."""
    B = D - miu * S
    B = (B + B.T) / 2
    sum_B = np.sum(B, axis=1)
    A = np.diag(sum_B)
    L = A - B

    X_t = X.T
    XLXt = X_t @ L @ X_t.T
    XLXt = np.maximum(XLXt, XLXt.T)
    XAXt = X_t @ A @ X_t.T
    XAXt = np.maximum(XAXt, XAXt.T)

    eigvalues, eigvectors = eig(XLXt, XAXt)
    eigvalues = np.real(eigvalues)
    eigvectors = np.real(eigvectors)

    idx = np.argsort(-eigvalues)
    eigvalues = eigvalues[idx]
    eigvectors = eigvectors[:, idx]

    for j in range(eigvectors.shape[1]):
        eigvectors[:, j] /= np.linalg.norm(eigvectors[:, j])

    proper_dim = get_proper_dim(eigvalues, thr)
    P = eigvectors[:, :proper_dim]
    X_proj = (P.T @ X_t).T
    return X_proj, P


# ---------------------------------------------------------------------------
# SDLPPReducer
# ---------------------------------------------------------------------------

class SDLPPReducer(BaseReducer):
    """SDLPP with pluggable disambiguation.

    Params: T, target_d, k, miu, thr (same semantics as original).
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.T = params.get('T', 100)
        self.target_d = params.get('target_d', 13)
        self.k = params.get('k', 8)
        self.miu = params.get('miu', 0.1)
        self.thr = params.get('thr', 0.95)

    def fit(self, X, partial_target, disambiguator=None):
        if disambiguator is None:
            from pll.disambig.knn_propagation import KNNPropagation
            disambiguator = KNNPropagation()

        if hasattr(partial_target, 'toarray'):
            partial_target = partial_target.toarray()

        candidate_mask = (partial_target > 0).astype(float)
        temp = np.sum(partial_target, axis=0, keepdims=True)
        temp[temp == 0] = 1
        Y = partial_target.astype(float) / temp
        D = 1 - (Y.T @ Y)
        np.fill_diagonal(D, 0)

        self.Y_history_ = {'Y': [Y.copy()]}

        X_iter = X.copy()
        S = None

        for it in range(self.T):
            d_old = X_iter.shape[1]
            nn_indices, nn_dists, S = construct_s(X_iter, D, self.k)

            Y, D = disambiguator.disambiguate(
                Y, nn_indices, nn_dists, self.k, candidate_mask, iteration=it
            )

            X_iter, _ = solve_projection(X_iter, D, S, self.thr, self.miu)
            self.Y_history_['Y'].append(Y.copy())

            if X_iter.shape[1] == d_old:
                break

        _, self.P_ = solve_projection(X, D, S, self.target_d, self.miu)
        return self

    def transform(self, X):
        return X @ self.P_
