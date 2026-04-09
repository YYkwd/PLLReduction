"""SDLPP: Semi-supervised Dimensionality Learning with Partial Labels.

Ported from original sdlpp.py. The disambiguation step is delegated
to a pluggable BaseDisambiguator instance.
"""

import numpy as np
from scipy.linalg import eig, eigh

from .base import BaseReducer

_DIAG = False


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
        positive = lambda_vals[lambda_vals > 0]
        if len(positive) == 0:
            return len(lambda_vals)
        sum_pos = float(np.sum(positive))
        tmp_lambda = 0.0
        for lind in range(len(lambda_vals)):
            if lambda_vals[lind] > 0:
                tmp_lambda += lambda_vals[lind]
            if tmp_lambda >= thr * sum_pos:
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
    """Generalized eigenvalue problem: maximize Rayleigh quotient.

    Regularizes XAXt and prefers eigh (symmetric solver) over eig for
    numerical stability, matching MATLAB behaviour on ill-conditioned data.
    """
    B = D - miu * S
    B = (B + B.T) / 2
    sum_B = np.sum(B, axis=1)
    L = np.diag(sum_B) - B

    X_t = X.T
    # Diagonal-aware multiplication: avoids materialising full m x m A
    XL = X_t @ L                             # (d, m)
    XLXt = XL @ X_t.T                        # (d, d)
    XLXt = np.fmax(XLXt, XLXt.T)            # symmetrise (fmax ignores NaN)

    XA = X_t * sum_B[np.newaxis, :]          # column scaling
    XAXt = XA @ X_t.T
    XAXt = np.fmax(XAXt, XAXt.T)

    # Additive regularisation: makes XAXt positive-definite so eigh works
    tr = np.trace(XAXt)
    n = XAXt.shape[0]
    reg = max(1e-10 * tr / n, 1e-14) if tr > 0 else 1e-10
    XAXt += reg * np.eye(n)

    if _DIAG:
        n_neg = int(np.sum(sum_B < 0))
        n_zero = int(np.sum(np.abs(sum_B) < 1e-15))
        print(f"  [DIAG solve] X shape={X.shape}, sum_B neg={n_neg} zero={n_zero}")
        print(f"  [DIAG solve] XAXt cond={np.linalg.cond(XAXt):.2e}, "
              f"XAXt NaN={np.sum(np.isnan(XAXt))}, Inf={np.sum(np.isinf(XAXt))}")

    try:
        eigvalues, eigvectors = eigh(XLXt, XAXt)
        # eigh returns ascending order; flip to descending
        eigvalues = eigvalues[::-1].copy()
        eigvectors = eigvectors[:, ::-1].copy()
    except np.linalg.LinAlgError:
        eigvalues, eigvectors = eig(XLXt, XAXt)
        eigvalues = np.real(eigvalues)
        eigvectors = np.real(eigvectors)
        idx = np.argsort(-eigvalues)
        eigvalues = eigvalues[idx]
        eigvectors = eigvectors[:, idx]

    # Drop non-finite eigenvalues and their eigenvectors
    finite_mask = np.isfinite(eigvalues)
    if not np.all(finite_mask):
        eigvalues = eigvalues[finite_mask]
        eigvectors = eigvectors[:, finite_mask]

    if _DIAG:
        n_pos = int(np.sum(eigvalues > 0))
        n_neg_ev = int(np.sum(eigvalues < 0))
        print(f"  [DIAG eig] eigvals(finite): n={len(eigvalues)} "
              f"min={eigvalues.min():.4e} max={eigvalues.max():.4e} "
              f"pos={n_pos} neg={n_neg_ev}")

    # Normalise eigenvectors; zero out degenerate ones
    for j in range(eigvectors.shape[1]):
        nrm = np.linalg.norm(eigvectors[:, j])
        if nrm > 1e-12:
            eigvectors[:, j] /= nrm
        else:
            eigvectors[:, j] = 0.0

    proper_dim = get_proper_dim(eigvalues, thr)
    proper_dim = min(proper_dim, eigvectors.shape[1])

    if _DIAG:
        print(f"  [DIAG dim] thr={thr}, proper_dim={proper_dim}")

    P = eigvectors[:, :proper_dim]
    X_proj = (P.T @ X_t).T

    if _DIAG:
        n_bad = int(np.sum(~np.isfinite(X_proj)))
        if n_bad:
            print(f"  [DIAG proj] WARNING X_proj has {n_bad} non-finite values")

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

            if not np.all(np.isfinite(X_iter)):
                break

            if X_iter.shape[1] == d_old:
                break

        _, self.P_ = solve_projection(X, D, S, self.target_d, self.miu)
        return self

    def transform(self, X):
        return X @ self.P_
