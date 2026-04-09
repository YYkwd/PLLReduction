"""CENDA: Confidence-based Dependence Maximization via HSIC.

Reference: Bao, Hang & Zhang, KDD'21.
Ported from MATLAB: CENDA.m, HSIC_Based_Solver.m, YUpdate_New.m.
"""

import numpy as np
from scipy.linalg import eig
from sklearn.neighbors import KDTree

from .base import BaseReducer


def _get_proper_dim(lambda_vals, dim_para):
    """Same logic as SDLPP/CENDA getProperDim.m."""
    if dim_para == 0:
        return len(lambda_vals)
    if dim_para < 1:
        total = np.sum(lambda_vals)
        if total <= 0:
            return len(lambda_vals)
        cumsum = 0.0
        for i, v in enumerate(lambda_vals):
            cumsum += v
            if cumsum >= dim_para * total:
                return i + 1
        return len(lambda_vals)
    return int(dim_para)


def _hsic_solver(X, L, mu, dim_para):
    """Solve generalized eigenvalue problem from adapted HSIC.

    X: (D, M) -- columns are samples
    L: (M, M) -- label kernel Y @ Y.T
    mu: trade-off
    dim_para: threshold

    Returns (P, eigenvalues)
    """
    D, N = X.shape

    # Center L: HLH where H = I - 1/N * 11^T
    col_mean = L.mean(axis=0, keepdims=True)  # (1, N)
    tmpL = L - col_mean  # broadcasting replaces np.tile
    row_mean = tmpL.mean(axis=1, keepdims=True)  # (N, 1)
    HLH = tmpL - row_mean  # broadcasting replaces np.tile

    S = X @ HLH @ X.T  # (D, D)
    B = mu * (X @ X.T) + (1 - mu) * np.eye(D)  # (D, D)

    eigvals, eigvecs = eig(S, B)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    proper_dim = _get_proper_dim(eigvals, dim_para)
    P = eigvecs[:, :proper_dim]
    lam = eigvals[:proper_dim]

    return P, lam


def _y_update_cenda(X_low_T, Y, k, cand_mask):
    """CENDA-style Y update: weighted KNN voting + accumulate original confidence.

    X_low_T: (D', M)
    Y: (M, Q)
    cand_mask: (M, Q) binary candidate mask (or list of arrays for back-compat)

    Returns Y_new (M, Q)
    """
    M, Q = Y.shape
    data = X_low_T.T  # (M, D')

    # Back-compat: if ins_set (list of arrays) is passed, convert to mask
    if isinstance(cand_mask, list):
        ins_set = cand_mask
        cand_mask = np.zeros((M, Q))
        for i in range(M):
            cand_mask[i, ins_set[i]] = 1.0

    tree = KDTree(data)
    _, indices = tree.query(data, k=k + 1)
    Idx = indices[:, 1:]  # (M, k)

    wr = np.arange(k, 0, -1, dtype=float)  # [k, k-1, ..., 1]

    # Vectorized: Z[i,q] = sum_j wr[j] * Y[Idx[i,j], q]
    # Y[nb,q] is already 0 for non-candidate q of sample nb, so no mask needed
    Y_nb = Y[Idx]  # (M, k, Q)
    Z = np.einsum('ijk,j->ik', Y_nb, wr)  # (M, Q)

    # Accumulate only for candidate labels, then normalize
    Z_new = Y + Z * cand_mask

    s = Z_new.sum(axis=1, keepdims=True)
    zero_rows = (s == 0).ravel()
    s[s == 0] = 1.0
    Z_new /= s
    Z_new[zero_rows] = Y[zero_rows]

    return Z_new


class CENDAReducer(BaseReducer):
    """CENDA dimensionality reduction via HSIC + KNN label update.

    Params: T, mu, dim_para, k
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.T = params.get('T', 50)
        self.mu = params.get('mu', 0.5)
        self.dim_para = params.get('dim_para', 0.999)
        self.k = params.get('k', 8)

    def fit(self, X, partial_target, disambiguator=None):
        if hasattr(partial_target, 'toarray'):
            partial_target = partial_target.toarray()

        M, D = X.shape
        Q = partial_target.shape[0]

        cand_mask = (partial_target.T > 0).astype(float)  # (M, Q)

        n_cands = cand_mask.sum(axis=1, keepdims=True)
        n_cands[n_cands == 0] = 1.0
        Y = cand_mask / n_cands  # uniform over candidates

        X_t = X.T  # (D, M)

        for t in range(self.T):
            L = Y @ Y.T  # (M, M) label kernel

            P, _ = _hsic_solver(X_t, L, self.mu, self.dim_para)
            X_low = P.T @ X_t  # (d', M)

            Y = _y_update_cenda(X_low, Y, self.k, cand_mask)

        # Final projection
        L = Y @ Y.T
        self.P_, _ = _hsic_solver(X_t, L, self.mu, self.dim_para)
        return self

    def transform(self, X):
        return X @ self.P_
