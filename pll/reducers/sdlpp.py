"""SDLPP: Semi-supervised Dimensionality Learning with Partial Labels.

Ported from original sdlpp.py. The disambiguation step is delegated
to a pluggable BaseDisambiguator instance.

Performance: all O(m^2) dense matrices (D, S, B, L) have been eliminated.
KNN uses sklearn NearestNeighbors; S is sparse; D is computed on-demand
via Y; solve_projection uses algebraic identities to avoid full L/B.
"""

import logging
import warnings

import numpy as np
from scipy.linalg import eig, eigh
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from .base import BaseReducer

log = logging.getLogger(__name__)

_DIAG = False


# ---------------------------------------------------------------------------
# Mathematical utilities
# ---------------------------------------------------------------------------

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


def _d_lookup_batch(Y, rows, cols):
    """Compute D[rows, cols] = 1 - sum_q Y[q, rows] * Y[q, cols] on-the-fly.

    Parameters
    ----------
    Y    : (Q, m) label confidence matrix
    rows : (m, 1) or broadcastable to cols.shape
    cols : (m, k) neighbor indices

    Returns
    -------
    D_vals : same shape as cols, float
    """
    r = np.broadcast_to(rows, cols.shape).ravel()
    c = cols.ravel()
    dot = np.einsum('ij,ij->j', Y[:, r], Y[:, c])
    return (1.0 - dot).reshape(cols.shape)


def construct_s(X, Y, k):
    """Build sparse similarity S and compact neighbor arrays via KNN.

    Uses sklearn NearestNeighbors (auto algorithm selection) instead of
    computing the full m x m distance matrix.  S is returned as a sparse
    CSR matrix with at most m*k nonzeros.

    Parameters
    ----------
    X : (m, d) feature matrix (will be L2-normalised internally)
    Y : (Q, m) label confidence matrix (for on-demand D computation)
    k : int, number of neighbors

    Returns
    -------
    nn_indices : (m, k) int  -- neighbor indices, -1 if filtered
    nn_dists   : (m, k) float -- distances, 0.0 if filtered
    S          : (m, m) sparse CSR -- heat-kernel similarity
    """
    m = X.shape[0]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    X_norm = X / norms

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
    nn.fit(X_norm)
    allDist, neighbor = nn.kneighbors(X_norm)
    neighbor = neighbor[:, 1:]          # (m, k)  -- exclude self
    allDist = allDist[:, 1:]            # (m, k)

    sigma = float(np.mean(allDist[:, -1]))
    if sigma < 1e-10:
        sigma = 1.0

    rows = np.arange(m)[:, None]                          # (m, 1)
    D_neighbors = _d_lookup_batch(Y, rows, neighbor)      # (m, k)
    valid_mask = D_neighbors < 1                           # (m, k)

    nn_indices = np.where(valid_mask, neighbor, -1)
    nn_dists = np.where(valid_mask, allDist, 0.0)

    vi, vj = np.where(valid_mask)
    vals = np.exp(-(allDist[vi, vj] ** 2) / (sigma * sigma))
    S = csr_matrix((vals, (vi, neighbor[vi, vj])), shape=(m, m))
    S = (S + S.T) * 0.5              # symmetrise (k-NN graph is directed)

    return nn_indices, nn_dists, S


def solve_projection(X, Y, S_sparse, thr, miu):
    """Generalized eigenvalue problem without materialising m x m matrices.

    Computes XLXt and XAXt using algebraic identities:
        B = D - miu*S,  L = diag(sum_B) - B
        XLXt = XAXt - XDXt + miu * XSXt

    where XDXt = outer(s,s) - (YX)^T(YX) - X^T diag(d_diag) X
    and   XAXt = X^T diag(sum_B) X.

    All terms are O(m*d^2) or O(m*d*Q) instead of O(m^2).
    """
    m, d = X.shape
    Q = Y.shape[0]

    # --- sum_B = sum_D - miu * sum_S  (m,) ---
    Y_col_sum = Y.sum(axis=1)                          # (Q,)
    y_sq = np.einsum('ij,ij->j', Y, Y)                 # ||Y[:,i]||^2  (m,)
    sum_D = (m - 1) - (Y.T @ Y_col_sum) + y_sq         # (m,)
    sum_S = np.asarray(S_sparse.sum(axis=1)).ravel()    # (m,)
    sum_B = sum_D - miu * sum_S                         # (m,)

    # --- XAXt = X^T diag(sum_B) X  (d, d) ---
    XAXt = (X * sum_B[:, None]).T @ X
    XAXt = np.fmax(XAXt, XAXt.T)

    # --- XDXt via algebraic identity ---
    # D = J - Y^TY - I + diag(y_sq)  where J=ones(m,m)
    # X^T D X = X^T J X - X^T Y^TY X - X^TX + X^T diag(y_sq) X
    s = X.sum(axis=0)                                   # (d,)
    XJXt = np.outer(s, s)                                # (d, d)
    YX = Y @ X                                           # (Q, d)
    XYtYXt = YX.T @ YX                                  # (d, d)
    XtX = X.T @ X                                        # (d, d)
    Xw = (X * y_sq[:, None]).T @ X                       # (d, d)
    XDXt = XJXt - XYtYXt - XtX + Xw                     # (d, d)

    # --- XSXt via sparse matmul ---
    SX = S_sparse @ X                                    # (m, d)  sparse @ dense
    XSXt = X.T @ SX                                      # (d, d)

    # --- XLXt = XAXt - XDXt + miu * XSXt ---
    XLXt = XAXt - XDXt + miu * XSXt
    XLXt = np.fmax(XLXt, XLXt.T)

    # Regularise XAXt for eigh
    tr = np.trace(XAXt)
    n = d
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
        eigvalues = eigvalues[::-1].copy()
        eigvectors = eigvectors[:, ::-1].copy()
    except np.linalg.LinAlgError:
        eigvalues, eigvectors = eig(XLXt, XAXt)
        eigvalues = np.real(eigvalues)
        eigvectors = np.real(eigvectors)
        idx = np.argsort(-eigvalues)
        eigvalues = eigvalues[idx]
        eigvectors = eigvectors[:, idx]

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
    X_proj = (P.T @ X.T).T

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

    Params: T, target_d, k, miu, thr, y_tol, max_stall.
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.T = params.get('T', 100)
        self.target_d = params.get('target_d', 13)
        self.k = params.get('k', 8)
        self.miu = params.get('miu', 0.1)
        self.thr = params.get('thr', 0.95)
        self.y_tol = params.get('y_tol', 1e-6)
        self.max_stall = params.get('max_stall', 0)

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

        self.Y_history_ = {'Y': [Y.copy()]}

        X_iter = X.copy()
        S_sparse = None
        stall_count = 0
        stop_reason = 'max_iter'

        for it in range(self.T):
            d_old = X_iter.shape[1]
            Y_old = Y.copy()

            nn_indices, nn_dists, S_sparse = construct_s(
                X_iter, Y, self.k)

            Y = disambiguator.disambiguate(
                Y, nn_indices, nn_dists, self.k, candidate_mask, iteration=it
            )

            X_iter, _ = solve_projection(
                X_iter, Y, S_sparse, self.thr, self.miu)
            self.Y_history_['Y'].append(Y.copy())

            d_new = X_iter.shape[1]
            y_norm_old = np.linalg.norm(Y_old)
            y_diff = (np.linalg.norm(Y - Y_old) / (y_norm_old + 1e-12)
                      if y_norm_old > 0 else 0.0)

            log.debug("iter %d/%d  dim %d->%d  Y_diff=%.2e",
                      it + 1, self.T, d_old, d_new, y_diff)

            if not np.all(np.isfinite(X_iter)):
                stop_reason = 'non_finite'
                break

            if d_new == d_old:
                stop_reason = 'dim_stable'
                break

            if self.y_tol > 0 and y_diff < self.y_tol:
                stop_reason = 'y_converged'
                break

            if d_new <= self.target_d:
                stop_reason = 'min_dim'
                break

            if self.max_stall > 0:
                if d_old - d_new <= 1:
                    stall_count += 1
                else:
                    stall_count = 0
                if stall_count >= self.max_stall:
                    stop_reason = 'stall'
                    break

        log.info("SDLPP fit: %d/%d iters, stop=%s, dim %d->%d",
                 it + 1, self.T, stop_reason,
                 X.shape[1], X_iter.shape[1])

        _, self.P_ = solve_projection(
            X, Y, S_sparse, self.target_d, self.miu)
        return self

    def transform(self, X):
        return X @ self.P_
