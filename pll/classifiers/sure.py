"""SURE: Self-guided retraining for partial label learning (Feng & An, AAAI'19).

Alternates closed-form updates of a kernel (or linear) least-squares model and
per-sample QP surrogates (paper Eq. (10)) for the confidence matrix P.

Kernel vs linear (``classifier.params``):

- ``use_kernel=True`` — always RBF kernel SURE (paper's main setup; O(m²) memory).
- ``use_kernel=False`` — always linear SURE (Eq. (2)–(4)); cheap but weak on
  very large / highly imbalanced data.
- ``use_kernel`` omitted / ``null`` — **auto**: kernel iff ``m <= kernel_max_samples``
  (default 2500), else linear.

Reference: Feng & An, "Partial Label Learning with Self-Guided Retraining", AAAI 2019.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

# Paper grid for λ, β (user should tune via validation).
DEFAULT_LAMBDA = 0.1
DEFAULT_BETA = 0.1
DEFAULT_MAX_ITER = 50
DEFAULT_TOL = 1e-4
# When ``use_kernel`` is unset: kernel iff train size m <= this threshold (O(m²) RAM).
DEFAULT_KERNEL_MAX_SAMPLES = 2500


def _initial_p_from_candidates(y_row: np.ndarray) -> np.ndarray:
    """Uniform distribution on candidate labels (binary mask y_row)."""
    y_row = np.asarray(y_row, dtype=np.float64)
    s = y_row.sum()
    if s <= 0:
        raise ValueError('SURE requires at least one candidate label per sample')
    return y_row / s


def _solve_p_surrogate_row(
    q: np.ndarray,
    y_mask: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Minimize surrogate OPS (paper Eq. (10)) for one sample; j = argmax q on candidates."""
    q = np.asarray(q, dtype=np.float64).ravel()
    y_mask = np.asarray(y_mask, dtype=np.float64).ravel()
    l = q.shape[0]
    cand = np.flatnonzero(y_mask > 0)
    c = cand.size
    if c == 1:
        p = np.zeros(l, dtype=np.float64)
        p[int(cand[0])] = 1.0
        return p

    q_c = q[cand]
    j_loc = int(np.argmax(q_c))
    j_global = int(cand[j_loc])

    def objective(pc: np.ndarray) -> float:
        return float(np.sum((pc - q_c) ** 2) - lam * pc[j_loc])

    def grad(pc: np.ndarray) -> np.ndarray:
        g = 2.0 * (pc - q_c)
        g[j_loc] -= lam
        return g

    cons = [{'type': 'eq', 'fun': lambda pc: np.sum(pc) - 1.0}]
    for k in range(c):
        if k != j_loc:
            cons.append({'type': 'ineq', 'fun': lambda pc, jl=j_loc, kk=k: pc[jl] - pc[kk]})

    p0 = np.ones(c, dtype=np.float64) / c
    res = minimize(
        objective,
        p0,
        method='SLSQP',
        jac=grad,
        bounds=[(0.0, None)] * c,
        constraints=cons,
        options={'maxiter': 400, 'ftol': 1e-12},
    )
    pc = res.x if res.success else p0
    pc = np.clip(pc, 0.0, None)
    s = pc.sum()
    if s > 0:
        pc = pc / s
    p = np.zeros(l, dtype=np.float64)
    p[cand] = pc
    return p


def _rbf_sigma(X: np.ndarray) -> float:
    """Gaussian bandwidth: mean pairwise Euclidean distance (paper)."""
    m = X.shape[0]
    if m < 2:
        return 1.0
    d = pdist(X, metric='euclidean')
    sigma = float(np.mean(d))
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = 1.0
    return sigma


def _kernel_matrix(X: np.ndarray, sigma: float) -> np.ndarray:
    from sklearn.metrics.pairwise import euclidean_distances

    d2 = euclidean_distances(X, X, squared=True)
    return np.exp(-d2 / (2.0 * sigma * sigma))


def _linear_solver(inv_block: np.ndarray):
    """Pre-factor symmetric inv_block; return solve(rhs) closure (same as np.linalg.solve)."""
    try:
        c, low = cho_factor(inv_block, lower=True, check_finite=False)

        def solve(rhs: np.ndarray) -> np.ndarray:
            return cho_solve((c, low), rhs, check_finite=False)
    except np.linalg.LinAlgError:
        lu, piv = lu_factor(inv_block, check_finite=False)

        def solve(rhs: np.ndarray) -> np.ndarray:
            return lu_solve((lu, piv), rhs, check_finite=False)

    return solve


def _kernel_solver(B: np.ndarray):
    """LU factor B once; each solve is O(m²) per RHS block (same result as np.linalg.solve)."""
    lu, piv = lu_factor(B, check_finite=False)

    def solve(rhs: np.ndarray) -> np.ndarray:
        return lu_solve((lu, piv), rhs, check_finite=False)

    return solve


class SUREClassifier:
    """Kernel or linear SURE (AAAI'19) for PLL after dimensionality reduction.

    Params (YAML / ``classifier.params``):

    - ``use_kernel`` (bool | omit): see module docstring; omit = auto by ``kernel_max_samples``.
    - ``kernel_max_samples`` (int): auto-mode threshold only.
    - ``lambda``, ``beta``, ``max_iter``, ``tol``, ``sigma`` (optional RBF bandwidth).
    """

    def __init__(self, params=None):
        params = params or {}
        self.lam = float(params.get('lambda', DEFAULT_LAMBDA))
        self.beta = float(params.get('beta', DEFAULT_BETA))
        self.max_iter = int(params.get('max_iter', DEFAULT_MAX_ITER))
        self.tol = float(params.get('tol', DEFAULT_TOL))
        self.kernel_max_samples = int(
            params.get('kernel_max_samples', DEFAULT_KERNEL_MAX_SAMPLES)
        )
        use_k = params.get('use_kernel', None)
        if use_k is None:
            self._use_kernel = None  # auto: kernel iff m <= kernel_max_samples
        else:
            self._use_kernel = bool(use_k)
        self._sigma = params.get('sigma', None)
        if self.lam < 0 or self.beta <= 0:
            raise ValueError('require lambda >= 0 and beta > 0')
        if self.max_iter < 1:
            raise ValueError('max_iter must be >= 1')

        self._X_train: np.ndarray | None = None
        self._use_kernel_fit: bool = False
        self._sigma_fit: float = 1.0
        self._A: np.ndarray | None = None
        self._b: np.ndarray | None = None
        self._W: np.ndarray | None = None
        self._n_classes: int = 0

    def fit(self, X, y=None, partial_target=None):
        X = np.asarray(X, dtype=np.float64, order='C')
        if X.ndim != 2:
            raise ValueError('X must be 2-dimensional')
        m = X.shape[0]

        if partial_target is None:
            raise ValueError('SURE requires partial_target for training')
        if hasattr(partial_target, 'toarray'):
            partial_target = partial_target.toarray()
        pt = np.asarray(partial_target, dtype=np.float64)
        if pt.ndim != 2:
            raise ValueError('partial_target must be 2-dimensional')
        n_classes, n_pt = pt.shape
        if n_pt != m:
            raise ValueError('partial_target second dim must match len(X)')
        self._n_classes = n_classes

        Y = (pt > 0).astype(np.float64).T
        P = np.array([_initial_p_from_candidates(Y[i]) for i in range(m)], dtype=np.float64)

        use_kernel = self._use_kernel if self._use_kernel is not None else (
            m <= self.kernel_max_samples
        )
        self._use_kernel_fit = use_kernel
        self._X_train = X.copy()

        if use_kernel:
            sigma = float(self._sigma) if self._sigma is not None else _rbf_sigma(X)
            self._sigma_fit = sigma
            self._W = None
            K = _kernel_matrix(X, sigma)
            m_ = X.shape[0]
            one = np.ones((m_, 1), dtype=np.float64)
            colsum = K.T @ one
            B = K + self.beta * np.eye(m_, dtype=np.float64) - (one @ colsum.T) / m_
            solve_B = _kernel_solver(B)
            k_ones = (K @ one).ravel()
            for _ in range(self.max_iter):
                rhs = P - (one @ (one.T @ P)) / m_
                A_it = solve_B(rhs)
                b_it = (P.T @ one.ravel() - A_it.T @ k_ones) / m_
                Q = K @ A_it + b_it[np.newaxis, :]
                P_new = np.array(
                    [_solve_p_surrogate_row(Q[i], Y[i], self.lam) for i in range(m)],
                    dtype=np.float64,
                )
                delta = np.linalg.norm(P_new - P, ord='fro')
                P = P_new
                if delta < self.tol:
                    break
            rhs = P - (one @ (one.T @ P)) / m_
            self._A = solve_B(rhs)
            self._b = (P.T @ one.ravel() - self._A.T @ k_ones) / m_
        else:
            self._sigma_fit = 0.0
            self._A = None
            m_ = X.shape[0]
            n = X.shape[1]
            XtX = X.T @ X
            Xt1 = X.T @ np.ones((m_, 1), dtype=np.float64)
            inv_block = XtX + self.beta * np.eye(n, dtype=np.float64) - (Xt1 @ Xt1.T) / m_
            ones_row = np.ones((1, m_), dtype=np.float64)
            solve_lin = _linear_solver(inv_block)
            x_sum = X.T @ np.ones(m_)
            for _ in range(self.max_iter):
                rhs = X.T @ P - (Xt1 @ (ones_row @ P)) / m_
                W_it = solve_lin(rhs)
                b_it = (P.T @ np.ones(m_) - W_it.T @ x_sum) / m_
                Q = X @ W_it + b_it[np.newaxis, :]
                P_new = np.array(
                    [_solve_p_surrogate_row(Q[i], Y[i], self.lam) for i in range(m)],
                    dtype=np.float64,
                )
                delta = np.linalg.norm(P_new - P, ord='fro')
                P = P_new
                if delta < self.tol:
                    break
            rhs = X.T @ P - (Xt1 @ (ones_row @ P)) / m_
            self._W = solve_lin(rhs)
            self._b = (P.T @ np.ones(m_) - self._W.T @ x_sum) / m_
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64, order='C')
        if X.ndim != 2:
            raise ValueError('X must be 2-dimensional')
        if self._X_train is None:
            raise RuntimeError('call fit before predict')

        if self._use_kernel_fit:
            from sklearn.metrics.pairwise import euclidean_distances

            d2 = euclidean_distances(X, self._X_train, squared=True)
            K_ts = np.exp(-d2 / (2.0 * self._sigma_fit * self._sigma_fit))
            scores = K_ts @ self._A + self._b[np.newaxis, :]
        else:
            scores = X @ self._W + self._b[np.newaxis, :]
        return np.argmax(scores, axis=1).astype(int)
