"""IPAL: Instance-based Partial Label Learning classifier.

Reference: Zhang & Yu, IJCAI'15.
Ported from MATLAB: IPAL_train.m, IPAL_predict.m.

Two phases:
  Train: iterative label propagation to disambiguate partial labels.
  Predict: minimum reconstruction error from KNN disambiguated labels.
"""

import numpy as np
from scipy.optimize import nnls
from sklearn.neighbors import KDTree


class IPALClassifier:
    """IPAL partial label classifier.

    Unlike KNN which needs ground truth y for fit(), IPAL takes partial_target
    directly and performs internal disambiguation.

    Params: k, alpha, max_iter
    """

    def __init__(self, params=None):
        params = params or {}
        self.k = params.get('k', 10)
        self.alpha = params.get('alpha', 0.95)
        self.max_iter = params.get('max_iter', 100)
        self.tol = params.get('tol', 1e-4)

    def fit(self, X, y=None, partial_target=None):
        """Train IPAL model.

        Parameters
        ----------
        X : (n_samples, n_features)
        y : ignored (kept for API compatibility)
        partial_target : (n_classes, n_samples) candidate label matrix
        """
        if partial_target is None:
            raise ValueError("IPAL requires partial_target for training")
        if hasattr(partial_target, 'toarray'):
            partial_target = partial_target.toarray()

        M, N = X.shape
        Q = partial_target.shape[0]

        # Row-normalize data (like MATLAB normr)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_norm = X / norms

        self._tree = KDTree(X_norm)
        self._X_norm = X_norm

        # Find KNN
        dists, indices = self._tree.query(X_norm, k=self.k + 1)
        neighbors = indices[:, 1:]  # (M, k)

        # Initialize y (label confidence), shape (M, Q)
        pt = partial_target.T.astype(float)  # (M, Q)
        row_sum = pt.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        y_conf = pt / row_sum

        # Build transition matrix via NNLS
        trans = np.zeros((M, M))
        for i in range(M):
            nb_data = X_norm[neighbors[i]].T  # (N, k)
            w, _ = nnls(nb_data, X_norm[i])
            trans[i, neighbors[i]] = w

        # Row-normalize transition matrix
        row_sum_t = trans.sum(axis=1, keepdims=True)
        row_sum_t[row_sum_t == 0] = 1
        trans = trans / row_sum_t

        # Iterative label propagation
        y0 = y_conf.copy()
        for iteration in range(self.max_iter):
            y_old = y_conf.copy()
            y_conf = self.alpha * (trans @ y_conf) + (1 - self.alpha) * y0
            y_conf = y_conf * pt  # mask to candidate labels
            row_sum_y = y_conf.sum(axis=1, keepdims=True)
            row_sum_y[row_sum_y == 0] = 1
            y_conf = y_conf / row_sum_y

            if np.linalg.norm(y_old - y_conf) < self.tol:
                break

        # Posterior calibration
        label_sum = y0.sum(axis=0)
        pred_sum = y_conf.sum(axis=0)
        pred_sum[pred_sum == 0] = 1
        poster = label_sum / pred_sum
        y_conf = y_conf * poster[np.newaxis, :]

        # Disambiguated labels: argmax per sample
        disambiguated = np.zeros((M, Q))
        for i in range(M):
            idx = np.argmax(y_conf[i])
            disambiguated[i, idx] = 1.0

        self._disambiguated = disambiguated  # (M, Q) one-hot
        self._Q = Q

        return self

    def predict(self, X):
        """Predict labels via minimum reconstruction error.

        Returns integer class labels (n_samples,).
        """
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_norm = X / norms

        dists, indices = self._tree.query(X_norm, k=self.k)

        n_test = X_norm.shape[0]
        pred = np.zeros(n_test, dtype=int)

        for i in range(n_test):
            nb_idx = indices[i]
            nb_data = self._X_norm[nb_idx].T  # (N, k)
            w, _ = nnls(nb_data, X_norm[i])

            nb_labels = self._disambiguated[nb_idx]  # (k, Q)

            min_residual = np.inf
            best_label = 0
            for label in range(self._Q):
                label_col = nb_labels[:, label]  # (k,)
                restore = nb_data @ (label_col * w)
                residual = np.linalg.norm(X_norm[i] - restore)
                if residual < min_residual:
                    min_residual = residual
                    best_label = label

            pred[i] = best_label

        return pred
