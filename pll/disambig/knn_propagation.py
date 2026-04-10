"""KNN-based label propagation disambiguation.

Extracted from original sdlpp.py: update_y, compute_sample_reliability,
compute_class_weights. Algorithm logic is preserved exactly.
"""

import numpy as np
from .base import BaseDisambiguator


def _compute_sample_reliability(Y, candidate_mask, eps=1e-8, r_min=0.0):
    """H_i -> H_tilde_i -> r_i = 1 - H_tilde_i"""
    n_candidates = np.sum(candidate_mask, axis=0).astype(float)
    H = -np.sum(Y * np.log(Y + eps), axis=0)
    H_tilde = np.zeros_like(H)
    multi = n_candidates > 1
    H_tilde[multi] = H[multi] / (np.log(n_candidates[multi] + eps) + eps)
    r = 1 - H_tilde
    return np.clip(r, r_min, 1)


def _compute_class_weights(Y, r, alpha=0.5, eps=1e-8):
    """N_k^soft = sum_i r_i q_ik, w_k = 1/(N_k^soft)^alpha, mean-normalized."""
    N_soft = np.sum(r[np.newaxis, :] * Y, axis=1)
    w_cls = 1.0 / np.power(N_soft + eps, alpha)
    w_cls = w_cls / (np.mean(w_cls) + eps)
    return w_cls


def _class_imbalance_cv(Y, r, eps=1e-8):
    """Coefficient of variation of class soft counts."""
    N_soft = np.sum(r[np.newaxis, :] * Y, axis=1)
    return float(np.std(N_soft) / (np.mean(N_soft) + eps))


def _update_y(nn_indices, nn_dists, Y_last, k, candidate_mask,
              r=None, w_cls=None, eps=1e-8):
    """KNN label propagation: aggregate neighbor confidences, mask, normalize.

    Vectorized: avoids rebuilding m x m E_dist and per-sample Python loops.
    Returns only Y_new; the caller computes D on-demand from Y when needed.
    """
    Q, m = Y_last.shape

    self_idx = np.arange(m).reshape(-1, 1)                    # (m, 1)
    all_nn = np.concatenate([self_idx, nn_indices], axis=1)    # (m, k+1)

    valid = np.ones((m, nn_indices.shape[1] + 1), dtype=bool)
    valid[:, 1:] = nn_indices >= 0                             # (m, k+1)

    safe_idx = np.where(all_nn >= 0, all_nn, 0)               # (m, k+1)

    Y_nb = Y_last[:, safe_idx]                                 # (Q, m, k+1)
    Y_nb = Y_nb * valid[np.newaxis, :, :]

    if r is not None:
        r_nb = r[safe_idx] * valid                             # (m, k+1)
        Y_nb = Y_nb * r_nb[np.newaxis, :, :]

    Y_new = np.sum(Y_nb, axis=2)                               # (Q, m)
    Y_new *= candidate_mask

    if w_cls is not None:
        Y_new *= w_cls[:, np.newaxis]

    col_sum = np.sum(Y_new, axis=0, keepdims=True) + eps       # (1, m)
    Y_new /= col_sum

    Y_new *= candidate_mask

    return Y_new


class KNNPropagation(BaseDisambiguator):
    """KNN label propagation with optional sample reliability and class balance.

    Params
    ------
    use_sample_reliability : bool
    use_class_balance : bool
    alpha : float   (class balance exponent, also serves as alpha_max)
    r_min : float   (lower bound for sample reliability)
    warmup : int    (disable SR before this iteration)
    cb_adaptive_alpha : bool
        When True, alpha is scaled by a ramp based on class imbalance CV:
        alpha_eff = alpha * clip((cv - cb_cv0) / (cb_cv1 - cb_cv0), 0, 1)
    cb_cv0 : float  (CV below which alpha_eff = 0, i.e. balanced -> no CB)
    cb_cv1 : float  (CV above which alpha_eff = alpha_max, full CB)
    eps : float
    """

    def __init__(self, params=None):
        super().__init__(params)
        p = self.params
        self.use_sample_reliability = p.get('use_sample_reliability', False)
        self.use_class_balance = p.get('use_class_balance', False)
        self.r_min = p.get('r_min', 0.0)
        self.warmup = p.get('warmup', 0)
        self.cb_adaptive_alpha = p.get('cb_adaptive_alpha', False)
        self.cb_cv0 = p.get('cb_cv0', 0.1)
        self.cb_cv1 = p.get('cb_cv1', 0.5)
        self.alpha = p.get('alpha', 0.5)
        self.eps = p.get('eps', 1e-8)

    def disambiguate(self, Y, nn_indices, nn_dists, k, candidate_mask, iteration=0):
        r, w_cls = None, None

        use_sr_now = self.use_sample_reliability and (iteration >= self.warmup)

        if use_sr_now:
            r = _compute_sample_reliability(Y, candidate_mask, self.eps, self.r_min)

        if self.use_class_balance:
            r_for_cls = r if r is not None else np.ones(Y.shape[1])
            if self.cb_adaptive_alpha:
                cv = _class_imbalance_cv(Y, r_for_cls, self.eps)
                ramp = (cv - self.cb_cv0) / (self.cb_cv1 - self.cb_cv0 + self.eps)
                alpha_eff = self.alpha * float(np.clip(ramp, 0, 1))
                if alpha_eff > 1e-12:
                    w_cls = _compute_class_weights(
                        Y, r_for_cls, alpha_eff, self.eps)
            else:
                w_cls = _compute_class_weights(
                    Y, r_for_cls, self.alpha, self.eps)

        return _update_y(nn_indices, nn_dists, Y, k, candidate_mask,
                         r=r, w_cls=w_cls, eps=self.eps)
