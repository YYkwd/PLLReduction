"""SDLPP: Semi-supervised Dimensionality Learning with Partial Labels"""

import numpy as np
from scipy.linalg import eig


def eudist2(fea_a, fea_b=None, b_sqrt=True):
    """Compute Euclidean distance matrix."""
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
    """Determine dimensionality of projected feature space."""
    if dim_para == 0:
        proper_dim = len(lambda_vals)
    elif dim_para < 1:
        thr = dim_para
        sum_lambda = np.sum(lambda_vals)
        tmp_lambda = 0
        proper_dim = len(lambda_vals)
        for lind in range(len(lambda_vals)):
            tmp_lambda += lambda_vals[lind]
            if tmp_lambda >= thr * sum_lambda:
                proper_dim = lind + 1
                break
    else:
        proper_dim = int(dim_para)
    
    return proper_dim


def compute_sample_reliability(Y, candidate_mask, eps=1e-8):
    """
    Compute sample reliability from normalized entropy.
    H_i = -sum_k q_ik log(q_ik + eps)
    H_tilde_i = H_i / log(|S_i| + eps), with H_tilde_i=0 when |S_i|=1
    r_i = 1 - H_tilde_i
    """
    n_candidates = np.sum(candidate_mask, axis=0).astype(float)  # |S_i|
    H = -np.sum(Y * np.log(Y + eps), axis=0)
    H_tilde = np.zeros_like(H)
    multi = n_candidates > 1
    H_tilde[multi] = H[multi] / (np.log(n_candidates[multi] + eps) + eps)
    r = 1 - H_tilde
    r = np.clip(r, 0, 1)
    return r


def compute_class_weights(Y, r, alpha=0.5, eps=1e-8):
    """
    N_k^soft = sum_i r_i q_ik
    w_k^cls = 1 / (N_k^soft + eps)^alpha, then mean-normalize
    """
    n_classes = Y.shape[0]
    N_soft = np.sum(r[np.newaxis, :] * Y, axis=1)
    w_cls = 1.0 / np.power(N_soft + eps, alpha)
    w_cls = w_cls / (np.mean(w_cls) + eps)
    return w_cls


def construct_s(X, D, k):
    """Construct similarity matrix S and adjacency matrix E_dist."""
    m = X.shape[0]
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    dist = eudist2(X_norm)
    sorted_indices = np.argsort(dist, axis=1)
    neighbor = sorted_indices[:, 1:k+1]
    allDist = np.sort(dist, axis=1)[:, 1:k+1]
    
    sigma = np.mean(allDist[:, -1])
    
    E_dist = np.zeros((m, m))
    S = np.zeros((m, m))
    
    for i in range(m):
        neighbor_idx = np.where(D[i, neighbor[i, :]] < 1)[0]
        
        if len(neighbor_idx) > 0:
            for j in neighbor_idx:
                neighbor_j = neighbor[i, j]
                distance = allDist[i, j]
                E_dist[i, neighbor_j] = distance
                S[i, neighbor_j] = np.exp(-distance * distance / (sigma * sigma))
    
    return E_dist, S


def update_y(E_dist, Y_last, k, candidate_mask, r=None, w_cls=None, S=None, eps=1e-8):
    """
    Update labeling confidences and semantic dissimilarity.
    candidate_mask: fixed binary mask from original partial_target, non-candidate must stay 0.
    r, w_cls: when both provided, use weighted propagation (r=None and w_cls=None for original loop).
    """
    m = E_dist.shape[0]
    n_classes = Y_last.shape[0]

    E_1 = (E_dist > 0).astype(float)
    sort_indices = np.argsort(-E_1, axis=1)

    use_weighted = (r is not None and w_cls is not None)
    if not use_weighted:
        Y_new = np.zeros_like(Y_last)
        for i in range(m):
            kNN_indices = sort_indices[i, :k]
            neighbor_indices = np.concatenate([[i], kNN_indices])
            sort_result = E_1[i, kNN_indices]
            sort_result = np.concatenate([[1], sort_result[:k]])[:k+1]
            kNN_Y = Y_last[:, neighbor_indices]
            kNN_Y = kNN_Y * sort_result[np.newaxis, :]
            mask = candidate_mask[:, i].reshape(-1, 1)
            Y_new[:, i] = np.sum(kNN_Y * mask, axis=1)
            Y_new[:, i] = Y_new[:, i] / (np.sum(Y_new[:, i]) + eps)
        Y_new = candidate_mask * Y_new
    else:
        if S is None:
            a_ij = E_1.copy() + np.eye(m)
        else:
            a_ij = S.copy()
            np.fill_diagonal(a_ij, np.diag(a_ij) + 1.0)
        a_row_sum = np.sum(a_ij, axis=1, keepdims=True)
        a_row_sum[a_row_sum == 0] = 1
        a_ij = a_ij / a_row_sum
        q_hat = (a_ij @ (r[:, np.newaxis] * Y_last.T)).T * w_cls[:, np.newaxis]
        q_tilde = candidate_mask * q_hat
        q_sum = np.sum(q_tilde, axis=0, keepdims=True) + eps
        Y_new = q_tilde / q_sum

    D_new = 1 - (Y_new.T @ Y_new)
    np.fill_diagonal(D_new, 0)

    return Y_new, D_new


def solver(X, D, S, thr, miu):
    """Solve generalized eigenvalue problem for dimensionality reduction."""
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
    
    # 取最大特征值: 最大化 Rayleigh 商 v'XLX'v/(v'XAX'v), 对应语义差异最大、局部结构保持的投影方向
    idx = np.argsort(-eigvalues)
    eigvalues = eigvalues[idx]
    eigvectors = eigvectors[:, idx]
    
    for j in range(eigvectors.shape[1]):
        eigvectors[:, j] = eigvectors[:, j] / np.linalg.norm(eigvectors[:, j])
    
    proper_dim = get_proper_dim(eigvalues, thr)
    P = eigvectors[:, :proper_dim]
    X_proj = (P.T @ X_t).T
    
    return X_proj, P


def sdlpp(data, partial_target, para):
    """
    SDLPP main function.
    
    Parameters:
    -----------
    data : ndarray
        Training data (n_samples, n_features)
    partial_target : ndarray or sparse matrix
        Partial label matrix (n_classes, n_samples)
    para : dict
        Hyperparameters: T, target_d, k, miu, thr
        
    Returns:
    --------
    X_proj : ndarray
        Projected data (n_samples, target_d)
    P : ndarray
        Projection matrix (n_features, target_d)
    Y_history : dict
        History of Y and D matrices
    """
    X = data.copy()
    
    if hasattr(partial_target, 'toarray'):
        partial_target = partial_target.toarray()
    
    q = partial_target.shape[0]
    T = para['T']
    target_d = para['target_d']
    k = para['k']
    miu = para['miu']
    thr = para['thr']
    use_sample_reliability = para.get('use_sample_reliability', False)
    use_class_balance = para.get('use_class_balance', False)
    alpha = para.get('imbalance_alpha', 0.5)
    eps = para.get('imbalance_eps', 1e-8)

    candidate_mask = (partial_target > 0).astype(float)
    temp = np.sum(partial_target, axis=0, keepdims=True)
    temp[temp == 0] = 1
    Y = partial_target.astype(float) / temp
    D = 1 - (Y.T @ Y)
    np.fill_diagonal(D, 0)

    Y_history = {'Y': [Y.copy()], 'D': [D.copy()]}

    for iteration in range(T):
        d_old = X.shape[1]
        E_dist, S = construct_s(X, D, k)

        if use_sample_reliability or use_class_balance:
            n_classes, m = Y.shape[0], Y.shape[1]
            r = compute_sample_reliability(Y, candidate_mask, eps) if use_sample_reliability else np.ones(m)
            w_cls = compute_class_weights(Y, r, alpha, eps) if use_class_balance else np.ones(n_classes)
            Y, D = update_y(E_dist, Y, k, candidate_mask, r=r, w_cls=w_cls, S=S, eps=eps)
        else:
            Y, D = update_y(E_dist, Y, k, candidate_mask)

        X_lower, _ = solver(X, D, S, thr, miu)
        d_new = X_lower.shape[1]
        X = X_lower
        Y_history['Y'].append(Y.copy())
        Y_history['D'].append(D.copy())
        if d_new == d_old:
            break
    
    X_proj, P = solver(data, D, S, target_d, miu)
    
    return X_proj, P, Y_history
