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


def update_y(E_dist, Y_last, k):
    """Update labeling confidences and semantic dissimilarity."""
    m = E_dist.shape[0]
    q = Y_last.shape[0]
    
    E_1 = (E_dist > 0).astype(float)
    sort_indices = np.argsort(-E_1, axis=1)
    
    Y_new = np.zeros_like(Y_last)
    
    for i in range(m):
        kNN_indices = sort_indices[i, :k]
        neighbor_indices = np.concatenate([[i], kNN_indices])
        sort_result = E_1[i, kNN_indices]
        sort_result = np.concatenate([[1], sort_result[:k]])[:k+1]
        
        kNN_Y = Y_last[:, neighbor_indices]
        kNN_Y = kNN_Y * sort_result[np.newaxis, :]
        
        mask = (Y_last[:, i] > 0).reshape(-1, 1)
        Y_new[:, i] = np.sum(kNN_Y * mask, axis=1)
        Y_new[:, i] = Y_new[:, i] / np.sum(Y_new[:, i])
    
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
    
    temp = np.sum(partial_target, axis=0, keepdims=True)
    Y = partial_target.astype(float) / temp
    D = 1 - (Y.T @ Y)
    np.fill_diagonal(D, 0)
    
    Y_history = {'Y': [Y.copy()], 'D': [D.copy()]}
    
    for iteration in range(T):
        d_old = X.shape[1]
        E_dist, S = construct_s(X, D, k)
        Y, D = update_y(E_dist, Y, k)
        X_lower, _ = solver(X, D, S, thr, miu)
        d_new = X_lower.shape[1]
        X = X_lower
        Y_history['Y'].append(Y.copy())
        Y_history['D'].append(D.copy())
        if d_new == d_old:
            break
    
    X_proj, P = solver(data, D, S, target_d, miu)
    
    return X_proj, P, Y_history
