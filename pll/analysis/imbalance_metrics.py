"""Class imbalance and candidate label metrics for PLL datasets.

All functions accept raw numpy arrays; no Dataset object dependency.
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Ground-truth class distribution
# ---------------------------------------------------------------------------

def class_distribution(target: np.ndarray, n_classes: int) -> dict:
    """Analyse true-label class distribution.

    Parameters
    ----------
    target : (n_classes, n_samples) one-hot or (n_samples,) int labels
    n_classes : total number of classes

    Returns dict with per-class counts and derived statistics.
    """
    if target.ndim == 2:
        y = np.argmax(target, axis=0)
    else:
        y = target.ravel().astype(int)

    counts = np.bincount(y, minlength=n_classes).astype(int)
    sorted_idx = np.argsort(-counts)

    n_max = int(counts.max())
    n_min = int(counts[counts > 0].min()) if np.any(counts > 0) else 0
    imbalance_ratio = n_max / n_min if n_min > 0 else float('inf')

    # Many / Medium / Few split (same rule as pll.eval.splits)
    n = len(sorted_idx)
    m_cut = (n + 2) // 3
    f_cut = (2 * n + 2) // 3
    many_idx = sorted_idx[:m_cut].tolist()
    medium_idx = sorted_idx[m_cut:f_cut].tolist()
    few_idx = sorted_idx[f_cut:].tolist()

    per_class = []
    for c in sorted_idx:
        group = ('many' if c in many_idx else
                 'medium' if c in medium_idx else 'few')
        per_class.append({
            'class': int(c),
            'count': int(counts[c]),
            'group': group,
        })

    return {
        'n_classes': n_classes,
        'counts': counts.tolist(),
        'n_max': n_max,
        'n_min': n_min,
        'imbalance_ratio': round(imbalance_ratio, 4),
        'mean_count': round(float(counts.mean()), 2),
        'std_count': round(float(counts.std()), 2),
        'cv': round(float(counts.std() / (counts.mean() + 1e-8)), 4),
        'many': {'classes': many_idx, 'total_samples': int(counts[many_idx].sum())},
        'medium': {'classes': medium_idx, 'total_samples': int(counts[medium_idx].sum())},
        'few': {'classes': few_idx, 'total_samples': int(counts[few_idx].sum())},
        'per_class': per_class,
    }


# ---------------------------------------------------------------------------
# Candidate (partial) label statistics
# ---------------------------------------------------------------------------

def candidate_label_stats(partial_target: np.ndarray,
                          target: Optional[np.ndarray] = None,
                          n_classes: int = 0) -> dict:
    """Statistics about candidate label sets.

    Parameters
    ----------
    partial_target : (n_classes, n_samples) binary matrix
    target : (n_classes, n_samples) one-hot, optional (for noise analysis)
    n_classes : fallback if partial_target shape is ambiguous
    """
    pt = (partial_target > 0).astype(int)
    n_cls, n_samples = pt.shape

    cands_per_sample = pt.sum(axis=0)  # how many candidates each sample has
    cls_as_candidate = pt.sum(axis=1)  # how often each class appears as candidate

    result = {
        'avg_candidates': round(float(cands_per_sample.mean()), 4),
        'min_candidates': int(cands_per_sample.min()),
        'max_candidates': int(cands_per_sample.max()),
        'median_candidates': float(np.median(cands_per_sample)),
        'std_candidates': round(float(cands_per_sample.std()), 4),
        'candidate_freq_per_class': cls_as_candidate.tolist(),
        'candidate_freq_cv': round(
            float(cls_as_candidate.std() / (cls_as_candidate.mean() + 1e-8)), 4),
    }

    if target is not None:
        if target.ndim == 2:
            y = np.argmax(target, axis=0)
        else:
            y = target.ravel().astype(int)
        # Number of *false* candidate labels per sample
        true_in_pt = np.array([pt[y[i], i] for i in range(n_samples)])
        noise_per_sample = cands_per_sample - true_in_pt
        result['avg_noise_candidates'] = round(float(noise_per_sample.mean()), 4)
        result['true_label_covered'] = float(true_in_pt.mean())

    return result


# ---------------------------------------------------------------------------
# Combined report for a single dataset
# ---------------------------------------------------------------------------

def compute_imbalance_report(
    X: np.ndarray,
    partial_target: np.ndarray,
    target: Optional[np.ndarray],
    name: str,
    n_classes: int,
) -> dict:
    """One-stop report combining class distribution and candidate stats."""
    report = {
        'dataset': name,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': n_classes,
        'target_format': f'one-hot ({target.shape})' if target is not None else 'missing',
        'partial_target_format': f'binary ({partial_target.shape})',
    }

    if target is not None:
        report['class_distribution'] = class_distribution(target, n_classes)
    else:
        report['class_distribution'] = None

    report['candidate_stats'] = candidate_label_stats(
        partial_target, target, n_classes)

    return report
