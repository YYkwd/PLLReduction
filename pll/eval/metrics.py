"""Evaluation metrics for PLL classification.

Key differences from the original implementation:
- balanced_acc averages recall only over classes *present* in the test set
  (aligned with MATLAB demo.m behaviour).
- many/medium/few group accuracies return NaN when the group has no samples
  in the test set (instead of 0).
- Records present_test_classes and missing_test_classes counts.
"""

import numpy as np
from .splitter import get_many_medium_few_splits


def compute_split_metrics(y_true, y_pred, y_train, n_classes):
    """Compute metrics for a single train/test split.

    Returns
    -------
    dict with keys:
        overall_acc, balanced_acc,
        present_test_classes, missing_test_classes,
        many_acc (float or NaN), medium_acc, few_acc
    """
    overall_acc = float(np.mean(y_true == y_pred))

    present = np.unique(y_true)
    recalls = []
    for c in present:
        mask = y_true == c
        recalls.append(float(np.mean(y_pred[mask] == c)))
    balanced_acc = float(np.mean(recalls)) if recalls else 0.0

    many, medium, few = get_many_medium_few_splits(y_train, n_classes)

    def _group_acc(group_classes):
        active = group_classes & set(present.tolist())
        if not active:
            return float('nan')
        group_recalls = []
        for c in active:
            mask = y_true == c
            group_recalls.append(float(np.mean(y_pred[mask] == c)))
        return float(np.mean(group_recalls))

    return {
        'overall_acc': overall_acc,
        'balanced_acc': balanced_acc,
        'present_test_classes': int(len(present)),
        'missing_test_classes': int(n_classes - len(present)),
        'many_acc': _group_acc(many),
        'medium_acc': _group_acc(medium),
        'few_acc': _group_acc(few),
    }


def aggregate_split_metrics(split_metrics, dataset_warning=None):
    """Aggregate repeated-split metrics into mean +/- std summary.

    Parameters
    ----------
    split_metrics : list[dict]
        Each dict is the output of ``compute_split_metrics``.
    dataset_warning : str or None
        Optional warning string for sparse datasets.

    Returns
    -------
    dict  – summary with *_mean, *_std, *_valid_runs keys.
    """
    n = len(split_metrics)
    summary = {'n_repeats': n}

    for key in ('overall_acc', 'balanced_acc'):
        vals = [m[key] for m in split_metrics]
        summary[f'{key}_mean'] = float(np.mean(vals))
        summary[f'{key}_std'] = float(np.std(vals, ddof=1)) if n > 1 else 0.0

    for key in ('many_acc', 'medium_acc', 'few_acc'):
        vals = [m[key] for m in split_metrics if not np.isnan(m[key])]
        valid = len(vals)
        if valid > 0:
            summary[f'{key}_mean'] = float(np.mean(vals))
            summary[f'{key}_std'] = float(np.std(vals, ddof=1)) if valid > 1 else 0.0
        else:
            summary[f'{key}_mean'] = float('nan')
            summary[f'{key}_std'] = float('nan')
        summary[f'{key}_valid_runs'] = valid

    summary['dataset_warning'] = dataset_warning
    return summary
