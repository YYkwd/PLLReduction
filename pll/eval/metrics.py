"""Evaluation metrics for PLL classification."""

import numpy as np


def compute_metrics(y_true, y_pred, n_classes,
                    many_classes=None, medium_classes=None, few_classes=None):
    """Compute Overall, Balanced, Many/Medium/Few accuracy.

    Parameters
    ----------
    y_true, y_pred : (n_samples,) integer class labels
    n_classes : int
    many_classes, medium_classes, few_classes : set of class indices or None
    """
    overall = float(np.mean(y_true == y_pred))

    per_class = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            per_class.append(float(np.mean(y_pred[mask] == y_true[mask])))
        else:
            per_class.append(0.0)
    balanced = float(np.mean(per_class))

    result = {'overall_acc': overall, 'balanced_acc': balanced}

    def _acc_for_set(classes):
        if not classes:
            return 0.0
        mask = np.isin(y_true, list(classes))
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(y_pred[mask] == y_true[mask]))

    if many_classes is not None:
        result['many_acc'] = _acc_for_set(many_classes)
    if medium_classes is not None:
        result['medium_acc'] = _acc_for_set(medium_classes)
    if few_classes is not None:
        result['few_acc'] = _acc_for_set(few_classes)

    return result
