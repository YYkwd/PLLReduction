"""Evaluator: orchestrate repeated-holdout evaluation pipeline.

Pipeline per split:
    preprocess -> reduce(+disambig) -> classify -> metrics
Then aggregate across splits.
"""

import inspect
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from pll.data.preprocessor import ZScorePreprocessor
from .splitter import RepeatedHoldout
from .metrics import compute_split_metrics, aggregate_split_metrics

log = logging.getLogger(__name__)

REDUCERS = {}
DISAMBIGUATORS = {}
CLASSIFIERS = {}


def _register_defaults():
    """Lazy import to avoid circular deps and keep registry in one place."""
    if REDUCERS:
        return
    from pll.reducers.sdlpp import SDLPPReducer
    from pll.reducers.delin import DELINReducer
    from pll.reducers.cenda import CENDAReducer
    from pll.disambig.knn_propagation import KNNPropagation
    from pll.classifiers.knn import KNNClassifier
    from pll.classifiers.ipal import IPALClassifier

    REDUCERS.update({'sdlpp': SDLPPReducer, 'delin': DELINReducer, 'cenda': CENDAReducer})
    DISAMBIGUATORS.update({'knn_propagation': KNNPropagation})
    CLASSIFIERS.update({'knn': KNNClassifier, 'ipal': IPALClassifier})


@dataclass
class EvalResult:
    """Container for evaluation output."""
    split_metrics: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    dataset_info: dict = field(default_factory=dict)


class Evaluator:
    """Orchestrate: repeated split -> preprocess -> reduce -> classify -> metrics -> aggregate."""

    def __init__(self, config: dict):
        _register_defaults()
        self.config = config

    def run(self, dataset) -> EvalResult:
        """Run full evaluation. Returns EvalResult with per-split and summary."""
        cfg = self.config
        eval_cfg = cfg.get('eval', {})

        X = dataset.X
        partial_target = dataset.partial_target
        target = dataset.target

        if target is None:
            raise ValueError("No ground truth target in dataset")

        y = np.argmax(target, axis=0) if target.ndim > 1 else target.ravel()
        y = y.astype(int)
        n_classes = dataset.n_classes

        splitter = RepeatedHoldout(
            n_repeats=eval_cfg.get('n_repeats', 10),
            test_ratio=eval_cfg.get('test_ratio', 0.2),
            stratified=eval_cfg.get('stratified', True),
            random_state=cfg.get('seed', 42),
        )

        dataset_warning = self._detect_sparse_warning(y, n_classes)
        note = cfg.get('dataset_note')
        if note:
            dataset_warning = note

        split_metrics = []
        for i, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            log.info("Split %d/%d  train=%d  test=%d",
                     i + 1, splitter.n_repeats, len(train_idx), len(test_idx))
            m = self._run_single_split(
                X, y, partial_target, train_idx, test_idx, n_classes, cfg,
            )
            m['split_idx'] = i
            split_metrics.append(m)
            log.info("  overall=%.4f  balanced=%.4f", m['overall_acc'], m['balanced_acc'])

        summary = aggregate_split_metrics(split_metrics, dataset_warning)
        summary['dataset'] = dataset.name
        summary['method'] = cfg.get('method_tag', cfg.get('model', {}).get('name', ''))
        summary['classifier'] = cfg.get('classifier', {}).get('name', '')

        return EvalResult(
            split_metrics=split_metrics,
            summary=summary,
            config=cfg,
            dataset_info={
                'name': dataset.name,
                'n_samples': dataset.n_samples,
                'n_features': dataset.n_features,
                'n_classes': n_classes,
            },
        )

    def _run_single_split(self, X, y, partial_target, train_idx, test_idx,
                          n_classes, cfg):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pt_train = partial_target[:, train_idx]

        prep = ZScorePreprocessor()
        X_train = prep.fit_transform(X_train)
        X_test = prep.transform(X_test)

        dis_cfg = cfg.get('disambig', {})
        disambiguator = None
        if dis_cfg.get('name'):
            dis_cls = DISAMBIGUATORS.get(dis_cfg['name'])
            if dis_cls:
                disambiguator = dis_cls(dis_cfg.get('params'))

        mod_cfg = cfg.get('model', {})
        reducer_cls = REDUCERS[mod_cfg['name']]
        reducer = reducer_cls(mod_cfg.get('params', {}))
        reducer.fit(X_train, pt_train, disambiguator)

        X_train_low = reducer.transform(X_train)
        X_test_low = reducer.transform(X_test)

        cls_cfg = cfg.get('classifier', {})
        clf_cls = CLASSIFIERS[cls_cfg['name']]
        clf = clf_cls(cls_cfg.get('params'))
        self._classify_fit(clf, X_train_low, y_train, pt_train)
        y_pred = clf.predict(X_test_low)

        return compute_split_metrics(y_test, y_pred, y_train, n_classes)

    @staticmethod
    def _classify_fit(clf, X_train, y_train, partial_target_train):
        """Call clf.fit with appropriate arguments depending on its signature."""
        sig = inspect.signature(clf.fit)
        if 'partial_target' in sig.parameters:
            clf.fit(X_train, y_train, partial_target=partial_target_train)
        else:
            clf.fit(X_train, y_train)

    @staticmethod
    def _detect_sparse_warning(y, n_classes):
        counts = np.bincount(y, minlength=n_classes)
        sparse_count = int(np.sum(counts < 5))
        if sparse_count > 0:
            return (
                f"Dataset has {sparse_count} classes with <5 samples; "
                "results are supplementary only, not recommended as primary benchmark."
            )
        return None
