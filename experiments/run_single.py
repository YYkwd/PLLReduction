"""Single experiment runner.

Pipeline: Load -> Split -> Preprocess -> Reduce(+Disambig) -> Classify -> Evaluate
"""

import sys
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pll.data.loader import load_dataset
from pll.data.preprocessor import ZScorePreprocessor
from pll.reducers.sdlpp import SDLPPReducer
from pll.reducers.delin import DELINReducer
from pll.reducers.cenda import CENDAReducer
from pll.disambig.knn_propagation import KNNPropagation
from pll.classifiers.knn import KNNClassifier
from pll.classifiers.ipal import IPALClassifier
from pll.eval.metrics import compute_metrics
from pll.eval.splits import get_many_medium_few_splits, create_cv_splitter
from pll.eval.reporter import print_metrics, save_results


# ---------------------------------------------------------------------------
# Registry: name -> class (extend here when adding new methods)
# ---------------------------------------------------------------------------
REDUCERS = {'sdlpp': SDLPPReducer, 'delin': DELINReducer, 'cenda': CENDAReducer}
DISAMBIGUATORS = {'knn_propagation': KNNPropagation}
CLASSIFIERS = {'knn': KNNClassifier, 'ipal': IPALClassifier}


def load_config(config_path=None):
    defaults = {
        'seed': 42,
        'data': {'name': 'lost', 'path': None, 'data_dir': 'datasets'},
        'preprocessing': {'method': 'zscore'},
        'model': {
            'name': 'sdlpp',
            'params': {'T': 100, 'target_d': 13, 'k': 8, 'miu': 0.1, 'thr': 0.95},
        },
        'disambig': {
            'name': 'knn_propagation',
            'params': {
                'use_sample_reliability': False,
                'use_class_balance': False,
                'r_min': 0.0,
                'warmup': 0,
                'warmup_by_dataset': {
                    'MSRCv2': 2,
                    'lost': 5,
                    'slashdotpl-f1': 0,
                },
                'sr_cb_policy_by_dataset': {
                    'lost': {
                        'use_sample_reliability': True,
                        'use_class_balance': True,
                        'r_min': 0.1,
                        'warmup': 5,
                        'cb_gate_enabled': False,
                    },
                    'MSRCv2': {
                        'use_sample_reliability': True,
                        'use_class_balance': True,
                        'r_min': 0.1,
                        'warmup': 2,
                        'cb_gate_enabled': False,
                    },
                    'slashdotpl-f1': {
                        'use_sample_reliability': False,
                        'use_class_balance': False,
                        'r_min': 0.1,
                        'warmup': 0,
                        'cb_gate_enabled': False,
                    },
                },
                'apply_sr_cb_policy': False,
                'cb_gate_enabled': False,
                'cb_gate_threshold': 0.15,
                'alpha': 0.5,
                'eps': 1e-8,
            },
        },
        'classifier': {'name': 'knn', 'params': {'n_neighbors': 5}},
        'eval': {'cv_folds': 5},
        'output': {'dir': 'results', 'formats': ['json']},
    }

    if config_path is not None:
        try:
            import yaml
            with open(config_path) as f:
                user_cfg = yaml.safe_load(f)
            if user_cfg:
                _deep_update(defaults, user_cfg)
        except ImportError:
            print("pyyaml not installed, using default config")

    return defaults


MODEL_DEFAULTS = {
    'sdlpp': {'T': 100, 'target_d': 13, 'k': 8, 'miu': 0.1, 'thr': 0.95},
    'delin': {'T': 75, 'ratio': 0.6, 'k': 8},
    'cenda': {'T': 50, 'mu': 0.5, 'dim_para': 0.999, 'k': 8},
}

CLASSIFIER_DEFAULTS = {
    'knn': {'n_neighbors': 5},
    'ipal': {'k': 10, 'alpha': 0.95},
}


def _deep_update(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _apply_defaults(config):
    """Fill in default params when model/classifier name changes."""
    model_name = config['model']['name']
    if model_name in MODEL_DEFAULTS:
        merged = dict(MODEL_DEFAULTS[model_name])
        merged.update(config['model'].get('params', {}))
        config['model']['params'] = merged

    cls_name = config['classifier']['name']
    if cls_name in CLASSIFIER_DEFAULTS:
        merged = dict(CLASSIFIER_DEFAULTS[cls_name])
        merged.update(config['classifier'].get('params', {}))
        config['classifier']['params'] = merged


def _apply_dataset_disambig_overrides(config, dataset_name):
    """Override disambiguation params with dataset-specific mapping."""
    dis_params = config['disambig'].setdefault('params', {})
    if not dis_params.get('apply_sr_cb_policy', False):
        return

    policy_map = dis_params.get('sr_cb_policy_by_dataset', {})
    if isinstance(policy_map, dict) and dataset_name in policy_map:
        policy = policy_map[dataset_name]
        if isinstance(policy, dict):
            allowed = {
                'use_sample_reliability',
                'use_class_balance',
                'r_min',
                'warmup',
                'cb_gate_enabled',
                'cb_gate_threshold',
                'alpha',
            }
            for k, v in policy.items():
                if k in allowed:
                    dis_params[k] = v
        return

    warmup_map = dis_params.get('warmup_by_dataset', {})

    if not isinstance(warmup_map, dict):
        return

    if dis_params.get('use_sample_reliability', False) and dataset_name in warmup_map:
        dis_params['warmup'] = int(warmup_map[dataset_name])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(config):
    _apply_defaults(config)
    np.random.seed(config['seed'])

    # 1. Load
    dcfg = config['data']
    dataset = load_dataset(dcfg['name'], dcfg.get('path'), dcfg.get('data_dir', 'datasets'))
    _apply_dataset_disambig_overrides(config, dataset.name)

    X = dataset.X
    partial_target = dataset.partial_target
    target = dataset.target

    if target is None:
        raise ValueError("No ground truth target in dataset")

    y = np.argmax(target, axis=0) if target.ndim > 1 else target.ravel()
    y = y.astype(int)
    n_classes = dataset.n_classes

    print(f"Dataset: {dataset.name} | "
          f"samples={dataset.n_samples} features={dataset.n_features} classes={n_classes}")

    # 2. CV splitter (auto-adapt folds to min class count)
    splitter = create_cv_splitter(
        n_splits=config['eval']['cv_folds'],
        random_state=config['seed'],
        y=y,
    )

    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pt_train = partial_target[:, train_idx]

        # 3. Preprocess
        prep = ZScorePreprocessor()
        X_train = prep.fit_transform(X_train)
        X_test = prep.transform(X_test)

        # 4. Disambiguator + Reducer
        dis_cfg = config['disambig']
        disambiguator = DISAMBIGUATORS[dis_cfg['name']](dis_cfg.get('params'))

        mod_cfg = config['model']
        reducer = REDUCERS[mod_cfg['name']](mod_cfg['params'])
        reducer.fit(X_train, pt_train, disambiguator)

        X_train_low = reducer.transform(X_train)
        X_test_low = reducer.transform(X_test)

        # 5. Classify
        cls_cfg = config['classifier']
        clf = CLASSIFIERS[cls_cfg['name']](cls_cfg.get('params'))
        clf.fit(X_train_low, y_train, partial_target=pt_train)
        y_pred = clf.predict(X_test_low)

        # 6. Evaluate
        many, medium, few = get_many_medium_few_splits(y_train, n_classes)
        m = compute_metrics(y_test, y_pred, n_classes, many, medium, few)
        fold_metrics.append(m)
        print_metrics(m, label=f"Fold {fold_idx + 1}")

    # 7. Average
    avg = {}
    for key in fold_metrics[0]:
        avg[key] = float(np.mean([fm[key] for fm in fold_metrics]))

    print("-" * 60)
    print_metrics(avg, label="Average")

    # 8. Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = config['model']['name']
    cls_name = config['classifier']['name']
    disambig_name = config['disambig']['name']
    dis_params = config['disambig'].get('params', {})
    if dis_params.get('use_sample_reliability') or dis_params.get('use_class_balance'):
        parts = []
        if dis_params.get('use_sample_reliability'):
            parts.append('SR')
        if dis_params.get('use_class_balance'):
            parts.append('CB')
        disambig_name += '_' + '+'.join(parts)

    run_tag = f"{model_name}_{cls_name}_{disambig_name}"
    output_dir = Path(config['output']['dir']) / dataset.name / run_tag / timestamp
    results = {
        'config': config,
        'avg_metrics': avg,
        'fold_metrics': fold_metrics,
    }
    save_results(results, output_dir, config['output'].get('formats', ['json']))
    print(f"Saved to {output_dir}")

    return avg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PLL Reduction Experiment')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default=None,
                        help=f'Reducer: {list(REDUCERS.keys())}')
    parser.add_argument('--classifier', type=str, default=None,
                        help=f'Classifier: {list(CLASSIFIERS.keys())}')
    parser.add_argument('--target-d', type=int, default=None)
    parser.add_argument('--sample-reliability', action='store_true')
    parser.add_argument('--class-balance', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.dataset:
        cfg['data']['name'] = args.dataset
    if args.model:
        cfg['model']['name'] = args.model
    if args.classifier:
        cfg['classifier']['name'] = args.classifier
    if args.target_d:
        cfg['model']['params']['target_d'] = args.target_d
    if args.sample_reliability:
        cfg['disambig']['params']['use_sample_reliability'] = True
    if args.class_balance:
        cfg['disambig']['params']['use_class_balance'] = True

    run(cfg)
