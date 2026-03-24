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
from pll.disambig.knn_propagation import KNNPropagation
from pll.classifiers.knn import KNNClassifier
from pll.eval.metrics import compute_metrics
from pll.eval.splits import get_many_medium_few_splits, create_cv_splitter
from pll.eval.reporter import print_metrics, save_results


# ---------------------------------------------------------------------------
# Registry: name -> class (extend here when adding new methods)
# ---------------------------------------------------------------------------
REDUCERS = {'sdlpp': SDLPPReducer}
DISAMBIGUATORS = {'knn_propagation': KNNPropagation}
CLASSIFIERS = {'knn': KNNClassifier}


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


def _deep_update(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(config):
    np.random.seed(config['seed'])

    # 1. Load
    dcfg = config['data']
    dataset = load_dataset(dcfg['name'], dcfg.get('path'), dcfg.get('data_dir', 'datasets'))

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

    # 2. CV splitter
    splitter = create_cv_splitter(
        n_splits=config['eval']['cv_folds'],
        random_state=config['seed'],
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
        clf.fit(X_train_low, y_train)
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
    disambig_name = config['disambig']['name']
    dis_params = config['disambig'].get('params', {})
    if dis_params.get('use_sample_reliability') or dis_params.get('use_class_balance'):
        parts = []
        if dis_params.get('use_sample_reliability'):
            parts.append('SR')
        if dis_params.get('use_class_balance'):
            parts.append('CB')
        disambig_name += '_' + '+'.join(parts)

    output_dir = Path(config['output']['dir']) / dataset.name / disambig_name / timestamp
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
    parser.add_argument('--target-d', type=int, default=None)
    parser.add_argument('--sample-reliability', action='store_true')
    parser.add_argument('--class-balance', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.dataset:
        cfg['data']['name'] = args.dataset
    if args.target_d:
        cfg['model']['params']['target_d'] = args.target_d
    if args.sample_reliability:
        cfg['disambig']['params']['use_sample_reliability'] = True
    if args.class_balance:
        cfg['disambig']['params']['use_class_balance'] = True

    run(cfg)
