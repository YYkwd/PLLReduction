"""Benchmark runner: multi-dataset x multi-method grid experiments.

Usage:
    python experiments/run_benchmark.py --fast --datasets lost MSRCv2
    python experiments/run_benchmark.py --models sdlpp delin cenda
    python experiments/run_benchmark.py --classifiers knn ipal
    python experiments/run_benchmark.py --fast   # all methods, quick mode
"""

import sys
import copy
import argparse
import time
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_single import (
    run, load_config, REDUCERS, CLASSIFIERS, MODEL_DEFAULTS,
)
from pll.eval.reporter import print_summary_table, save_summary_csv, save_summary_latex


# ---------------------------------------------------------------------------
# Experiment grid definition
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    'lost', 'MSRCv2', 'FG-NET', 'Mirflickr',
    'Soccer Player', 'Yahoo! News',
    'slashdotpl-f1', 'slashdotpl-f2', 'slashdotpl-f3',
]

ALL_MODELS = list(REDUCERS.keys())         # ['sdlpp', 'delin', 'cenda']
ALL_CLASSIFIERS = list(CLASSIFIERS.keys()) # ['knn', 'ipal']

# SDLPP-specific disambig variants (other models ignore these)
SDLPP_DISAMBIG_VARIANTS = {
    'baseline': {'use_sample_reliability': False, 'use_class_balance': False},
    'SR+CB': {
        'use_sample_reliability': True,
        'use_class_balance': True,
        'r_min': 0.1,
    },
}


def build_configs(datasets, models, classifiers, base_config):
    """Generate (dataset, method_tag, config) for all grid combinations.

    For SDLPP: also iterate over disambig variants (baseline / SR+CB).
    For DELIN/CENDA: disambig is built-in, only run once per classifier.
    """
    configs = []
    for ds_name in datasets:
        for model_name in models:
            for cls_name in classifiers:
                if model_name == 'sdlpp':
                    for var_name, var_params in SDLPP_DISAMBIG_VARIANTS.items():
                        cfg = copy.deepcopy(base_config)
                        cfg['data']['name'] = ds_name
                        cfg['model']['name'] = model_name
                        cfg['classifier']['name'] = cls_name
                        cfg['disambig']['params'].update(var_params)
                        cfg['output']['dir'] = str(
                            Path(base_config['output']['dir']) / 'benchmark')
                        tag = f"{model_name}({var_name})+{cls_name}"
                        configs.append((ds_name, tag, cfg))
                else:
                    cfg = copy.deepcopy(base_config)
                    cfg['data']['name'] = ds_name
                    cfg['model']['name'] = model_name
                    cfg['classifier']['name'] = cls_name
                    cfg['output']['dir'] = str(
                        Path(base_config['output']['dir']) / 'benchmark')
                    tag = f"{model_name}+{cls_name}"
                    configs.append((ds_name, tag, cfg))
    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(datasets, models, classifiers, base_config):
    configs = build_configs(datasets, models, classifiers, base_config)
    total = len(configs)

    all_results = []

    for idx, (ds_name, method_tag, cfg) in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{total}] {ds_name} / {method_tag}")
        print(f"{'='*70}")

        t0 = time.time()
        try:
            avg = run(cfg)
            elapsed = time.time() - t0
            all_results.append({
                'dataset': ds_name,
                'method': method_tag,
                'elapsed_s': round(elapsed, 1),
                **avg,
            })
        except Exception as e:
            print(f"FAILED: {e}")
            all_results.append({
                'dataset': ds_name,
                'method': method_tag,
                'elapsed_s': 0,
                'error': str(e),
            })

    # Summary
    print(f"\n\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}\n")
    print_summary_table(all_results)

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(base_config['output']['dir']) / 'benchmark' / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    save_summary_csv(all_results, out_dir / 'summary.csv')
    save_summary_latex(all_results, out_dir / 'summary.tex')
    print(f"\nSaved to {out_dir}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PLL Benchmark (grid experiments)')
    parser.add_argument('--config', type=str, default=None, help='Base YAML config')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help=f'Datasets (default: all). Choices: {ALL_DATASETS}')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help=f'Reducers (default: all). Choices: {ALL_MODELS}')
    parser.add_argument('--classifiers', type=str, nargs='+', default=None,
                        help=f'Classifiers (default: all). Choices: {ALL_CLASSIFIERS}')
    parser.add_argument('--fast', action='store_true',
                        help='Quick mode: T=10, cv_folds=3')
    args = parser.parse_args()

    base = load_config(args.config)
    datasets = args.datasets or ALL_DATASETS
    models = args.models or ALL_MODELS
    classifiers = args.classifiers or ALL_CLASSIFIERS

    if args.fast:
        base['eval']['cv_folds'] = 3
        for m in MODEL_DEFAULTS:
            if 'T' in MODEL_DEFAULTS[m]:
                MODEL_DEFAULTS[m]['T'] = 10

    run_benchmark(datasets, models, classifiers, base)
