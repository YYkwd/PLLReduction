"""Benchmark runner: multi-dataset x multi-config grid experiments.

Usage:
    python experiments/run_benchmark.py
    python experiments/run_benchmark.py --datasets lost slashdotpl-f1
    python experiments/run_benchmark.py --fast   # T=10 for quick sanity check
"""

import sys
import copy
import argparse
import time
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_single import run, load_config, REDUCERS, DISAMBIGUATORS, CLASSIFIERS
from pll.eval.reporter import print_summary_table, save_summary_csv, save_summary_latex


# ---------------------------------------------------------------------------
# Experiment grid definition
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    'lost', 'MSRCv2', 'FG-NET', 'Mirflickr',
    'Soccer Player', 'Yahoo! News',
    'slashdotpl-f1', 'slashdotpl-f2', 'slashdotpl-f3',
]

DISAMBIG_CONFIGS = {
    'baseline': {
        'use_sample_reliability': False,
        'use_class_balance': False,
    },
    'SR_only': {
        'use_sample_reliability': True,
        'use_class_balance': False,
    },
    'CB_only': {
        'use_sample_reliability': False,
        'use_class_balance': True,
    },
    'SR+CB': {
        'use_sample_reliability': True,
        'use_class_balance': True,
    },
}


def build_configs(datasets, disambig_configs, base_config, override_model_params=None):
    """Generate all (dataset, disambig_name, config) combinations."""
    configs = []
    for ds_name in datasets:
        for dis_name, dis_params in disambig_configs.items():
            cfg = copy.deepcopy(base_config)
            cfg['data']['name'] = ds_name
            cfg['disambig']['params'].update(dis_params)
            if override_model_params:
                cfg['model']['params'].update(override_model_params)
            cfg['output']['dir'] = str(Path(base_config['output']['dir']) / 'benchmark')
            configs.append((ds_name, dis_name, cfg))
    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(datasets, disambig_configs, base_config, override_model_params=None):
    configs = build_configs(datasets, disambig_configs, base_config, override_model_params)
    total = len(configs)

    all_results = []

    for idx, (ds_name, dis_name, cfg) in enumerate(configs, 1):
        tag = f"{ds_name} / {dis_name}"
        print(f"\n{'='*70}")
        print(f"[{idx}/{total}] {tag}")
        print(f"{'='*70}")

        t0 = time.time()
        try:
            avg = run(cfg)
            elapsed = time.time() - t0
            all_results.append({
                'dataset': ds_name,
                'disambig': dis_name,
                'elapsed_s': round(elapsed, 1),
                **avg,
            })
        except Exception as e:
            print(f"FAILED: {e}")
            all_results.append({
                'dataset': ds_name,
                'disambig': dis_name,
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
                        help=f'Datasets to run (default: all). Choices: {ALL_DATASETS}')
    parser.add_argument('--fast', action='store_true',
                        help='Quick mode: T=10, cv_folds=3')
    args = parser.parse_args()

    base = load_config(args.config)
    datasets = args.datasets or ALL_DATASETS

    override = None
    if args.fast:
        override = {'T': 10}
        base['eval']['cv_folds'] = 3

    run_benchmark(datasets, DISAMBIG_CONFIGS, base, override_model_params=override)
