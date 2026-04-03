"""Ablation: adaptive alpha for class balance vs baseline / fixed alpha.

Variants:
  baseline       – no SR, no CB
  sr_only        – SR only (r_min=0.1, warmup=dataset-specific)
  sr_cb_fixed    – SR + CB with fixed alpha=0.5
  sr_cb_adapt_*  – SR + CB with adaptive alpha (various cv0/cv1)

Datasets: lost, MSRCv2, slashdotpl-f1
Mode: --fast (T=10, cv_folds=3) by default for quick iteration.
"""

import sys
import copy
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_single import run, load_config, MODEL_DEFAULTS
from pll.eval.reporter import print_summary_table, save_summary_csv

DATASETS = ['lost', 'MSRCv2', 'slashdotpl-f1']

WARMUP_MAP = {'lost': 5, 'MSRCv2': 2, 'slashdotpl-f1': 0}

COMMON_SR = {
    'use_sample_reliability': True,
    'r_min': 0.1,
}

VARIANTS = {
    'baseline': {
        'use_sample_reliability': False,
        'use_class_balance': False,
        'cb_adaptive_alpha': False,
    },
    'sr_only': {
        **COMMON_SR,
        'use_class_balance': False,
        'cb_adaptive_alpha': False,
    },
    'sr+cb_fixed': {
        **COMMON_SR,
        'use_class_balance': True,
        'cb_adaptive_alpha': False,
        'alpha': 0.5,
    },
    'sr+cb_adapt(0.1,0.5)': {
        **COMMON_SR,
        'use_class_balance': True,
        'cb_adaptive_alpha': True,
        'cb_cv0': 0.1,
        'cb_cv1': 0.5,
        'alpha': 0.5,
    },
    'sr+cb_adapt(0.05,0.3)': {
        **COMMON_SR,
        'use_class_balance': True,
        'cb_adaptive_alpha': True,
        'cb_cv0': 0.05,
        'cb_cv1': 0.3,
        'alpha': 0.5,
    },
    'sr+cb_adapt(0.15,0.6)': {
        **COMMON_SR,
        'use_class_balance': True,
        'cb_adaptive_alpha': True,
        'cb_cv0': 0.15,
        'cb_cv1': 0.6,
        'alpha': 0.5,
    },
    'sr+cb_adapt(0.05,0.5)': {
        **COMMON_SR,
        'use_class_balance': True,
        'cb_adaptive_alpha': True,
        'cb_cv0': 0.05,
        'cb_cv1': 0.5,
        'alpha': 0.5,
    },
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Adaptive alpha ablation')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Override dataset list')
    parser.add_argument('--fast', action='store_true', default=True)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()

    base = load_config(args.config)
    if args.full:
        args.fast = False
    if args.fast:
        base['eval']['cv_folds'] = 3
        for m in MODEL_DEFAULTS:
            if 'T' in MODEL_DEFAULTS[m]:
                MODEL_DEFAULTS[m]['T'] = 10

    datasets = args.datasets or DATASETS
    configs = []
    for ds in datasets:
        warmup = WARMUP_MAP.get(ds, 0)
        for var_name, var_params in VARIANTS.items():
            cfg = copy.deepcopy(base)
            cfg['data']['name'] = ds
            cfg['model']['name'] = 'sdlpp'
            cfg['disambig']['params'].update(var_params)
            cfg['disambig']['params']['warmup'] = warmup
            cfg['disambig']['params']['cb_gate_enabled'] = False
            cfg['output']['dir'] = str(
                Path(base['output']['dir']) / 'benchmark' / 'adaptive_alpha_ablation')
            configs.append((ds, var_name, cfg))

    total = len(configs)
    all_results = []

    for idx, (ds, var_name, cfg) in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{total}] {ds} / {var_name}")
        print(f"{'='*70}")

        t0 = time.time()
        try:
            avg = run(cfg)
            elapsed = time.time() - t0
            all_results.append({
                'dataset': ds,
                'method': var_name,
                'elapsed_s': round(elapsed, 1),
                **avg,
            })
        except Exception as e:
            print(f"FAILED: {e}")
            all_results.append({
                'dataset': ds,
                'method': var_name,
                'elapsed_s': 0,
                'error': str(e),
            })

    print(f"\n\n{'='*70}")
    print("ADAPTIVE ALPHA ABLATION SUMMARY")
    print(f"{'='*70}\n")
    print_summary_table(all_results)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = (Path(base['output']['dir'])
               / 'benchmark' / 'adaptive_alpha_ablation' / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_summary_csv(all_results, out_dir / 'summary.csv')
    print(f"\nSaved to {out_dir}")

    return all_results


if __name__ == '__main__':
    main()
