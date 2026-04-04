"""4-way ablation: SR x CB factorial design on all datasets.

Variants (2x2 factorial):
  baseline  – no SR, no CB
  sr_only   – SR only  (r_min=0.1, warmup=per-dataset)
  cb_only   – CB only  (alpha=0.5)
  sr+cb     – SR + CB  (r_min=0.1, warmup=per-dataset, alpha=0.5)

Usage:
    python experiments/run_ablation_sr_cb.py                       # fast, all 9 datasets
    python experiments/run_ablation_sr_cb.py --datasets lost MSRCv2
    python experiments/run_ablation_sr_cb.py --seeds 42 43 44      # multi-seed
    python experiments/run_ablation_sr_cb.py --full                 # T=100, cv_folds=5
"""

import sys
import copy
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_single import run, load_config, MODEL_DEFAULTS
from pll.eval.reporter import print_summary_table, save_summary_csv

ALL_DATASETS = [
    'lost', 'MSRCv2', 'FG-NET', 'Mirflickr',
    'Soccer Player', 'Yahoo! News',
    'slashdotpl-f1', 'slashdotpl-f2', 'slashdotpl-f3',
]

WARMUP_MAP = {
    'lost': 5, 'MSRCv2': 2, 'Soccer Player': 2, 'Yahoo! News': 3,
    'FG-NET': 0, 'Mirflickr': 0,
    'slashdotpl-f1': 0, 'slashdotpl-f2': 0, 'slashdotpl-f3': 0,
}

VARIANTS = {
    'baseline': {
        'use_sample_reliability': False,
        'use_class_balance': False,
    },
    'sr_only': {
        'use_sample_reliability': True,
        'use_class_balance': False,
        'r_min': 0.1,
    },
    'cb_only': {
        'use_sample_reliability': False,
        'use_class_balance': True,
        'alpha': 0.5,
    },
    'sr+cb': {
        'use_sample_reliability': True,
        'use_class_balance': True,
        'r_min': 0.1,
        'alpha': 0.5,
    },
}

OUTPUT_TAG = 'sr_cb_ablation'


def build_configs(datasets, seeds, base_config):
    """Generate (dataset, variant, seed, config) tuples."""
    configs = []
    for ds in datasets:
        warmup = WARMUP_MAP.get(ds, 0)
        for var_name, var_params in VARIANTS.items():
            for seed in seeds:
                cfg = copy.deepcopy(base_config)
                cfg['seed'] = seed
                cfg['data']['name'] = ds
                cfg['model']['name'] = 'sdlpp'
                cfg['disambig']['params'].update(var_params)
                if var_params.get('use_sample_reliability', False):
                    cfg['disambig']['params']['warmup'] = warmup
                else:
                    cfg['disambig']['params']['warmup'] = 0
                cfg['output']['dir'] = str(
                    Path(base_config['output']['dir']) / 'benchmark' / OUTPUT_TAG)
                configs.append((ds, var_name, seed, cfg))
    return configs


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SR x CB 4-way ablation')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help=f'Override dataset list (default: all 9)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='Random seeds (default: 42)')
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

    datasets = args.datasets or ALL_DATASETS
    configs = build_configs(datasets, args.seeds, base)
    total = len(configs)
    multi_seed = len(args.seeds) > 1
    all_results = []

    for idx, (ds, var_name, seed, cfg) in enumerate(configs, 1):
        label = f"{ds} / {var_name}"
        if multi_seed:
            label += f" / seed={seed}"
        print(f"\n{'='*70}")
        print(f"[{idx}/{total}] {label}")
        print(f"{'='*70}")

        t0 = time.time()
        try:
            avg = run(cfg)
            elapsed = time.time() - t0
            row = {
                'dataset': ds,
                'method': var_name,
                'elapsed_s': round(elapsed, 1),
                **avg,
            }
            if multi_seed:
                row['seed'] = seed
            all_results.append(row)
        except Exception as e:
            print(f"FAILED: {e}")
            row = {
                'dataset': ds,
                'method': var_name,
                'elapsed_s': 0,
                'error': str(e),
            }
            if multi_seed:
                row['seed'] = seed
            all_results.append(row)

    print(f"\n\n{'='*70}")
    print("SR x CB ABLATION SUMMARY")
    print(f"{'='*70}\n")
    print_summary_table(all_results)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(base['output']['dir']) / 'benchmark' / OUTPUT_TAG / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    save_summary_csv(all_results, out_dir / 'summary.csv')
    print(f"\nSaved to {out_dir}")

    return all_results


if __name__ == '__main__':
    main()
