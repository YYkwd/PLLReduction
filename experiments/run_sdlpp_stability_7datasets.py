"""SDLPP baseline stability test on 7 mid-size datasets.

Runs pure SDLPP baseline (no SR, no CB) with default hyper-parameters
on every available mid-size dataset and prints a summary table.

Usage:
    python experiments/run_sdlpp_stability_7datasets.py
    python experiments/run_sdlpp_stability_7datasets.py --fast          # T=10, 3-fold
    python experiments/run_sdlpp_stability_7datasets.py --datasets lost MSRCv2 Mirflickr
"""

import copy
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_single import (
    run,
    load_config,
    apply_fast_training_overrides,
)
from pll.eval.reporter import print_summary_table, save_summary_csv

ALL_DATASETS = [
    'lost',
    'MSRCv2',
    'FG-NET',
    'Mirflickr',
    'slashdotpl-f1',
    'slashdotpl-f2',
    'slashdotpl-f3',
]


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='SDLPP baseline stability test on 7 mid-size datasets')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help=f'Override dataset list (default: {ALL_DATASETS})')
    parser.add_argument('--fast', action='store_true',
                        help='Quick mode: T=10, cv_folds=3')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    datasets = args.datasets or ALL_DATASETS
    base = load_config(None)

    base['seed'] = args.seed
    base['model']['name'] = 'sdlpp'
    base['disambig']['params']['use_sample_reliability'] = False
    base['disambig']['params']['use_class_balance'] = False
    base['disambig']['apply_sr_cb_policy'] = False

    if args.fast:
        apply_fast_training_overrides(base)

    mode = 'fast' if args.fast else 'full'
    print(f"{'='*70}")
    print(f"SDLPP BASELINE STABILITY TEST  |  mode={mode}  |  seed={args.seed}")
    print(f"Datasets: {datasets}")
    print(f"{'='*70}\n")

    all_results = []
    session_t0 = time.time()

    for idx, ds_name in enumerate(datasets, 1):
        cfg = copy.deepcopy(base)
        cfg['data']['name'] = ds_name
        cfg['_run_meta'] = {
            'run_id': f"stab_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx:02d}",
            'script_name': 'run_sdlpp_stability_7datasets',
            'mode': mode,
        }

        print(f"\n{'='*70}")
        print(f"[{idx}/{len(datasets)}]  {ds_name}")
        print(f"{'='*70}")

        t0 = time.time()
        try:
            avg = run(cfg)
            elapsed = time.time() - t0
            row = {
                'dataset': ds_name,
                'method': 'sdlpp_baseline',
                'elapsed_s': round(elapsed, 1),
                **avg,
            }
        except FileNotFoundError:
            print(f"  [SKIP] {ds_name}: .mat file not found")
            row = {
                'dataset': ds_name,
                'method': 'sdlpp_baseline',
                'elapsed_s': 0,
                'error': 'file_not_found',
            }
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [FAIL] {ds_name}: {e}")
            row = {
                'dataset': ds_name,
                'method': 'sdlpp_baseline',
                'elapsed_s': round(elapsed, 1),
                'error': str(e),
            }
        all_results.append(row)

    wall = time.time() - session_t0

    print(f"\n\n{'='*70}")
    print("SDLPP BASELINE STABILITY SUMMARY")
    print(f"{'='*70}\n")
    print_summary_table(all_results)

    n_ok = sum(1 for r in all_results if 'error' not in r)
    n_fail = len(all_results) - n_ok
    print(f"\nTotal: {n_ok} ok, {n_fail} fail/skip  |  wall time: {wall:.1f}s ({wall/60:.1f}min)")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(base['output']['dir']) / 'stability_test' / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    save_summary_csv(all_results, out_dir / 'summary.csv')
    print(f"Saved CSV to {out_dir / 'summary.csv'}")

    return all_results


if __name__ == '__main__':
    main()
