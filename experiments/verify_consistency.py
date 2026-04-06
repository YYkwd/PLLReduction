"""Consistency verification: run SDLPP on small datasets and compare results.

Usage:
    # Save baseline reference (run once before any optimization)
    python experiments/verify_consistency.py --save-baseline

    # Verify current code against saved baseline
    python experiments/verify_consistency.py --verify

    # Verify with a custom tag
    python experiments/verify_consistency.py --verify --tag "after_p5"
"""

import copy
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_single import run, load_config, MODEL_DEFAULTS

VERIFY_DIR = Path(__file__).resolve().parent.parent / 'results' / 'consistency_baseline'

DATASETS = ['lost', 'MSRCv2']
SEED = 42
CV_FOLDS = 3
T_FAST = 10

VARIANTS = {
    'baseline': {
        'use_sample_reliability': False,
        'use_class_balance': False,
    },
    'sr+cb': {
        'use_sample_reliability': True,
        'use_class_balance': True,
        'r_min': 0.1,
        'alpha': 0.5,
        'warmup': 5,
    },
}

METRIC_KEYS = ['balanced_acc', 'overall_acc']
TOLERANCE = 1e-4


def _build_config(dataset, variant_params):
    base = load_config(None)
    base['seed'] = SEED
    base['eval']['cv_folds'] = CV_FOLDS
    base['data']['name'] = dataset
    base['model']['name'] = 'sdlpp'
    MODEL_DEFAULTS['sdlpp']['T'] = T_FAST
    base['disambig']['params'].update(variant_params)
    return base


def run_all():
    results = {}
    for ds in DATASETS:
        results[ds] = {}
        for var_name, var_params in VARIANTS.items():
            cfg = _build_config(ds, var_params)
            print(f"\n{'='*60}")
            print(f"Running: {ds} / {var_name}")
            print(f"{'='*60}")
            avg = run(cfg)
            results[ds][var_name] = {k: float(avg[k]) for k in METRIC_KEYS}
            print(f"  -> {results[ds][var_name]}")
    return results


def save_baseline():
    results = run_all()
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    out = VERIFY_DIR / 'baseline.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline saved to {out}")
    return results


def verify(tag=""):
    baseline_path = VERIFY_DIR / 'baseline.json'
    if not baseline_path.exists():
        print(f"ERROR: baseline not found at {baseline_path}")
        print("Run with --save-baseline first.")
        sys.exit(1)

    with open(baseline_path) as f:
        baseline = json.load(f)

    current = run_all()

    all_pass = True
    print(f"\n{'='*60}")
    print(f"CONSISTENCY CHECK{' (' + tag + ')' if tag else ''}")
    print(f"{'='*60}")
    for ds in DATASETS:
        for var_name in VARIANTS:
            b = baseline[ds][var_name]
            c = current[ds][var_name]
            for metric in METRIC_KEYS:
                diff = abs(b[metric] - c[metric])
                status = "PASS" if diff < TOLERANCE else "FAIL"
                if status == "FAIL":
                    all_pass = False
                print(f"  {ds:12s} / {var_name:10s} / {metric:15s}: "
                      f"baseline={b[metric]:.6f}  current={c[metric]:.6f}  "
                      f"diff={diff:.2e}  [{status}]")

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — investigate before proceeding")
    print(f"{'='*60}")

    if tag:
        log_path = VERIFY_DIR / f'verify_{tag}.json'
        with open(log_path, 'w') as f:
            json.dump({'tag': tag, 'results': current, 'all_pass': all_pass}, f, indent=2)
        print(f"Log saved to {log_path}")

    return all_pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-baseline', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    if args.save_baseline:
        save_baseline()
    elif args.verify:
        ok = verify(args.tag)
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()
