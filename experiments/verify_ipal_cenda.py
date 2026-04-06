"""Consistency verification for IPAL and CENDA optimizations.

Usage:
    python experiments/verify_ipal_cenda.py --save-baseline
    python experiments/verify_ipal_cenda.py --verify --tag "after_sparse"
"""

import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_single import run, load_config, MODEL_DEFAULTS, CLASSIFIER_DEFAULTS

VERIFY_DIR = Path(__file__).resolve().parent.parent / 'results' / 'consistency_baseline'

DATASETS = ['lost', 'MSRCv2']
SEED = 42
CV_FOLDS = 3

METRIC_KEYS = ['balanced_acc', 'overall_acc']
TOLERANCE = 1e-4

PIPELINES = {
    'ipal': {
        'model': 'sdlpp',
        'model_params': {'T': 10},
        'classifier': 'ipal',
    },
    'cenda': {
        'model': 'cenda',
        'model_params': {'T': 10},
        'classifier': 'knn',
    },
}


def _build_config(dataset, pipeline_cfg):
    base = load_config(None)
    base['seed'] = SEED
    base['eval']['cv_folds'] = CV_FOLDS
    base['data']['name'] = dataset
    base['model']['name'] = pipeline_cfg['model']
    base['model']['params'] = dict(pipeline_cfg['model_params'])
    base['classifier']['name'] = pipeline_cfg['classifier']
    base['disambig']['params']['use_sample_reliability'] = False
    base['disambig']['params']['use_class_balance'] = False
    return base


def run_all():
    results = {}
    for ds in DATASETS:
        results[ds] = {}
        for pipe_name, pipe_cfg in PIPELINES.items():
            cfg = _build_config(ds, pipe_cfg)
            print(f"\n{'='*60}")
            print(f"Running: {ds} / {pipe_name}")
            print(f"{'='*60}")
            avg = run(cfg)
            results[ds][pipe_name] = {k: float(avg[k]) for k in METRIC_KEYS}
            print(f"  -> {results[ds][pipe_name]}")
    return results


def save_baseline():
    results = run_all()
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    out = VERIFY_DIR / 'baseline_ipal_cenda.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline saved to {out}")
    return results


def verify(tag=""):
    baseline_path = VERIFY_DIR / 'baseline_ipal_cenda.json'
    if not baseline_path.exists():
        print(f"ERROR: baseline not found at {baseline_path}")
        print("Run with --save-baseline first.")
        sys.exit(1)

    with open(baseline_path) as f:
        baseline = json.load(f)

    current = run_all()

    all_pass = True
    print(f"\n{'='*60}")
    print(f"CONSISTENCY CHECK (IPAL/CENDA){' (' + tag + ')' if tag else ''}")
    print(f"{'='*60}")
    for ds in DATASETS:
        for pipe_name in PIPELINES:
            b = baseline[ds][pipe_name]
            c = current[ds][pipe_name]
            for metric in METRIC_KEYS:
                diff = abs(b[metric] - c[metric])
                status = "PASS" if diff < TOLERANCE else "FAIL"
                if status == "FAIL":
                    all_pass = False
                print(f"  {ds:12s} / {pipe_name:10s} / {metric:15s}: "
                      f"baseline={b[metric]:.6f}  current={c[metric]:.6f}  "
                      f"diff={diff:.2e}  [{status}]")

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — investigate before proceeding")
    print(f"{'='*60}")

    if tag:
        log_path = VERIFY_DIR / f'verify_ipal_cenda_{tag}.json'
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
