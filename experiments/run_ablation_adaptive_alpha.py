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
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_single import (
    run,
    load_config,
    MODEL_DEFAULTS,
    canonical_json,
    compute_config_hash,
    collect_env_snapshot,
)
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

_LOG_FMT = '%(asctime)s | %(levelname)-8s | %(message)s'
_LOG_DATEFMT = '%Y-%m-%d %H:%M:%S'
LOG = logging.getLogger('adaptive_alpha_ablation')


def setup_logging(log_file=None):
    LOG.handlers.clear()
    LOG.setLevel(logging.DEBUG)
    LOG.propagate = False

    fmt = logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    LOG.addHandler(ch)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        LOG.addHandler(fh)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Adaptive alpha ablation')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Override dataset list')
    parser.add_argument('--fast', action='store_true', default=True)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Append logs to this file (UTF-8), in addition to stdout')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Write logs to <log-dir>/adaptive_alpha_ablation_<timestamp>.log')
    args = parser.parse_args()

    log_path = args.log_file
    if log_path is None and args.log_dir:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = str(Path(args.log_dir) / f'adaptive_alpha_ablation_{ts}.log')
    setup_logging(log_path)

    session_t0 = time.time()
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
            cfg['output']['dir'] = str(
                Path(base['output']['dir']) / 'benchmark' / 'adaptive_alpha_ablation')
            configs.append((ds, var_name, cfg))

    total = len(configs)
    all_results = []
    env_snapshot = collect_env_snapshot()
    LOG.info(
        'Session start | mode=%s | cv_folds=%s | T(sdlpp)=%s | datasets=%s | total_runs=%d',
        'fast' if args.fast else 'full',
        base['eval']['cv_folds'],
        MODEL_DEFAULTS.get('sdlpp', {}).get('T', '?'),
        datasets,
        total,
    )
    if log_path:
        LOG.info('Log file: %s', log_path)
    LOG.debug('EnvMeta: %s', canonical_json(env_snapshot))
    LOG.debug('BaseConfig: %s', canonical_json(base))

    for idx, (ds, var_name, cfg) in enumerate(configs, 1):
        run_id = f"adaptive_alpha_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx:03d}"
        cfg['_run_meta'] = {
            'run_id': run_id,
            'script_name': 'run_ablation_adaptive_alpha',
            'mode': 'fast' if args.fast else 'full',
            'env_snapshot': env_snapshot,
        }
        cfg_hash = compute_config_hash(cfg)
        LOG.info('=' * 70)
        LOG.info('START [%d/%d] %s / %s', idx, total, ds, var_name)
        LOG.debug('RunMeta | run_id=%s | config_hash=%s', run_id, cfg_hash)
        LOG.debug('RunConfig: %s', canonical_json(cfg))

        t0 = time.time()
        try:
            avg = run(cfg)
            elapsed = time.time() - t0
            all_results.append({
                'dataset': ds,
                'method': var_name,
                'elapsed_s': round(elapsed, 1),
                'run_id': run_id,
                'config_hash': cfg_hash,
                'script_name': 'run_ablation_adaptive_alpha',
                'mode': 'fast' if args.fast else 'full',
                **avg,
            })
            LOG.info(
                'DONE  [%d/%d] %s / %s | elapsed=%.1fs | balanced_acc=%.4f | overall_acc=%.4f',
                idx, total, ds, var_name, elapsed,
                float(avg['balanced_acc']), float(avg['overall_acc']),
            )
        except Exception as e:
            LOG.exception('FAIL  [%d/%d] %s / %s', idx, total, ds, var_name)
            all_results.append({
                'dataset': ds,
                'method': var_name,
                'elapsed_s': 0,
                'run_id': run_id,
                'config_hash': cfg_hash,
                'script_name': 'run_ablation_adaptive_alpha',
                'mode': 'fast' if args.fast else 'full',
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
    LOG.info('=' * 70)
    LOG.info('Session wall time: %.1fs (%.2f min)', time.time() - session_t0, (time.time() - session_t0) / 60.0)
    LOG.info('Saved summary CSV: %s', out_dir / 'summary.csv')
    print(f"\nSaved to {out_dir}")

    return all_results


if __name__ == '__main__':
    main()
