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
    python experiments/run_ablation_sr_cb.py --log-file run.log     # also append to file
    python experiments/run_ablation_sr_cb.py --log-dir results/logs  # auto-named log under dir
"""

import copy
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_single import (
    run,
    load_config,
    MODEL_DEFAULTS,
    apply_fast_training_overrides,
    canonical_json,
    compute_config_hash,
    collect_env_snapshot,
    project_root,
)
from experiments.tools.session_manifest import write_session_manifest, write_runs_index
from pll.eval.reporter import print_summary_table, save_summary_csv

ALL_DATASETS = [
    'lost', 'MSRCv2', 'FG-NET', 'Mirflickr',
    'Soccer Player', 'Yahoo! News',
    'slashdotpl-f1', 'slashdotpl-f2', 'slashdotpl-f3',
]

WARMUP_MAP = {
    'lost': 5, 'MSRCv2': 2, 'Soccer Player': 2, 'Yahoo! News': 3,
    'FG-NET': 5, 'Mirflickr': 0,
    'slashdotpl-f1': 0, 'slashdotpl-f2': 0, 'slashdotpl-f3': 2,
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

_LOG_FMT = '%(asctime)s | %(levelname)-8s | %(message)s'
_LOG_DATEFMT = '%Y-%m-%d %H:%M:%S'

LOG = logging.getLogger('sr_cb_ablation')


def setup_logging(log_file=None):
    """Console + optional UTF-8 file; does not alter root logger."""
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
    parser.add_argument('--log-file', type=str, default=None,
                        help='Append logs to this file (UTF-8), in addition to stdout')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Write logs to <log-dir>/sr_cb_ablation_<timestamp>.log')
    args = parser.parse_args()

    log_path = args.log_file
    if log_path is None and args.log_dir:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = str(Path(args.log_dir) / f'sr_cb_ablation_{ts}.log')
    setup_logging(log_path)

    session_t0 = time.time()
    session_started_iso = datetime.now().isoformat(timespec='seconds')
    model_defaults_at_start = copy.deepcopy(MODEL_DEFAULTS)
    base = load_config(args.config)
    if args.full:
        args.fast = False
    if args.fast:
        base = copy.deepcopy(base)
        apply_fast_training_overrides(base)

    datasets = args.datasets or ALL_DATASETS
    configs = build_configs(datasets, args.seeds, base)
    total = len(configs)
    multi_seed = len(args.seeds) > 1
    all_results = []
    runs_index_rows = []
    env_snapshot = collect_env_snapshot()

    t_sdlpp = base.get('model', {}).get('params', {}).get(
        'T', MODEL_DEFAULTS.get('sdlpp', {}).get('T', '?'))
    LOG.info(
        'Session start | mode=%s | cv_folds=%s | T(sdlpp)=%s | datasets=%d | seeds=%s | '
        'total_runs=%d',
        'fast' if args.fast else 'full',
        base['eval']['cv_folds'],
        t_sdlpp,
        len(datasets),
        args.seeds,
        total,
    )
    if log_path:
        LOG.info('Log file: %s', log_path)
    LOG.debug('EnvMeta: %s', canonical_json(env_snapshot))
    LOG.debug('BaseConfig: %s', canonical_json(base))

    for idx, (ds, var_name, seed, cfg) in enumerate(configs, 1):
        run_id = f"sr_cb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx:03d}"
        cfg['_run_meta'] = {
            'run_id': run_id,
            'script_name': 'run_ablation_sr_cb',
            'mode': 'fast' if args.fast else 'full',
            'env_snapshot': env_snapshot,
        }
        label = f"{ds} / {var_name}"
        if multi_seed:
            label += f" / seed={seed}"
        LOG.info('=' * 70)
        LOG.info('START [%d/%d] %s', idx, total, label)
        LOG.debug('RunConfig: %s', canonical_json(cfg))

        t0 = time.time()
        try:
            avg, meta = run(cfg, return_meta=True)
            cfg_hash = meta['config_hash']
            elapsed = time.time() - t0
            try:
                results_rel = str(
                    Path(meta['output_dir']).resolve().relative_to(project_root())
                )
            except ValueError:
                results_rel = meta['output_dir']
            row = {
                'dataset': ds,
                'method': var_name,
                'seed': seed,
                'elapsed_s': round(elapsed, 1),
                'run_id': run_id,
                'config_hash': cfg_hash,
                'script_name': 'run_ablation_sr_cb',
                'mode': 'fast' if args.fast else 'full',
                **avg,
            }
            all_results.append(row)
            runs_index_rows.append({
                'dataset': ds,
                'method': var_name,
                'seed': seed,
                'run_id': run_id,
                'config_hash': cfg_hash,
                'results_json_rel': f'{results_rel}/results.json',
            })
            LOG.debug(
                'RunMeta | run_id=%s | config_hash=%s | dataset=%s | method=%s | seed=%s',
                run_id, cfg_hash, ds, var_name, seed,
            )
            LOG.info(
                'DONE  [%d/%d] %s | elapsed=%.1fs | balanced_acc=%.4f | overall_acc=%.4f',
                idx, total, label, elapsed,
                float(avg['balanced_acc']), float(avg['overall_acc']),
            )
        except Exception as e:
            LOG.exception('FAIL  [%d/%d] %s', idx, total, label)
            cfg_hash = compute_config_hash(cfg)
            row = {
                'dataset': ds,
                'method': var_name,
                'seed': seed,
                'elapsed_s': 0,
                'run_id': run_id,
                'config_hash': cfg_hash,
                'script_name': 'run_ablation_sr_cb',
                'mode': 'fast' if args.fast else 'full',
                'error': str(e),
            }
            all_results.append(row)
            runs_index_rows.append({
                'dataset': ds,
                'method': var_name,
                'seed': seed,
                'run_id': run_id,
                'config_hash': cfg_hash,
                'results_json_rel': '',
            })

    wall = time.time() - session_t0
    LOG.info('=' * 70)
    LOG.info('Session wall time: %.1fs (%.2f min)', wall, wall / 60.0)

    print(f"\n\n{'='*70}")
    print("SR x CB ABLATION SUMMARY")
    print(f"{'='*70}\n")
    print_summary_table(all_results)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(base['output']['dir']) / 'benchmark' / OUTPUT_TAG / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    save_summary_csv(all_results, out_dir / 'summary.csv')
    write_runs_index(out_dir, runs_index_rows)
    n_ok = sum(1 for r in all_results if 'error' not in r)
    write_session_manifest(
        out_dir,
        script_name='run_ablation_sr_cb',
        campaign=OUTPUT_TAG,
        mode_fast=bool(args.fast),
        base_config=base,
        env_snapshot=env_snapshot,
        model_defaults_snapshot=model_defaults_at_start,
        started_at_iso=session_started_iso,
        ended_at_iso=datetime.now().isoformat(timespec='seconds'),
        wall_seconds=time.time() - session_t0,
        config_path=args.config,
        n_runs=len(all_results),
        n_success=n_ok,
    )
    LOG.info('Saved summary CSV: %s', out_dir / 'summary.csv')
    LOG.info('Saved session_manifest.json and runs_index.csv under %s', out_dir)
    print(f"\nSaved to {out_dir}")

    return all_results


if __name__ == '__main__':
    main()
