"""Run CIFAR10 long-tailed PLL benchmark on generated dataset names.

This script assumes datasets are already generated into datasets/*.mat with naming:
    cifar10-lt-g{gamma}-r{r}-{dataset_mode}-s{dataset_seed}

Supported disambiguation variants:
    baseline  - no SR, no CB
    sr_only   - SR only
    cb_only   - CB only
    sr+cb     - SR + CB

Important: dataset-generation params are independent from training params.
- dataset params: dataset_mode, dataset_seeds (used to resolve dataset filenames)
- training params: train_mode, run_seeds (used in cfg['seed'] and fast/full overrides)

Example:
    conda run -n PLLRec python experiments/run_cifar10_lt.py --train-mode fast --dataset-mode fast
    conda run -n PLLRec python experiments/run_cifar10_lt.py --variants baseline,sr+cb
    conda run -n PLLRec python experiments/run_cifar10_lt.py --train-mode full --dataset-mode full --gammas 100,200 --rs 1,2,3 --dataset-seeds 42,43,44 --run-seeds 7,8
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_single import (
    REDUCERS,
    CLASSIFIERS,
    apply_fast_training_overrides,
    collect_env_snapshot,
    compute_config_hash,
    load_config,
    project_root,
    run,
)
from experiments.tools.session_manifest import write_runs_index, write_session_manifest
from pll.eval.reporter import print_summary_table, save_summary_csv


VARIANTS = {
    'baseline': {
        'use_sample_reliability': False,
        'use_class_balance': False,
    },
    'sr_only': {
        'use_sample_reliability': True,
        'use_class_balance': False,
    },
    'cb_only': {
        'use_sample_reliability': False,
        'use_class_balance': True,
    },
    'sr+cb': {
        'use_sample_reliability': True,
        'use_class_balance': True,
    },
}


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(',') if x.strip()]


def _build_dataset_name(gamma: int, r: int, dataset_mode: str, dataset_seed: int) -> str:
    return f'cifar10-lt-g{gamma}-r{r}-{dataset_mode}-s{dataset_seed}'


def _build_dataset_jobs(
    gammas: list[int],
    rs: list[int],
    dataset_seeds: list[int],
    dataset_mode: str,
) -> list[tuple[int, int, int, str]]:
    jobs = []
    for gamma in gammas:
        for r in rs:
            for dataset_seed in dataset_seeds:
                jobs.append(
                    (
                        gamma,
                        r,
                        dataset_seed,
                        _build_dataset_name(gamma, r, dataset_mode, dataset_seed),
                    )
                )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description='CIFAR10 LT-PLL benchmark runner')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--train-mode', choices=['fast', 'full'], default=None,
                        help='Training budget mode (controls fast/full overrides)')
    parser.add_argument('--dataset-mode', choices=['fast', 'full'], default=None,
                        help='Dataset naming mode used to resolve generated .mat files')
    parser.add_argument('--fast', action='store_true', default=True)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--gammas', type=str, default='100,200')
    parser.add_argument('--rs', type=str, default='1,2,3')
    parser.add_argument('--seeds', type=str, default='42,43,44',
                        help='[compat] same as --run-seeds if --run-seeds is omitted')
    parser.add_argument('--run-seeds', type=str, default=None,
                        help='Training random seeds (cfg["seed"])')
    parser.add_argument('--dataset-seeds', type=str, default=None,
                        help='Dataset seeds encoded in dataset filenames')
    parser.add_argument('--models', type=str, default='sdlpp')
    parser.add_argument('--classifiers', type=str, default='knn')
    parser.add_argument('--variants', type=str, default='baseline,sr+cb',
                        help=f'Comma-separated variant list. Choices: {list(VARIANTS.keys())}')
    parser.add_argument('--data-dir', type=str, default='datasets/cifar10',
                        help='Directory containing generated CIFAR10 LT .mat files')
    parser.add_argument('--output-tag', type=str, default='cifar10_lt')
    parser.add_argument('--sr-r-min', type=float, default=0.1,
                        help='r_min used when SR is enabled')
    parser.add_argument('--sr-warmup', type=int, default=0,
                        help='warmup used when SR is enabled')
    parser.add_argument('--cb-alpha', type=float, default=0.5,
                        help='alpha used when CB is enabled')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print expanded jobs and exit without training')
    parser.add_argument('--require-data', action='store_true',
                        help='Fail if any dataset .mat is missing before running')
    args = parser.parse_args()

    if args.full:
        args.fast = False

    train_mode = args.train_mode or ('fast' if args.fast else 'full')
    dataset_mode = args.dataset_mode or train_mode

    gammas = _parse_int_list(args.gammas)
    rs = _parse_int_list(args.rs)
    run_seeds = _parse_int_list(args.run_seeds) if args.run_seeds else _parse_int_list(args.seeds)
    dataset_seeds = (
        _parse_int_list(args.dataset_seeds)
        if args.dataset_seeds
        else list(run_seeds)
    )
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    classifiers = [c.strip() for c in args.classifiers.split(',') if c.strip()]
    variants = _parse_str_list(args.variants)

    bad_models = [m for m in models if m not in REDUCERS]
    bad_cls = [c for c in classifiers if c not in CLASSIFIERS]
    bad_variants = [v for v in variants if v not in VARIANTS]
    if bad_models:
        raise ValueError(f'Unknown models: {bad_models}, choices={list(REDUCERS.keys())}')
    if bad_cls:
        raise ValueError(f'Unknown classifiers: {bad_cls}, choices={list(CLASSIFIERS.keys())}')
    if bad_variants:
        raise ValueError(f'Unknown variants: {bad_variants}, choices={list(VARIANTS.keys())}')

    dataset_jobs = _build_dataset_jobs(gammas, rs, dataset_seeds, dataset_mode)
    if not dataset_jobs:
        raise ValueError('No dataset jobs built from provided grids')

    if args.require_data:
        missing = []
        for _, _, _, ds in dataset_jobs:
            p = Path(args.data_dir) / f'{ds}.mat'
            if not p.is_file():
                missing.append(str(p))
        if missing:
            print('[ERROR] Missing dataset files:')
            for x in missing:
                print(f'  - {x}')
            raise SystemExit(2)

    base = load_config(args.config)
    if train_mode == 'fast':
        base = copy.deepcopy(base)
        apply_fast_training_overrides(base)

    env_snapshot = collect_env_snapshot()
    session_t0 = time.time()
    session_started_iso = datetime.now().isoformat(timespec='seconds')

    jobs = []
    for gamma, r, dataset_seed, ds_name in dataset_jobs:
        for model in models:
            for clf in classifiers:
                for variant in variants:
                    for run_seed in run_seeds:
                        jobs.append((
                            gamma,
                            r,
                            dataset_seed,
                            run_seed,
                            ds_name,
                            model,
                            clf,
                            variant,
                        ))

    if args.dry_run:
        print(
            f'[DRY-RUN] train_mode={train_mode} dataset_mode={dataset_mode} '
            f'data_dir={args.data_dir} total_jobs={len(jobs)}'
        )
        for idx, (gamma, r, dataset_seed, run_seed, ds_name, model, clf, variant) in enumerate(jobs, 1):
            print(
                f'[{idx}/{len(jobs)}] dataset={ds_name} gamma={gamma} r={r} '
                f'dataset_seed={dataset_seed} run_seed={run_seed} '
                f'model={model} clf={clf} variant={variant}'
            )
        return

    all_results = []
    runs_index_rows = []
    total = len(jobs)

    for idx, (gamma, r, dataset_seed, run_seed, ds_name, model, clf, variant) in enumerate(jobs, 1):
        print(f"\n{'='*70}")
        print(
            f'[{idx}/{total}] dataset={ds_name} model={model} clf={clf} variant={variant} '
            f'dataset_seed={dataset_seed} run_seed={run_seed}'
        )
        print(f"{'='*70}")

        cfg = copy.deepcopy(base)
        cfg['seed'] = run_seed
        cfg['data']['name'] = ds_name
        cfg['data']['data_dir'] = args.data_dir
        cfg['model']['name'] = model
        cfg['classifier']['name'] = clf
        cfg['output']['dir'] = str(Path(base['output']['dir']) / 'benchmark' / args.output_tag)

        var_params = VARIANTS[variant]
        cfg['disambig']['params']['use_sample_reliability'] = var_params['use_sample_reliability']
        cfg['disambig']['params']['use_class_balance'] = var_params['use_class_balance']
        if var_params['use_sample_reliability']:
            cfg['disambig']['params']['r_min'] = args.sr_r_min
            cfg['disambig']['params']['warmup'] = args.sr_warmup
        else:
            cfg['disambig']['params']['warmup'] = 0
        if var_params['use_class_balance']:
            cfg['disambig']['params']['alpha'] = args.cb_alpha

        run_id = f"cifar10lt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx:03d}"
        cfg['_run_meta'] = {
            'run_id': run_id,
            'script_name': 'run_cifar10_lt',
            'mode': train_mode,
            'dataset_mode': dataset_mode,
            'dataset_seed': dataset_seed,
            'run_seed': run_seed,
            'env_snapshot': env_snapshot,
        }

        t0 = time.time()
        try:
            avg, meta = run(cfg, return_meta=True)
            cfg_hash = meta['config_hash']
            elapsed = time.time() - t0
            try:
                results_rel = str(Path(meta['output_dir']).resolve().relative_to(project_root()))
            except ValueError:
                results_rel = meta['output_dir']

            all_results.append({
                'dataset': ds_name,
                'method': f'{model}({variant})+{clf}',
                'gamma': gamma,
                'r': r,
                'dataset_seed': dataset_seed,
                'seed': run_seed,
                'model': model,
                'classifier': clf,
                'variant': variant,
                'elapsed_s': round(elapsed, 1),
                'run_id': run_id,
                'config_hash': cfg_hash,
                'mode': train_mode,
                **avg,
            })
            runs_index_rows.append({
                'dataset': ds_name,
                'method': f'{model}({variant})+{clf}',
                'seed': run_seed,
                'run_id': run_id,
                'config_hash': cfg_hash,
                'results_json_rel': f'{results_rel}/results.json',
            })
        except Exception as e:
            cfg_hash = compute_config_hash(cfg)
            all_results.append({
                'dataset': ds_name,
                'method': f'{model}({variant})+{clf}',
                'gamma': gamma,
                'r': r,
                'dataset_seed': dataset_seed,
                'seed': run_seed,
                'model': model,
                'classifier': clf,
                'variant': variant,
                'elapsed_s': 0,
                'run_id': run_id,
                'config_hash': cfg_hash,
                'mode': train_mode,
                'error': str(e),
            })
            runs_index_rows.append({
                'dataset': ds_name,
                'method': f'{model}({variant})+{clf}',
                'seed': run_seed,
                'run_id': run_id,
                'config_hash': cfg_hash,
                'results_json_rel': '',
            })
            print(f'[FAIL] {e}')

    print(f"\n\n{'='*70}")
    print('CIFAR10 LT SUMMARY')
    print(f"{'='*70}\n")
    print_summary_table(all_results)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(base['output']['dir']) / 'benchmark' / args.output_tag / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    save_summary_csv(all_results, out_dir / 'summary.csv')
    write_runs_index(out_dir, runs_index_rows)
    n_ok = sum(1 for r in all_results if 'error' not in r)
    write_session_manifest(
        out_dir,
        script_name='run_cifar10_lt',
        campaign=args.output_tag,
        mode_fast=(train_mode == 'fast'),
        base_config=base,
        env_snapshot=env_snapshot,
        model_defaults_snapshot={},
        started_at_iso=session_started_iso,
        ended_at_iso=datetime.now().isoformat(timespec='seconds'),
        wall_seconds=time.time() - session_t0,
        config_path=args.config,
        n_runs=len(all_results),
        n_success=n_ok,
    )
    print(f'\nSaved to {out_dir}')


if __name__ == '__main__':
    main()
