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
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_single import (
    run,
    load_config,
    REDUCERS,
    CLASSIFIERS,
    MODEL_DEFAULTS,
    apply_fast_training_overrides,
    collect_env_snapshot,
    compute_config_hash,
    project_root,
)
from experiments.tools.session_manifest import write_session_manifest, write_runs_index
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

CAMPAIGN_TAG = 'benchmark_grid'

# SDLPP-specific disambig variants (other models ignore these)
SDLPP_DISAMBIG_VARIANTS = {
    'baseline': {'use_sample_reliability': False, 'use_class_balance': False},
    'SR+CB': {
        'use_sample_reliability': True,
        'use_class_balance': True,
        'r_min': 0.1,
    },
}


def build_configs(datasets, models, classifiers, base_config, fast=False):
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
                        if fast:
                            apply_fast_training_overrides(cfg)
                        configs.append((ds_name, tag, cfg))
                else:
                    cfg = copy.deepcopy(base_config)
                    cfg['data']['name'] = ds_name
                    cfg['model']['name'] = model_name
                    cfg['classifier']['name'] = cls_name
                    cfg['output']['dir'] = str(
                        Path(base_config['output']['dir']) / 'benchmark')
                    tag = f"{model_name}+{cls_name}"
                    if fast:
                        apply_fast_training_overrides(cfg)
                    configs.append((ds_name, tag, cfg))
    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(datasets, models, classifiers, base_config, fast=False, config_path=None):
    session_t0 = time.time()
    session_started_iso = datetime.now().isoformat(timespec='seconds')
    model_defaults_at_start = copy.deepcopy(MODEL_DEFAULTS)
    env_snapshot = collect_env_snapshot()
    manifest_base = copy.deepcopy(base_config)
    if fast:
        apply_fast_training_overrides(manifest_base)

    configs = build_configs(datasets, models, classifiers, base_config, fast=fast)
    total = len(configs)

    all_results = []
    runs_index_rows = []

    for idx, (ds_name, method_tag, cfg) in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{total}] {ds_name} / {method_tag}")
        print(f"{'='*70}")

        run_id = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx:03d}"
        cfg['_run_meta'] = {
            'run_id': run_id,
            'script_name': 'run_benchmark',
            'mode': 'fast' if fast else 'full',
            'env_snapshot': env_snapshot,
        }

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
            all_results.append({
                'dataset': ds_name,
                'method': method_tag,
                'elapsed_s': round(elapsed, 1),
                'run_id': run_id,
                'config_hash': cfg_hash,
                'script_name': 'run_benchmark',
                'mode': 'fast' if fast else 'full',
                **avg,
            })
            runs_index_rows.append({
                'dataset': ds_name,
                'method': method_tag,
                'seed': cfg.get('seed', ''),
                'run_id': run_id,
                'config_hash': cfg_hash,
                'results_json_rel': f'{results_rel}/results.json',
            })
        except Exception as e:
            print(f"FAILED: {e}")
            cfg_hash = compute_config_hash(cfg)
            all_results.append({
                'dataset': ds_name,
                'method': method_tag,
                'elapsed_s': 0,
                'run_id': run_id,
                'config_hash': cfg_hash,
                'script_name': 'run_benchmark',
                'mode': 'fast' if fast else 'full',
                'error': str(e),
            })
            runs_index_rows.append({
                'dataset': ds_name,
                'method': method_tag,
                'seed': cfg.get('seed', ''),
                'run_id': run_id,
                'config_hash': cfg_hash,
                'results_json_rel': '',
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
    write_runs_index(out_dir, runs_index_rows)
    n_ok = sum(1 for r in all_results if 'error' not in r)
    write_session_manifest(
        out_dir,
        script_name='run_benchmark',
        campaign=CAMPAIGN_TAG,
        mode_fast=bool(fast),
        base_config=manifest_base,
        env_snapshot=env_snapshot,
        model_defaults_snapshot=model_defaults_at_start,
        started_at_iso=session_started_iso,
        ended_at_iso=datetime.now().isoformat(timespec='seconds'),
        wall_seconds=time.time() - session_t0,
        config_path=config_path,
        n_runs=len(all_results),
        n_success=n_ok,
    )
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
                        help='Quick mode: T=10, cv_folds=3 (per config, no global dict mutation)')
    args = parser.parse_args()

    base = load_config(args.config)
    datasets = args.datasets or ALL_DATASETS
    models = args.models or ALL_MODELS
    classifiers = args.classifiers or ALL_CLASSIFIERS

    run_benchmark(datasets, models, classifiers, base, fast=args.fast, config_path=args.config)
