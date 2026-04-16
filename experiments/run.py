"""Unified experiment runner.

Supports:
- Single dataset + method
- Batch: datasets x methods x classifiers x sweep grid
- Campaign mode with aggregated results

Examples
--------
# Single run
python experiments/run.py --dataset lost --method sdlpp_baseline

# Batch benchmark
python experiments/run.py \\
    --datasets lost MSRCv2 FG-NET Mirflickr \\
    --methods sdlpp_baseline sdlpp_sr_cb delin cenda \\
    --campaign benchmark_v2

# Classifier comparison
python experiments/run.py \\
    --datasets lost MSRCv2 --methods sdlpp_baseline \\
    --classifiers knn ipal --campaign clf_compare

# SR x CB ablation (Cartesian product)
python experiments/run.py \\
    --datasets lost MSRCv2 --method sdlpp_sr_cb \\
    --sweep disambig.params.use_sample_reliability=true,false \\
    --sweep disambig.params.use_class_balance=true,false \\
    --campaign ablation_sr_cb

# Parameter sweep
python experiments/run.py \\
    --datasets lost --method sdlpp_sr_cb \\
    --sweep disambig.params.r_min=0.0,0.05,0.1,0.2,0.3 \\
    --campaign sweep_rmin

# Baseline SDLPP: reduced dimension target_d (same as run_all.sh batch 4c)
python experiments/run.py \\
    --datasets lost MSRCv2 --method sdlpp_baseline \\
    --sweep model.params.target_d=5,8,13,20,30,50 \\
    --n-repeats 10 --campaign sweep_baseline_target_d_v1
"""

import argparse
import copy
import itertools
from collections import OrderedDict
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from pll.config import (
    load_experiment_config, set_nested, auto_cast,
    canonical_json, compute_config_hash, collect_env_snapshot,
    reduction_identity_hash,
)
from pll.data.loader import load_dataset
from pll.eval.evaluator import Evaluator
from pll.eval.reporter import (
    print_summary_table, save_splits_csv, save_summary_json,
    save_config_json, save_summary_csv, save_summary_latex,
    SWEEP_ALIAS,
)

log = logging.getLogger('run')


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

def parse_sweeps(sweep_args):
    """Parse ['path=v1,v2', ...] into [(dotted_path, [val, ...]), ...]."""
    axes = []
    for s in (sweep_args or []):
        path, vals_str = s.split('=', 1)
        vals = [auto_cast(v) for v in vals_str.split(',')]
        axes.append((path, vals))
    return axes


def sweep_short_name(dotted_path):
    """Produce short column name for a sweep parameter."""
    short = dotted_path.rsplit('.', 1)[-1]
    return SWEEP_ALIAS.get(short, short)


def expand_sweep_grid(cfg, sweep_axes):
    """Yield (config_copy, sweep_label_dict) for each point in the Cartesian product."""
    if not sweep_axes:
        yield cfg, {}
        return
    keys = [p for p, _ in sweep_axes]
    val_lists = [v for _, v in sweep_axes]
    for combo in itertools.product(*val_lists):
        cfg_copy = copy.deepcopy(cfg)
        sweep_label = {}
        for path, val in zip(keys, combo):
            set_nested(cfg_copy, path, val)
            sweep_label[sweep_short_name(path)] = val
        yield cfg_copy, sweep_label


def sweep_suffix(sweep_label):
    """Build a filesystem-safe directory suffix from sweep labels."""
    if not sweep_label:
        return ''
    parts = []
    for k, v in sweep_label.items():
        if isinstance(v, bool):
            parts.append(f'{k}={str(v).lower()}')
        else:
            parts.append(f'{k}={v}')
    return '_'.join(parts)


# ---------------------------------------------------------------------------
# Experiment specification
# ---------------------------------------------------------------------------

@dataclass
class ExperimentSpec:
    dataset: str
    method: str
    classifier: str
    sweep_label: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)


def build_experiment_grid(args):
    """Build the full (dataset x method x classifier x sweep) grid."""
    sweep_axes = parse_sweeps(args.sweep)
    datasets = args.datasets or ([args.dataset] if args.dataset else ['lost'])
    methods = args.methods or ([args.method] if args.method else ['sdlpp_baseline'])

    cli_overrides = {}
    if args.seed is not None:
        cli_overrides['seed'] = args.seed
    if args.n_repeats is not None:
        cli_overrides.setdefault('eval', {})['n_repeats'] = args.n_repeats

    grid = []
    for ds in datasets:
        for m in methods:
            cfg = load_experiment_config(ds, m, cli_overrides or None)

            if args.fast:
                cfg.setdefault('eval', {})['n_repeats'] = 3
                mp = cfg.get('model', {}).get('params', {})
                if 'T' in mp:
                    mp['T'] = 10

            classifiers = args.classifiers or [cfg.get('classifier', {}).get('name', 'knn')]
            for clf_name in classifiers:
                cfg_clf = copy.deepcopy(cfg)
                cfg_clf.setdefault('classifier', {})['name'] = clf_name
                if args.clf_params:
                    for kv in args.clf_params:
                        k, v = kv.split('=', 1)
                        cfg_clf['classifier'].setdefault('params', {})[k] = auto_cast(v)

                if args.fast and clf_name == 'plsvm':
                    cp = cfg_clf.setdefault('classifier', {}).setdefault('params', {})
                    t0 = int(cp.get('T', 2000))
                    cp['T'] = min(t0, 400)

                for cfg_swept, slabel in expand_sweep_grid(cfg_clf, sweep_axes):
                    grid.append(ExperimentSpec(
                        dataset=ds, method=m, classifier=clf_name,
                        sweep_label=slabel, config=cfg_swept,
                    ))
    return grid


def group_grid_for_shared_reduction(grid: list[ExperimentSpec]):
    """Group specs that share the same dataset and reduction (all but classifier).

    Yields batches: either ``('single', spec)`` or ``('multi', list[spec])``.
    Order follows first occurrence of each group key in *grid*.
    """
    groups: OrderedDict[tuple, list[ExperimentSpec]] = OrderedDict()
    for spec in grid:
        key = (spec.dataset, reduction_identity_hash(spec.config))
        if key not in groups:
            groups[key] = []
        groups[key].append(spec)
    for specs in groups.values():
        if len(specs) == 1:
            yield 'single', specs[0]
        else:
            yield 'multi', specs


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def result_dir(base_dir, campaign, dataset, method_tag, classifier, sweep_label_dict,
               timestamp):
    """Build the output directory path."""
    tag = f'{method_tag}+{classifier}'
    parts = [base_dir, campaign, dataset, tag]
    sfx = sweep_suffix(sweep_label_dict)
    if sfx:
        parts.append(sfx)
    parts.append(timestamp)
    return Path(*parts)


def save_experiment(eval_result, sweep_label, campaign, base_dir, timestamp):
    """Persist a single experiment's results to disk."""
    cfg = eval_result.config
    summary = eval_result.summary
    method_tag = cfg.get('method_tag', cfg.get('model', {}).get('name', 'unknown'))
    clf_name = cfg.get('classifier', {}).get('name', 'knn')

    out = result_dir(
        base_dir, campaign, summary['dataset'], method_tag, clf_name,
        sweep_label, timestamp,
    )
    out.mkdir(parents=True, exist_ok=True)

    save_config_json(cfg, out / 'config.json')
    save_splits_csv(eval_result.split_metrics, out / 'splits.csv')

    summary_with_sweep = dict(summary)
    summary_with_sweep['sweep_params'] = sweep_label or {}
    summary_with_sweep['timestamp'] = timestamp
    save_summary_json(summary_with_sweep, out / 'summary.json')

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = '%(asctime)s | %(levelname)-8s | %(message)s'
    logging.basicConfig(level=level, format=fmt, datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout)


def main():
    parser = argparse.ArgumentParser(
        description='Unified PLL experiment runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Dataset / method selection
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--dataset', type=str, help='Single dataset name')
    g.add_argument('--datasets', type=str, nargs='+', help='Multiple dataset names')

    g2 = parser.add_mutually_exclusive_group()
    g2.add_argument('--method', type=str, help='Single method config name')
    g2.add_argument('--methods', type=str, nargs='+', help='Multiple method config names')

    parser.add_argument('--classifiers', type=str, nargs='+', default=None,
                        help='Override classifier(s); creates dataset x method x classifier grid')
    parser.add_argument('--clf-params', type=str, nargs='+', default=None,
                        help='Classifier param overrides: key=value')

    # Sweep
    parser.add_argument('--sweep', type=str, action='append', default=None,
                        help='Parameter sweep: dotted.path=v1,v2,...  (repeatable)')

    # Campaign & output
    parser.add_argument('--campaign', type=str, default='single',
                        help='Campaign name for result grouping (default: single)')

    # Overrides
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n-repeats', type=int, default=None)
    parser.add_argument('--fast', action='store_true', help='Quick smoke test mode')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    setup_logging(args.verbose)

    session_t0 = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    env_snapshot = collect_env_snapshot()

    grid = build_experiment_grid(args)
    total = len(grid)
    log.info('Campaign=%s  experiments=%d  timestamp=%s', args.campaign, total, timestamp)

    all_summaries = []
    all_outputs = []
    idx = 0

    for batch_kind, batch in group_grid_for_shared_reduction(grid):
        if batch_kind == 'single':
            spec = batch
            idx += 1
            label = f'{spec.dataset} / {spec.method} / {spec.classifier}'
            if spec.sweep_label:
                label += f' / {spec.sweep_label}'

            log.info('=' * 70)
            log.info('START [%d/%d] %s', idx, total, label)

            t0 = time.time()
            try:
                np.random.seed(spec.config.get('seed', 42))
                dataset = load_dataset(
                    spec.config['data']['name'],
                    data_dir=spec.config.get('data', {}).get('data_dir', 'datasets'),
                )
                evaluator = Evaluator(spec.config)
                result = evaluator.run(dataset)
                elapsed = time.time() - t0

                out_dir = save_experiment(
                    result, spec.sweep_label, args.campaign,
                    spec.config.get('output', {}).get('base_dir', 'results'),
                    timestamp,
                )

                row = dict(result.summary)
                row['sweep_params'] = spec.sweep_label
                row['elapsed_s'] = round(elapsed, 1)
                all_summaries.append(row)
                all_outputs.append(str(out_dir))

                log.info(
                    'DONE  [%d/%d] %s | %.1fs | overall=%.4f  balanced=%.4f',
                    idx, total, label, elapsed,
                    row.get('overall_acc_mean', 0), row.get('balanced_acc_mean', 0),
                )
            except Exception:
                log.exception('FAIL  [%d/%d] %s', idx, total, label)
                all_summaries.append({
                    'dataset': spec.dataset,
                    'method': spec.method,
                    'classifier': spec.classifier,
                    'sweep_params': spec.sweep_label,
                    'error': True,
                })
        else:
            specs = batch
            clfs = ', '.join(s.classifier for s in specs)
            base = specs[0]
            idx_start = idx + 1
            idx_end = idx + len(specs)
            idx = idx_end
            group_label = f'{base.dataset} / {base.method} / [{clfs}]'
            if base.sweep_label:
                group_label += f' / {base.sweep_label}'
            log.info('=' * 70)
            log.info(
                'START [%d-%d/%d] %s (shared reduction)',
                idx_start, idx_end, total, group_label,
            )

            t0 = time.time()
            try:
                np.random.seed(base.config.get('seed', 42))
                dataset = load_dataset(
                    base.config['data']['name'],
                    data_dir=base.config.get('data', {}).get('data_dir', 'datasets'),
                )
                evaluator = Evaluator(base.config)
                results = evaluator.run_multi_classifier(
                    dataset, [s.config for s in specs])
                elapsed = time.time() - t0
                per_elapsed = elapsed / len(specs)

                for spec, result in zip(specs, results):
                    label = f'{spec.dataset} / {spec.method} / {spec.classifier}'
                    if spec.sweep_label:
                        label += f' / {spec.sweep_label}'
                    out_dir = save_experiment(
                        result, spec.sweep_label, args.campaign,
                        spec.config.get('output', {}).get('base_dir', 'results'),
                        timestamp,
                    )
                    row = dict(result.summary)
                    row['sweep_params'] = spec.sweep_label
                    row['elapsed_s'] = round(per_elapsed, 1)
                    row['shared_reduction_group'] = True
                    all_summaries.append(row)
                    all_outputs.append(str(out_dir))
                    log.info(
                        'DONE  %s | ~%.1fs (shared batch) | overall=%.4f  balanced=%.4f',
                        label, per_elapsed,
                        row.get('overall_acc_mean', 0), row.get('balanced_acc_mean', 0),
                    )
            except Exception:
                log.exception('FAIL  [%d-%d/%d] %s', idx_start, idx_end, total, group_label)
                for spec in specs:
                    all_summaries.append({
                        'dataset': spec.dataset,
                        'method': spec.method,
                        'classifier': spec.classifier,
                        'sweep_params': spec.sweep_label,
                        'error': True,
                    })

    wall = time.time() - session_t0
    log.info('=' * 70)
    log.info('Total wall time: %.1fs (%.2f min)', wall, wall / 60)

    print(f"\n{'=' * 70}")
    print(f"CAMPAIGN: {args.campaign}  ({total} experiments)")
    print(f"{'=' * 70}\n")

    valid = [r for r in all_summaries if 'error' not in r]
    if valid:
        print_summary_table(valid)

    failed = [r for r in all_summaries if 'error' in r]
    if failed:
        print(f"\n{len(failed)} experiment(s) FAILED:")
        for r in failed:
            print(f"  - {r['dataset']} / {r['method']} / {r['classifier']}")

    # Campaign-level files
    if total > 1 or args.campaign != 'single':
        base_dir = Path(grid[0].config.get('output', {}).get('base_dir', 'results'))
        campaign_dir = base_dir / args.campaign
        campaign_dir.mkdir(parents=True, exist_ok=True)

        save_summary_csv(valid, campaign_dir / 'campaign_summary.csv')
        save_summary_latex(valid, campaign_dir / 'campaign_summary.tex')

        manifest = {
            'campaign': args.campaign,
            'timestamp': timestamp,
            'n_experiments': total,
            'n_success': len(valid),
            'n_failed': len(failed),
            'wall_seconds': round(wall, 1),
            'env_snapshot': env_snapshot,
            'output_dirs': all_outputs,
        }
        save_summary_json(manifest, campaign_dir / 'session_manifest.json')

        log.info('Campaign summary saved to %s', campaign_dir)

    return all_summaries


if __name__ == '__main__':
    main()
