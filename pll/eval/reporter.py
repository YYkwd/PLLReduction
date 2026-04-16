"""Result formatting, printing, and persistence.

Supports:
- mean +/- std display for repeated-holdout results
- sweep parameters as independent columns
- valid_runs counts for many/medium/few
- dataset warning markers (*)
- CSV, JSON, and LaTeX export
"""

import csv
import json
import math
import numpy as np
from pathlib import Path

METRIC_KEYS = ['overall_acc', 'balanced_acc', 'many_acc', 'medium_acc', 'few_acc']

SWEEP_ALIAS = {
    'use_sample_reliability': 'SR',
    'use_class_balance': 'CB',
    'r_min': 'r_min',
    'alpha': 'alpha',
    'warmup': 'warmup',
    'cb_cv0': 'cv0',
    'cb_cv1': 'cv1',
    'target_d': 'target_d',
    'miu': 'miu',
}


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return super().default(obj)


def _sweep_col_name(key):
    """Convert a dotted sweep path to a short column name."""
    short = key.rsplit('.', 1)[-1]
    return SWEEP_ALIAS.get(short, short)


def _detect_sweep_keys(rows):
    """Auto-detect sweep parameter keys from result rows."""
    keys = []
    seen = set()
    for r in rows:
        sp = r.get('sweep_params') or {}
        for k in sp:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def _fmt_metric(mean, std):
    """Format a metric as 'mean+-std'."""
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return '  N/A'
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f'{mean:.4f}'
    return f'{mean:.4f}+-{std:.3f}'


def print_summary_table(rows, sweep_keys=None):
    """Pretty-print benchmark summary as a console table.

    Each row is a summary dict (from Evaluator/aggregate).
    Sweep parameters automatically become independent columns.
    """
    if not rows:
        print("(no results)")
        return

    if sweep_keys is None:
        sweep_keys = _detect_sweep_keys(rows)
    sweep_cols = [_sweep_col_name(k) for k in sweep_keys]

    fixed_cols = ['Dataset', 'Method', 'Classifier']
    metric_cols = ['overall_acc', 'balanced_acc', 'many_acc', 'medium_acc', 'few_acc']

    all_cols = fixed_cols + sweep_cols + metric_cols
    widths = {c: len(c) for c in all_cols}

    formatted_rows = []
    for r in rows:
        ds = r.get('dataset', '')
        warn = r.get('dataset_warning') or r.get('summary', {}).get('dataset_warning')
        if warn:
            ds += '*'
        method = r.get('method', r.get('method_tag', ''))
        clf = r.get('classifier', '')

        vals = {'Dataset': ds, 'Method': method, 'Classifier': clf}

        sp = r.get('sweep_params') or {}
        for raw_k, col_name in zip(sweep_keys, sweep_cols):
            v = sp.get(raw_k, '')
            vals[col_name] = str(v)

        for mk in metric_cols:
            mean_k = f'{mk}_mean'
            std_k = f'{mk}_std'
            if mean_k in r:
                vals[mk] = _fmt_metric(r[mean_k], r.get(std_k))
            elif mk in r:
                v = r[mk]
                vals[mk] = f'{v:.4f}' if isinstance(v, (int, float)) and not math.isnan(v) else 'N/A'
            else:
                vals[mk] = ''

        for c in all_cols:
            widths[c] = max(widths[c], len(str(vals.get(c, ''))))
        formatted_rows.append(vals)

    for mk in metric_cols:
        widths[mk] = max(widths[mk], 13)

    header = '  '.join(f'{c:<{widths[c]}s}' for c in all_cols)
    print(header)
    print('-' * len(header))

    for vals in formatted_rows:
        parts = []
        for c in all_cols:
            v = str(vals.get(c, ''))
            if c in metric_cols:
                parts.append(f'{v:>{widths[c]}s}')
            else:
                parts.append(f'{v:<{widths[c]}s}')
        print('  '.join(parts))

    # valid_runs line for group metrics
    has_valid = any(f'{mk}_valid_runs' in r for r in rows for mk in ('many_acc', 'medium_acc', 'few_acc'))
    if has_valid:
        print()
        for r in rows:
            parts = []
            for c in fixed_cols + sweep_cols:
                parts.append(' ' * widths[c])
            for mk in metric_cols:
                vr_key = f'{mk}_valid_runs'
                nr = r.get('n_repeats', '')
                if vr_key in r:
                    parts.append(f'{"(" + str(r[vr_key]) + "/" + str(nr) + ")":>{widths[mk]}s}')
                else:
                    parts.append(' ' * widths[mk])
            print('  '.join(parts))

    warned = [r for r in rows if r.get('dataset_warning') or r.get('summary', {}).get('dataset_warning')]
    if warned:
        print()
        print("* = dataset with sparse classes (supplementary results)")


def save_splits_csv(split_metrics, path):
    """Save per-split metrics as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not split_metrics:
        return
    fieldnames = list(split_metrics[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in split_metrics:
            row = {}
            for k, v in m.items():
                if isinstance(v, float) and math.isnan(v):
                    row[k] = ''
                elif isinstance(v, float):
                    row[k] = f'{v:.6f}'
                else:
                    row[k] = v
            writer.writerow(row)


def save_summary_json(summary, path):
    """Save summary dict as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)


def save_config_json(config, path):
    """Save full config snapshot as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)


def save_summary_csv(rows, path):
    """Save campaign summary as CSV (one row per experiment)."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sweep_keys = _detect_sweep_keys(rows)
    sweep_cols = [_sweep_col_name(k) for k in sweep_keys]

    meta_fields = ['dataset', 'method', 'classifier']
    metric_fields = []
    for mk in METRIC_KEYS:
        metric_fields.extend([f'{mk}_mean', f'{mk}_std'])
        if any(f'{mk}_valid_runs' in r for r in rows):
            metric_fields.append(f'{mk}_valid_runs')
    has_dim_out = any('dim_out' in r for r in rows)
    dim_out_col = ['dim_out'] if has_dim_out else []
    fieldnames = meta_fields + sweep_cols + dim_out_col + metric_fields + ['n_repeats', 'dataset_warning']

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, '') for k in fieldnames}
            sp = r.get('sweep_params') or {}
            for raw_k, col_name in zip(sweep_keys, sweep_cols):
                out[col_name] = sp.get(raw_k, '')
            for mk in METRIC_KEYS:
                for suffix in ('_mean', '_std'):
                    key = f'{mk}{suffix}'
                    v = r.get(key)
                    if isinstance(v, float) and not math.isnan(v):
                        out[key] = f'{v:.6f}'
                    elif isinstance(v, float):
                        out[key] = ''
            writer.writerow(out)


def save_summary_latex(rows, path):
    """Save campaign summary as a LaTeX booktabs table."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sweep_keys = _detect_sweep_keys(rows)
    sweep_cols = [_sweep_col_name(k) for k in sweep_keys]

    data_cols = ['Dataset', 'Method', 'Classifier'] + sweep_cols
    metric_cols_display = ['overall\\_acc', 'balanced\\_acc', 'many\\_acc', 'medium\\_acc', 'few\\_acc']

    col_spec = 'l' * len(data_cols) + 'c' * len(metric_cols_display)
    header = ' & '.join(data_cols + metric_cols_display) + r' \\'

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Benchmark Results}',
        rf'\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        header,
        r'\midrule',
    ]

    prev_ds = None
    for r in rows:
        ds = r.get('dataset', '')
        ds_col = ds if ds != prev_ds else ''
        prev_ds = ds
        method = r.get('method', '')
        clf = r.get('classifier', '')

        parts = [ds_col, method, clf]
        sp = r.get('sweep_params') or {}
        for raw_k in sweep_keys:
            parts.append(str(sp.get(raw_k, '')))

        for mk in METRIC_KEYS:
            mean_v = r.get(f'{mk}_mean')
            std_v = r.get(f'{mk}_std')
            if mean_v is not None and not (isinstance(mean_v, float) and math.isnan(mean_v)):
                if std_v is not None and not (isinstance(std_v, float) and math.isnan(std_v)):
                    parts.append(f'{mean_v:.4f}$\\pm${std_v:.3f}')
                else:
                    parts.append(f'{mean_v:.4f}')
            else:
                parts.append('--')

        lines.append(' & '.join(parts) + r' \\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
