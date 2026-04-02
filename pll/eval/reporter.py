"""Result formatting, printing, and persistence."""

import csv
import json
import numpy as np
from pathlib import Path

METRIC_KEYS = ['overall_acc', 'balanced_acc', 'many_acc', 'medium_acc', 'few_acc']


def print_metrics(metrics: dict, label: str = ''):
    prefix = f"[{label}] " if label else ""
    parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
    print(f"{prefix}{' | '.join(parts)}")


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_results(results: dict, output_dir, formats=None):
    """Save experiment results to disk.

    Supported formats: 'json', 'csv' (fold metrics table).
    """
    formats = formats or ['json']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if 'json' in formats:
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    if 'csv' in formats and 'fold_metrics' in results:
        rows = results['fold_metrics']
        fieldnames = ['fold'] + list(rows[0].keys())
        with open(output_dir / 'fold_metrics.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, m in enumerate(rows):
                row = {'fold': i + 1}
                row.update({k: f"{v:.6f}" for k, v in m.items()})
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Benchmark summary utilities
# ---------------------------------------------------------------------------

_META_KEYS = {'dataset', 'method', 'disambig', 'elapsed_s', 'error'}


def _method_col(row):
    """Return the method/config identifier from a result row."""
    return row.get('method', row.get('disambig', ''))


def _collect_metric_keys(rows):
    """Return metric keys present in rows (preserve METRIC_KEYS order)."""
    if not rows:
        return []
    available = set(rows[0].keys()) - _META_KEYS
    return [k for k in METRIC_KEYS if k in available]


def print_summary_table(rows):
    """Pretty-print benchmark results as a console table."""
    keys = _collect_metric_keys(rows)
    if not keys:
        print("(no valid results)")
        return

    max_method_len = max(len(_method_col(r)) for r in rows)
    mw = max(max_method_len + 2, 14)
    hdr_fmt = f"{{:<20s}} {{:<{mw}s}}" + " {:>12s}" * len(keys)
    row_fmt = f"{{:<20s}} {{:<{mw}s}}" + " {:>12s}" * len(keys)
    print(hdr_fmt.format("Dataset", "Method", *keys))
    print("-" * (22 + mw + 13 * len(keys)))

    for r in rows:
        if 'error' in r:
            vals = ['FAIL'] * len(keys)
        else:
            vals = [f"{r[k]:.4f}" for k in keys]
        print(row_fmt.format(r['dataset'], _method_col(r), *vals))


def save_summary_csv(rows, path):
    """Save benchmark summary as CSV."""
    keys = _collect_metric_keys(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ['dataset', 'method'] + keys + ['elapsed_s']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, '') for k in fieldnames}
            out['method'] = _method_col(r)
            for k in keys:
                if k in r and not isinstance(r[k], str):
                    out[k] = f"{r[k]:.6f}"
            writer.writerow(out)


def save_summary_latex(rows, path):
    """Save benchmark summary as a LaTeX booktabs table."""
    keys = _collect_metric_keys(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    col_spec = "ll" + "c" * len(keys)
    header_labels = [k.replace('_', r'\_') for k in keys]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Benchmark Results}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Dataset & Method & " + " & ".join(header_labels) + r" \\",
        r"\midrule",
    ]

    prev_ds = None
    for r in rows:
        ds_col = r['dataset'] if r['dataset'] != prev_ds else ""
        prev_ds = r['dataset']
        if 'error' in r:
            vals = ['--'] * len(keys)
        else:
            vals = [f"{r[k]:.4f}" for k in keys]
        lines.append(f"{ds_col} & {_method_col(r)} & " + " & ".join(vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
