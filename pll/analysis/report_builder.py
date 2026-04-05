"""Build, print, and persist dataset analysis reports."""

import csv
import json
import numpy as np
from pathlib import Path
from typing import Optional

from .loaders import AnalysisDataset, discover_datasets, load_analysis_dataset
from .imbalance_metrics import compute_imbalance_report


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_full_report(data_dir: str = 'datasets',
                      names: Optional[list[str]] = None) -> list[dict]:
    """Build imbalance reports for all (or selected) datasets in data_dir."""
    entries = discover_datasets(data_dir)
    if names:
        name_set = set(names)
        entries = [e for e in entries if e['name'] in name_set]

    reports = []
    for entry in entries:
        try:
            ds = load_analysis_dataset(entry['name'], entry['path'])
            r = compute_imbalance_report(
                ds.X, ds.partial_target, ds.target, ds.name, ds.n_classes)
            r['file_format'] = ds.file_format
            reports.append(r)
        except Exception as e:
            print(f"[ERROR] {entry['name']}: {e}")
            reports.append({
                'dataset': entry['name'],
                'error': str(e),
            })
    return reports


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

_HDR = (
    f"{'Dataset':<18s} {'Samples':>7s} {'Feats':>6s} {'Classes':>7s} "
    f"{'IR':>7s} {'CV':>6s} {'AvgCand':>8s} {'MinC':>5s} {'MaxC':>5s} "
    f"{'Many':>5s} {'Med':>5s} {'Few':>5s}"
)


def print_summary(reports: list[dict]):
    """Print compact console table for all datasets."""
    print()
    print(_HDR)
    print('-' * len(_HDR))

    for r in reports:
        if 'error' in r:
            print(f"{r['dataset']:<18s}  ** ERROR: {r['error']}")
            continue

        cd = r.get('class_distribution')
        cs = r.get('candidate_stats', {})

        ir_str = f"{cd['imbalance_ratio']:.1f}" if cd else '?'
        cv_str = f"{cd['cv']:.3f}" if cd else '?'
        many_n = str(len(cd['many']['classes'])) if cd else '?'
        med_n = str(len(cd['medium']['classes'])) if cd else '?'
        few_n = str(len(cd['few']['classes'])) if cd else '?'

        print(
            f"{r['dataset']:<18s} "
            f"{r['n_samples']:>7d} "
            f"{r['n_features']:>6d} "
            f"{r['n_classes']:>7d} "
            f"{ir_str:>7s} "
            f"{cv_str:>6s} "
            f"{cs.get('avg_candidates', 0):>8.2f} "
            f"{cs.get('min_candidates', 0):>5d} "
            f"{cs.get('max_candidates', 0):>5d} "
            f"{many_n:>5s} "
            f"{med_n:>5s} "
            f"{few_n:>5s}"
        )

    print()


def print_detail(report: dict):
    """Print detailed per-class breakdown for one dataset."""
    name = report['dataset']
    if 'error' in report:
        print(f"\n[{name}] ERROR: {report['error']}\n")
        return

    cd = report.get('class_distribution')
    cs = report.get('candidate_stats', {})

    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    print(f"  Samples: {report['n_samples']}  |  Features: {report['n_features']}  "
          f"|  Classes: {report['n_classes']}")
    print(f"  Target format: {report['target_format']}")
    print(f"  Partial target format: {report['partial_target_format']}")

    if cd:
        print(f"\n  Class distribution:")
        print(f"    Imbalance ratio (max/min): {cd['imbalance_ratio']}")
        print(f"    Mean count: {cd['mean_count']:.1f}  |  Std: {cd['std_count']:.1f}  "
              f"|  CV: {cd['cv']:.4f}")
        print(f"    Many ({len(cd['many']['classes'])} classes, "
              f"{cd['many']['total_samples']} samples)  |  "
              f"Medium ({len(cd['medium']['classes'])} classes, "
              f"{cd['medium']['total_samples']} samples)  |  "
              f"Few ({len(cd['few']['classes'])} classes, "
              f"{cd['few']['total_samples']} samples)")

        print(f"\n    {'Class':>5s} {'Count':>7s} {'Group':>8s}  Bar")
        print(f"    {'-'*45}")
        max_count = cd['n_max'] if cd['n_max'] > 0 else 1
        for item in cd['per_class']:
            bar_len = int(30 * item['count'] / max_count)
            bar = '#' * bar_len
            print(f"    {item['class']:>5d} {item['count']:>7d} {item['group']:>8s}  {bar}")

    print(f"\n  Candidate label stats:")
    print(f"    Avg candidates/sample: {cs.get('avg_candidates', '?')}")
    print(f"    Range: [{cs.get('min_candidates', '?')}, {cs.get('max_candidates', '?')}]  "
          f"|  Median: {cs.get('median_candidates', '?')}  "
          f"|  Std: {cs.get('std_candidates', '?')}")
    print(f"    Candidate freq CV (across classes): {cs.get('candidate_freq_cv', '?')}")
    if 'avg_noise_candidates' in cs:
        print(f"    Avg noise candidates/sample: {cs['avg_noise_candidates']}")
        print(f"    True label in candidate set: {cs['true_label_covered']:.4f}")
    print()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_report(reports: list[dict], output_dir: str, formats=('json', 'csv')):
    """Save reports as JSON and/or flat CSV.

    JSON: full detail.  CSV: one row per dataset, key metrics only.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if 'json' in formats:
        path = out / 'dataset_analysis.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(reports, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        print(f"Saved JSON: {path}")

    if 'csv' in formats:
        path = out / 'dataset_analysis.csv'
        fieldnames = [
            'dataset', 'n_samples', 'n_features', 'n_classes',
            'imbalance_ratio', 'cv', 'mean_count', 'std_count',
            'many_classes', 'many_samples', 'medium_classes', 'medium_samples',
            'few_classes', 'few_samples',
            'avg_candidates', 'min_candidates', 'max_candidates',
            'avg_noise_candidates', 'true_label_covered',
        ]
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in reports:
                if 'error' in r:
                    writer.writerow({'dataset': r['dataset']})
                    continue
                cd = r.get('class_distribution') or {}
                cs = r.get('candidate_stats', {})
                row = {
                    'dataset': r['dataset'],
                    'n_samples': r['n_samples'],
                    'n_features': r['n_features'],
                    'n_classes': r['n_classes'],
                    'imbalance_ratio': cd.get('imbalance_ratio', ''),
                    'cv': cd.get('cv', ''),
                    'mean_count': cd.get('mean_count', ''),
                    'std_count': cd.get('std_count', ''),
                    'many_classes': len(cd.get('many', {}).get('classes', [])),
                    'many_samples': cd.get('many', {}).get('total_samples', ''),
                    'medium_classes': len(cd.get('medium', {}).get('classes', [])),
                    'medium_samples': cd.get('medium', {}).get('total_samples', ''),
                    'few_classes': len(cd.get('few', {}).get('classes', [])),
                    'few_samples': cd.get('few', {}).get('total_samples', ''),
                    'avg_candidates': cs.get('avg_candidates', ''),
                    'min_candidates': cs.get('min_candidates', ''),
                    'max_candidates': cs.get('max_candidates', ''),
                    'avg_noise_candidates': cs.get('avg_noise_candidates', ''),
                    'true_label_covered': cs.get('true_label_covered', ''),
                }
                writer.writerow(row)
        print(f"Saved CSV:  {path}")
