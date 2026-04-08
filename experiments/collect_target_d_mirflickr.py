"""Collect target_d ablation results for Mirflickr from completed run folders."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path('results/benchmark/target_d_mirflickr/Mirflickr')


def iter_results(variant_dir: str, variant_name: str):
    base = ROOT / variant_dir
    for run_dir in sorted(base.glob('*/')):
        p = run_dir / 'results.json'
        if not p.is_file():
            continue
        data = json.loads(p.read_text(encoding='utf-8'))
        cfg = data.get('config', {})
        d = cfg.get('model', {}).get('params', {}).get('target_d')
        avg = data.get('avg_metrics', {})
        yield {
            'dataset': cfg.get('data', {}).get('name', 'Mirflickr'),
            'target_d': int(d) if d is not None else -1,
            'variant': variant_name,
            'seed': cfg.get('seed', 42),
            'mode': data.get('mode', ''),
            'overall_acc': float(avg.get('overall_acc', 0.0)),
            'balanced_acc': float(avg.get('balanced_acc', 0.0)),
            'many_acc': float(avg.get('many_acc', 0.0)),
            'medium_acc': float(avg.get('medium_acc', 0.0)),
            'few_acc': float(avg.get('few_acc', 0.0)),
            'run_dir': str(run_dir),
        }


def main() -> None:
    rows = []
    rows.extend(iter_results('sdlpp_knn_knn_propagation', 'baseline'))
    rows.extend(iter_results('sdlpp_knn_knn_propagation_SR+CB', 'sr+cb'))

    if not rows:
        raise SystemExit('No results found to collect.')

    latest = {}
    for row in rows:
        key = (row['target_d'], row['variant'])
        prev = latest.get(key)
        if prev is None or row['run_dir'] > prev['run_dir']:
            latest[key] = row

    rows = sorted(latest.values(), key=lambda r: (r['target_d'], r['variant']))

    out_dir = Path('results/analysis')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"target_d_impact_mirflickr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f'Saved: {out_path}')
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
