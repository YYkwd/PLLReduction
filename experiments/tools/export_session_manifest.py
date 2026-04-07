"""Backfill runs_index.csv (and optional minimal manifest) by scanning results.json files.

Usage:
  python experiments/tools/export_session_manifest.py --scan-root results/benchmark/sr_cb_ablation/20260101_120000
  python experiments/tools/export_session_manifest.py --scan-root results --max-depth 6
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    ap = argparse.ArgumentParser(description='Build runs_index.csv from results.json files')
    ap.add_argument('--scan-root', type=str, required=True)
    ap.add_argument('--out-dir', type=str, default=None,
                    help='Directory for runs_index.csv (default: scan-root if it is a session folder)')
    ap.add_argument('--max-depth', type=int, default=8)
    args = ap.parse_args()

    scan = Path(args.scan_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else scan
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in scan.rglob('results.json'):
        try:
            rel = p.relative_to(scan)
        except ValueError:
            continue
        if len(rel.parts) > args.max_depth:
            continue
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            continue
        cfg = data.get('config') or {}
        data_cfg = cfg.get('data') or {}
        ds = data_cfg.get('name', '')
        run_id = data.get('run_id', '')
        ch = data.get('config_hash', '')
        try:
            rel_json = str(p.resolve().relative_to(ROOT))
        except ValueError:
            rel_json = str(p)
        seed = cfg.get('seed', '')
        dis = cfg.get('disambig') or {}
        params = dis.get('params') or {}
        parts = []
        if params.get('use_sample_reliability'):
            parts.append('SR')
        if params.get('use_class_balance'):
            parts.append('CB')
        method_guess = '+'.join(parts) if parts else 'baseline'
        rows.append({
            'dataset': ds,
            'method': method_guess,
            'seed': seed,
            'run_id': run_id,
            'config_hash': ch,
            'results_json_rel': rel_json,
        })

    out_csv = out_dir / 'runs_index.csv'
    fieldnames = [
        'dataset', 'method', 'seed', 'run_id', 'config_hash', 'results_json_rel',
    ]
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    mini = {
        'schema_version': 1,
        'generated_by': 'experiments/tools/export_session_manifest.py',
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'scan_root': str(scan),
        'n_results_json': len(rows),
    }
    with open(out_dir / 'session_manifest_scanned.json', 'w', encoding='utf-8') as f:
        json.dump(mini, f, indent=2)
    print(f'Wrote {out_csv} ({len(rows)} rows) and session_manifest_scanned.json')


if __name__ == '__main__':
    main()
