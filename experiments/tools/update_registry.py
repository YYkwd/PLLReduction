"""Append best-per-dataset records from a summary.csv into configs/registry/."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

# experiments/tools/ -> project root
ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY = ROOT / 'configs' / 'registry'


def _load_summary_rows(path: Path) -> list:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def best_per_dataset(rows, metric: str):
    """Return dict dataset -> row with max metric (skip errors)."""
    best = {}
    for row in rows:
        if row.get('error'):
            continue
        if metric not in row or row[metric] == '':
            continue
        try:
            v = float(row[metric])
        except ValueError:
            continue
        ds = row.get('dataset', '')
        if not ds:
            continue
        if ds not in best or v > best[ds][0]:
            best[ds] = (v, row)
    return {ds: pair[1] for ds, pair in best.items()}


def append_jsonl(path: Path, records: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def merge_best_configs_yaml(entries_to_add: list):
    try:
        import yaml
    except ImportError:
        print('pyyaml not installed; skip best_configs.yaml merge (jsonl still updated)')
        return
    path = REGISTRY / 'best_configs.yaml'
    if path.is_file() and path.stat().st_size > 0:
        shutil.copy2(path, path.with_suffix('.yaml.bak'))
    with open(path, encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    data.setdefault('schema_version', 1)
    data.setdefault('entries', [])
    existing_ids = {e.get('id') for e in data['entries'] if isinstance(e, dict)}
    for e in entries_to_add:
        eid = e.get('id')
        if eid and eid not in existing_ids:
            data['entries'].append(e)
            existing_ids.add(eid)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def append_summary_index(summary_path: Path, campaign: str, n_rows: int, notes: str):
    path = REGISTRY / 'summary_index.csv'
    session_dir = summary_path.parent
    recorded = datetime.now(timezone.utc).isoformat(timespec='seconds')
    manifest = session_dir / 'session_manifest.json'
    script_name, cfg_path = '', ''
    if manifest.is_file():
        try:
            meta = json.loads(manifest.read_text(encoding='utf-8'))
            script_name = meta.get('script_name', '')
            cfg_path = str(meta.get('config_path') or '')
        except json.JSONDecodeError:
            pass
    row = {
        'recorded_at': recorded,
        'session_dir': str(session_dir.resolve().relative_to(ROOT)),
        'script_name': script_name,
        'campaign': campaign,
        'summary_csv': str(summary_path.resolve().relative_to(ROOT)),
        'config_path': cfg_path,
        'n_rows': str(n_rows),
        'notes': notes.replace(',', ';'),
    }
    write_header = not path.is_file() or path.stat().st_size == 0
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
            extrasaction='ignore',
        )
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description='Update configs/registry from a summary CSV')
    ap.add_argument('--summary', type=str, required=True, help='Path to summary.csv')
    ap.add_argument('--metric', type=str, default='balanced_acc')
    ap.add_argument('--campaign', type=str, default='', help='Campaign label, e.g. sr_cb_ablation')
    ap.add_argument(
        '--experiment-family', type=str, default='',
        dest='experiment_family',
        help='Tag for best_configs entries',
    )
    ap.add_argument(
        '--append', action='store_true',
        help='Write to best_runs.jsonl, best_configs.yaml, summary_index.csv',
    )
    ap.add_argument('--notes', type=str, default='')
    args = ap.parse_args()

    summary_path = Path(args.summary).resolve()
    rows = _load_summary_rows(summary_path)
    winners = best_per_dataset(rows, args.metric)
    iso = datetime.now(timezone.utc).isoformat(timespec='seconds')
    evidence_rel = str(summary_path.relative_to(ROOT))

    jsonl_recs = []
    yaml_entries = []
    for ds, row in sorted(winners.items()):
        ch = row.get('config_hash', '')
        method = row.get('method', '')
        try:
            val = float(row[args.metric])
        except (TypeError, ValueError):
            val = None
        jsonl_recs.append({
            'recorded_at': iso,
            'campaign': args.campaign,
            'experiment_family': args.experiment_family,
            'dataset': ds,
            'method': method,
            'metric': args.metric,
            'value': val,
            'config_hash': ch,
            'run_id': row.get('run_id', ''),
            'seed': row.get('seed', ''),
            'evidence_summary_csv': evidence_rel,
        })
        safe_id = f"{args.campaign or 'run'}_{ds}_{method}_{ch[:8] if ch else 'nohash'}".replace(
            ' ', '_').replace('/', '_')
        yaml_entries.append({
            'id': safe_id,
            'experiment_family': args.experiment_family or args.campaign,
            'dataset': ds,
            'method': method,
            'metric': args.metric,
            'value': val,
            'config_hash': ch or None,
            'evidence_summary_csv': evidence_rel,
            'recorded_at': iso,
            'notes': args.notes or None,
        })

    print(f'Best per dataset ({args.metric}): {len(winners)} datasets')
    for ds, row in sorted(winners.items()):
        print(f"  {ds}: {row.get('method')} -> {row.get(args.metric)} hash={row.get('config_hash', '')[:16]}...")

    if args.append:
        append_jsonl(REGISTRY / 'best_runs.jsonl', jsonl_recs)
        merge_best_configs_yaml(yaml_entries)
        append_summary_index(
            summary_path,
            args.campaign,
            len(rows),
            args.notes,
        )
        print(f'Appended to {REGISTRY / "best_runs.jsonl"} and updated registry files.')
    else:
        print('Dry run (pass --append to write files)')


if __name__ == '__main__':
    main()
