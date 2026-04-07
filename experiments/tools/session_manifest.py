"""Session-level artifacts: manifest + runs index for benchmark/ablation batches."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, List, Mapping, Optional


def config_copy_for_manifest(cfg: Mapping[str, Any]) -> dict:
    """Deep copy of config without per-run _run_meta (for session snapshots)."""
    c = copy.deepcopy(dict(cfg))
    c.pop('_run_meta', None)
    return c


def write_session_manifest(
    out_dir: Path | str,
    *,
    script_name: str,
    campaign: str,
    mode_fast: bool,
    base_config: Mapping[str, Any],
    env_snapshot: Mapping[str, Any],
    model_defaults_snapshot: Mapping[str, Any],
    started_at_iso: str,
    ended_at_iso: str,
    wall_seconds: float,
    config_path: Optional[str],
    n_runs: int,
    n_success: int,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema_version': 1,
        'script_name': script_name,
        'campaign': campaign,
        'mode_fast': mode_fast,
        'config_path': config_path,
        'session': {
            'started_at': started_at_iso,
            'ended_at': ended_at_iso,
            'wall_seconds': round(wall_seconds, 3),
            'n_runs': n_runs,
            'n_success': n_success,
        },
        'env_snapshot': dict(env_snapshot),
        'model_defaults_at_start': copy.deepcopy(dict(model_defaults_snapshot)),
        'base_config': config_copy_for_manifest(base_config),
    }
    path = out_dir / 'session_manifest.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return path


def write_runs_index(out_dir: Path | str, rows: List[Mapping[str, Any]]) -> Path:
    """Write runs_index.csv linking summary rows to per-run results.json paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'runs_index.csv'
    fieldnames = [
        'dataset', 'method', 'seed', 'run_id', 'config_hash', 'results_json_rel',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    return path
