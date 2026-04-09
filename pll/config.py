"""Hierarchical configuration loading.

Merge order (later wins):
    base.yaml -> datasets/<name>.yaml -> methods/<name>.yaml -> CLI overrides
"""

import copy
import hashlib
import json
import os
import platform
import socket
import sys
from pathlib import Path

import numpy as np

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / 'configs'


def _load_yaml(path: Path):
    try:
        import yaml
    except ImportError:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else None


def deep_merge(base, override):
    """Recursively merge *override* into *base* (in-place)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v


def load_experiment_config(dataset_name, method_name, cli_overrides=None,
                           configs_dir=None):
    """Build a fully merged config dict.

    Parameters
    ----------
    dataset_name : str   – must match a file under configs/datasets/
    method_name  : str   – must match a file under configs/methods/
    cli_overrides : dict  – flat or nested overrides from CLI
    configs_dir  : Path   – override default configs/ location
    """
    cdir = Path(configs_dir) if configs_dir else _CONFIGS_DIR
    cfg = _load_yaml(cdir / 'base.yaml') or _base_skeleton()
    cfg = copy.deepcopy(cfg)

    ds_yaml = _load_yaml(cdir / 'datasets' / f'{dataset_name}.yaml')
    if ds_yaml:
        deep_merge(cfg, ds_yaml)
    else:
        cfg.setdefault('data', {})['name'] = dataset_name

    method_yaml = _load_yaml(cdir / 'methods' / f'{method_name}.yaml')
    if method_yaml:
        deep_merge(cfg, method_yaml)

    if cli_overrides:
        deep_merge(cfg, cli_overrides)

    resolve_warmup_by_dataset(cfg, dataset_name)
    return cfg


def resolve_warmup_by_dataset(cfg, dataset_name):
    """Replace warmup_by_dataset mapping with the concrete warmup value."""
    dp = cfg.get('disambig', {}).get('params', {})
    by_ds = dp.pop('warmup_by_dataset', None)
    if by_ds and dataset_name in by_ds:
        dp['warmup'] = by_ds[dataset_name]


def set_nested(d, dotted_path, value):
    """Set a value in a nested dict using a dotted path.

    Example: set_nested(cfg, 'disambig.params.r_min', 0.1)
    """
    keys = dotted_path.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def auto_cast(s):
    """Auto-cast a string to bool / int / float if possible."""
    if s.lower() == 'true':
        return True
    if s.lower() == 'false':
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


# ---- JSON / hashing helpers ------------------------------------------------

def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def canonical_json(data):
    """Stable JSON serialization for hashing/logging."""
    return json.dumps(data, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), default=_json_default)


def compute_config_hash(config):
    return hashlib.sha256(canonical_json(config).encode('utf-8')).hexdigest()


def collect_env_snapshot():
    snap = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'hostname': socket.gethostname(),
        'numpy': np.__version__,
    }
    try:
        import scipy; snap['scipy'] = scipy.__version__
    except Exception:
        snap['scipy'] = 'unknown'
    try:
        import sklearn; snap['sklearn'] = sklearn.__version__
    except Exception:
        snap['sklearn'] = 'unknown'
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        snap[k] = os.environ.get(k, '')
    return snap


def _base_skeleton():
    """Fallback if base.yaml is missing."""
    return {
        'seed': 42,
        'data': {'data_dir': 'datasets'},
        'preprocessing': {'method': 'zscore'},
        'eval': {
            'protocol': 'repeated_holdout',
            'n_repeats': 10,
            'test_ratio': 0.2,
            'stratified': True,
        },
        'output': {'base_dir': 'results', 'formats': ['json']},
    }
