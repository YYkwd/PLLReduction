"""Auto-discover and load PLL datasets from various file formats."""

import numpy as np
import scipy.io as sio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AnalysisDataset:
    """Lightweight container for dataset analysis (no preprocessing needed)."""
    name: str
    path: Path
    file_format: str
    X: np.ndarray                        # (n_samples, n_features)
    partial_target: np.ndarray           # (n_classes, n_samples) binary
    target: Optional[np.ndarray]         # (n_classes, n_samples) binary or None
    n_samples: int
    n_features: int
    n_classes: int


SUPPORTED_EXTENSIONS = {'.mat', '.csv', '.txt', '.npz'}


def discover_datasets(data_dir: str = 'datasets') -> list[dict]:
    """Scan directory for loadable dataset files.

    Returns list of {'name': ..., 'path': ..., 'format': ...} sorted by name.
    Skips hidden files and subdirectories.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root.resolve()}")

    found = []
    for p in sorted(root.iterdir()):
        if p.is_dir() or p.name.startswith('.'):
            continue
        ext = p.suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            found.append({
                'name': p.stem,
                'path': p,
                'format': ext.lstrip('.'),
            })
        else:
            print(f"[SKIP] Unsupported format: {p.name}")
    return found


def _load_mat(path: Path) -> dict:
    """Load .mat file, return raw arrays dict."""
    mat = sio.loadmat(str(path))

    def _dense(arr):
        if hasattr(arr, 'toarray'):
            return arr.toarray()
        return np.asarray(arr, dtype=float)

    result = {}
    if 'data' in mat:
        result['X'] = _dense(mat['data'])
    elif 'features' in mat:
        result['X'] = _dense(mat['features'])
    else:
        raise KeyError(f"No 'data' or 'features' key in {path.name}")

    if 'partial_target' in mat:
        result['partial_target'] = _dense(mat['partial_target'])
    elif 'partial_targets' in mat:
        result['partial_target'] = _dense(mat['partial_targets'])
    else:
        raise KeyError(f"No 'partial_target' key in {path.name}")

    target = mat.get('target', mat.get('targets', None))
    if target is not None:
        result['target'] = _dense(target)
    else:
        result['target'] = None

    return result


def _load_npz(path: Path) -> dict:
    """Load .npz file."""
    data = np.load(str(path), allow_pickle=True)
    result = {}
    for key_x in ('X', 'data', 'features'):
        if key_x in data:
            result['X'] = np.asarray(data[key_x], dtype=float)
            break
    else:
        raise KeyError(f"No feature array found in {path.name}")

    for key_pt in ('partial_target', 'partial_targets'):
        if key_pt in data:
            result['partial_target'] = np.asarray(data[key_pt], dtype=float)
            break
    else:
        raise KeyError(f"No partial_target array found in {path.name}")

    for key_t in ('target', 'targets'):
        if key_t in data:
            result['target'] = np.asarray(data[key_t], dtype=float)
            break
    else:
        result['target'] = None

    return result


def _load_csv_txt(path: Path) -> dict:
    """Best-effort CSV/TXT loader.

    Expected layout: first column = true label (int), remaining = features.
    Partial targets are not available in plain text — raises with guidance.
    """
    raise NotImplementedError(
        f"Plain CSV/TXT loading for '{path.name}' is not yet supported. "
        f"PLL datasets require a partial_target matrix which plain text files "
        f"typically don't provide. Convert to .mat or .npz first, or add a "
        f"custom loader in pll/analysis/loaders.py."
    )


_FORMAT_LOADERS = {
    'mat': _load_mat,
    'npz': _load_npz,
    'csv': _load_csv_txt,
    'txt': _load_csv_txt,
}


def load_analysis_dataset(name: str, path=None, data_dir='datasets') -> AnalysisDataset:
    """Load a single dataset for analysis.

    Parameters
    ----------
    name : dataset name (stem of filename)
    path : explicit file path (overrides data_dir + name)
    data_dir : directory to search in
    """
    if path is None:
        candidates = list(Path(data_dir).glob(f'{name}.*'))
        candidates = [c for c in candidates if c.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not candidates:
            raise FileNotFoundError(
                f"No dataset file found for '{name}' in {data_dir}/")
        path = candidates[0]
    else:
        path = Path(path)

    ext = path.suffix.lower().lstrip('.')
    loader = _FORMAT_LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"No loader for format '.{ext}' ({path.name})")

    raw = loader(path)
    X = raw['X']
    pt = raw['partial_target']

    return AnalysisDataset(
        name=name,
        path=path,
        file_format=ext,
        X=X,
        partial_target=pt,
        target=raw['target'],
        n_samples=X.shape[0],
        n_features=X.shape[1],
        n_classes=pt.shape[0],
    )
