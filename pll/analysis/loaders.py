"""Discover and load PLL datasets from .mat / bundle / heuristic CSV."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import scipy.io as sio
except ImportError:
    sio = None  # type: ignore


@dataclass
class LoadedPLL:
    """Unified representation after a successful load."""

    name: str
    X: np.ndarray
    partial_target: np.ndarray
    target: Optional[np.ndarray]
    source_path: str
    format: str  # 'mat' | 'npy_bundle' | 'csv_bundle'
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadSkip:
    """Record when a path could not be loaded (non-fatal)."""

    path: str
    reason: str
    hint: str = ""


LoaderFn = Callable[[Path], Tuple[Optional[LoadedPLL], Optional[str]]]


def discover_dataset_paths(data_dir: Path) -> Tuple[List[Path], List[str]]:
    """Return candidate paths (files or dirs) and informational notes."""
    data_dir = Path(data_dir)
    notes: List[str] = []
    if not data_dir.is_dir():
        return [], [f"Data directory not found: {data_dir}"]

    candidates: List[Path] = []

    # 1) *.mat in root
    for p in sorted(data_dir.glob("*.mat")):
        candidates.append(p)

    # 2) Subdirectories that look like dataset bundles (no .mat inside root name)
    for sub in sorted(data_dir.iterdir()):
        if sub.is_dir() and not sub.name.startswith("."):
            if _looks_like_bundle(sub):
                candidates.append(sub)

    # 3) Loose .csv / .txt (usually need bundle; we still list for clear skip messages)
    for pattern in ("*.csv", "*.txt"):
        for p in sorted(data_dir.glob(pattern)):
            if p.is_file() and not p.name.startswith("."):
                candidates.append(p)

    if not candidates:
        notes.append(
            f"No datasets found under {data_dir}. Expected *.mat or subdirs with "
            "data.npy/X.npy + partial_target.npy (+ optional target.npy)."
        )
    return candidates, notes


def _looks_like_bundle(d: Path) -> bool:
    names = {x.name.lower() for x in d.iterdir() if x.is_file()}
    has_pt = any(
        n in names
        for n in ("partial_target.npy", "partial_target.csv", "partial_target.txt")
    )
    has_data = any(
        n in names for n in ("data.npy", "x.npy", "data.csv", "x.csv")
    )
    return has_pt and has_data


def try_load_path(path: Path) -> Tuple[Optional[LoadedPLL], Optional[LoadSkip]]:
    """Try all registered loaders until one succeeds."""
    path = Path(path)
    loaders: List[LoaderFn] = [
        _try_load_mat,
        _try_load_npy_bundle,
        _try_load_csv_bundle,
    ]
    errors: List[str] = []
    for loader in loaders:
        loaded, err = loader(path)
        if loaded is not None:
            return loaded, None
        if err:
            errors.append(err)

    if path.is_file():
        _, msg = try_load_single_csv_txt(path)
        if msg:
            errors.append(msg)

    hint = (
        "For .mat: need variables `data`, `partial_target` (and optional `target`). "
        "For bundle dir: place data.npy + partial_target.npy (+ target.npy), "
        "or data.csv + partial_target.csv (+ optional target.csv)."
    )
    return None, LoadSkip(
        str(path),
        "; ".join(errors) if errors else "unsupported format",
        hint,
    )


def _try_load_mat(path: Path) -> Tuple[Optional[LoadedPLL], Optional[str]]:
    if path.suffix.lower() != ".mat" or not path.is_file():
        return None, None
    if sio is None:
        return None, "scipy.io.loadmat unavailable (install scipy)"
    try:
        mat = sio.loadmat(str(path))
    except Exception as e:
        return None, f"mat read failed: {e}"

    keys = [k for k in mat.keys() if not k.startswith("__")]
    if "data" not in mat or "partial_target" not in mat:
        return None, f"mat missing data/partial_target (keys: {keys[:12]}...)"

    X = np.asarray(mat["data"], dtype=float)
    pt = mat["partial_target"]
    if hasattr(pt, "toarray"):
        pt = pt.toarray()
    pt = np.asarray(pt, dtype=float)

    tgt = mat.get("target", None)
    if tgt is not None:
        if hasattr(tgt, "toarray"):
            tgt = tgt.toarray()
        tgt = np.asarray(tgt, dtype=float)

    name = path.stem
    return (
        LoadedPLL(
            name=name,
            X=X,
            partial_target=pt,
            target=tgt,
            source_path=str(path.resolve()),
            format="mat",
            meta={"mat_keys": keys},
        ),
        None,
    )


def _try_load_npy_bundle(path: Path) -> Tuple[Optional[LoadedPLL], Optional[str]]:
    if not path.is_dir():
        return None, None

    def _find(name_options: Tuple[str, ...]) -> Optional[Path]:
        for n in name_options:
            p = path / n
            if p.is_file():
                return p
        return None

    f_data = _find(("data.npy", "X.npy", "x.npy"))
    f_pt = _find(("partial_target.npy",))
    if f_data is None or f_pt is None:
        return None, None

    try:
        X = np.load(f_data)
        pt = np.load(f_pt)
    except Exception as e:
        return None, f"npy load failed: {e}"

    f_tgt = _find(("target.npy",))
    tgt = None
    if f_tgt is not None:
        tgt = np.load(f_tgt)

    return (
        LoadedPLL(
            name=path.name,
            X=np.asarray(X, dtype=float),
            partial_target=np.asarray(pt, dtype=float),
            target=np.asarray(tgt, dtype=float) if tgt is not None else None,
            source_path=str(path.resolve()),
            format="npy_bundle",
            meta={},
        ),
        None,
    )


def _try_load_csv_bundle(path: Path) -> Tuple[Optional[LoadedPLL], Optional[str]]:
    if not path.is_dir():
        return None, None

    def _find(name_options: Tuple[str, ...]) -> Optional[Path]:
        for n in name_options:
            p = path / n
            if p.is_file():
                return p
        return None

    f_data = _find(("data.csv", "X.csv", "x.csv"))
    f_pt = _find(("partial_target.csv",))
    if f_data is None or f_pt is None:
        return None, None

    try:
        X = _read_numeric_2d_csv(f_data)
        pt = _read_numeric_2d_csv(f_pt)
    except Exception as e:
        return None, f"csv bundle read failed: {e}"

    f_tgt = _find(("target.csv",))
    tgt = None
    if f_tgt is not None:
        tgt = _read_numeric_2d_csv(f_tgt)

    return (
        LoadedPLL(
            name=path.name,
            X=np.asarray(X, dtype=float),
            partial_target=np.asarray(pt, dtype=float),
            target=np.asarray(tgt, dtype=float) if tgt is not None else None,
            source_path=str(path.resolve()),
            format="csv_bundle",
            meta={},
        ),
        None,
    )


def _read_numeric_2d_csv(path: Path) -> np.ndarray:
    """Load CSV as float matrix (no header)."""
    try:
        import numpy as _np

        return _np.loadtxt(str(path), delimiter=",", dtype=float)
    except Exception:
        pass
    try:
        import pandas as pd

        df = pd.read_csv(path, header=None)
        return df.values.astype(float)
    except ImportError:
        raise RuntimeError("need numpy.loadtxt or pandas to read CSV") from None


def try_load_single_csv_txt(path: Path) -> Tuple[Optional[LoadedPLL], Optional[str]]:
    """Best-effort: single file with one matrix (rarely sufficient for full PLL)."""
    if not path.is_file():
        return None, None
    suf = path.suffix.lower()
    if suf not in (".csv", ".txt"):
        return None, None

    try:
        if suf == ".csv":
            arr = _read_numeric_2d_csv(path)
        else:
            arr = np.loadtxt(str(path), dtype=float)
    except Exception as e:
        return None, str(e)

    return (
        None,
        f"single-file {suf}: loaded shape {arr.shape} but cannot infer PLL roles "
        "(need data + partial_target in .mat or bundle). Use directory bundle.",
    )
