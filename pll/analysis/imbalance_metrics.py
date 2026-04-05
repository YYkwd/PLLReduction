"""Class imbalance and PLL ambiguity metrics (for experiment design)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


EPS = 1e-12


def _safe_ratio(num: float, den: float) -> float:
    return float(num / (den + EPS))


@dataclass
class TrueLabelStats:
    """Statistics from ground-truth labels (n_samples,)."""

    n_samples: int
    n_classes: int
    counts: np.ndarray  # shape (n_classes,)
    classes_present: int
    n_min: int
    n_max: int
    n_mean: float
    imbalance_ratio: float  # N_max / N_min among classes with count > 0
    cv_counts: float  # std(N) / mean(N), coefficient of variation
    entropy_norm: float  # H / log(K), uniformity proxy
    hhi: float  # Herfindahl index sum p_k^2 (high = concentrated / long-tail)
    effective_num_classes: float  # exp(H), diversity analogue
    many: List[int]
    medium: List[int]
    few: List[int]
    n_samples_many: int
    n_samples_medium: int
    n_samples_few: int
    frac_classes_tail: float  # fraction of classes with count <= median count


def compute_true_label_stats(y: np.ndarray, n_classes: int) -> TrueLabelStats:
    """y: integer labels (n_samples,) in [0, n_classes)."""
    y = np.asarray(y, dtype=np.int64).ravel()
    counts = np.bincount(y, minlength=n_classes)
    present = int(np.sum(counts > 0))
    nz = counts[counts > 0]
    n_min = int(nz.min()) if len(nz) else 0
    n_max = int(nz.max()) if len(nz) else 0
    n_mean = float(np.mean(counts)) if len(counts) else 0.0
    ir = _safe_ratio(float(n_max), float(n_min)) if n_min > 0 else float('inf')
    cv = float(np.std(counts) / (np.mean(counts) + EPS))

    p = counts.astype(float) / (counts.sum() + EPS)
    h = -np.sum(p * np.log(p + EPS))
    h_max = np.log(float(n_classes))
    entropy_norm = float(h / (h_max + EPS)) if h_max > 0 else 0.0

    hhi = float(np.sum(p ** 2))
    effective_num = float(np.exp(h))

    many, medium, few = _many_medium_few_from_counts(counts)
    n_sm = int(sum(counts[list(many)]))
    n_md = int(sum(counts[list(medium)]))
    n_fw = int(sum(counts[list(few)]))
    med_c = float(np.median(counts[counts > 0])) if present else 0.0
    tail = int(np.sum((counts > 0) & (counts <= med_c))) if present else 0
    frac_tail = tail / present if present else 0.0

    return TrueLabelStats(
        n_samples=int(y.shape[0]),
        n_classes=int(n_classes),
        counts=counts,
        classes_present=present,
        n_min=n_min,
        n_max=n_max,
        n_mean=n_mean,
        imbalance_ratio=ir,
        cv_counts=cv,
        entropy_norm=entropy_norm,
        hhi=hhi,
        effective_num_classes=effective_num,
        many=many,
        medium=medium,
        few=few,
        n_samples_many=n_sm,
        n_samples_medium=n_md,
        n_samples_few=n_fw,
        frac_classes_tail=frac_tail,
    )


def _many_medium_few_from_counts(counts: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
    """Same rule as pll.eval.splits.get_many_medium_few_splits but from counts."""
    n_classes = len(counts)
    sorted_idx = np.argsort(-counts)
    n = len(sorted_idx)
    m = (n + 2) // 3
    f = (2 * n + 2) // 3
    many = sorted_idx[:m].tolist()
    medium = sorted_idx[m:f].tolist()
    few = sorted_idx[f:].tolist()
    return many, medium, few


@dataclass
class CandidateStats:
    """PLL candidate set statistics."""

    cand_per_sample: np.ndarray  # (n_samples,) count of positive entries per column
    mean_cand: float
    std_cand: float
    min_cand: int
    max_cand: int
    median_cand: float
    frac_singleton: float  # exactly one candidate
    histogram: Dict[str, int]  # bucket -> count
    per_class_as_candidate: np.ndarray  # (n_classes,) how often class k appears in candidate set


def compute_candidate_stats(
    partial_target: np.ndarray,
    threshold: float = 0.5,
) -> CandidateStats:
    """partial_target: (n_classes, n_samples), binary or soft."""
    pt = np.asarray(partial_target, dtype=float)
    binary = pt > threshold
    cand = np.sum(binary, axis=0).astype(int)
    n_classes = pt.shape[0]
    per_class = np.sum(binary, axis=1).astype(int)

    hist: Dict[str, int] = {}
    for c in range(int(cand.min()), int(cand.max()) + 1):
        hist[str(c)] = int(np.sum(cand == c))

    return CandidateStats(
        cand_per_sample=cand,
        mean_cand=float(np.mean(cand)),
        std_cand=float(np.std(cand)),
        min_cand=int(cand.min()),
        max_cand=int(cand.max()),
        median_cand=float(np.median(cand)),
        frac_singleton=float(np.mean(cand == 1)),
        histogram=hist,
        per_class_as_candidate=per_class,
    )


def to_json_safe(obj: Any) -> Any:
    """Convert numpy / dataclass tree to JSON-serializable types."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(x) for x in obj]
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
        return to_json_safe(asdict(obj))
    return str(obj)
