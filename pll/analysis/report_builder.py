"""Assemble PLL imbalance report dict from LoadedPLL."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pll.analysis.imbalance_metrics import (
    compute_candidate_stats,
    compute_true_label_stats,
    to_json_safe,
)


def extract_y_from_target(target: Optional[np.ndarray], n_classes: int) -> Optional[np.ndarray]:
    """Return integer labels (n_samples,) or None if unavailable."""
    if target is None:
        return None
    t = np.asarray(target, dtype=float)
    if t.ndim == 1:
        return t.astype(np.int64)
    if t.ndim == 2:
        if t.shape[0] == n_classes:
            return np.argmax(t, axis=0).astype(np.int64)
        if t.shape[1] == n_classes:
            return np.argmax(t, axis=1).astype(np.int64)
    return None


def validate_shapes(X: np.ndarray, pt: np.ndarray) -> Optional[str]:
    n = X.shape[0]
    if pt.ndim != 2:
        return f"partial_target must be 2D, got {pt.ndim}D"
    if pt.shape[1] != n:
        return f"partial_target second dim {pt.shape[1]} != n_samples {n}"
    return None


def build_report(
    name: str,
    X: np.ndarray,
    partial_target: np.ndarray,
    target: Optional[np.ndarray],
    source_path: str,
    source_format: str,
    cand_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Full report for JSON/CSV export and terminal summary."""
    X = np.asarray(X, dtype=float)
    pt = np.asarray(partial_target, dtype=float)
    n_classes = pt.shape[0]
    n_samples = X.shape[0]
    n_features = X.shape[1]

    err = validate_shapes(X, pt)
    if err:
        return {
            "dataset": name,
            "source_path": source_path,
            "source_format": source_format,
            "error": err,
        }

    y = extract_y_from_target(target, n_classes)

    rep: Dict[str, Any] = {
        "dataset": name,
        "source_path": source_path,
        "source_format": source_format,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "target_available": y is not None,
        "target_shape": list(target.shape) if target is not None else None,
        "partial_target_shape": list(pt.shape),
        "partial_target_value_range": {
            "min": float(np.min(pt)),
            "max": float(np.max(pt)),
        },
    }

    tls = None
    if y is not None:
        tls = compute_true_label_stats(y, n_classes)
        ir = tls.imbalance_ratio
        rep["true_label"] = {
            "classes_present": tls.classes_present,
            "count_per_class": tls.counts.tolist(),
            "n_min": tls.n_min,
            "n_max": tls.n_max,
            "n_mean": round(tls.n_mean, 4),
            "imbalance_ratio_Nmax_over_Nmin": round(float(ir), 4)
            if np.isfinite(ir)
            else None,
            "cv_counts": round(tls.cv_counts, 6),
            "entropy_norm": round(tls.entropy_norm, 6),
            "hhi": round(tls.hhi, 6),
            "effective_num_classes": round(tls.effective_num_classes, 4),
            "many_medium_few": {
                "many_class_ids": tls.many,
                "medium_class_ids": tls.medium,
                "few_class_ids": tls.few,
                "n_samples_in_many_classes": tls.n_samples_many,
                "n_samples_in_medium_classes": tls.n_samples_medium,
                "n_samples_in_few_classes": tls.n_samples_few,
            },
            "frac_classes_in_tail_by_median_count": round(tls.frac_classes_tail, 6),
        }
    else:
        rep["true_label"] = None
        rep["warnings"] = [
            "No ground-truth target: class-imbalance metrics skipped (candidate stats only).",
        ]

    cs = compute_candidate_stats(pt, threshold=cand_threshold)
    rep["candidates"] = {
        "threshold_for_positive": cand_threshold,
        "per_sample_candidate_count": {
            "mean": round(cs.mean_cand, 6),
            "std": round(cs.std_cand, 6),
            "min": cs.min_cand,
            "max": cs.max_cand,
            "median": round(cs.median_cand, 6),
            "histogram_count_by_k": cs.histogram,
            "fraction_singleton": round(cs.frac_singleton, 6),
        },
        "per_class_as_candidate_frequency": cs.per_class_as_candidate.tolist(),
        "mean_frequency_per_class": float(np.mean(cs.per_class_as_candidate)),
    }

    # Cross-hint: expected ambiguity vs imbalance (for experiment planning)
    if tls is not None:
        ir = tls.imbalance_ratio
        ir_ok = np.isfinite(ir)
        rep["experiment_hints"] = {
            "use_class_balance_relevant": bool(
                tls.cv_counts > 0.15 or (ir_ok and ir > 3.0)
            ),
            "use_sample_reliability_relevant": bool(cs.mean_cand > 1.5),
            "notes": [
                "cv_counts high -> class reweighting (CB) may matter",
                "mean_cand high -> label ambiguity high; SR may matter",
            ],
        }

    return rep


def report_to_flat_row(rep: Dict[str, Any]) -> Dict[str, Any]:
    """One flat dict per dataset for CSV (scalar columns only)."""
    if rep.get("error"):
        return {
            "dataset": rep.get("dataset"),
            "error": rep["error"],
            "source_path": rep.get("source_path"),
        }

    row: Dict[str, Any] = {
        "dataset": rep["dataset"],
        "n_samples": rep["n_samples"],
        "n_features": rep["n_features"],
        "n_classes": rep["n_classes"],
        "target_available": rep["target_available"],
        "source_format": rep["source_format"],
    }

    tl = rep.get("true_label")
    if tl:
        row["imbalance_ratio"] = tl.get("imbalance_ratio_Nmax_over_Nmin")
        row["cv_class_counts"] = tl.get("cv_counts")
        row["entropy_norm"] = tl.get("entropy_norm")
        row["hhi"] = tl.get("hhi")
        row["effective_num_classes"] = tl.get("effective_num_classes")
        row["n_samples_many_bucket"] = tl["many_medium_few"]["n_samples_in_many_classes"]
        row["n_samples_medium_bucket"] = tl["many_medium_few"]["n_samples_in_medium_classes"]
        row["n_samples_few_bucket"] = tl["many_medium_few"]["n_samples_in_few_classes"]
        row["frac_classes_tail"] = tl.get("frac_classes_in_tail_by_median_count")

    cand = rep.get("candidates", {})
    ps = cand.get("per_sample_candidate_count", {})
    row["mean_cand_per_sample"] = ps.get("mean")
    row["max_cand_per_sample"] = ps.get("max")
    row["frac_singleton_cand"] = ps.get("fraction_singleton")

    hints = rep.get("experiment_hints", {})
    if hints:
        row["hint_cb_relevant"] = hints.get("use_class_balance_relevant")
        row["hint_sr_relevant"] = hints.get("use_sample_reliability_relevant")

    return row
