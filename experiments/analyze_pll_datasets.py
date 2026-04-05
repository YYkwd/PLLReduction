#!/usr/bin/env python3
"""Analyze PLL datasets under datasets/ for class imbalance & candidate ambiguity.

Outputs JSON (full report) + CSV (flat scalars) for SR/CB experiment planning.

Usage:
    python experiments/analyze_pll_datasets.py
    python experiments/analyze_pll_datasets.py --data-dir datasets --out-dir results/dataset_analysis
    python experiments/analyze_pll_datasets.py --cand-threshold 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pll.analysis.imbalance_metrics import to_json_safe
from pll.analysis.loaders import LoadSkip, discover_dataset_paths, try_load_path
from pll.analysis.report_builder import build_report, report_to_flat_row


def _print_table(rows: list[dict], keys: list[str]) -> None:
    if not rows:
        print("(no rows)")
        return
    widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    header = " | ".join(k.ljust(widths[k]) for k in keys)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r.get(k, "")).ljust(widths[k]) for k in keys))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PLL dataset imbalance & candidate ambiguity analysis",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets",
        help="Root directory containing .mat files and/or bundle subdirs",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/dataset_analysis",
        help="Output directory for JSON and CSV reports",
    )
    parser.add_argument(
        "--cand-threshold",
        type=float,
        default=0.5,
        help="Threshold on partial_target for treating a class as candidate (default 0.5)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less terminal output (still writes files)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates, notes = discover_dataset_paths(data_dir)
    if not args.quiet:
        for n in notes:
            print(f"[note] {n}")

    reports: list[dict] = []
    skips: list[dict] = []

    for path in candidates:
        loaded, skip = try_load_path(path)
        if skip is not None:
            skips.append(
                {
                    "path": skip.path,
                    "reason": skip.reason,
                    "hint": skip.hint,
                }
            )
            if not args.quiet:
                print(f"[skip] {skip.path}\n       reason: {skip.reason}")
            continue

        assert loaded is not None
        rep = build_report(
            loaded.name,
            loaded.X,
            loaded.partial_target,
            loaded.target,
            loaded.source_path,
            loaded.format,
            cand_threshold=args.cand_threshold,
        )
        reports.append(rep)

    # JSON full report
    payload = {
        "generated_at": ts,
        "data_dir": str(data_dir.resolve()),
        "cand_threshold": args.cand_threshold,
        "reports": reports,
        "skipped": skips,
    }
    json_path = out_dir / "pll_dataset_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(payload), f, indent=2, ensure_ascii=False)

    # CSV flat
    flat_rows = [report_to_flat_row(r) for r in reports]
    csv_path = out_dir / "pll_dataset_summary.csv"
    if flat_rows:
        fieldnames = sorted({k for row in flat_rows for k in row.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in flat_rows:
                w.writerow(row)
    else:
        csv_path.write_text("dataset,note\n", encoding="utf-8")

    skips_path = out_dir / "skipped_paths.json"
    with open(skips_path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(skips), f, indent=2, ensure_ascii=False)

    if not args.quiet:
        print(f"\nSaved: {json_path}")
        print(f"Saved: {csv_path}")
        print(f"Saved: {skips_path}")

        # Terminal summary (experiment-oriented)
        print("\n=== Summary (for SR/CB experiments) ===\n")
        summary_rows = []
        for r in reports:
            if r.get("error"):
                summary_rows.append(
                    {
                        "dataset": r.get("dataset"),
                        "error": r.get("error"),
                    }
                )
                continue
            tl = r.get("true_label") or {}
            mmf = (tl.get("many_medium_few") or {})
            cand = (r.get("candidates") or {}).get("per_sample_candidate_count") or {}
            hints = r.get("experiment_hints") or {}
            summary_rows.append(
                {
                    "dataset": r["dataset"],
                    "n": r["n_samples"],
                    "d": r["n_features"],
                    "K": r["n_classes"],
                    "IR": tl.get("imbalance_ratio_Nmax_over_Nmin", ""),
                    "cv_N": round(tl.get("cv_counts", 0), 4) if tl else "",
                    "mean_cand": round(cand.get("mean", 0), 3) if cand else "",
                    "hint_CB": hints.get("use_class_balance_relevant", ""),
                    "hint_SR": hints.get("use_sample_reliability_relevant", ""),
                }
            )
        _print_table(
            summary_rows,
            [
                "dataset",
                "n",
                "d",
                "K",
                "IR",
                "cv_N",
                "mean_cand",
                "hint_CB",
                "hint_SR",
            ],
        )

        if skips:
            print(f"\n[info] {len(skips)} path(s) skipped (see skipped_paths.json)")


if __name__ == "__main__":
    main()
