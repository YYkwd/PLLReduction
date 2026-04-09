"""Analyze PLL datasets for class imbalance and candidate label properties.

Usage:
    python experiments/analyze_datasets.py                          # all datasets
    python experiments/analyze_datasets.py --datasets lost MSRCv2   # specific ones
    python experiments/analyze_datasets.py --detail                 # per-class breakdown
    python experiments/analyze_datasets.py --output-dir results/analysis
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pll.analysis.report_builder import (
    build_full_report, print_summary, print_detail, save_report,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='PLL dataset imbalance analysis')
    parser.add_argument('--data-dir', type=str, default='datasets',
                        help='Directory containing dataset files')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Specific dataset names (default: all found)')
    parser.add_argument('--detail', action='store_true',
                        help='Print per-class breakdown for each dataset')
    parser.add_argument('--output-dir', type=str, default='results/analysis',
                        help='Directory to save JSON/CSV reports')
    parser.add_argument('--no-save', action='store_true',
                        help='Skip saving reports to disk')
    args = parser.parse_args()

    reports = build_full_report(args.data_dir, args.datasets)

    if not reports:
        print(f"No datasets found in '{args.data_dir}/'")
        return

    print_summary(reports)

    if args.detail:
        for r in reports:
            print_detail(r)

    if not args.no_save:
        save_report(reports, args.output_dir)


if __name__ == '__main__':
    main()
