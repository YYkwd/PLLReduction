"""Batch generate CIFAR10 long-tailed PLL datasets.

Default quick grid follows current decision:
- gamma in {100, 200}
- r in {1, 2, 3}
- seeds in {42, 43, 44}
- mode=fast with n1_fast=1000

Use --mode full for n1_full=3000 runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description='Batch generate CIFAR10 LT-PLL datasets')
    parser.add_argument('--script', type=str, default='experiments/generate_cifar10_lt_pll.py')
    parser.add_argument('--mode', choices=['fast', 'full'], default='fast')
    parser.add_argument('--gammas', type=str, default='100,200')
    parser.add_argument('--rs', type=str, default='1,2,3')
    parser.add_argument('--seeds', type=str, default='42,43,44')
    parser.add_argument('--cifar-mat-dir', type=str, default='datasets/cifar-10-batches-mat')
    parser.add_argument('--output-dir', type=str, default='datasets/cifar10')
    parser.add_argument('--n1-fast', type=int, default=1000)
    parser.add_argument('--n1-full', type=int, default=3000)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    gammas = _parse_int_list(args.gammas)
    rs = _parse_int_list(args.rs)
    seeds = _parse_int_list(args.seeds)

    script_path = Path(args.script)
    if not script_path.is_file():
        raise FileNotFoundError(f'Generator script not found: {script_path}')

    jobs = list(product(gammas, rs, seeds))
    total = len(jobs)
    if total == 0:
        print('[WARN] No jobs to run')
        return

    print(f'[INFO] mode={args.mode} jobs={total}')
    for idx, (gamma, r, seed) in enumerate(jobs, 1):
        dataset_name = f'cifar10-lt-g{gamma}-r{r}-{args.mode}-s{seed}'
        cmd = [
            sys.executable,
            str(script_path),
            '--cifar-mat-dir', args.cifar_mat_dir,
            '--output-dir', args.output_dir,
            '--mode', args.mode,
            '--n1-fast', str(args.n1_fast),
            '--n1-full', str(args.n1_full),
            '--gamma', str(gamma),
            '--r', str(r),
            '--seed', str(seed),
            '--name', dataset_name,
            '--save-meta',
        ]

        print(f'[{idx}/{total}] {dataset_name}')
        if args.dry_run:
            print('  ' + ' '.join(cmd))
            continue

        subprocess.run(cmd, check=True)

    print('[OK] Grid generation finished')


if __name__ == '__main__':
    main()
