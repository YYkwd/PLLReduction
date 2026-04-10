"""Batch generate CIFAR10 long-tailed PLL datasets.

Default grid (eta mode, matching LT-PLL literature):
- gamma in {100, 150, 200, 250}
- eta in {0.3, 0.5}
- seeds in {42, 43, 44}
- mode=full with n1_full=5000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description='Batch generate CIFAR10 LT-PLL datasets')
    parser.add_argument('--script', type=str, default='experiments/generate_cifar10_lt_pll.py')
    parser.add_argument('--mode', choices=['fast', 'full'], default='full')
    parser.add_argument('--gammas', type=str, default='100,150,200,250')

    noise_group = parser.add_mutually_exclusive_group(required=True)
    noise_group.add_argument('--etas', type=str, default=None,
                             help='Comma-separated eta values (eta mode)')
    noise_group.add_argument('--rs', type=str, default=None,
                             help='Comma-separated r values (legacy r mode)')

    parser.add_argument('--seeds', type=str, default='42')
    parser.add_argument('--cifar-mat-dir', type=str, default='datasets/cifar-10-batches-mat')
    parser.add_argument('--output-dir', type=str, default='datasets/cifar10')
    parser.add_argument('--n1-fast', type=int, default=3000)
    parser.add_argument('--n1-full', type=int, default=5000)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    gammas = _parse_int_list(args.gammas)
    seeds = _parse_int_list(args.seeds)
    use_eta = args.etas is not None

    if use_eta:
        noise_vals = _parse_float_list(args.etas)
    else:
        noise_vals = _parse_int_list(args.rs)

    script_path = Path(args.script)
    if not script_path.is_file():
        raise FileNotFoundError(f'Generator script not found: {script_path}')

    jobs = list(product(gammas, noise_vals, seeds))
    total = len(jobs)
    if total == 0:
        print('[WARN] No jobs to run')
        return

    print(f'[INFO] mode={args.mode} noise={"eta" if use_eta else "r"} jobs={total}')
    for idx, (gamma, noise, seed) in enumerate(jobs, 1):
        if use_eta:
            noise_tag = f'eta{noise}'
            noise_args = ['--eta', str(noise)]
        else:
            noise_tag = f'r{int(noise)}'
            noise_args = ['--r', str(int(noise))]

        dataset_name = f'cifar10-lt-g{gamma}-{noise_tag}-s{seed}'
        cmd = [
            sys.executable,
            str(script_path),
            '--cifar-mat-dir', args.cifar_mat_dir,
            '--output-dir', args.output_dir,
            '--mode', args.mode,
            '--n1-fast', str(args.n1_fast),
            '--n1-full', str(args.n1_full),
            '--gamma', str(gamma),
            *noise_args,
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
