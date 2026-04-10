"""Generate long-tailed PLL CIFAR10 datasets from MATLAB CIFAR10 batches.

Two candidate-label generation modes:
  --eta  (recommended): each negative label independently flipped with
         probability eta.  Candidate set size ~ 1 + Binomial(C-1, eta).
         Standard LT-PLL protocol (HTCF, LSPL, etc.).
  --r    (legacy):      exactly r false-positive labels per sample.

Long-tail sampling follows n_j = n1 * gamma^(-j/(C-1)), j=0..C-1.

Output .mat format (compatible with pll/data/loader.py):
- data: (n_samples, n_features)
- target: (n_classes, n_samples), one-hot
- partial_target: (n_classes, n_samples), binary candidate labels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio


def _load_one_batch(batch_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mat = sio.loadmat(str(batch_path))
    keys = [k for k in mat.keys() if not k.startswith('__')]

    if 'data' not in mat or 'labels' not in mat:
        print(f"[ERROR] Failed reading standard keys in {batch_path}")
        print(f"[ERROR] Expected keys: ['data', 'labels']")
        print(f"[ERROR] Actual keys: {keys}")
        raise KeyError(f"Missing 'data'/'labels' in {batch_path.name}")

    X = np.asarray(mat['data'])
    y = np.asarray(mat['labels']).ravel().astype(int)

    if X.ndim != 2:
        raise ValueError(f"Unexpected data shape in {batch_path.name}: {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Unexpected labels shape in {batch_path.name}: {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Sample mismatch in {batch_path.name}: data={X.shape[0]} labels={y.shape[0]}"
        )

    return X, y


def load_cifar10_train(cifar_mat_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for i in range(1, 6):
        p = cifar_mat_dir / f'data_batch_{i}.mat'
        if not p.is_file():
            raise FileNotFoundError(f"Missing CIFAR batch file: {p}")
        X_i, y_i = _load_one_batch(p)
        xs.append(X_i)
        ys.append(y_i)

    X = np.vstack(xs)
    y = np.concatenate(ys)

    if X.shape[0] != 50000:
        print(f"[WARN] CIFAR10 train size expected 50000, got {X.shape[0]}")

    return X.astype(np.float64), y.astype(int)


def long_tail_counts(n1: int, gamma: float, n_classes: int = 10) -> np.ndarray:
    """Compute class counts for labels 0..9 with label-0 as head.

    gamma is defined as n_head / n_tail (> 1).
    count[j] = n1 * gamma^(-j/(C-1)), j in [0, C-1]
    """
    if gamma <= 0:
        raise ValueError('gamma must be > 0')
    if n_classes <= 1:
        raise ValueError('n_classes must be > 1')

    j = np.arange(n_classes, dtype=float)
    raw = n1 * np.power(gamma, -j / (n_classes - 1))
    cnt = np.rint(raw).astype(int)

    cnt[0] = int(n1)
    cnt[cnt < 1] = 1

    # Ensure non-increasing counts by class index.
    for k in range(1, n_classes):
        if cnt[k] > cnt[k - 1]:
            cnt[k] = cnt[k - 1]

    return cnt


def sample_long_tail(
    X: np.ndarray,
    y: np.ndarray,
    counts: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    idx_all = []
    n_classes = len(counts)

    for c in range(n_classes):
        idx_c = np.where(y == c)[0]
        want = int(counts[c])
        if want > len(idx_c):
            raise ValueError(
                f"Class {c}: requested {want}, but only {len(idx_c)} samples available"
            )
        pick = rng.choice(idx_c, size=want, replace=False)
        idx_all.append(pick)

    idx = np.concatenate(idx_all)
    rng.shuffle(idx)
    return X[idx], y[idx]


def build_targets(y: np.ndarray, n_classes: int = 10) -> np.ndarray:
    n = y.shape[0]
    target = np.zeros((n_classes, n), dtype=np.float64)
    target[y, np.arange(n)] = 1.0
    return target


def build_partial_target_r(
    y: np.ndarray,
    r: int,
    rng: np.random.Generator,
    n_classes: int = 10,
) -> np.ndarray:
    """Fixed-r mode: exactly r false-positive labels per sample."""
    if r < 0 or r >= n_classes:
        raise ValueError(f'r must be in [0, {n_classes - 1}], got {r}')

    n = y.shape[0]
    partial = np.zeros((n_classes, n), dtype=np.float64)
    partial[y, np.arange(n)] = 1.0

    all_classes = np.arange(n_classes)
    for i in range(n):
        neg = all_classes[all_classes != y[i]]
        if r > 0:
            false_pos = rng.choice(neg, size=r, replace=False)
            partial[false_pos, i] = 1.0

    return partial


def build_partial_target_eta(
    y: np.ndarray,
    eta: float,
    rng: np.random.Generator,
    n_classes: int = 10,
) -> np.ndarray:
    """Eta-flip mode: each negative label independently flipped with prob eta.

    Follows the standard LT-PLL protocol (e.g. HTCF, LSPL).
    Candidate set always contains the true label; each of the C-1 negative
    classes is added independently with probability eta.
    """
    if not 0 < eta < 1:
        raise ValueError(f'eta must be in (0, 1), got {eta}')

    n = y.shape[0]
    partial = np.zeros((n_classes, n), dtype=np.float64)
    partial[y, np.arange(n)] = 1.0

    flip_mask = rng.random((n_classes, n)) < eta
    true_label_mask = np.zeros_like(flip_mask)
    true_label_mask[y, np.arange(n)] = True
    flip_mask &= ~true_label_mask

    partial[flip_mask] = 1.0
    return partial


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate CIFAR10 long-tailed PLL .mat dataset')
    parser.add_argument('--cifar-mat-dir', type=str, default='datasets/cifar-10-batches-mat')
    parser.add_argument('--output-dir', type=str, default='datasets/cifar10')
    parser.add_argument('--mode', choices=['fast', 'full'], default='full')
    parser.add_argument('--n1-fast', type=int, default=3000)
    parser.add_argument('--n1-full', type=int, default=5000)
    parser.add_argument('--gamma', type=int, required=True)

    noise_group = parser.add_mutually_exclusive_group(required=True)
    noise_group.add_argument('--eta', type=float, default=None,
                             help='Flip probability for each negative label (eta-mode)')
    noise_group.add_argument('--r', type=int, default=None,
                             help='Exact number of false-positive labels (legacy r-mode)')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name', type=str, default=None,
                        help='Optional dataset name override')
    parser.add_argument('--save-meta', action='store_true',
                        help='Save sidecar metadata JSON next to output .mat')
    args = parser.parse_args()

    cifar_mat_dir = Path(args.cifar_mat_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n1 = args.n1_fast if args.mode == 'fast' else args.n1_full
    use_eta = args.eta is not None

    rng = np.random.default_rng(args.seed)

    X_all, y_all = load_cifar10_train(cifar_mat_dir)

    counts = long_tail_counts(n1=n1, gamma=float(args.gamma), n_classes=10)
    X_lt, y_lt = sample_long_tail(X_all, y_all, counts=counts, rng=rng)

    target = build_targets(y_lt, n_classes=10)

    if use_eta:
        partial_target = build_partial_target_eta(
            y_lt, eta=args.eta, rng=rng, n_classes=10)
        noise_tag = f'eta{args.eta}'
    else:
        partial_target = build_partial_target_r(
            y_lt, r=args.r, rng=rng, n_classes=10)
        noise_tag = f'r{args.r}'

    if args.name:
        dataset_name = args.name
    else:
        dataset_name = f'cifar10-lt-g{args.gamma}-{noise_tag}-s{args.seed}'
    out_path = output_dir / f'{dataset_name}.mat'

    sio.savemat(
        str(out_path),
        {
            'data': X_lt,
            'target': target,
            'partial_target': partial_target,
        },
        do_compression=True,
    )

    per_class = np.bincount(y_lt, minlength=10)
    ir = float(per_class.max() / per_class.min())
    cand_sizes = partial_target.sum(axis=0)
    avg_cands = float(cand_sizes.mean())
    min_cands = int(cand_sizes.min())
    max_cands = int(cand_sizes.max())

    if not use_eta:
        expected = int(1 + args.r)
        if min_cands != expected or max_cands != expected:
            raise ValueError(
                'Candidate cardinality mismatch: '
                f'min={min_cands}, max={max_cands}, expected={expected}'
            )

    if args.save_meta:
        meta_path = output_dir / f'{dataset_name}.meta.json'
        payload = {
            'dataset_name': dataset_name,
            'mode': args.mode,
            'seed': int(args.seed),
            'gamma': int(args.gamma),
            'noise_mode': 'eta' if use_eta else 'r',
            'eta': args.eta,
            'r': args.r,
            'n1': int(n1),
            'n_samples': int(X_lt.shape[0]),
            'n_features': int(X_lt.shape[1]),
            'n_classes': 10,
            'class_counts': per_class.tolist(),
            'imbalance_ratio': ir,
            'avg_candidates': avg_cands,
            'candidate_min': min_cands,
            'candidate_max': max_cands,
            'source_dir': str(cifar_mat_dir),
            'output_mat': str(out_path),
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        print(f'[INFO] Saved metadata: {meta_path}')

    noise_info = f'eta={args.eta}' if use_eta else f'r={args.r}'
    print(f'[OK] Saved: {out_path}')
    print(f'[INFO] mode={args.mode} seed={args.seed} n1={n1} gamma={args.gamma} {noise_info}')
    print(f'[INFO] class_counts={per_class.tolist()}')
    print(f'[INFO] imbalance_ratio={ir:.4f} avg_candidates={avg_cands:.4f}'
          f' min={min_cands} max={max_cands}')


if __name__ == '__main__':
    main()
