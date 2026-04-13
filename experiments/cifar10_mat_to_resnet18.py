"""Replace CIFAR rows in a PLL .mat with ImageNet-pretrained ResNet18 embeddings.

Expects ``data`` shaped (N, 3072) in standard CIFAR-10 MATLAB layout
(1024 red + 1024 green + 1024 blue per row, row-major 32x32).

Requires: torch, torchvision

Example (run on GPU machine; CPU is slower but works):

    python experiments/cifar10_mat_to_resnet18.py \\
        --input-mat datasets/cifar10/cifar10-lt-g100-eta0.3-s42.mat \\
        --output-mat datasets/cifar10/cifar10-lt-g100-eta0.3-s42-resnet18.mat
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio


def cifar_rows_to_nchw(X: np.ndarray) -> np.ndarray:
    """(N, 3072) -> (N, 3, 32, 32) float32."""
    if X.ndim != 2 or X.shape[1] != 3072:
        raise ValueError(f'Expected (N, 3072), got {X.shape}')
    r = X[:, :1024].reshape(-1, 32, 32)
    g = X[:, 1024:2048].reshape(-1, 32, 32)
    b = X[:, 2048:].reshape(-1, 32, 32)
    # (N, H, W, 3) -> (N, 3, H, W)
    hwc = np.stack([r, g, b], axis=-1).astype(np.float32)
    return np.transpose(hwc, (0, 3, 1, 2))


def main() -> None:
    parser = argparse.ArgumentParser(description='CIFAR .mat -> ResNet18 embedding .mat')
    parser.add_argument('--input-mat', type=str, required=True)
    parser.add_argument('--output-mat', type=str, required=True)
    parser.add_argument('--device', type=str, default=None,
                        help='cuda, cpu, or omit for auto')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    try:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as e:
        raise SystemExit(
            'Need torch and torchvision. Install with: pip install torch torchvision'
        ) from e

    in_path = Path(args.input_mat)
    out_path = Path(args.output_mat)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mat = sio.loadmat(str(in_path))
    X = np.asarray(mat['data'], dtype=np.float64)
    images = cifar_rows_to_nchw(X)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).to(device)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    feats = []
    bs = args.batch_size
    with torch.no_grad():
        for start in range(0, len(images), bs):
            chunk = images[start : start + bs]
            t = torch.from_numpy(chunk).to(device)
            t = torch.nn.functional.interpolate(
                t / 255.0, size=(224, 224), mode='bilinear', align_corners=False
            )
            t = (t - mean) / std
            z = model(t)
            feats.append(z.cpu().numpy())

    Z = np.vstack(feats).astype(np.float64)
    save = {k: mat[k] for k in mat if not k.startswith('__')}
    save['data'] = Z

    sio.savemat(str(out_path), save, do_compression=True)
    print(f'[OK] wrote {out_path}  data shape {Z.shape}')


if __name__ == '__main__':
    main()
