"""Ablation on target_d impact for lost dataset (baseline vs sr+cb)."""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_single import load_config, run

D_LIST = [8, 13, 20, 36, 64]
VARIANTS = {
    'baseline': {
        'use_sample_reliability': False,
        'use_class_balance': False,
    },
    'sr+cb': {
        'use_sample_reliability': True,
        'use_class_balance': True,
        'r_min': 0.1,
        'warmup': 0,
        'alpha': 0.5,
    },
}


def main() -> None:
    rows = []

    for d in D_LIST:
        for variant_name, variant_params in VARIANTS.items():
            cfg = load_config(None)
            cfg['seed'] = 42
            cfg['data']['name'] = 'lost'
            cfg['model']['name'] = 'sdlpp'
            cfg['model']['params']['target_d'] = d
            cfg['disambig']['params'].update(variant_params)
            cfg['output']['dir'] = 'results/benchmark/target_d_lost'
            cfg['_run_meta'] = {
                'run_id': f"td_lost_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{d}_{variant_name}",
                'script_name': 'target_d_impact_lost',
                'mode': 'full',
            }

            print(f"\n=== running d={d}, variant={variant_name} ===")
            avg = run(cfg)
            rows.append(
                {
                    'dataset': 'lost',
                    'target_d': d,
                    'variant': variant_name,
                    'seed': 42,
                    'overall_acc': float(avg['overall_acc']),
                    'balanced_acc': float(avg['balanced_acc']),
                    'many_acc': float(avg['many_acc']),
                    'medium_acc': float(avg['medium_acc']),
                    'few_acc': float(avg['few_acc']),
                }
            )

    out_dir = Path('results/analysis')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"target_d_impact_lost_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_path}")
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
