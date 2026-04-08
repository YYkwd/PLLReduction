"""Run the missing Mirflickr target_d=64, sr+cb fast configuration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_single import apply_fast_training_overrides, load_config, run


def main() -> None:
    cfg = load_config(None)
    cfg['seed'] = 42
    cfg['data']['name'] = 'Mirflickr'
    cfg['model']['name'] = 'sdlpp'
    cfg['model']['params']['target_d'] = 64
    cfg['disambig']['params'].update(
        {
            'use_sample_reliability': True,
            'use_class_balance': True,
            'r_min': 0.1,
            'warmup': 0,
            'alpha': 0.5,
        }
    )
    cfg['output']['dir'] = 'results/benchmark/target_d_mirflickr'
    apply_fast_training_overrides(cfg)
    cfg['_run_meta'] = {
        'run_id': f"td_mirflickr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_64_sr+cb",
        'script_name': 'run_one_mirflickr_d64_sr_cb',
        'mode': 'fast',
    }

    avg = run(cfg)
    print(avg)


if __name__ == '__main__':
    main()
