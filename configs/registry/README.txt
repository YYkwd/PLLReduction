Registry (versioned under configs/registry/)
============================================

Purpose
-------
- best_runs.jsonl: append-only audit log of "best" or notable runs (machine-readable).
- best_configs.yaml: human-editable summary of recommended params per dataset / experiment family.
- summary_index.csv: index of benchmark/ablation session summaries (path, time, script).

Typical workflow
----------------
1. After a batch finishes, optionally append a row to summary_index.csv (or use experiments/tools/update_registry.py).
2. Prefer a dry run first (no writes):
     python experiments/tools/update_registry.py --summary path/to/summary.csv --metric balanced_acc
3. To persist:
     python experiments/tools/update_registry.py --summary path/to/summary.csv --metric balanced_acc --append
   This appends to best_runs.jsonl, merges new rows into best_configs.yaml (existing entries kept),
   appends summary_index.csv, and writes best_configs.yaml.bak before overwriting the YAML.

4. Keep configs/default.yaml as the single source for default training YAML; registry files cite evidence (config_hash + paths) only.

Acceptance tests
----------------
  python -m unittest experiments.tests.test_acceptance_config_registry -v

Note: results/ is usually gitignored; this directory is meant to be committed so tracking survives clean result wipes.
