"""Acceptance tests for config loading and registry merge (no full training runs).

Run from repo root:
  python -m unittest experiments.tests.test_acceptance_config_registry -v

Requires PyYAML for YAML-related tests; skipped otherwise.
Requires sklearn etc. for run_single import tests; skipped otherwise.
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


class TestLoadConfigFromDefaultYaml(unittest.TestCase):
    def setUp(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest('PyYAML not installed')

    def test_load_config_none_matches_default_yaml_file(self):
        from experiments.run_single import load_config, _DEFAULT_CONFIG_YAML

        if not _DEFAULT_CONFIG_YAML.is_file():
            self.skipTest('configs/default.yaml missing')
        with open(_DEFAULT_CONFIG_YAML, encoding='utf-8') as f:
            y = __import__('yaml').safe_load(f)
        c = load_config(None)
        self.assertEqual(c['disambig']['sr_cb_policy_by_dataset'], y['disambig']['sr_cb_policy_by_dataset'])
        self.assertEqual(c['model']['params'], y['model']['params'])

    def test_explicit_config_merges_onto_default(self):
        import yaml

        from experiments.run_single import load_config, _DEFAULT_CONFIG_YAML

        if not _DEFAULT_CONFIG_YAML.is_file():
            self.skipTest('configs/default.yaml missing')
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.safe_dump({'seed': 12345}, f)
            p = f.name
        try:
            c = load_config(p)
            self.assertEqual(c['seed'], 12345)
            with open(_DEFAULT_CONFIG_YAML, encoding='utf-8') as df:
                y = yaml.safe_load(df)
            self.assertEqual(c['disambig']['sr_cb_policy_by_dataset']['lost'],
                             y['disambig']['sr_cb_policy_by_dataset']['lost'])
        finally:
            Path(p).unlink(missing_ok=True)


class TestLoadConfigAndFast(unittest.TestCase):
    def setUp(self):
        try:
            from experiments import run_single  # noqa: F401
        except ImportError as e:
            self.skipTest(f'run_single import failed (env deps): {e}')

    def test_apply_fast_does_not_mutate_model_defaults(self):
        from experiments.run_single import load_config, apply_fast_training_overrides, MODEL_DEFAULTS

        t0 = MODEL_DEFAULTS['sdlpp']['T']
        cfg = load_config(None)
        apply_fast_training_overrides(cfg)
        self.assertEqual(cfg['model']['params']['T'], 10)
        self.assertEqual(cfg['eval']['cv_folds'], 3)
        self.assertEqual(MODEL_DEFAULTS['sdlpp']['T'], t0)

    def test_config_hash_stable(self):
        from experiments.run_single import compute_config_hash

        cfg = {'a': 1, '_run_meta': {'run_id': 'x'}}
        cfg2 = copy.deepcopy(cfg)
        self.assertEqual(compute_config_hash(cfg), compute_config_hash(cfg2))


class TestRegistryMergePreservesHistory(unittest.TestCase):
    def setUp(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest('PyYAML not installed')

    def test_merge_appends_without_dropping_prior_entries(self):
        import yaml

        from experiments.tools.update_registry import merge_best_configs_yaml

        with tempfile.TemporaryDirectory() as tmp:
            reg = Path(tmp)
            path = reg / 'best_configs.yaml'
            import experiments.tools.update_registry as ur

            orig_reg = ur.REGISTRY
            ur.REGISTRY = reg
            try:
                path.write_text(
                    'schema_version: 1\nentries:\n'
                    '  - id: keep_me\n    dataset: lost\n    method: baseline\n',
                    encoding='utf-8',
                )
                merge_best_configs_yaml(
                    [
                        {
                            'id': 'new_one',
                            'dataset': 'MSRCv2',
                            'method': 'sr+cb',
                            'metric': 'balanced_acc',
                            'value': 0.5,
                            'config_hash': 'abc',
                            'evidence_summary_csv': 'x.csv',
                            'recorded_at': 't',
                            'notes': None,
                        }
                    ]
                )
                data = yaml.safe_load(path.read_text(encoding='utf-8'))
                ids = [e['id'] for e in data['entries']]
                self.assertIn('keep_me', ids)
                self.assertIn('new_one', ids)
                self.assertTrue((path.with_suffix('.yaml.bak')).is_file())
            finally:
                ur.REGISTRY = orig_reg


if __name__ == '__main__':
    unittest.main()
