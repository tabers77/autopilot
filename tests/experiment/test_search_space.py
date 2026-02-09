"""Tests for SearchSpace: enumeration count, sampling, cost ordering."""

import unittest

from taberspilotml.experiment.config import ExperimentConfig, ModelSpec, CVSpec
from taberspilotml.experiment.search_space import SearchSpace, MODEL_COST_ESTIMATES


class TestSearchSpaceEnumeration(unittest.TestCase):

    def test_total_combinations_single_dim(self):
        space = SearchSpace(model_names=['RF', 'XGB', 'KNN'])
        self.assertEqual(space.total_combinations, 3)

    def test_total_combinations_multi_dim(self):
        space = SearchSpace(
            model_names=['RF', 'XGB'],
            scaler_names=[None, 'Standard'],
            imputation_strategies=['mean', 'median'],
        )
        # 2 models * 2 scalers * 1 transformer * 2 imputations * 1 cv * 1 stacking = 8
        self.assertEqual(space.total_combinations, 8)

    def test_enumerate_configs_count(self):
        base = ExperimentConfig(name='base', target_label='target')
        space = SearchSpace(
            model_names=['RF', 'XGB'],
            scaler_names=[None, 'MinMax'],
        )
        configs = space.enumerate_configs(base)
        self.assertEqual(len(configs), space.total_combinations)

    def test_enumerate_configs_unique_fingerprints(self):
        base = ExperimentConfig(name='base', target_label='target')
        space = SearchSpace(
            model_names=['RF', 'XGB', 'KNN'],
            scaler_names=[None, 'Standard'],
        )
        configs = space.enumerate_configs(base)
        fingerprints = [c.fingerprint for c in configs]
        self.assertEqual(len(fingerprints), len(set(fingerprints)))

    def test_enumerate_preserves_base_config(self):
        base = ExperimentConfig(
            name='base',
            target_label='my_target',
            evaluation_metric='f1_score',
            cv=CVSpec(n_splits=10),
        )
        space = SearchSpace(model_names=['RF', 'XGB'])
        configs = space.enumerate_configs(base)

        for config in configs:
            self.assertEqual(config.target_label, 'my_target')
            self.assertEqual(config.evaluation_metric, 'f1_score')
            self.assertEqual(config.cv.n_splits, 10)

    def test_enumerate_names_unique(self):
        base = ExperimentConfig(name='exp', target_label='target')
        space = SearchSpace(model_names=['RF', 'XGB'])
        configs = space.enumerate_configs(base)
        names = [c.name for c in configs]
        self.assertEqual(len(names), len(set(names)))


class TestSearchSpaceSampling(unittest.TestCase):

    def test_sample_count(self):
        base = ExperimentConfig(name='base', target_label='target')
        space = SearchSpace(model_names=['RF', 'XGB', 'KNN', 'NB'])
        configs = space.sample_configs(base, n=2, seed=42)
        self.assertEqual(len(configs), 2)

    def test_sample_capped_at_total(self):
        base = ExperimentConfig(name='base', target_label='target')
        space = SearchSpace(model_names=['RF', 'XGB'])
        configs = space.sample_configs(base, n=100, seed=42)
        self.assertEqual(len(configs), space.total_combinations)

    def test_sample_reproducible(self):
        base = ExperimentConfig(name='base', target_label='target')
        space = SearchSpace(model_names=['RF', 'XGB', 'KNN', 'NB'])
        configs1 = space.sample_configs(base, n=2, seed=42)
        configs2 = space.sample_configs(base, n=2, seed=42)
        self.assertEqual(
            [c.model.model_name for c in configs1],
            [c.model.model_name for c in configs2],
        )


class TestCostOrdering(unittest.TestCase):

    def test_cost_ordered_cheap_first(self):
        base = ExperimentConfig(name='base', target_label='target')
        space = SearchSpace(model_names=['XGB', 'NB', 'RF', 'LR'])
        configs = space.cost_ordered_configs(base)

        costs = [MODEL_COST_ESTIMATES.get(c.model.model_name, 5) for c in configs]
        self.assertEqual(costs, sorted(costs))

    def test_cost_ordering_stacking_increases_cost(self):
        base = ExperimentConfig(name='base', target_label='target')
        space = SearchSpace(
            model_names=['NB'],
            stacking_options=[False, True],
        )
        configs = space.cost_ordered_configs(base)
        # Non-stacking should come before stacking
        self.assertFalse(configs[0].model.stacking)
        self.assertTrue(configs[1].model.stacking)


if __name__ == '__main__':
    unittest.main()
