"""Tests for ExperimentRunner: run configs, verify leaderboard ordering."""

import unittest

import pandas as pd

from taberspilotml.experiment.config import (
    TaskType,
    ExperimentConfig,
    ModelSpec,
    CVSpec,
)
from taberspilotml.experiment.runner import ExperimentRunner, run_single, ExperimentResult


def _make_dataset(n=100):
    """Create a simple classification dataset."""
    import random
    random.seed(42)
    rows = []
    for _ in range(n):
        a = random.random()
        b = random.random()
        c = 1 if a > b else 0
        rows.append((a, b, c))
    return pd.DataFrame(data=rows, columns=['a', 'b', 'target'])


class TestRunSingle(unittest.TestCase):

    def test_run_single_completes(self):
        df = _make_dataset()
        config = ExperimentConfig(
            name='test_rf',
            target_label='target',
            task_type=TaskType.CLASSIFICATION,
            model=ModelSpec(model_name='RF'),
            cv=CVSpec(n_splits=3, policy_type='k_fold', shuffle=True, random_state=42),
            evaluation_metric='accuracy',
        )
        result = run_single(config, df)

        self.assertEqual(result.status, 'completed')
        self.assertIn('accuracy', result.scores)
        self.assertIsNotNone(result.primary_score)
        self.assertGreater(result.execution_time, 0)

    def test_run_single_failed_bad_model(self):
        df = _make_dataset()
        config = ExperimentConfig(
            name='bad_model',
            target_label='target',
            model=ModelSpec(model_name='NONEXISTENT'),
        )
        result = run_single(config, df)
        self.assertEqual(result.status, 'failed')
        self.assertIsNotNone(result.error_message)


class TestExperimentRunner(unittest.TestCase):

    def test_run_two_configs_leaderboard(self):
        df = _make_dataset(n=200)
        configs = [
            ExperimentConfig(
                name='rf_exp',
                target_label='target',
                model=ModelSpec(model_name='RF'),
                cv=CVSpec(n_splits=3, shuffle=True, random_state=42),
            ),
            ExperimentConfig(
                name='knn_exp',
                target_label='target',
                model=ModelSpec(model_name='KNN'),
                cv=CVSpec(n_splits=3, shuffle=True, random_state=42),
            ),
        ]
        runner = ExperimentRunner(df, configs)
        results = runner.run_all()

        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.status == 'completed' for r in results))

        lb = runner.leaderboard()
        self.assertIsInstance(lb, pd.DataFrame)
        self.assertEqual(len(lb), 2)
        self.assertIn('accuracy', lb.columns)
        # Verify sorted descending (classification)
        if len(lb) >= 2:
            self.assertGreaterEqual(lb['accuracy'].iloc[0], lb['accuracy'].iloc[1])

    def test_empty_leaderboard(self):
        runner = ExperimentRunner(pd.DataFrame(), [])
        lb = runner.leaderboard()
        self.assertTrue(lb.empty)

    def test_compare_is_leaderboard(self):
        df = _make_dataset()
        configs = [
            ExperimentConfig(name='exp1', target_label='target', model=ModelSpec(model_name='RF'),
                             cv=CVSpec(n_splits=3, shuffle=True, random_state=42)),
        ]
        runner = ExperimentRunner(df, configs)
        runner.run_all()
        compare = runner.compare()
        lb = runner.leaderboard()
        pd.testing.assert_frame_equal(compare, lb)


class TestExperimentResult(unittest.TestCase):

    def test_to_dict(self):
        config = ExperimentConfig(name='test', target_label='target')
        result = ExperimentResult(config=config, scores={'accuracy': 0.95}, score_stds={'accuracy': 0.02},
                                  execution_time=1.5, status='completed')
        d = result.to_dict()
        self.assertEqual(d['config_name'], 'test')
        self.assertEqual(d['primary_score'], 0.95)
        self.assertEqual(d['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
