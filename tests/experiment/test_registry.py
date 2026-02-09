"""Tests for ExperimentRegistry: register, query, compare, persistence."""

import os
import tempfile
import unittest

from taberspilotml.experiment.config import ExperimentConfig, ModelSpec, FeatureSpec
from taberspilotml.experiment.runner import ExperimentResult
from taberspilotml.experiment.registry import ExperimentRegistry


def _make_result(name='test', model='RF', scaler=None, score=0.9, status='completed'):
    config = ExperimentConfig(
        name=name,
        target_label='target',
        model=ModelSpec(model_name=model),
        features=FeatureSpec(scaler_name=scaler),
    )
    return ExperimentResult(
        config=config,
        scores={'accuracy': score},
        score_stds={'accuracy': 0.02},
        execution_time=1.0,
        status=status,
    )


class TestExperimentRegistry(unittest.TestCase):

    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.tmpfile.close()
        self.registry = ExperimentRegistry(self.tmpfile.name)

    def tearDown(self):
        os.unlink(self.tmpfile.name)

    def test_register_and_get(self):
        result = _make_result(name='exp1', model='RF', score=0.92)
        exp_id = self.registry.register(result)

        retrieved = self.registry.get(exp_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['name'], 'exp1')
        self.assertEqual(retrieved['model_name'], 'RF')
        self.assertAlmostEqual(retrieved['primary_score'], 0.92)

    def test_get_nonexistent(self):
        result = self.registry.get(999)
        self.assertIsNone(result)

    def test_query_by_model(self):
        self.registry.register(_make_result(name='rf1', model='RF', score=0.90))
        self.registry.register(_make_result(name='xgb1', model='XGB', score=0.92))
        self.registry.register(_make_result(name='rf2', model='RF', score=0.91))

        rf_results = self.registry.query(model_name='RF')
        self.assertEqual(len(rf_results), 2)
        self.assertTrue(all(r == 'RF' for r in rf_results['model_name']))

    def test_query_by_status(self):
        self.registry.register(_make_result(name='ok', status='completed'))
        self.registry.register(_make_result(name='bad', status='failed'))

        completed = self.registry.query(status='completed')
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed.iloc[0]['name'], 'ok')

    def test_compare_ids(self):
        id1 = self.registry.register(_make_result(name='exp1', model='RF'))
        id2 = self.registry.register(_make_result(name='exp2', model='XGB'))
        id3 = self.registry.register(_make_result(name='exp3', model='KNN'))

        comparison = self.registry.compare([id1, id3])
        self.assertEqual(len(comparison), 2)

    def test_best(self):
        self.registry.register(_make_result(name='low', model='KNN', score=0.70))
        self.registry.register(_make_result(name='mid', model='RF', score=0.85))
        self.registry.register(_make_result(name='high', model='XGB', score=0.95))

        top = self.registry.best(n=2, classification=True)
        self.assertEqual(len(top), 2)
        self.assertAlmostEqual(top.iloc[0]['primary_score'], 0.95)

    def test_best_regression(self):
        self.registry.register(_make_result(name='low', score=0.5))
        self.registry.register(_make_result(name='high', score=2.0))

        top = self.registry.best(n=1, classification=False)
        self.assertEqual(len(top), 1)
        self.assertAlmostEqual(top.iloc[0]['primary_score'], 0.5)

    def test_history(self):
        self.registry.register(_make_result(name='first', score=0.80))
        self.registry.register(_make_result(name='second', score=0.85))
        self.registry.register(_make_result(name='third', score=0.90))

        history = self.registry.history()
        self.assertEqual(len(history), 3)
        # Should be ordered by creation time ascending
        self.assertEqual(history.iloc[0]['name'], 'first')
        self.assertEqual(history.iloc[2]['name'], 'third')

    def test_persistence(self):
        """Data persists across registry instances."""
        self.registry.register(_make_result(name='persistent', score=0.99))

        # Create new registry pointing to same DB
        registry2 = ExperimentRegistry(self.tmpfile.name)
        results = registry2.query(name='persistent')
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results.iloc[0]['primary_score'], 0.99)

    def test_scores_json_round_trip(self):
        result = _make_result(name='scores_test', score=0.88)
        exp_id = self.registry.register(result)

        retrieved = self.registry.get(exp_id)
        self.assertIn('accuracy', retrieved['scores'])
        self.assertAlmostEqual(retrieved['scores']['accuracy'], 0.88)


if __name__ == '__main__':
    unittest.main()
