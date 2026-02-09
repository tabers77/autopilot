"""Tests for comparison utilities: ablation analysis, decision impact matrix."""

import unittest

from taberspilotml.experiment.config import (
    ExperimentConfig,
    ModelSpec,
    FeatureSpec,
    PreprocessingSpec,
    CVSpec,
)
from taberspilotml.experiment.runner import ExperimentResult
from taberspilotml.experiment.comparison import (
    pipeline_comparison_table,
    ablation_analysis,
    decision_impact_matrix,
)


def _make_result(name, model='RF', scaler=None, score=0.9, std=0.02):
    config = ExperimentConfig(
        name=name,
        target_label='target',
        model=ModelSpec(model_name=model),
        features=FeatureSpec(scaler_name=scaler),
        cv=CVSpec(n_splits=5, shuffle=True, random_state=42),
    )
    return ExperimentResult(
        config=config,
        scores={'accuracy': score},
        score_stds={'accuracy': std},
        execution_time=1.0,
        status='completed',
    )


class TestPipelineComparisonTable(unittest.TestCase):

    def test_basic_table(self):
        results = [
            _make_result('exp1', model='RF', score=0.90),
            _make_result('exp2', model='XGB', score=0.92),
        ]
        table = pipeline_comparison_table(results)

        self.assertEqual(len(table), 2)
        self.assertIn('model', table.columns)
        self.assertIn('accuracy', table.columns)

    def test_excludes_failed(self):
        results = [
            _make_result('exp1', score=0.9),
            ExperimentResult(
                config=ExperimentConfig(name='failed', target_label='target'),
                status='failed',
                error_message='boom',
            ),
        ]
        table = pipeline_comparison_table(results)
        self.assertEqual(len(table), 1)


class TestAblationAnalysis(unittest.TestCase):

    def test_ablation_deltas(self):
        baseline = _make_result('baseline', model='RF', score=0.85)
        variant1 = _make_result('variant_xgb', model='XGB', score=0.90)
        variant2 = _make_result('variant_knn', model='KNN', score=0.80)

        results = [baseline, variant1, variant2]
        ablation = ablation_analysis(results, baseline.fingerprint)

        self.assertEqual(len(ablation), 2)
        # variant_xgb should have positive delta
        xgb_row = ablation[ablation['name'] == 'variant_xgb']
        self.assertGreater(xgb_row['delta'].iloc[0], 0)
        # variant_knn should have negative delta
        knn_row = ablation[ablation['name'] == 'variant_knn']
        self.assertLess(knn_row['delta'].iloc[0], 0)

    def test_ablation_missing_baseline(self):
        results = [_make_result('exp1')]
        with self.assertRaises(ValueError):
            ablation_analysis(results, 'nonexistent_fingerprint')

    def test_ablation_sorted_by_delta(self):
        baseline = _make_result('baseline', model='RF', score=0.85)
        results = [
            baseline,
            _make_result('worse', model='KNN', score=0.70),
            _make_result('better', model='XGB', score=0.95),
        ]
        ablation = ablation_analysis(results, baseline.fingerprint)
        # Should be sorted descending by delta
        self.assertGreater(ablation['delta'].iloc[0], ablation['delta'].iloc[1])


class TestDecisionImpactMatrix(unittest.TestCase):

    def test_impact_matrix(self):
        results = [
            _make_result('rf1', model='RF', score=0.90),
            _make_result('rf2', model='RF', scaler='Standard', score=0.92),
            _make_result('xgb1', model='XGB', score=0.88),
        ]
        matrix = decision_impact_matrix(results)

        self.assertGreater(len(matrix), 0)
        self.assertIn('decision', matrix.columns)
        self.assertIn('value', matrix.columns)
        self.assertIn('mean_score', matrix.columns)
        self.assertIn('n_experiments', matrix.columns)

    def test_empty_results(self):
        matrix = decision_impact_matrix([])
        self.assertTrue(matrix.empty)


if __name__ == '__main__':
    unittest.main()
