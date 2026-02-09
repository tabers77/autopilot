"""Tests for automated experiment reports: ExperimentCard, ComparisonReport, SearchReport."""

import unittest

from taberspilotml.experiment.config import ExperimentConfig, ModelSpec, FeatureSpec
from taberspilotml.experiment.runner import ExperimentResult
from taberspilotml.experiment.reports import ExperimentCard, ComparisonReport, SearchReport


def _make_result(name='test', model='RF', score=0.9, std=0.02, time_s=1.0, status='completed'):
    config = ExperimentConfig(
        name=name,
        target_label='target',
        model=ModelSpec(model_name=model),
    )
    return ExperimentResult(
        config=config,
        scores={'accuracy': score},
        score_stds={'accuracy': std},
        execution_time=time_s,
        status=status,
    )


class TestExperimentCard(unittest.TestCase):

    def test_card_generation(self):
        result = _make_result(name='my_exp', model='RF', score=0.92)
        card = ExperimentCard(result)
        md = card.to_markdown()

        self.assertIn('# Experiment: my_exp', md)
        self.assertIn('RF', md)
        self.assertIn('0.9200', md)
        self.assertIn('Reproducibility', md)

    def test_card_with_error(self):
        result = _make_result(name='failed_exp', status='failed')
        result.error_message = 'Something went wrong'
        card = ExperimentCard(result)
        md = card.to_markdown()

        self.assertIn('Error', md)
        self.assertIn('Something went wrong', md)

    def test_card_contains_config_table(self):
        result = _make_result(name='config_test')
        card = ExperimentCard(result)
        md = card.to_markdown()

        self.assertIn('Configuration', md)
        self.assertIn('Model', md)
        self.assertIn('CV Policy', md)


class TestComparisonReport(unittest.TestCase):

    def test_comparison_report(self):
        results = [
            _make_result(name='rf_exp', model='RF', score=0.90, time_s=2.0),
            _make_result(name='xgb_exp', model='XGB', score=0.92, time_s=5.0),
            _make_result(name='knn_exp', model='KNN', score=0.85, time_s=1.0),
        ]
        report = ComparisonReport(results)
        md = report.to_markdown()

        self.assertIn('Pipeline Comparison Report', md)
        self.assertIn('Leaderboard', md)
        self.assertIn('Decision Impact', md)
        self.assertIn('Stability Analysis', md)
        self.assertIn('Key Insights', md)
        self.assertIn('xgb_exp', md)
        self.assertIn('Best pipeline', md)

    def test_empty_comparison(self):
        report = ComparisonReport([])
        md = report.to_markdown()
        self.assertIn('No completed experiments', md)

    def test_comparison_has_rankings(self):
        results = [
            _make_result(name='worse', score=0.80),
            _make_result(name='better', score=0.95),
        ]
        report = ComparisonReport(results)
        md = report.to_markdown()
        # Better should appear before worse in leaderboard
        better_pos = md.find('better')
        worse_pos = md.find('worse')
        self.assertLess(better_pos, worse_pos)


class TestSearchReport(unittest.TestCase):

    def test_search_report_basic(self):
        results = [
            _make_result(name='trial_1', model='RF', score=0.85),
            _make_result(name='trial_2', model='XGB', score=0.92),
        ]
        report = SearchReport(results)
        md = report.to_markdown()

        self.assertIn('Pipeline Search Report', md)
        self.assertIn('Search Summary', md)
        self.assertIn('Total configs evaluated', md)
        self.assertIn('Winner', md)

    def test_search_report_with_budget(self):
        results = [_make_result()]
        budget_summary = {
            'total_trials': 10,
            'elapsed_time': 45.2,
            'stopped_by': 'trial_limit',
        }
        report = SearchReport(results, budget_summary)
        md = report.to_markdown()

        self.assertIn('Budget', md)
        self.assertIn('trial_limit', md)

    def test_search_report_with_failures(self):
        results = [
            _make_result(name='ok', status='completed', score=0.9),
            _make_result(name='bad', status='failed'),
        ]
        report = SearchReport(results)
        md = report.to_markdown()

        self.assertIn('Failed', md)


if __name__ == '__main__':
    unittest.main()
