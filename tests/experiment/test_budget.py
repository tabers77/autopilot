"""Tests for BudgetConfig and BudgetTracker: time cap, trial limit, early stopping."""

import time
import unittest

from taberspilotml.experiment.budget import BudgetConfig, BudgetTracker
from taberspilotml.experiment.config import ExperimentConfig, ModelSpec
from taberspilotml.experiment.runner import ExperimentResult


def _make_result(score=0.9, status='completed', name='test'):
    config = ExperimentConfig(name=name, target_label='target', model=ModelSpec(model_name='RF'))
    return ExperimentResult(
        config=config,
        scores={'accuracy': score},
        score_stds={'accuracy': 0.01},
        status=status,
        execution_time=0.1,
    )


class TestBudgetTrackerTrialLimit(unittest.TestCase):

    def test_trial_limit_stops(self):
        config = BudgetConfig(max_trials=3)
        tracker = BudgetTracker(config)

        self.assertFalse(tracker.should_stop())
        for i in range(3):
            tracker.record_trial(_make_result(score=0.8 + i * 0.01))

        self.assertTrue(tracker.should_stop())

    def test_trial_limit_not_reached(self):
        config = BudgetConfig(max_trials=5)
        tracker = BudgetTracker(config)

        tracker.record_trial(_make_result())
        tracker.record_trial(_make_result())
        self.assertFalse(tracker.should_stop())

    def test_no_trial_limit(self):
        config = BudgetConfig()  # No limits
        tracker = BudgetTracker(config)
        for _ in range(100):
            tracker.record_trial(_make_result())
        self.assertFalse(tracker.should_stop())


class TestBudgetTrackerTimeCap(unittest.TestCase):

    def test_time_limit_stops(self):
        config = BudgetConfig(max_time_seconds=0.1)
        tracker = BudgetTracker(config)

        time.sleep(0.15)
        self.assertTrue(tracker.should_stop())

    def test_time_limit_not_reached(self):
        config = BudgetConfig(max_time_seconds=60)
        tracker = BudgetTracker(config)
        self.assertFalse(tracker.should_stop())

    def test_remaining_time(self):
        config = BudgetConfig(max_time_seconds=60)
        tracker = BudgetTracker(config)
        remaining = tracker.remaining_time
        self.assertIsNotNone(remaining)
        self.assertGreater(remaining, 0)

    def test_remaining_time_none_if_no_limit(self):
        config = BudgetConfig()
        tracker = BudgetTracker(config)
        self.assertIsNone(tracker.remaining_time)


class TestBudgetTrackerEarlyStopping(unittest.TestCase):

    def test_early_stopping_triggers(self):
        config = BudgetConfig(early_stopping_patience=3, early_stopping_threshold=0.01)
        tracker = BudgetTracker(config)

        # First trial sets baseline
        tracker.record_trial(_make_result(score=0.90))
        self.assertFalse(tracker.should_stop())

        # Three trials without sufficient improvement
        tracker.record_trial(_make_result(score=0.900))  # no improvement
        tracker.record_trial(_make_result(score=0.901))  # < threshold
        tracker.record_trial(_make_result(score=0.902))  # < threshold

        self.assertTrue(tracker.should_stop())

    def test_early_stopping_resets_on_improvement(self):
        config = BudgetConfig(early_stopping_patience=3, early_stopping_threshold=0.01)
        tracker = BudgetTracker(config)

        tracker.record_trial(_make_result(score=0.85))
        tracker.record_trial(_make_result(score=0.855))  # < threshold
        tracker.record_trial(_make_result(score=0.856))  # < threshold
        # Significant improvement resets counter
        tracker.record_trial(_make_result(score=0.90))
        self.assertFalse(tracker.should_stop())

    def test_failed_trials_count_as_no_improvement(self):
        config = BudgetConfig(early_stopping_patience=2)
        tracker = BudgetTracker(config)

        tracker.record_trial(_make_result(score=0.90))
        tracker.record_trial(_make_result(status='failed'))
        tracker.record_trial(_make_result(status='failed'))

        self.assertTrue(tracker.should_stop())


class TestBudgetTrackerSummary(unittest.TestCase):

    def test_summary_basic(self):
        config = BudgetConfig(max_trials=10, max_time_seconds=60)
        tracker = BudgetTracker(config)
        tracker.record_trial(_make_result(score=0.9))

        summary = tracker.summary()
        self.assertEqual(summary['total_trials'], 1)
        self.assertEqual(summary['best_score'], 0.9)
        self.assertIsNotNone(summary['elapsed_time'])
        self.assertIsNone(summary['stopped_by'])

    def test_summary_stopped_by_trials(self):
        config = BudgetConfig(max_trials=1)
        tracker = BudgetTracker(config)
        tracker.record_trial(_make_result())

        summary = tracker.summary()
        self.assertEqual(summary['stopped_by'], 'trial_limit')


class TestBudgetConfig(unittest.TestCase):

    def test_defaults(self):
        config = BudgetConfig()
        self.assertIsNone(config.max_time_seconds)
        self.assertIsNone(config.max_trials)
        self.assertIsNone(config.max_time_per_trial_seconds)
        self.assertIsNone(config.early_stopping_patience)
        self.assertEqual(config.early_stopping_threshold, 0.001)


if __name__ == '__main__':
    unittest.main()
