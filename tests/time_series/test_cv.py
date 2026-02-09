"""Tests for TemporalSplitPolicy."""

import unittest

import numpy as np

from taberspilotml.time_series.cv import TemporalSplitPolicy


class TestExpandingWindow(unittest.TestCase):

    def test_basic_splits(self):
        policy = TemporalSplitPolicy(strategy='expanding_window', n_splits=3)
        splits = list(policy.split(100))

        self.assertGreater(len(splits), 0)
        self.assertLessEqual(len(splits), 3)

        for train_idx, test_idx in splits:
            # Train should come before test
            self.assertLess(train_idx[-1], test_idx[0])

    def test_expanding_train_grows(self):
        policy = TemporalSplitPolicy(strategy='expanding_window', n_splits=3)
        splits = list(policy.split(100))

        if len(splits) >= 2:
            self.assertGreater(len(splits[1][0]), len(splits[0][0]))

    def test_gap_respected(self):
        policy = TemporalSplitPolicy(strategy='expanding_window', n_splits=3, gap=5)
        splits = list(policy.split(100))

        for train_idx, test_idx in splits:
            gap = test_idx[0] - train_idx[-1]
            self.assertGreater(gap, 1)  # gap > 0

    def test_no_overlap(self):
        policy = TemporalSplitPolicy(strategy='expanding_window', n_splits=5)
        splits = list(policy.split(200))

        for train_idx, test_idx in splits:
            overlap = set(train_idx) & set(test_idx)
            self.assertEqual(len(overlap), 0)


class TestSlidingWindow(unittest.TestCase):

    def test_basic_splits(self):
        policy = TemporalSplitPolicy(strategy='sliding_window', n_splits=3, max_train_size=30)
        splits = list(policy.split(100))

        self.assertGreater(len(splits), 0)

        for train_idx, test_idx in splits:
            self.assertLess(train_idx[-1], test_idx[0])

    def test_max_train_size_respected(self):
        policy = TemporalSplitPolicy(strategy='sliding_window', n_splits=3, max_train_size=20)
        splits = list(policy.split(100))

        for train_idx, test_idx in splits:
            self.assertLessEqual(len(train_idx), 20)

    def test_no_overlap(self):
        policy = TemporalSplitPolicy(strategy='sliding_window', n_splits=3, max_train_size=30)
        splits = list(policy.split(100))

        for train_idx, test_idx in splits:
            overlap = set(train_idx) & set(test_idx)
            self.assertEqual(len(overlap), 0)


class TestBlockedCV(unittest.TestCase):

    def test_basic_splits(self):
        policy = TemporalSplitPolicy(strategy='blocked', n_splits=3)
        splits = list(policy.split(100))

        self.assertGreater(len(splits), 0)

        for train_idx, test_idx in splits:
            self.assertLess(train_idx[-1], test_idx[0])

    def test_gap_respected(self):
        policy = TemporalSplitPolicy(strategy='blocked', n_splits=3, gap=5)
        splits = list(policy.split(100))

        for train_idx, test_idx in splits:
            gap = test_idx[0] - train_idx[-1]
            self.assertGreater(gap, 1)

    def test_no_overlap(self):
        policy = TemporalSplitPolicy(strategy='blocked', n_splits=3)
        splits = list(policy.split(100))

        for train_idx, test_idx in splits:
            overlap = set(train_idx) & set(test_idx)
            self.assertEqual(len(overlap), 0)


class TestTemporalSplitPolicyGeneral(unittest.TestCase):

    def test_unknown_strategy_raises(self):
        policy = TemporalSplitPolicy(strategy='unknown')
        with self.assertRaises(ValueError):
            list(policy.split(100))

    def test_get_n_splits(self):
        policy = TemporalSplitPolicy(n_splits=7)
        self.assertEqual(policy.get_n_splits(), 7)


if __name__ == '__main__':
    unittest.main()
