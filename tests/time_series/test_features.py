"""Tests for LagFeatureEngine."""

import unittest

import numpy as np
import pandas as pd

from taberspilotml.time_series.features import LagFeatureEngine


def _make_ts_df(n=50):
    np.random.seed(42)
    return pd.DataFrame({
        'value': np.random.randn(n).cumsum(),
        'feature1': np.random.randn(n),
    })


class TestLagFeatureEngine(unittest.TestCase):

    def test_basic_lag_creation(self):
        df = _make_ts_df()
        engine = LagFeatureEngine(lags=[1, 2, 3], target_column='value', drop_na=True)
        result = engine.create_features(df)

        self.assertIn('value_lag_1', result.columns)
        self.assertIn('value_lag_2', result.columns)
        self.assertIn('value_lag_3', result.columns)
        # After dropping NaN, should have fewer rows
        self.assertLess(len(result), len(df))

    def test_lag_values_correct(self):
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})
        engine = LagFeatureEngine(lags=[1], target_column='value', drop_na=True)
        result = engine.create_features(df)

        # After lag 1 and drop_na, first row should have value=20 and lag_1=10
        self.assertEqual(result.iloc[0]['value'], 20)
        self.assertEqual(result.iloc[0]['value_lag_1'], 10)

    def test_rolling_features(self):
        df = _make_ts_df(100)
        engine = LagFeatureEngine(
            lags=[1],
            target_column='value',
            rolling_windows=[3, 5],
            drop_na=True,
        )
        result = engine.create_features(df)

        self.assertIn('value_rolling_mean_3', result.columns)
        self.assertIn('value_rolling_std_3', result.columns)
        self.assertIn('value_rolling_mean_5', result.columns)
        self.assertIn('value_rolling_std_5', result.columns)

    def test_no_drop_na(self):
        df = _make_ts_df(20)
        engine = LagFeatureEngine(lags=[1, 2], target_column='value', drop_na=False)
        result = engine.create_features(df)

        # Should preserve all rows
        self.assertEqual(len(result), len(df))
        # First rows should have NaN in lag columns
        self.assertTrue(pd.isna(result.iloc[0]['value_lag_1']))

    def test_all_numeric_columns_lagged(self):
        df = _make_ts_df()
        engine = LagFeatureEngine(lags=[1], target_column=None, drop_na=True)
        result = engine.create_features(df)

        self.assertIn('value_lag_1', result.columns)
        self.assertIn('feature1_lag_1', result.columns)

    def test_validate_no_leakage(self):
        df = _make_ts_df(50)
        engine = LagFeatureEngine(lags=[1, 2], target_column='value')
        result = engine.create_features(df)

        # Validate no leakage at cutoff point
        is_valid = engine.validate_no_leakage(result, cutoff_idx=30)
        self.assertTrue(is_valid)

    def test_feature_names(self):
        engine = LagFeatureEngine(
            lags=[1, 2],
            target_column='value',
            rolling_windows=[3],
        )
        names = engine.feature_names
        self.assertIn('value_lag_1', names)
        self.assertIn('value_lag_2', names)
        self.assertIn('value_rolling_mean_3', names)
        self.assertIn('value_rolling_std_3', names)

    def test_feature_names_empty_without_target(self):
        engine = LagFeatureEngine(lags=[1], target_column=None)
        self.assertEqual(engine.feature_names, [])


if __name__ == '__main__':
    unittest.main()
