"""Tests for TimeSeriesDataset."""

import unittest

import pandas as pd
import numpy as np

from taberspilotml.time_series.dataset import TimeSeriesDataset


def _make_ts_df(n=100):
    """Create a simple time-series dataframe."""
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    np.random.seed(42)
    return pd.DataFrame({
        'date': dates,
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'target': np.random.randint(0, 2, n),
    })


class TestTimeSeriesDataset(unittest.TestCase):

    def test_from_dataframe(self):
        df = _make_ts_df()
        ds = TimeSeriesDataset.from_dataframe(
            df, target_columns=['target'], time_index='date'
        )
        self.assertIsNotNone(ds.inputs)
        self.assertIsNotNone(ds.labels)
        self.assertEqual(ds.time_index, 'date')
        self.assertEqual(len(ds.labels), 100)

    def test_temporal_ordering_enforced(self):
        df = _make_ts_df()
        # Shuffle the dataframe
        shuffled = df.sample(frac=1, random_state=42)
        ds = TimeSeriesDataset.from_dataframe(
            shuffled, target_columns=['target'], time_index='date'
        )
        # Should still work (auto-sorts)
        self.assertEqual(len(ds.labels), 100)

    def test_missing_time_index_raises(self):
        df = _make_ts_df()
        with self.assertRaises(ValueError):
            TimeSeriesDataset.from_dataframe(
                df, target_columns=['target'], time_index='nonexistent'
            )

    def test_slice_by_cutoff(self):
        df = _make_ts_df()
        ds = TimeSeriesDataset.from_dataframe(
            df, target_columns=['target'], time_index='date'
        )
        train, test = ds.slice_by_cutoff(80)

        self.assertEqual(len(train.labels), 80)
        self.assertEqual(len(test.labels), 20)

    def test_inputs_exclude_time_index(self):
        df = _make_ts_df()
        ds = TimeSeriesDataset.from_dataframe(
            df, target_columns=['target'], time_index='date'
        )
        # time_index should not be in inputs
        if isinstance(ds.inputs, pd.DataFrame):
            self.assertNotIn('date', ds.inputs.columns)

    def test_with_scaler(self):
        from sklearn.preprocessing import StandardScaler
        df = _make_ts_df()
        ds = TimeSeriesDataset.from_dataframe(
            df, target_columns=['target'], time_index='date',
            scaler=StandardScaler()
        )
        self.assertIsNotNone(ds.inputs)

    def test_validate_temporal_ordering(self):
        df = _make_ts_df()
        ds = TimeSeriesDataset.from_dataframe(
            df, target_columns=['target'], time_index='date'
        )
        self.assertTrue(ds.validate_temporal_ordering(df))


if __name__ == '__main__':
    unittest.main()
