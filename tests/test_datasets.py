import unittest

import pandas as pd
import numpy as np

from sklearn import preprocessing as prep
from taberspilotml.scoring_funcs import datasets as d


class DatasetTestCase(unittest.TestCase):
    df = pd.DataFrame(data=[(1, 2, 'x'), (3, 4, 'y')], columns=['a', 'b', 'c'])

    def test_from_dataframe(self):
        ds = d.Dataset.from_dataframe(self.df, target_columns=['c'])

        pd.testing.assert_frame_equal(self.df.drop('c', axis=1), ds.inputs)
        pd.testing.assert_series_equal(self.df['c'], ds.labels)

    def test_from_inputs_labels(self):
        ds = d.Dataset(inputs=self.df.drop(['c', 'b'], axis=1), labels=self.df[['c', 'b']])

        pd.testing.assert_frame_equal(self.df.drop(['c', 'b'], axis=1), ds.inputs)
        pd.testing.assert_frame_equal(self.df[['c', 'b']], ds.labels)

    def test_from_dataframe_scaled(self):
        ds = d.Dataset.from_dataframe(self.df, target_columns=['c'], scaler=prep.StandardScaler())

        scaler = prep.StandardScaler()

        expected_in = scaler.fit_transform(self.df.drop('c', axis=1))

        np.testing.assert_array_equal(expected_in, ds.inputs)


if __name__ == '__main__':

    unittest.main()
