import math
import random
import unittest
from unittest import mock
from unittest.mock import ANY, call, patch

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from tuiautopilotml import base_helpers as bh
from tuiautopilotml.pre_modelling import imbalance
from tuiautopilotml.scoring_funcs import (cross_validation as cv,
                                          datasets as d,
                                          evaluation_metrics as ev,
                                          scorers)


from tests import utils


class ComputeEntropyTestCase(unittest.TestCase):

    def test_single_element(self):
        self.assertTrue(math.isnan(imbalance.compute_entropy(pd.Series([1]))))

    def test_two_elements(self):
        self.assertTrue(math.isnan(imbalance.compute_entropy(pd.Series([1, 1]))))

    def test_two_different_elements(self):
        self.assertEqual(1.0, imbalance.compute_entropy(pd.Series([1, 2])))

    def test_multiple_different_elements(self):
        self.assertAlmostEqual(0.92, imbalance.compute_entropy(pd.Series([1, 2, 1, 3, 6, 3, 1])), places=2)


class TrainVsTestTestCase(utils.PandasTestCase):

    @patch.object(scorers, 'get_cross_validation_score', return_value=(0.5, 0.1))
    def test_covariance_shift_score(self, mock_crossval: mock.MagicMock):
        dataset = utils.generate_dataset(50)
        ds = d.Dataset.from_dataframe(dataset, ['c'])

        train = dataset.sample(frac=0.8, random_state=37)
        test = dataset.drop(train.index)

        tvt = imbalance.TrainVsTest(train, test)

        model = RandomForestClassifier()
        with patch.object(d.Dataset, 'from_dataframe', return_value=ds) as mock_fromdf:
            mean, std = tvt.get_covariance_shift_score('c', estimator=model, random_state=37)
            self.assertEqual((0.5, 0.0), (mean, std))

        mock_fromdf.assert_called_with(ANY, ['is_train'])
        self.assertEqual(4, mock_fromdf.call_count)

        mock_crossval.assert_called_with(ds, model=model,
                                         split_policy=cv.SplitPolicy(policy_type='k_fold',
                                                                     shuffle=True,
                                                                     n_splits=5,
                                                                     n_repeats=3,
                                                                     random_state=37),
                                         evaluation_metrics=[ev.EvalMetrics.ROC_AUC])
        self.assertEqual(4, mock_crossval.call_count)

    @patch.object(scorers, 'get_cross_validation_score', return_value=(0.5, 0.1))
    def test_covariance_shift_score_per_feature(self, mock_crossval: mock.MagicMock):
        dataset = utils.generate_dataset(50)
        ds = d.Dataset.from_dataframe(dataset, ['c'])

        train = dataset.sample(frac=0.8, random_state=37)
        test = dataset.drop(train.index)

        tvt = imbalance.TrainVsTest(train, test)

        model = RandomForestClassifier()
        with patch.object(d.Dataset, 'from_dataframe', return_value=d.Dataset(ds.inputs, ds.labels)) as mock_fromdf:
            result = tvt.get_covariance_shift_score_per_feature(estimator=model, random_state=37, n_repeats=3)

        self.assertEqual(({'a': (0.5, 0.0), 'b': (0.5, 0.0)}, []), result)
        self.assertEqual(4, mock_fromdf.call_count)

        mock_fromdf.assert_called_with(ANY, ['is_train'])

        self.assertEqual(8, mock_crossval.call_count)
        mock_crossval.assert_called_with(d.Dataset(inputs=ANY, labels=ds.labels), model=model,
                                         split_policy=cv.SplitPolicy(policy_type='k_fold', n_splits=5, n_repeats=3,
                                         random_state=37, shuffle=True),
                                         evaluation_metrics=[ev.EvalMetrics.ROC_AUC])


if __name__ == '__main__':
    unittest.main()
