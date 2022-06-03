import unittest
from unittest import mock
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn import model_selection as ms
from sklearn.ensemble import RandomForestRegressor

from tests import utils
from taberspilotml import base_helpers as bh
from taberspilotml.scoring_funcs import (cross_validation as cv,
                                         evaluation_metrics as ev,
                                         datasets,
                                         scorers)
from taberspilotml.scoring_funcs import datasets as d
from taberspilotml.configs import models


class GetScaledXScoreTestCase(unittest.TestCase):
    scaled_dataframe = pd.DataFrame(data=[(0.5, 1, 2)], columns=['a', 'b', 'c'])
    dataset = d.Dataset(inputs=scaled_dataframe.drop('c', axis=1), labels=scaled_dataframe.drop(['a', 'b'], axis=1))

    @patch.object(d.Dataset, 'from_dataframe', return_value=dataset)
    @patch.object(bh, 'scale_x', return_value=scaled_dataframe)
    @patch.object(scorers, 'get_cross_validation_score', return_value=(0.5, 0.1))
    def test_calls_get_cross_val_score_wrapper_correctly(self, mock_cvwrapper, mock_scale_x, mock_ds_from_df):
        raw_dataframe = pd.DataFrame(data=[(1, 2, 2)], columns=['a', 'b', 'c'])
        target_label = 'c'

        mean, std = scorers.get_scaled_x_score(raw_dataframe, target_label)

        expected_policy = cv.SplitPolicy(policy_type='k_fold', n_splits=5, n_repeats=3, random_state=0, shuffle=True)

        self.assertTupleEqual((0.5, 0.1), (mean, std))
        mock_scale_x.assert_called_once_with(raw_dataframe, target_label, 'MinMax', transformer_name=None,
                                             use_transformers=False)

        mock_ds_from_df.assert_called_once_with(self.scaled_dataframe, [target_label])

        mock_cvwrapper.assert_called_once_with(
            dataset=self.dataset,
            model=models['clf']['RF'],
            evaluation_metrics=[ev.EvalMetrics.ACCURACY],
            split_policy=expected_policy)


class RegressionDFTestCase:
    reg_df = pd.DataFrame(
        columns=['productSKU', 'hotel_name', 'list_position', 'departure_date',
                 'party_composition', 'impressions', 'clicks', 'CTR'],
        data=[(80568, 463, 1, 1, 1, 4288, 677, 15.79),
              (631290, 41, 1, 12, 1, 14085, 670, 4.76),
              (78004, 357, 1, 87, 1, 5476, 487, 8.89),
              (38261, 449, 1, 113, 1, 2718, 449, 16.52),
              (80568, 463, 2, 1, 1, 13133, 369, 2.81),
              (38261, 449, 6, 113, 1, 12393, 327, 2.64),
              (56541, 106, 2, 15, 1, 4462, 315, 7.06),
              (56541, 106, 3, 15, 1, 9659, 295, 3.05),
              (574837, 157, 1, 1, 1, 1497, 270, 18.04),
              ]
    )


class GetCrossValidationScoreTestCase(unittest.TestCase, RegressionDFTestCase):

    @patch.object(ms, 'cross_val_score', return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
    def test_classification(self, mock_cross_val_score: mock.MagicMock):
        inputs = pd.DataFrame(data=[(1, 2), (2, 1)], columns=['f1', 'f2'])
        labels = pd.Series(['a', 'b'])

        mock_model = mock.MagicMock(spec=[])

        policy = cv.SplitPolicy.kfold_default()

        k_fold = ms.KFold(n_splits=5, random_state=0, shuffle=True)
        with patch.object(policy, 'build', return_value=k_fold) as mock_build:
            result = scorers.get_cross_validation_score(datasets.Dataset(inputs=inputs, labels=labels),
                                                        model=mock_model,
                                                        split_policy=policy)

        self.assertListEqual([0.3, np.std([0.1, 0.2, 0.3, 0.4, 0.5])], result)

        mock_cross_val_score.assert_called_once_with(
            mock_model, inputs, labels, cv=k_fold, n_jobs=-1,
            scoring='accuracy', verbose=0)
        mock_build.assert_called_once_with()

    def test_regression(self):
        # Need random_state set both here and below to get deterministic result
        model = RandomForestRegressor(random_state=3)
        dataframe = self.reg_df.copy()
        target_label = 'clicks'

        ds = datasets.Dataset.from_dataframe(dataframe, [target_label])
        split_policy = cv.SplitPolicy(n_splits=5, n_repeats=3, random_state=3, shuffle=True, policy_type='k_fold')

        results = scorers.get_cross_validation_score(
            dataset=ds, model=model, split_policy=split_policy,
            evaluation_metrics=[ev.EvalMetrics.NEG_MEAN_SQUARED_ERROR], n_jobs=-1,
            verbose=0)

        expected = -49757.012  # This is the consistent result (rounded) we get using the data and params above.
        self.assertEqual(expected, round(results[0], 3), f'result should be {expected:.3f}')


class GetTrainTestSplitScoreTestCase(utils.PandasTestCase):

    def test_provide_df(self):
        data = [(1, 1, 1),
                (2, 2, 0),
                (3, 3, 1),
                (4, 4, 0),
                (5, 5, 1)]
        df = pd.DataFrame(data=data, columns=['a', 'b', 'target'])

        model = mock.Mock()
        with mock.patch.object(model, 'fit') as mock_fit:
            with mock.patch.object(model, 'predict', return_value=[1, 1]) as mock_predict:
                result = scorers.get_hold_out_score(df, 'target', model=model,
                                                    evaluation_metric='accuracy', test_size=0.4)

                self.assertEqual(0.5, result)

        captured_trainx_df, captured_trainy_df = mock_fit.call_args[0]
        self.assertPandasEqual(df.drop('target', axis=1)[2:], captured_trainx_df)
        self.assertPandasEqual(df.drop('target', axis=1)[2:], captured_trainx_df)
        self.assertPandasEqual(df['target'][2:], captured_trainy_df)

        mock_predict.assert_called_once()

        captured_test_df = mock_predict.call_args[0][0]
        self.assertPandasEqual(df.drop('target', axis=1)[:2], captured_test_df)


if __name__ == '__main__':
    unittest.main()
