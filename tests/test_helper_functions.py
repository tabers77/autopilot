import unittest
from unittest import mock

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from tuiautopilotml import helper_functions as hf


class DFSanityCheckTestCase(unittest.TestCase):

    def test_empty_df(self):
        # Is this the behaviour want?
        self.assertEqual(0, hf.df_sanity_check(pd.DataFrame()))

    def test_simple_df(self):
        self.assertEqual(0, hf.df_sanity_check(
            pd.DataFrame(data=[(1, 'a')], columns=['a', 'b'])))

    def test_nulls_in_df(self):
        self.assertEqual(1, hf.df_sanity_check(
            pd.DataFrame(data=[(1, None)], columns=['a', 'b'])))

    def test_numericstrings_in_df(self):
        self.assertEqual(1, hf.df_sanity_check(
            pd.DataFrame(data=[(1, '23')], columns=['a', 'b'])))

    def test_nulls_and_numericstrings_in_df(self):
        self.assertEqual(2, hf.df_sanity_check(
            pd.DataFrame(data=[(2, '23', None)], columns=['a', 'b', 'c'])))


class GetTrainTestSplitScoreTestCase(unittest.TestCase):

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
                result = hf.get_hold_out_score(df, 'target', model=model, evaluation_metric='accuracy', test_size=0.4)

                self.assertEqual(0.5, result)

        captured_trainx_df, captured_trainy_df = mock_fit.call_args[0]
        pd.testing.assert_frame_equal(df.drop('target', axis=1)[2:], captured_trainx_df)
        pd.testing.assert_series_equal(df['target'][2:], captured_trainy_df)

        mock_predict.assert_called_once()

        captured_test_df = mock_predict.call_args[0][0]
        pd.testing.assert_frame_equal(df.drop('target', axis=1)[:2], captured_test_df)


class GetCrossValScoreWrapperTestCase(unittest.TestCase):
    def test_regression(self):
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

        # Need random_state set both here and below to get deterministic result
        model = RandomForestRegressor(random_state=3)
        classification = True
        evaluation_metric = 'neg_mean_squared_error'
        dataframe = reg_df.copy()
        target_label = 'clicks'

        results = hf.get_cross_val_score_wrapper(
            dataframe=dataframe, target_label=target_label, model=model, n_folds=5, n_repeats=3, random_state=3,
            classification=classification, multi_classif=False, evaluation_metric=evaluation_metric, n_jobs=-1,
            verbose=0)

        expected = -49757.012  # This is the consistent result (rounded) we get using the data and params above.
        self.assertEqual(expected, round(results[0], 3), f'result should be {expected:.3f}')


if __name__ == '__main__':

    unittest.main()
