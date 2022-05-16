import unittest
from unittest import mock
from unittest.mock import patch

import pandas as pd

from tuiautopilotml import base_helpers as bh
from tuiautopilotml.scoring_funcs import (cross_validation as cv,
                                          datasets as d,
                                          evaluation_metrics as ev,
                                          scorers)
from tuiautopilotml.pre_modelling import outliers


class HandleOutliersTestCase(unittest.TestCase):
    output_df = pd.DataFrame(data=[(0.1, 0.2, 0.3)] * 10, columns=['a', 'b', 'c'])
    dataset = d.Dataset(inputs=output_df.drop('c', axis=1), labels=output_df.drop(['a', 'b'], axis=1))

    @patch.object(d.Dataset, 'from_dataframe', return_value=dataset)
    @patch.object(bh, 'get_output_df_wrapper', return_value=output_df)
    @patch.object(scorers, 'get_cross_validation_score', return_value=(0.5, 0.1))
    @patch.object(scorers, 'get_custom_cv_score', return_value=(0.5, 0.1))
    def test_calls_get_cross_val_wrapper_correctly(
            self, mock_custom_cv, mock_cv_wrapper, mock_outputdf_wrapper, mock_fromdf):
        dataframe = pd.DataFrame(data=[(1, 2, 3)] * 10, columns=['a', 'b', 'c'])
        target_label = 'target'

        mock_model = mock.MagicMock(spec=['foo'])
        result_dict, result_df = outliers.handle_outliers(dataframe, target_label, model=mock_model)
        self.assertEqual({'replace_outliers-mean': (0.5, 0.1)}, result_dict)
        pd.testing.assert_frame_equal(self.output_df, result_df)

        mock_cv_wrapper.assert_called_once_with(dataset=self.dataset, evaluation_metrics=[ev.EvalMetrics.ACCURACY],
                                                split_policy=cv.SplitPolicy(policy_type='k_fold', n_splits=5,
                                                                            n_repeats=10, random_state=0,
                                                                            shuffle=True),
                                                model=mock_model, n_jobs=-1, verbose=0)


if __name__ == '__main__':
    unittest.main()
