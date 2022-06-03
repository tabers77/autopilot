from contextlib import redirect_stdout
import io
import random
import unittest
from unittest import mock
from unittest.mock import call, patch

import pandas as pd
from sklearn.linear_model import LogisticRegression
import taberspilotml.pre_modelling.feature_importance as fi
from taberspilotml.configs import models
from taberspilotml.scoring_funcs import evaluation_metrics as em
import taberspilotml.scoring_funcs.scorers as sc

from tests import utils


class FeatureImportanceFixture:
    df = pd.DataFrame(data=[
        (0.1, 1, 1.1, 1),
        (0.1, 0.0, 1.1, 0),
        (0.2, 0.5, 1., 0),
        (0.2, 0.8, 1., 1)
    ], columns=['f1', 'f2', 'f3', 'c'])


class AutoFeatureImportanceFromEstimatorTestCase(unittest.TestCase, FeatureImportanceFixture):
    def test_select_correlating_column(self):

        columns = fi.auto_feature_selection_from_estimator(
            self.df, 'c', estimator=LogisticRegression())

        self.assertSequenceEqual(['f2'], columns)


class GetFeatureImportanceL1TestCase(unittest.TestCase, FeatureImportanceFixture):
    def test_select_correlating_column(self):
        case = fi.BestFeatures(df=self.df, target_label='c', classification=True)
        columns = case.l1_feature_selection(penalty='l2')

        self.assertSequenceEqual(['f2'], columns)


class GetFeatureImportanceUniTestCase(unittest.TestCase, FeatureImportanceFixture):
    def test_select_correlating_column(self):
        case = fi.BestFeatures(df=self.df, target_label='c', classification=True)
        columns = case.univariate_feature_selection(k=1)

        self.assertListEqual(['f2'], columns)


class GetReduceFeaturesCVScoresTestCase(utils.PandasTestCase, FeatureImportanceFixture):

    def test_actual_run(self):

        random.seed(37)
        dataframe = utils.generate_dataset(15)

        with redirect_stdout(io.StringIO()):
            scores, output_df = fi.get_reduced_features_cv_scores(dataframe, 'c', 'XGB', classification=True)

        for key in ('reduced_x', 'x_all'):
            self.assertListEqual([0.7333, 0.3887], [round(v, 4) for v in scores[key]])

        self.assertPandasEqual(dataframe[['a', 'c']], output_df)

    @patch.object(fi, 'auto_feature_selection_from_estimator', return_value=['f1', 'f2'])
    @patch.object(sc, 'get_cross_validation_score', return_value='mock_score')
    def test_select_correlating_column(self, mock_cvscore, mock_autofs):
        expected_df = pd.DataFrame(
            data=[(0.1, 1.0, 1),
                  (0.1, 0.0, 0),
                  (0.2, 0.5, 0),
                  (0.2, 0.8, 1)],
            columns=['f1', 'f2', 'c']
        )
        model = models['clf']['KNN']
        # define method

        scores_dict, output_df = fi.get_reduced_features_cv_scores(self.df, 'c', 'KNN', classification=True)

        pd.testing.assert_frame_equal(expected_df, output_df)
        self.assertDictEqual({'reduced_x': 'mock_score',
                              'x_all': 'mock_score'}, scores_dict)

        mock_autofs.assert_called_once_with(mock.ANY, 'c', estimator=model)
        mock_cvscore.assert_has_calls([call(dataset=mock.ANY,  # Specifying this raises error about ambiguous comparison
                                            model=model,
                                            evaluation_metrics=[em.EvalMetrics.ACCURACY]),
                                       call(dataset=mock.ANY,
                                            model=model,
                                            evaluation_metrics=[em.EvalMetrics.ACCURACY])])

        captured_ds = mock_cvscore.call_args_list[0][1]['dataset']
        self.assertPandasEqual(self.df[['f1', 'f2']], captured_ds.inputs)
        self.assertPandasEqual(self.df['c'], captured_ds.labels)

        captured_ds = mock_cvscore.call_args_list[1][1]['dataset']
        self.assertPandasEqual(self.df[['f1', 'f2', 'f3']], captured_ds.inputs)
        self.assertPandasEqual(self.df['c'], captured_ds.labels)


if __name__ == '__main__':
    unittest.main()
