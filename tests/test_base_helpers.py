import contextlib
import datetime as dt
import io
import random

import unittest
from unittest import mock
from unittest.mock import ANY, patch

import mlflow
import numpy as np
import pandas as pd
from sklearn import preprocessing as prep

from xgboost import XGBClassifier

from taberspilotml import base_helpers as bh
from taberspilotml.scoring_funcs import (cross_validation as cv,
                                         datasets as d,
                                         evaluation_metrics as ev,
                                         scorers)

from taberspilotml import mlflow_uploader as mf

from tests import utils


def test_adder(a, b):
    # This function is used in the unit tests below for get_params_from_config()
    return a + b


class GetParamFromConfigTestCase(unittest.TestCase):
    def test_get_params_from_config__noargs(self):
        result = bh.get_params_from_config(test_adder, {})
        self.assertDictEqual({}, result)

    def test_get_params_from_config__one_arg(self):
        result = bh.get_params_from_config(test_adder, {'a': 1})
        self.assertDictEqual({'a': 1}, result)

    def test_get_params_from_config__one_relevant_arg(self):
        result = bh.get_params_from_config(test_adder, {'a': 1, 'z': 2})
        self.assertDictEqual({'a': 1}, result)

    def test_get_params_from_config__both_relevant_args(self):
        result = bh.get_params_from_config(test_adder, {'a': 1, 'b': 2})
        self.assertDictEqual({'a': 1, 'b': 2}, result)

    def test_get_params_from_config__irrelevant_args(self):
        result = bh.get_params_from_config(test_adder, {'aa': 1, 'bb': 2})
        self.assertDictEqual({}, result)


class GetBestScoreTestCase(unittest.TestCase):

    def test_classification_true(self):
        score_dict = {'method1': [0.5, 0.04], 'method2': [0.6, 0.03]}
        result = bh.get_best_score(score_dict, classification=True)
        self.assertTupleEqual((0.6, 0.03, 'method2'), result)

    def test_classification_false(self):
        score_dict = {'method1': [0.5, 0.04], 'method2': [0.6, 0.03]}
        result = bh.get_best_score(score_dict, classification=False)
        self.assertTupleEqual((0.5, 0.04, 'method1'), result)


class GetLatestScoreTestCase(unittest.TestCase):

    @patch.object(mlflow, 'search_runs', return_value=pd.DataFrame(
        columns=['metrics.accuracy', 'status', 'end_time']))
    def test_missing_columns(self, mock_search_runs: mock.MagicMock):
        with self.assertRaises(AssertionError) as err:
            self.assertIsNone(bh.get_latest_score({'evaluation_metric': 'accuracy'}))

        mock_search_runs.assert_called_once_with(['0'])
        self.assertTupleEqual(("Some expected columns are missing from mlflow: "
                               "['metrics.std', 'tags.mlflow.runName']",),
                              err.exception.args)

    @patch.object(mlflow, 'search_runs', return_value=pd.DataFrame(
        columns=['metrics.std', 'metrics.accuracy', 'tags.mlflow.runName', 'status', 'end_time']))
    def test_no_run_id_number(self, mock_search_runs: mock.MagicMock):
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            self.assertIsNone(bh.get_latest_score({'evaluation_metric': 'accuracy'}))

        self.assertEqual("The parameter 'run_id_number' is missing, you need to generate a baseline score first\n",
                         captured.getvalue())
        mock_search_runs.assert_called_once_with(['0'])

    @patch.object(mlflow, 'search_runs', return_value=pd.DataFrame(
        columns=['metrics.std', 'metrics.accuracy', 'tags.mlflow.runName', 'status', 'end_time']))
    def test_no_evaluation_metric(self, _):
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            self.assertIsNone(bh.get_latest_score({'run_id_number': 1}))

        self.assertEqual(
            "The parameter 'evaluation_metric' is missing, you need to generate a baseline score first\n",
            captured.getvalue())

    @patch.object(mlflow, 'search_runs', return_value=pd.DataFrame(
        columns=['metrics.std', 'metrics.accuracy', 'tags.mlflow.runName', 'status', 'end_time']))
    def test_no_data(self, _):
        config = {
            'run_id_number': 1,
            'evaluation_metric': 'accuracy'
        }
        with self.assertRaises(AssertionError) as err:
            self.assertIsNone(bh.get_latest_score(config))

        self.assertTupleEqual(('Did not find any scores for run_id_number: 1',), err.exception.args)

    @patch.object(mlflow, 'search_runs', return_value=pd.DataFrame(
        data=[(0.04, 0.6, 'runName_1', 'FINISHED', dt.datetime(year=2022, month=4, day=13, hour=11, minute=31)),
              ],
        columns=['metrics.std', 'metrics.accuracy', 'tags.mlflow.runName', 'status', 'end_time']))
    def test_one_relevant_row(self, _):
        config = {
            'run_id_number': 1,
            'evaluation_metric': 'accuracy'
        }
        self.assertTupleEqual((0.6, 0.04), bh.get_latest_score(config))

    @patch.object(mlflow, 'search_runs', return_value=pd.DataFrame(
        data=[(0.04, 0.6, 'runName_1', 'FINISHED', dt.datetime(year=2022, month=4, day=13, hour=11, minute=31)),
              (0.03, 0.7, 'runName_1', 'FINISHED', dt.datetime(year=2022, month=4, day=13, hour=11, minute=38)),
              ],
        columns=['metrics.std', 'metrics.accuracy', 'tags.mlflow.runName', 'status', 'end_time']))
    def test_two_relevant_rows(self, _):
        config = {
            'run_id_number': 1,
            'evaluation_metric': 'accuracy'
        }
        self.assertTupleEqual((0.7, 0.03), bh.get_latest_score(config))

    @patch.object(mlflow, 'search_runs', return_value=pd.DataFrame(
        data=[(0.04, 0.6, 'runName_1', 'FINISHED', dt.datetime(year=2022, month=4, day=13, hour=11, minute=31)),
              (0.03, 0.7, 'runName_1', 'FINISHED', dt.datetime(year=2022, month=4, day=13, hour=11, minute=38)),
              ],
        columns=['metrics.std', 'metrics.accuracy', 'tags.mlflow.runName', 'status', 'end_time']))
    def test_no_relevant_rows(self, _):
        config = {
            'run_id_number': 2,
            'evaluation_metric': 'accuracy'
        }
        with self.assertRaises(AssertionError) as err:
            self.assertIsNone(bh.get_latest_score(config))

        self.assertTupleEqual(('Did not find any scores for run_id_number: 2',), err.exception.args)


class NewScoreSufficientlyBetterTestCase(unittest.TestCase):
    scores = {}
    config = {}

    @patch.object(bh, 'get_latest_score', return_value=(0.5, 0.05))
    @patch.object(bh, 'get_best_score', return_value=(0.5, 0.05, 'method'))
    def test_identical_scores__return_false(self, mock_best, mock_latest):
        self.assertFalse(bh.new_score_sufficiently_better(self.scores, self.config))

        mock_best.assert_called_once_with(self.config, classification=True)
        mock_latest.assert_called_once_with(config_dict=self.config)

    @patch.object(bh, 'get_latest_score', return_value=(0.5, 0.01))
    @patch.object(bh, 'get_best_score', return_value=(0.51, 0.01, 'method'))
    def test_improvement_smaller_std__return_true(self, _1, _2):
        self.assertTrue(bh.new_score_sufficiently_better(self.scores, self.config))

    @patch.object(bh, 'get_latest_score', return_value=(0.5, 0.02))
    @patch.object(bh, 'get_best_score', return_value=(0.51, 0.02, 'method'))
    def test_improvement_larger_std__return_false(self, _1, _2):
        self.assertFalse(bh.new_score_sufficiently_better(self.scores, self.config))

    @patch.object(bh, 'get_latest_score', return_value=(0.5, None))
    @patch.object(bh, 'get_best_score', return_value=(0.51, None, 'method'))
    def test_improvement_no_stds__return_true(self, _1, _2):
        self.assertTrue(bh.new_score_sufficiently_better(self.scores, self.config))

    @patch.object(bh, 'get_latest_score', return_value=(0.51, None))
    @patch.object(bh, 'get_best_score', return_value=(0.5, None, 'method'))
    def test_no_improvement_no_stds__return_false(self, _1, _2):
        self.assertFalse(bh.new_score_sufficiently_better(self.scores, self.config))

    @patch.object(bh, 'get_latest_score', return_value=(0.51, None))
    @patch.object(bh, 'get_best_score', return_value=(0.5, None, 'method'))
    def test_regression_improvement_no_stds__return_true(self, mock_best, _):
        self.assertTrue(bh.new_score_sufficiently_better(self.scores, self.config, classification=False))

        mock_best.assert_called_once_with(self.config, classification=False)

    @patch.object(bh, 'get_latest_score', return_value=(0.5, None))
    @patch.object(bh, 'get_best_score', return_value=(0.51, None, 'method'))
    def test_regression_no_improvement_no_stds__return_false(self, _1, _2):
        self.assertFalse(bh.new_score_sufficiently_better(self.scores, self.config, classification=False))

    @patch.object(bh, 'get_latest_score', return_value=(0.51, 0.005))
    @patch.object(bh, 'get_best_score', return_value=(0.5, 0.005, 'method'))
    def test_regression_improvement_smaller_stds__return_true(self, _1, _2):
        self.assertTrue(bh.new_score_sufficiently_better(self.scores, self.config, classification=False))

    @patch.object(bh, 'get_latest_score', return_value=(0.51, 0.05))
    @patch.object(bh, 'get_best_score', return_value=(0.5, 0.06, 'method'))
    def test_regression_improvement_larger_stds__return_false(self, _1, _2):
        self.assertFalse(bh.new_score_sufficiently_better(self.scores, self.config, classification=False))


class UploadUpdateConfigTestCase(utils.PandasTestCase):

    @patch.object(bh, 'new_score_sufficiently_better', return_value=False)
    def test_score_not_better(self, mock_sufficiently_better: mock.MagicMock):
        with contextlib.redirect_stdout(io.StringIO()) as captured_stdout:
            scores = {}
            config = {
                'evaluation_metric': 'accuracy', 'classification': True}
            bh.update_upload_config(scores, config, 'runName')

        self.assertEqual('We keep previous results since the new results are not significant\n',
                         captured_stdout.getvalue())
        mock_sufficiently_better.assert_called_once_with(scores, config, classification=True)

    @patch.object(bh, 'new_score_sufficiently_better', return_value=True)
    @patch.object(bh, 'get_best_score', return_value=(0.5, 0.05, 'method'))
    def test_score_better(self, mock_best_score: mock.MagicMock, mock_sufficiently_better: mock.MagicMock):
        scores = {}
        config = {
            'evaluation_metric': 'accuracy',
            'classification': True
        }

        mock_uploader = mock.Mock(spec=['upload_config_file'])
        with patch.object(mf, 'MLFlow', return_value=mock_uploader) as mock_mlflow:
            bh.update_upload_config(scores, config, 'runName')

        self.assertDictEqual({
            'accuracy': 0.5,
            'std': 0.05,
            'model_name': 'method',
            'evaluation_metric': 'accuracy',
            'classification': True
        }, config)

        mock_mlflow.assert_called_once_with(config,
                                            ['accuracy', 'std', 'k_fold_method', 'n_folds', 'n_repeats',
                                             'seed', 'n_jobs', 'num_rows', 'model_name', 'best_method', 'tuned_params'])
        mock_uploader.upload_config_file.assert_called_once_with(run_name='runName')
        mock_sufficiently_better.assert_called_once_with(scores, config, classification=True)
        mock_best_score.assert_called_once_with(scores, classification=True)

    @patch.object(bh, 'new_score_sufficiently_better', return_value=True)
    @patch.object(bh, 'get_best_score', return_value=(0.5, 0.05, 'method'))
    def test_score_better__has_model(self, _1, _2):
        scores = {}
        config = {
            'evaluation_metric': 'accuracy',
            'classification': True
        }

        fake_model = mock.Mock(spec=['get_params'])
        fake_model.get_params.return_value = {'param1': 1, 'param2': 10}

        mock_uploader = mock.Mock(spec=['upload_config_file'])
        with patch.object(mf, 'MLFlow', return_value=mock_uploader):
            with patch.object(mf, 'upload_artifacts') as mock_upload_artifacts:
                bh.update_upload_config(scores, config, 'runName', model=fake_model)

        self.assertDictEqual(config['best_model_params'], {'param1': 1, 'param2': 10})

        mock_upload_artifacts.assert_called_once_with(model=fake_model, model_name='method')

    @patch.object(bh, 'new_score_sufficiently_better', return_value=True)
    @patch.object(bh, 'get_best_score', return_value=(0.5, 0.05, 'method'))
    def test_score_better__has_result_df(self, _1, _2):
        scores = {}
        config = {
            'evaluation_metric': 'accuracy',
            'classification': True
        }

        mock_uploader = mock.Mock(spec=['upload_config_file'])
        with patch.object(mf, 'MLFlow', return_value=mock_uploader) as mock_mlflow:
            bh.update_upload_config(scores, config, 'runName', result_df=pd.DataFrame())

        self.assertEqual(config['num_rows'], 0)
        self.assertEqual(config['runName_best_method'], 'method')
        self.assertPandasEqual(config['df'], pd.DataFrame())

        mock_mlflow.assert_called_once_with(ANY,
                                            ['accuracy', 'std', 'k_fold_method', 'n_folds', 'n_repeats',
                                             'seed', 'n_jobs', 'num_rows', 'model_name', 'best_method', 'tuned_params',
                                             'runName_best_method'])

    @patch.object(bh, 'new_score_sufficiently_better', return_value=True)
    @patch.object(bh, 'get_best_score', return_value=(0.5, 0.05, 'method'))
    def test_score_better__has_tuned_params(self, _1, _2):
        scores = {}
        config = {
            'evaluation_metric': 'accuracy',
            'classification': True
        }

        fake_model = mock.Mock(spec=['get_params'])
        fake_model.get_params.return_value = {'param1': 1, 'param2': 10}
        mock_uploader = mock.Mock(spec=['upload_config_file'])
        with patch.object(mf, 'MLFlow', return_value=mock_uploader) as mock_mlflow:
            with patch.object(mf, 'upload_artifacts'):
                bh.update_upload_config(scores, config, 'runName',
                                        tuned_params={'param1': 1, 'param2': 10}, model=fake_model)

        self.assertDictEqual(config['tuned_params'], {'param1': 1, 'param2': 10})
        self.assertDictEqual(config['best_model_params'], {'param1': 1, 'param2': 10})

        mock_mlflow.assert_called_once_with(ANY,
                                            ['accuracy', 'std', 'k_fold_method', 'n_folds', 'n_repeats',
                                             'seed', 'n_jobs', 'num_rows', 'model_name', 'best_method', 'tuned_params'])


class ContainsNullsTestCase(unittest.TestCase):
    def test_has_null__returns_true(self):

        df = pd.DataFrame(data=[(None, 'hello')], columns=['a', 'b'])
        self.assertTrue(bh.contains_nulls(df))

    def test_no_nulls__returns_false(self):
        df = pd.DataFrame(data=[(1, 'hello')], columns=['a', 'b'])
        self.assertIsNotNone(bh.contains_nulls(df))
        self.assertFalse(bh.contains_nulls(df))


class ContainsObjectTestCase(unittest.TestCase):
    def test_has_object__returns_true(self):
        df = pd.DataFrame(data=[(1, 'hello')], columns=['a', 'b'])
        self.assertTrue(bh.contains_object(df))

    def test_no_object__returns_false(self):
        df = pd.DataFrame(data=[(1, 1)], columns=['a', 'b'])
        self.assertIsNotNone(bh.contains_object(df))
        self.assertFalse(bh.contains_object(df))


class IsColDate(unittest.TestCase):

    def test_col_has_timedelta_returns_true(self):
        df = pd.DataFrame(data=[(1, dt.timedelta(days=2))], columns=['a', 'b'])
        df['b'] = pd.to_timedelta(df['b'])

        self.assertTrue(bh.is_col_date(df, 'b'))

    def test_col_has_date_returns_true(self):
        df = pd.DataFrame(data=[(1, dt.date(2021, 10, 10))], columns=['a', 'b'])
        df['b'] = pd.to_datetime(df['b'])

        self.assertTrue(bh.is_col_date(df, 'b'))

    def test_col_has_datetime_returns_true(self):
        df = pd.DataFrame(data=[(1, dt.datetime(2021, 10, 10))], columns=['a', 'b'])
        df['b'] = pd.to_datetime(df['b'])

        self.assertTrue(bh.is_col_date(df, 'b'))

    def test_col_is_int_returns_False(self):
        df = pd.DataFrame(data=[(1, dt.datetime(2021, 10, 10))], columns=['a', 'b'])
        df['b'] = pd.to_datetime(df['b'])
        self.assertFalse(bh.is_col_date(df, 'a'))


class DFSanityCheckTestCase(unittest.TestCase):

    def test_empty_df(self):
        # Is this the behaviour want?
        self.assertEqual(0, bh.df_sanity_check(pd.DataFrame()))

    def test_simple_df(self):
        self.assertEqual(0, bh.df_sanity_check(
            pd.DataFrame(data=[(1, 'a')], columns=['a', 'b'])))

    def test_nulls_in_df(self):
        self.assertEqual(1, bh.df_sanity_check(
            pd.DataFrame(data=[(1, None)], columns=['a', 'b'])))

    def test_numeric_strings_in_df(self):
        self.assertEqual(1, bh.df_sanity_check(
            pd.DataFrame(data=[(1, '23')], columns=['a', 'b'])))

    def test_nulls_and_numeric_strings_in_df(self):
        self.assertEqual(2, bh.df_sanity_check(
            pd.DataFrame(data=[(2, '23', None)], columns=['a', 'b', 'c'])))


class GetSplitsWrapperTestCase(utils.PandasTestCase):
    df = pd.DataFrame(data=[(1, 2, 'x'),
                            (3, 4, 'y'),
                            (5, 6, 'z'),
                            (7, 8, 'u'),
                            (9, 10, 'v')], columns=['a', 'b', 'c'])

    def test_notscaled_notsplit(self):
        inputs, label = bh.get_splits_wrapper(self.df, 'c', False, False, validation_set=False)

        self.assertPandasEqual(self.df[['a', 'b']], inputs)
        self.assertPandasEqual(self.df['c'], label)

    def test_scaled_notsplit(self):
        inputs, label = bh.get_splits_wrapper(self.df, 'c', scaled=True, train_split=False, validation_set=False)

        scaler = prep.StandardScaler()

        expected_inputs = scaler.fit_transform(self.df[['a', 'b']])

        np.testing.assert_array_equal(expected_inputs, inputs)
        self.assertPandasEqual(self.df['c'], label)

    def test_notscaled_trainsplit(self):
        train_inputs, test_inputs, train_label, test_label = bh.get_splits_wrapper(
            self.df, 'c', scaled=False, train_split=True, validation_set=False, random_seed=0)

        self.assertPandasEqual(pd.DataFrame(
            data=[(1, 2),
                  (3, 4),
                  (7, 8),
                  (9, 10)], columns=['a', 'b'], index=[0, 1, 3, 4]), train_inputs)

        self.assertPandasEqual(pd.DataFrame(
            data=[(5, 6)], columns=['a', 'b'], index=[2]), test_inputs)

        self.assertPandasEqual(pd.Series(
            data=['x', 'y', 'u', 'v'], name='c', index=[0, 1, 3, 4]), train_label)

        self.assertPandasEqual(pd.Series(
            data=['z'], name='c', index=[2]), test_label)

    def test_scaled_trainsplit(self):
        train_inputs, test_inputs, train_label, test_label = bh.get_splits_wrapper(
            self.df, 'c', scaled=True, train_split=True, validation_set=False, random_seed=0)

        scaler = prep.StandardScaler()

        all_inputs = scaler.fit_transform(self.df[['a', 'b']])

        expected_train_in = np.array([[a, b] for i, (a, b) in enumerate(all_inputs) if i != 2])

        np.testing.assert_array_equal(expected_train_in, train_inputs)
        np.testing.assert_array_equal(np.array([all_inputs[2]]), test_inputs)

        self.assertPandasEqual(pd.Series(
            data=['x', 'y', 'u', 'v'], name='c', index=[0, 1, 3, 4]), train_label)

        self.assertPandasEqual(pd.Series(
            data=['z'], name='c', index=[2]), test_label)

    def test_notscaled_trainsplit_withvalidation(self):
        train_inputs, validate_inputs, train_label, validate_label, test_inputs, test_label = bh.get_splits_wrapper(
            self.df, 'c', scaled=False, train_split=True, validation_set=True, random_seed=0)

        self.assertPandasEqual(pd.DataFrame(
            data=[(9, 10),
                  (3, 4),
                  (1, 2)], columns=['a', 'b'], index=[4, 1, 0]), train_inputs)

        self.assertPandasEqual(pd.DataFrame(
            data=[(5, 6)], columns=['a', 'b'], index=[2]), test_inputs)

        self.assertPandasEqual(pd.DataFrame(
            data=[(7, 8)], columns=['a', 'b'], index=[3]), validate_inputs)

        self.assertPandasEqual(pd.Series(
            data=['v', 'y', 'x'], name='c', index=[4, 1, 0]), train_label)

        self.assertPandasEqual(pd.Series(
            data=['z'], name='c', index=[2]), test_label)

        self.assertPandasEqual(pd.Series(
            data=['u'], name='c', index=[3]), validate_label)


class BaselineScoreCVTestCase(utils.PandasTestCase):

    @patch.object(scorers, 'get_cross_validation_score', return_value=[0.6, 0.3266])
    @patch.object(mf, 'upload_baseline_score')
    def test_calls_cross_validation_correctly(self, mock_mfupload: mock.MagicMock, mock_crossval: mock.MagicMock):
        df = utils.generate_dataset(5)

        # This dict contains the key, value pairs you would supply in autopilot_mode's config_dict,
        # if you use baseline_score_cv in place of get_baseline_score
        kwargs = {
            'df': df,
            'target_label': 'c',
            'evaluation_metric': ev.EvalMetrics.ACCURACY,
            'model': XGBClassifier(use_label_encoder=False, random_state=0),  # Do we really need configs.py ?
            'run_id_number': 1,
            'policy': cv.SplitPolicy(policy_type='k_fold', n_splits=3, n_repeats=10, shuffle=True, random_state=0)
        }

        ds = d.Dataset(inputs=df[['a', 'b']].copy(), labels=df['c'].copy())
        with patch.object(d.Dataset, 'from_dataframe', return_value=ds) as mock_fromdf:
            score, std = bh.baseline_score_cv(**kwargs)

        self.assertTupleEqual((0.6, 0.3266), (round(score, 4), round(std, 4)))

        mock_mfupload.assert_called_once_with(df, 1, 'accuracy', [score, std], ANY)
        mock_crossval.assert_called_once_with(ds,
                                              model=kwargs['model'],
                                              split_policy=kwargs['policy'],
                                              evaluation_metrics=[kwargs['evaluation_metric']],
                                              n_jobs=-1, verbose=3)
        mock_fromdf.assert_called_once_with(df, ['c'])


if __name__ == '__main__':
    unittest.main()
