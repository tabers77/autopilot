import unittest
from unittest import mock
from unittest.mock import patch, DEFAULT, call

import pandas as pd

from keras.models import Functional, Sequential
import mlflow

from taberspilotml import mlflow_uploader as mfu


class UploadArtifactsTestCase(unittest.TestCase):

    @patch.object(mlflow, 'start_run')
    @patch.object(mlflow, 'log_param')
    @patch.object(mlflow.sklearn, 'log_model')
    def test_sklearn_model(self, mock_log_model, mock_log_param, mock_start_run: mock.MagicMock):
        mock_model = mock.Mock(spec=['get_params'])
        mock_model.get_params.return_value = {'p1': 1}

        mfu.upload_artifacts('modelName', model=mock_model)

        mock_log_model.assert_called_once_with(mock_model, 'modelName')
        mock_log_param.assert_called_once_with('p1', 1)
        mock_start_run.assert_called_once_with(run_name='modelName')

    @patch.object(mlflow, 'start_run')
    @patch.object(mlflow.keras, 'log_model')
    def test_keras_sequential_model(self, mock_log_model, mock_start_run: mock.MagicMock):
        mock_model = mock.Mock(spec=Sequential)

        mfu.upload_artifacts('modelName', model=mock_model)

        mock_log_model.assert_called_once_with(mock_model, 'modelName')
        mock_start_run.assert_called_once_with(run_name='modelName')

    def test_keras_functional_model(self):
        mock_model = mock.Mock(spec=Functional)

        with self.assertRaises(AssertionError) as err:
            mfu.upload_artifacts('modelName', model=mock_model)

        self.assertEqual(('Keras Functional models not supported.',), err.exception.args)


class UploadBaseLineScoreTestCase(unittest.TestCase):

    @patch.multiple('mlflow',
                    start_run=DEFAULT,
                    log_metric=DEFAULT,
                    log_param=DEFAULT,
                    end_run=DEFAULT)
    @patch.object(mlflow.sklearn, 'log_model')
    def test_something(self, log_model: mock.MagicMock, start_run, log_metric, log_param, end_run):

        mock_model = mock.MagicMock(spec=['get_params'])
        mock_model.get_params.return_value = {'p': 1}
        mfu.upload_baseline_score(pd.DataFrame(), 1, 'accuracy', [0, 1], mock_model)

        start_run.assert_called_once_with(run_name='1_baseline_score_stage')
        log_model.assert_called_once_with(mock_model, '1_baseline_score_stage')
        log_metric.assert_has_calls(calls=[
            call('accuracy', 0),
            call('std', 1)
        ])

        log_param.assert_has_calls(calls=[
            call('p', 1),
            call('num_rows', 0),
            call('num_features', 0),
            call('features_inc_target', ()),
        ])
        end_run.assert_called_once_with()


class MLFlowTestCase(unittest.TestCase):

    @patch.multiple('mlflow', start_run=DEFAULT, end_run=DEFAULT, log_metric=DEFAULT)
    def test_upload_config_file__evaluation_metric(self, start_run, end_run, log_metric):
        mlf = mfu.MLFlow({'p1': 1, 'p2': 2, 'evaluation_metric': 'p1'}, ['p1'])
        mlf.upload_config_file('runName')

        start_run.assert_called_once_with(run_name='runName')
        end_run.assert_called_once_with()
        log_metric.assert_called_once_with('p1', 1)

    @patch.multiple('mlflow', start_run=DEFAULT, end_run=DEFAULT, log_metric=DEFAULT)
    def test_upload_config_file__std(self, start_run, end_run, log_metric):
        mlf = mfu.MLFlow({'p1': 1, 'p2': 2, 'evaluation_metric': 'p1', 'std': 0.1}, ['std'])
        mlf.upload_config_file('runName')

        start_run.assert_called_once_with(run_name='runName')
        end_run.assert_called_once_with()
        log_metric.assert_has_calls([
            call('std', 0.1)
        ])

    @patch.multiple('mlflow', start_run=DEFAULT, end_run=DEFAULT, log_metric=DEFAULT, log_param=DEFAULT)
    def test_upload_config_file__param(self, start_run, end_run, log_metric: mock.MagicMock, log_param):
        mlf = mfu.MLFlow({'p1': 1, 'p2': 2, 'evaluation_metric': 'p1', 'std': 0.1}, ['p2'])
        mlf.upload_config_file('runName')

        start_run.assert_called_once_with(run_name='runName')
        end_run.assert_called_once_with()
        log_metric.assert_not_called()
        log_param.assert_has_calls([
            call('p2', 2)
        ])


class UploadFromFunctionDFTestCase(unittest.TestCase):

    @patch.multiple('mlflow', start_run=DEFAULT, log_param=DEFAULT)
    @patch.object(mfu.MLFlow, '_upload_scores')
    def test_score_with_dataframe__singlescore(self, upload_scores: mock.MagicMock, start_run, log_param):
        mock_model = mock.Mock(spec=[])
        mock_scorer = mock.MagicMock()
        mock_scorer.return_value = {'score': 0.5}

        df = pd.DataFrame(data=[(1, 2)], columns=['a', 'b'])
        mfu.MLFlow.upload_from_function_df(
            mock_scorer, 'modelName', 'metric', target_label='targetLabel', model=mock_model, df=df)

        start_run.assert_called_once_with(run_name='modelName')
        mock_scorer.assert_called_once_with(evaluation_metric='metric', model=mock_model, df=df,
                                            target_label='targetLabel')

        self.assertEqual(3, log_param.call_count)
        log_param.assert_has_calls([call('Num rows', 1),
                                    call('N features', 2),
                                    call('Features', ('a', 'b'))])
        upload_scores.assert_called_once_with({'score': 0.5}, 'metric', mock_model, 'modelName')


class UploadFromFunctionXYTestCase(unittest.TestCase):

    @patch.multiple('mlflow', start_run=DEFAULT, log_param=DEFAULT)
    @patch.object(mfu.MLFlow, '_upload_scores')
    def test_score_with_xy__singlescore(self, upload_scores, start_run, log_param):
        mock_model = mock.Mock(spec=[])

        mock_scorer = mock.MagicMock()
        mock_scorer.return_value = {'score': 0.5}

        x = pd.DataFrame(data=[(1,), (2,)], columns=['a'])
        y = pd.DataFrame(data=[(1,), (0,)], columns=['b'])

        mfu.MLFlow.upload_from_function_xy(
            mock_scorer, 'modelName', 'metric', model=mock_model, x=x, y=y)

        mock_scorer.assert_called_once_with(evaluation_metric='metric', model=mock_model, x=x, y=y)
        upload_scores.assert_called_once_with({'score': 0.5}, 'metric', mock_model, 'modelName')

        self.assertEqual(3, log_param.call_count)
        log_param.assert_has_calls([call('num_rows', 2),
                                    call('num_features', 1),
                                    call('features', ('a',))])

        start_run.assert_called_once_with(run_name='modelName')


class UploadScoresTestCase(unittest.TestCase):

    @patch.multiple('mlflow', log_metric=DEFAULT, end_run=DEFAULT)
    @patch.object(mlflow.sklearn, 'log_model')
    def test_singlescore(self, log_model, log_metric, end_run):
        mock_model = mock.Mock(spec=['get_params'])

        mfu.MLFlow._upload_scores(scores=0.5, evaluation_metric='metric', model=mock_model, model_name='modelName')

        self.assertEqual(1, log_metric.call_count)
        log_metric.assert_has_calls([call('metric', 0.5)])
        log_model.assert_called_once_with(mock_model, 'modelName')

        end_run.assert_called_once_with()
        mock_model.get_params.assert_not_called()

    @patch.multiple('mlflow', start_run=DEFAULT, log_metric=DEFAULT, end_run=DEFAULT)
    @patch.object(mlflow.sklearn, 'log_model')
    def test_score_with_dataframe__scorelist(self, log_model, start_run, log_metric, end_run):
        mock_model = mock.Mock(spec=['get_params'])

        mfu.MLFlow._upload_scores([0.5, 0.05], 'metric', mock_model, 'modelName')

        self.assertEqual(2, log_metric.call_count)
        log_metric.assert_has_calls([call('metric', 0.5),
                                     call('std', 0.05)])
        mock_model.get_params.assert_not_called()

    @patch.multiple('mlflow', log_metric=DEFAULT, log_param=DEFAULT, end_run=DEFAULT)
    @patch.object(mlflow.sklearn, 'log_model')
    def test_score_with_dataframe__scorelistdict(self, log_model, log_metric, log_param, end_run):
        mock_model = mock.Mock(spec=['get_params'])
        mock_model.get_params.return_value = {}

        mfu.MLFlow._upload_scores([0.5, {'p1': 0.05}], 'metric', mock_model, 'modelName')

        self.assertEqual(1, log_metric.call_count)
        log_metric.assert_has_calls([call('metric', 0.5)])

        self.assertEqual(1, log_param.call_count)
        log_param.assert_has_calls([call('p1', 0.05)])
        mock_model.get_params.assert_not_called()

    @patch.multiple('mlflow', log_metric=DEFAULT, log_param=DEFAULT, end_run=DEFAULT)
    @patch.object(mlflow.sklearn, 'log_model')
    def test_score_with_dataframe__scoredict(self, log_model, log_metric, log_param, end_run):
        mock_model = mock.Mock(spec=['get_params'])
        mock_model.get_params.return_value = {'p1': 1, 'p2': 0.1}

        mfu.MLFlow._upload_scores({'metric': 0.6, 'std': 0.04}, 'metric', mock_model, 'modelName')

        self.assertEqual(2, log_metric.call_count)
        log_metric.assert_has_calls([call('metric', 0.6),
                                     call('std', 0.04)])

        self.assertEqual(2, log_param.call_count)
        log_param.assert_has_calls([call('p1', 1),
                                    call('p2', 0.1)])
        mock_model.get_params.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
