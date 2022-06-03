import unittest
from unittest import mock
from unittest.mock import patch

import pandas as pd
from xgboost import XGBClassifier

import taberspilotml
from taberspilotml import auto_mode as auto
from taberspilotml import base_helpers as bh
import taberspilotml.pre_modelling.encoders as enc
from taberspilotml import preprocessing as dv
from taberspilotml.scoring_funcs import cross_validation as cv, evaluation_metrics as ev
from taberspilotml import mlflow_uploader as mf

from tests import utils


class AutopilotModeTestCase(unittest.TestCase):
    test_default_steps = {
        'dataframe_transformation': (dv.dataframe_transformation, auto.initial_checkpoint_handler),
        'handle_missing_values': (taberspilotml.pre_modelling.handle_nulls.eval_imputation_method_wrapper,
                                  auto.initial_checkpoint_handler),
        'encoding': (enc.get_encoded_wrapper, auto.initial_checkpoint_handler),
        'baseline_score': ((lambda classification, evaluation_metric, model_name: 0.5), auto.support_handler)
    }

    df = pd.DataFrame(
        data=[(1, 2), (2, 3)], columns=['a', 'b'])

    config = {
        'df': df,
        'run_id_number': 1,
        'target_label': 'b',
        'classification': True,
        'evaluation_metric': 'accuracy',
        'model_name': 'KNN'
    }

    def test_default_steps__noargs(self):
        steps_list = list(auto.default_steps.items())
        errors = (
            "dataframe_transformation() missing 1 required positional argument: 'df'",
            "eval_imputation_method_wrapper() missing 2 required positional arguments: 'df' and 'target_label'",
            "get_encoded_wrapper() missing 1 required positional argument: 'df'",
            "get_baseline_score() missing 5 required positional arguments: 'df', 'target_label', "
            "'classification', 'evaluation_metric', and 'model_name'",
        )

        summary_report = auto.autopilot_mode(list(auto.default_steps.keys()), {'run_id_number': 1})

        self.assertEqual(4, len(summary_report))

        for i, (process_str, func_name, result) in enumerate(summary_report):
            self.assertEqual('not processed', process_str)
            expected_name, func = steps_list[i]
            self.assertEqual(expected_name, func_name)
            self.assertIsInstance(result, TypeError)
            error, = result.args
            self.assertEqual(errors[i], error)

    # Patch out call to get_baseline_score - requires patching the dict rather than the function itself due
    # to the way the auto_mode() function works.
    @patch.dict(auto.default_steps, values=test_default_steps)
    def test_default_step__with_df_arg(self):
        config = dict(self.config)  # Preserve original config
        expected_keys = set(config.keys()).union({'base_encoded_df', 'target_label'})

        summary = auto.autopilot_mode(auto.default_steps.keys(), config)

        self.assertEqual(4, len(summary))
        self.assertEqual(expected_keys, set(config.keys()))
        pd.testing.assert_frame_equal(config['base_encoded_df'], config['df'])

        for (message, fname), expected_fname in zip(summary,
                                                    ['dataframe_transformation',
                                                     'handle_missing_values',
                                                     'encoding',
                                                     'baseline_score']):
            self.assertEqual('successfully processed', message)
            self.assertEqual(expected_fname, fname)

        print(summary)

    @patch.dict(auto.default_steps, values=test_default_steps)
    @patch.dict(auto.all_pipeline_steps, values={'evaluate_models': (lambda fake_arg: ((0.4, 0.2), 'model'),
                                                                     auto.scoring_handler)})
    @patch.object(taberspilotml.base_helpers, 'update_upload_config')
    def test_modelling_step(self, mock_update_upload_config: mock.MagicMock):
        config = dict(self.config)
        config['fake_arg'] = 2
        expected_keys = set(config.keys()).union({'base_encoded_df', 'target_label'})

        summary = auto.autopilot_mode(['evaluate_models'], config)
        for (mesg, fname), expected_name in zip(summary, list(auto.default_steps) + ['evaluate_models']):
            self.assertEqual(expected_name, fname)
            self.assertEqual('successfully processed', mesg)

        self.assertEqual(expected_keys, set(config.keys()))
        mock_update_upload_config.assert_called_once_with(
            config_dict=config, model='model', run_name='1_evaluate_models_stage', scores=(0.4, 0.2))

    @patch.dict(auto.default_steps, values=test_default_steps)
    @patch.dict(auto.all_pipeline_steps, values={'grid_search': (lambda fake_arg: ((0.4, 0.2), 'params', 'model'),
                                                                 auto.hyper_p_handler)})
    @patch.object(taberspilotml.base_helpers, 'update_upload_config')
    def test_hyperparam_step(self, mock_update_upload_config: mock.MagicMock):
        config = dict(self.config)
        config['fake_arg'] = 2

        expected_keys = set(config.keys()).union({'base_encoded_df', 'target_label'})

        summary = auto.autopilot_mode(['grid_search'], config)
        for (mesg, fname), expected_name in zip(summary, list(auto.default_steps) + ['grid_search']):
            self.assertEqual(expected_name, fname)
            self.assertEqual('successfully processed', mesg)

        self.assertEqual(expected_keys, set(config.keys()))
        mock_update_upload_config.assert_called_once_with(
            config_dict=config, model='model', run_name='1_grid_search_stage', scores=(0.4, 0.2), tuned_params='params')

    @patch.dict(auto.default_steps, values=test_default_steps)
    @patch.dict(auto.all_pipeline_steps, values={'handle_outliers': (lambda fake_arg: ((0.4, 0.2), 'result_df'),
                                                                     auto.mixed_handler)})
    @patch.object(taberspilotml.base_helpers, 'update_upload_config')
    def test_mixed_step(self, mock_update_upload_config: mock.MagicMock):
        config = dict(self.config)
        config['fake_arg'] = 2

        expected_keys = set(config.keys()).union({'base_encoded_df', 'target_label'})

        summary = auto.autopilot_mode(['handle_outliers'], config)
        for (mesg, fname), expected_name in zip(summary, list(auto.default_steps) + ['handle_outliers']):
            self.assertEqual(expected_name, fname)
            self.assertEqual('successfully processed', mesg)

        self.assertEqual(expected_keys, set(config.keys()))
        mock_update_upload_config.assert_called_once_with(
            config_dict=config, result_df='result_df', run_name='1_handle_outliers_stage', scores=(0.4, 0.2))

    @patch.object(mf, 'upload_baseline_score')
    def test_execute_steps(self, mock_upload):
        df = utils.generate_dataset(15)
        config_dict = {
            'df': df,
            'target_label': 'c',
            'evaluation_metrics': [ev.EvalMetrics.ACCURACY],
            'run_id_number': 1,
            'policy': cv.SplitPolicy(policy_type='k_fold', n_splits=3, n_repeats=10, shuffle=True, random_state=0),
            'baseline_score': {
                'model_build_args': {
                    'use_label_encoder': False,
                    'random_state': 0,
                },
                'model_builder': XGBClassifier,
            }
        }

        def handler(step_name, function, parameters, config):
            step_config = config[step_name]
            model = step_config['model_builder'](**step_config['model_build_args'])
            eval_metric = config['evaluation_metrics'][0]
            result = function(model=model, evaluation_metric=eval_metric, **parameters)

            config['baseline_score'] = result
            config['built_model'] = model

        steps = dict(auto.default_steps)
        steps.update({'baseline_score': (bh.baseline_score_cv, handler)})

        auto.execute_steps(steps, config_dict)
        mock_upload.assert_called_once_with(mock.ANY, 1, 'accuracy', config_dict['baseline_score'],
                                            config_dict['built_model'])


if __name__ == '__main__':
    unittest.main()
