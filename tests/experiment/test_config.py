"""Tests for ExperimentConfig serialization, fingerprinting, and to_config_dict mapping."""

import json
import unittest

import pandas as pd

from taberspilotml.experiment.config import (
    TaskType,
    PreprocessingSpec,
    FeatureSpec,
    ModelSpec,
    CVSpec,
    ExperimentConfig,
)
from taberspilotml.scoring_funcs.cross_validation import SplitPolicy


class TestExperimentConfigSerialization(unittest.TestCase):
    """Test JSON serialization round-trip."""

    def test_to_dict_and_from_dict_round_trip(self):
        config = ExperimentConfig(
            name='test_exp',
            task_type=TaskType.CLASSIFICATION,
            target_label='target',
            preprocessing=PreprocessingSpec(imputation_strategy='median'),
            features=FeatureSpec(scaler_name='Standard'),
            model=ModelSpec(model_name='XGB', stacking=True),
            cv=CVSpec(n_splits=10, policy_type='stratified_k_fold'),
            evaluation_metric='f1_score',
        )

        d = config.to_dict()
        restored = ExperimentConfig.from_dict(d)

        self.assertEqual(restored.name, config.name)
        self.assertEqual(restored.task_type, config.task_type)
        self.assertEqual(restored.target_label, config.target_label)
        self.assertEqual(restored.preprocessing.imputation_strategy, 'median')
        self.assertEqual(restored.features.scaler_name, 'Standard')
        self.assertEqual(restored.model.model_name, 'XGB')
        self.assertEqual(restored.model.stacking, True)
        self.assertEqual(restored.cv.n_splits, 10)
        self.assertEqual(restored.cv.policy_type, 'stratified_k_fold')
        self.assertEqual(restored.evaluation_metric, 'f1_score')

    def test_to_json_and_from_json_round_trip(self):
        config = ExperimentConfig(
            name='json_test',
            task_type=TaskType.REGRESSION,
            target_label='price',
            model=ModelSpec(model_name='LR'),
            evaluation_metric='r2',
        )

        json_str = config.to_json()
        restored = ExperimentConfig.from_json(json_str)

        self.assertEqual(restored.name, 'json_test')
        self.assertEqual(restored.task_type, TaskType.REGRESSION)
        self.assertEqual(restored.target_label, 'price')
        self.assertEqual(restored.model.model_name, 'LR')
        self.assertEqual(restored.evaluation_metric, 'r2')

    def test_json_is_valid(self):
        config = ExperimentConfig()
        json_str = config.to_json()
        parsed = json.loads(json_str)
        self.assertIsInstance(parsed, dict)


class TestFingerprintDeterminism(unittest.TestCase):
    """Test that fingerprint is deterministic and reflects decisions."""

    def test_same_config_same_fingerprint(self):
        config1 = ExperimentConfig(name='exp1', model=ModelSpec(model_name='RF'))
        config2 = ExperimentConfig(name='exp2', model=ModelSpec(model_name='RF'))
        # Name is excluded from fingerprint
        self.assertEqual(config1.fingerprint, config2.fingerprint)

    def test_different_config_different_fingerprint(self):
        config1 = ExperimentConfig(model=ModelSpec(model_name='RF'))
        config2 = ExperimentConfig(model=ModelSpec(model_name='XGB'))
        self.assertNotEqual(config1.fingerprint, config2.fingerprint)

    def test_fingerprint_is_string(self):
        config = ExperimentConfig()
        self.assertIsInstance(config.fingerprint, str)
        self.assertEqual(len(config.fingerprint), 16)

    def test_fingerprint_stable_across_calls(self):
        config = ExperimentConfig(model=ModelSpec(model_name='CART'))
        fp1 = config.fingerprint
        fp2 = config.fingerprint
        self.assertEqual(fp1, fp2)


class TestToConfigDict(unittest.TestCase):
    """Test backward-compatible to_config_dict mapping."""

    def setUp(self):
        self.df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'target': [0, 1, 0]})

    def test_basic_keys_present(self):
        config = ExperimentConfig(target_label='target')
        config_dict = config.to_config_dict(self.df)

        self.assertIn('df', config_dict)
        self.assertIn('target_label', config_dict)
        self.assertIn('classification', config_dict)
        self.assertIn('evaluation_metric', config_dict)
        self.assertIn('k_fold_method', config_dict)
        self.assertIn('n_folds', config_dict)
        self.assertIn('n_repeats', config_dict)
        self.assertIn('model_name', config_dict)
        self.assertIn('models_list', config_dict)
        self.assertIn('run_id_number', config_dict)

    def test_classification_flag(self):
        clf_config = ExperimentConfig(task_type=TaskType.CLASSIFICATION)
        self.assertTrue(clf_config.to_config_dict(self.df)['classification'])

        reg_config = ExperimentConfig(task_type=TaskType.REGRESSION)
        self.assertFalse(reg_config.to_config_dict(self.df)['classification'])

    def test_cv_params_mapped(self):
        config = ExperimentConfig(
            cv=CVSpec(policy_type='stratified_k_fold', n_splits=10, n_repeats=5)
        )
        config_dict = config.to_config_dict(self.df)
        self.assertEqual(config_dict['k_fold_method'], 'stratified_k_fold')
        self.assertEqual(config_dict['n_folds'], 10)
        self.assertEqual(config_dict['n_repeats'], 5)

    def test_model_name_mapped(self):
        config = ExperimentConfig(model=ModelSpec(model_name='XGB'))
        config_dict = config.to_config_dict(self.df)
        self.assertEqual(config_dict['model_name'], 'XGB')


class TestSplitPolicyFromCVSpec(unittest.TestCase):
    """Test the SplitPolicy.from_cv_spec classmethod."""

    def test_from_cv_spec(self):
        cv_spec = CVSpec(
            policy_type='repeated_k_fold',
            n_splits=10,
            n_repeats=3,
            shuffle=True,
            random_state=42,
        )
        policy = SplitPolicy.from_cv_spec(cv_spec)

        self.assertEqual(policy.policy_type, 'repeated_k_fold')
        self.assertEqual(policy.n_splits, 10)
        self.assertEqual(policy.n_repeats, 3)
        self.assertTrue(policy.shuffle)
        self.assertEqual(policy.random_state, 42)

    def test_from_cv_spec_builds_splitter(self):
        cv_spec = CVSpec(policy_type='k_fold', n_splits=5, shuffle=True, random_state=0)
        policy = SplitPolicy.from_cv_spec(cv_spec)
        splitter = policy.build()
        self.assertEqual(splitter.n_splits, 5)


class TestTaskType(unittest.TestCase):

    def test_classification_property(self):
        config = ExperimentConfig(task_type=TaskType.CLASSIFICATION)
        self.assertTrue(config.classification)

    def test_regression_property(self):
        config = ExperimentConfig(task_type=TaskType.REGRESSION)
        self.assertFalse(config.classification)

    def test_time_series_property(self):
        config = ExperimentConfig(task_type=TaskType.TIME_SERIES)
        self.assertFalse(config.classification)


class TestDefaultValues(unittest.TestCase):
    """Test that default values are sensible."""

    def test_default_config(self):
        config = ExperimentConfig()
        self.assertEqual(config.task_type, TaskType.CLASSIFICATION)
        self.assertEqual(config.model.model_name, 'RF')
        self.assertEqual(config.cv.n_splits, 5)
        self.assertEqual(config.evaluation_metric, 'accuracy')


if __name__ == '__main__':
    unittest.main()
