import unittest
import unittest.mock as mock

import numpy as np
import pandas as pd
from sklearn import model_selection as ms
import sklearn.metrics
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer

from tuiautopilotml import cross_validation as cv
from tuiautopilotml import datasets as d
from tuiautopilotml.cross_validation import EvalMetrics


class MetricsToScoringDictTestCase(unittest.TestCase):

    def test_string_cases(self):
        metrics = [
            EvalMetrics.ACCURACY,
            EvalMetrics.NEG_MEAN_SQUARED_ERROR,
            EvalMetrics.NEG_MEAN_ABSOLUTE_ERROR,
            EvalMetrics.NEG_ROOT_MEAN_SQUARED_ERROR,
            EvalMetrics.R2
        ]

        scoring_dict = cv.metrics_to_scoringdict(metrics, None)

        self.assertEqual(len(metrics), len(scoring_dict))
        for m in metrics:
            self.assertTrue(m.value in scoring_dict)
            self.assertEqual(m.value, scoring_dict[m.value])

    def test_make_scorer_cases(self):
        for name, score in (('f1_score', f1_score),
                            ('precision_score', precision_score),
                            ('recall_score', recall_score)):
            scorer = make_scorer(score, average='macro')
            metrics = [
                EvalMetrics.from_str(name)
            ]

            with mock.patch.object(sklearn.metrics, 'make_scorer', return_value=scorer) as mock_makescorer:
                scoring_dict = cv.metrics_to_scoringdict(metrics, 'macro')

                self.assertEqual(1, len(scoring_dict))
                self.assertIs(scoring_dict[name], scorer)

            mock_makescorer.assert_called_once_with(score, average='macro')


class SplitPolicyTestCase(unittest.TestCase):

    def test_defaults(self):
        policy = cv.SplitPolicy()
        self.assertEqual(3, policy.n_splits)
        self.assertEqual('k_fold', policy.type)
        self.assertIsNone(policy.random_state)
        self.assertEqual(1, policy.n_repeats)

    def test_set_type(self):
        policy = cv.SplitPolicy()
        method = policy.set_policy_type('repeated_k_fold')
        self.assertIs(policy, method)
        self.assertEqual('repeated_k_fold', policy.type)

    def test_set_n_repeats(self):
        policy = cv.SplitPolicy().set_n_repeats(3)
        self.assertEqual(3, policy.n_repeats)

    def test_build_defaults(self):
        built = cv.SplitPolicy().build()

        self.assertIsInstance(built, ms.KFold)
        self.assertEqual(3, built.n_splits)
        self.assertIsNone(built.random_state)
        self.assertFalse(built.shuffle)

    def test_build_kfold_nondefaults(self):
        built = cv.SplitPolicy.kfold_default().set_n_splits(5).set_shuffle(True).set_random_state(7).build()

        self.assertIsInstance(built, ms.KFold)
        self.assertEqual(5, built.n_splits)
        self.assertEqual(7, built.random_state)
        self.assertTrue(built.shuffle)

    def test_build_repeated_kfold_nondefaults(self):
        built = cv.SplitPolicy.repeated_kfold_default().set_shuffle(True).set_random_state(2).build()
        self.assertIsInstance(built, ms.RepeatedKFold)
        self.assertEqual(2, built.random_state)

    def test_build_stratified_kfold_nondefaults(self):
        built = cv.SplitPolicy().stratified_kfold_default().set_shuffle(True).set_random_state(2).build()
        self.assertIsInstance(built, ms.StratifiedKFold)
        self.assertEqual(3, built.n_splits)
        self.assertTrue(built.shuffle)
        self.assertEqual(2, built.random_state)

    def test_build_repeated_stratified_kfold_nondefaults(self):
        built = (cv.SplitPolicy().repeated_stratified_kfold_default()
                 .set_shuffle(True).set_random_state(21).set_n_repeats(4).build())
        self.assertIsInstance(built, ms.RepeatedStratifiedKFold)
        self.assertEqual(4, built.n_repeats)
        self.assertEqual(21, built.random_state)


class CrossValidationConfigTestCase(unittest.TestCase):

    def test_defaults(self):
        config = cv.CrossValidatorConfig()

        expected_fold_type = cv.SplitPolicy()
        self.assertEqual(expected_fold_type.n_splits, config.policy.n_splits)
        self.assertEqual((EvalMetrics.ACCURACY,), config.eval_metrics)
        self.assertEqual(-1, config.n_jobs)
        self.assertEqual(0, config.verbose)

    def test_set_n_jobs(self):
        config = cv.CrossValidatorConfig().set_n_jobs(5)
        self.assertEqual(5, config.n_jobs)

    def test_set_verbose(self):
        config = cv.CrossValidatorConfig().set_verbose(1)

        self.assertEqual(1, config.verbose)

    def test_set_eval_metrics(self):
        config = cv.CrossValidatorConfig().set_eval_metrics([EvalMetrics.F1_SCORE, EvalMetrics.PRECISION_SCORE])

        self.assertListEqual([EvalMetrics.F1_SCORE, EvalMetrics.PRECISION_SCORE], config.eval_metrics)


class CrossValidatorTestCase(unittest.TestCase):
    X = pd.DataFrame(data=[(1, 2), (3, 4), (5, 6)], columns=['a', 'b'])
    y = pd.DataFrame(data=[1, 2, 3], columns=['label'])
    model = object()  # For purposes of test we don't care what the model is...
    kfold = ms.KFold(n_splits=3, shuffle=True)
    dataset = d.Dataset(inputs=X, labels=y)

    def test_default_setup(self):
        config = cv.CrossValidatorConfig()
        validator = cv.CrossValidator(config, self.model, self.dataset)

        with mock.patch.object(ms, 'cross_val_score', return_value=[0.1, 0.3, 0.5]) as mock_cv_score:
            scoring = validator.get_cross_validation_scores()

            self.assertDictEqual({'accuracy': [0.1, 0.3, 0.5]}, scoring.scores)

        mock_cv_score.assert_called_once_with(self.model, self.X, self.y,
                                              cv=mock.ANY, scoring='accuracy', n_jobs=-1, verbose=0)
        captured_cv = mock_cv_score.call_args[1]['cv']

        self.assertIsInstance(captured_cv, ms.KFold)
        self.assertEqual(3, captured_cv.n_splits)
        self.assertFalse(captured_cv.shuffle)
        self.assertIsNone(captured_cv.random_state)

    def test_multimetrics(self):
        cv_scores = {
            'accuracy': np.array([0.1, 0.3, 0.5]),
            'f1_score': np.array([0.2, 0.6, 1.0]),
            'precision_score': np.array([0.2, 0.6, 1.0]),
            'recall_score': np.array([0.2, 0.6, 1.0])
        }

        config = cv.CrossValidatorConfig().set_eval_metrics([EvalMetrics.ACCURACY,
                                                             EvalMetrics.F1_SCORE,
                                                             EvalMetrics.PRECISION_SCORE,
                                                             EvalMetrics.RECALL_SCORE])
        validator = cv.CrossValidator(config, self.model, self.dataset)

        with mock.patch.object(ms, 'cross_validate', return_value=cv_scores) as mock_cv:
            scoring = validator.get_cross_validation_scores()

            self.assertDictEqual(cv_scores, scoring.scores)

        mock_cv.assert_called_once_with(self.model, self.X, self.y, return_train_score=False,
                                        cv=mock.ANY, scoring=mock.ANY, n_jobs=-1, verbose=0)
        captured_scoring = mock_cv.call_args[1]['scoring']
        print(captured_scoring)


if __name__ == '__main__':
    unittest.main()
