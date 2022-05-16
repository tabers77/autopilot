import unittest
import unittest.mock as mock

import numpy as np
import pandas as pd
from sklearn import model_selection as ms
import sklearn.metrics
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer

from tuiautopilotml.scoring_funcs import (cross_validation as cv,
                                          datasets as d,
                                          evaluation_metrics as ev)


class MetricsToScoringDictTestCase(unittest.TestCase):

    def test_string_cases(self):
        metrics = [
            ev.EvalMetrics.ACCURACY,
            ev.EvalMetrics.NEG_MEAN_SQUARED_ERROR,
            ev.EvalMetrics.NEG_MEAN_ABSOLUTE_ERROR,
            ev.EvalMetrics.NEG_ROOT_MEAN_SQUARED_ERROR,
            ev.EvalMetrics.R2
        ]

        scoring_dict = ev.metrics_to_scoringdict(metrics, None)

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
                ev.EvalMetrics.from_str(name)
            ]

            with mock.patch.object(sklearn.metrics, 'make_scorer', return_value=scorer) as mock_makescorer:
                scoring_dict = ev.metrics_to_scoringdict(metrics, 'macro')

                self.assertEqual(1, len(scoring_dict))
                self.assertIs(scoring_dict[name], scorer)

            mock_makescorer.assert_called_once_with(score, average='macro')


class SplitPolicyTestCase(unittest.TestCase):

    def test_defaults(self):
        policy = cv.SplitPolicy()
        self.assertEqual(3, policy.n_splits)
        self.assertEqual('k_fold', policy.policy_type)
        self.assertIsNone(policy.random_state)
        self.assertEqual(1, policy.n_repeats)

    def test_build_defaults(self):
        built = cv.SplitPolicy().build()

        self.assertIsInstance(built, ms.KFold)
        self.assertEqual(3, built.n_splits)
        self.assertIsNone(built.random_state)
        self.assertFalse(built.shuffle)

    def test_build_repeated_kfold_defaults(self):
        built = cv.SplitPolicy.repeated_kfold_default().build()
        self.assertIsInstance(built, ms.RepeatedKFold)
        self.assertIsNone(built.random_state)

    def test_build_stratified_kfold_defaults(self):
        built = cv.SplitPolicy.stratified_kfold_default().build()
        self.assertIsInstance(built, ms.StratifiedKFold)
        self.assertEqual(3, built.n_splits)
        self.assertFalse(built.shuffle)
        self.assertIsNone(built.random_state)

    def test_build_repeated_stratified_kfold_defaults(self):
        built = cv.SplitPolicy().repeated_stratified_kfold_default().build()
        self.assertIsInstance(built, ms.RepeatedStratifiedKFold)
        self.assertEqual(3, built.n_repeats)
        self.assertIsNone(built.random_state)


class CrossValidatorTestCase(unittest.TestCase):
    X = pd.DataFrame(data=[(1, 2), (3, 4), (5, 6)], columns=['a', 'b'])
    y = pd.DataFrame(data=[1, 2, 3], columns=['label'])
    model = object()  # For purposes of test we don't care what the model is...
    kfold = ms.KFold(n_splits=3, shuffle=True)
    dataset = d.Dataset(inputs=X, labels=y)

    def test_default_setup(self):
        with mock.patch.object(ms, 'cross_val_score', return_value=[0.1, 0.3, 0.5]) as mock_cv_score:
            scoring = cv.get_cv_scores(self.model, self.dataset, [ev.EvalMetrics.ACCURACY], self.kfold, -1, 0, None)

            self.assertDictEqual({'accuracy': [0.1, 0.3, 0.5]}, scoring.scores)

        mock_cv_score.assert_called_once_with(self.model, self.X, self.y,
                                              cv=self.kfold, scoring='accuracy', n_jobs=-1, verbose=0)

    def test_multimetrics(self):
        cv_scores = {
            'accuracy': np.array([0.1, 0.3, 0.5]),
            'f1_score': np.array([0.2, 0.6, 1.0]),
            'precision_score': np.array([0.2, 0.6, 1.0]),
            'recall_score': np.array([0.2, 0.6, 1.0])
        }

        with mock.patch.object(ms, 'cross_validate', return_value=cv_scores) as mock_cv:
            scoring = cv.get_cv_scores(self.model,
                                       self.dataset,
                                       [ev.EvalMetrics.ACCURACY,
                                        ev.EvalMetrics.F1_SCORE,
                                        ev.EvalMetrics.PRECISION_SCORE,
                                        ev.EvalMetrics.RECALL_SCORE],
                                       self.kfold, -1, 0, 'macro')

            self.assertDictEqual(cv_scores, scoring.scores)

        mock_cv.assert_called_once_with(self.model, self.X, self.y, return_train_score=False,
                                        cv=self.kfold, scoring=mock.ANY, n_jobs=-1, verbose=0)
        captured_scoring = mock_cv.call_args[1]['scoring']
        for key in cv_scores.keys():
            if key == 'accuracy':
                self.assertEqual('accuracy', captured_scoring[key])
            else:
                self.assertEqual(f'make_scorer({key}, average=macro)', str(captured_scoring[key]))


if __name__ == '__main__':
    unittest.main()
