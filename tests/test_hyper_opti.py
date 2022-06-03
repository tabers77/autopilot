from contextlib import redirect_stdout
import io
import math
import os
import random
import unittest

from xgboost import XGBClassifier

from tests.utils import generate_dataset
from taberspilotml import hyper_opti as ho


class HyperOptParameterTuningCvTestCase(unittest.TestCase):

    # This is an integration test - a unit test would require major alteration of the underlying code.
    def test_actual_run(self):
        random.seed(37)
        os.environ['HYPEROPT_FMIN_SEED'] = '37'

        df = generate_dataset(15)
        target_label = 'c'

        # Suppress print statements from the call - there's a lot of them!
        with redirect_stdout(io.StringIO()):
            result, best_params, model = ho.hyperopt_parameter_tuning_cv(df, target_label, n_folds=3, model_name='XGB',
                                                                         timeout_minutes=(1/60))

        self.assertIsInstance(result, dict)
        mean, std = result['XGB']

        self.assertTupleEqual((0.6, 0.3266), (round(mean, 4), round(std, 4)))

        # math.nan == math.nan returns False so we can't do an equals check on this item of the dictionary, hence
        # we do this instead.
        self.assertTrue(math.isnan(best_params['missing']))
        del best_params['missing']

        expected_params = {
            'gamma': 0.1,
            'n_estimators': 300,
            'max_depth': 3,
            'min_child_weight': 1,
            'objective': 'binary:logistic',
            'random_state': 0,
            'use_label_encoder': False
        }
        # For some reason the dictionary can have extra entries when this test runs alongside other unit tests,
        # not sure what the source of the interference is, but these dictionary values are the same in both cases.
        for k in expected_params.keys():
            self.assertEqual(expected_params[k], best_params[k])

        self.assertIsInstance(model, XGBClassifier)


if __name__ == '__main__':
    unittest.main()