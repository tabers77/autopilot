import random
import unittest
from unittest import mock
from unittest.mock import patch

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


from tuiautopilotml.modelling import ml_models
from tuiautopilotml import visualization as vs

from tests import utils


class EvaluateModelWrapperTestCase(utils.PandasTestCase):
    # Main aim of these tests currently is to ensure cross validation is called correctly
    # - not all execution paths are tested.
    random.seed(37)
    df = utils.generate_dataset(15)

    @patch.object(vs, 'get_graph')  # Suppress graph generation as it pops up a window the user needs to respond to.
    def test_calls_cross_validation__one_model(self, mock_get_graph: mock.MagicMock):
        scores, best_model = ml_models.evaluate_models_wrapper(self.df, 'c', ['XGB'])

        self.assertTrue('XGB' in scores)
        mean, std = scores['XGB']

        self.assertEqual([0.6, 0.3266], [round(mean, 4), round(std, 4)])

        self.assertIsInstance(best_model, XGBClassifier)
        mock_get_graph.assert_called_once_with(color='firebrick',
                                               fig_title='Best Models Scores',
                                               figsize=(6, 4),
                                               file_name='best_models_scores',
                                               horizontal=True,
                                               input_data={'XGB': 0.6},
                                               save_figure=True,
                                               stage='Models',
                                               style='seaborn-darkgrid',
                                               x_title='Params', y_title='Scores')

    @patch.object(vs, 'get_graph')
    def test_calls_cross_validation__two_models(self, _):
        scores, best_model = ml_models.evaluate_models_wrapper(self.df, 'c', ['XGB', 'LR'])

        self.assertTrue('XGB' in scores)
        mean, std = scores['XGB']
        self.assertEqual([0.6, 0.3266], [round(mean, 4), round(std, 4)])

        self.assertTrue('LR' in scores)
        mean, std = scores['LR']
        self.assertEqual([0.7333, 0.2494], [round(mean, 4), round(std, 4)])

        self.assertIsInstance(best_model, LogisticRegression)


if __name__ == '__main__':
    unittest.main()
