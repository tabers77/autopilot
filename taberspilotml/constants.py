""" Various constants and defaults used in the library. """
from taberspilotml.scoring_funcs import evaluation_metrics as em
from hyperopt import hp

models_list_default = ['KNN', 'NB', 'SVC', 'RF', 'XGB', 'ADA', 'MLP']

DEFAULT_COLOR = 'firebrick'
DEFAULT_PALETTE = 'mako'
DEFAULT_SEED = 0

DEFAULT_CLASSIFICATION_METRICS = (em.EvalMetrics.ACCURACY,
                                  em.EvalMetrics.F1_SCORE,
                                  em.EvalMetrics.PRECISION_SCORE,
                                  em.EvalMetrics.RECALL_SCORE)

DEFAULT_REGRESSION_METRICS = (em.EvalMetrics.NEG_MEAN_ABSOLUTE_ERROR,
                              em.EvalMetrics.NEG_MEAN_SQUARED_ERROR,
                              em.EvalMetrics.NEG_ROOT_MEAN_SQUARED_ERROR,
                              em.EvalMetrics.R2)


class HyperoptDefaultParameterSpaces:

    XGB = {'n_estimators': hp.choice('n_estimators', [10, 50, 300, 750, 1200, 1300, 1500]),
           'max_depth': hp.choice('max_depth', [i for i in range(1, 11, 2)]),
           'min_child_weight': hp.choice('min_child_weight', [i for i in range(1, 6, 2)]),
           'gamma': hp.choice('gamma', [i / 10.0 for i in range(0, 5)])}
