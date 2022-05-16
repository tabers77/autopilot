import enum
from typing import Optional, Sequence, Text

import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score


class EvalMetrics(enum.Enum):
    ACCURACY = 'accuracy'
    F1_SCORE = 'f1_score'
    PRECISION_SCORE = 'precision_score'
    RECALL_SCORE = 'recall_score'
    ROC_AUC = 'roc_auc'

    NEG_MEAN_ABSOLUTE_ERROR = 'neg_mean_absolute_error'
    NEG_MEAN_SQUARED_ERROR = 'neg_mean_squared_error'
    NEG_ROOT_MEAN_SQUARED_ERROR = 'neg_root_mean_squared_error'
    R2 = 'r2'

    @staticmethod
    def from_str(val: str) -> 'EvalMetrics':
        return EvalMetrics[val.upper()]


def metrics_to_scoringdict(metrics: Sequence[EvalMetrics], averaging_policy: Optional[Text]):
    """ Converts a list of EvalMetrics into a scoring dict for use with sklearn's cross_validate method.

    :param metrics:
        The metrics.
    :param averaging_policy:
        The averaging method to use.
    """

    scoring_dict = {}
    for m in metrics:
        if m in [EvalMetrics.ACCURACY,
                 EvalMetrics.NEG_MEAN_SQUARED_ERROR,
                 EvalMetrics.NEG_ROOT_MEAN_SQUARED_ERROR,
                 EvalMetrics.NEG_MEAN_ABSOLUTE_ERROR,
                 EvalMetrics.R2]:

            scoring_dict[m.value] = m.value
        else:
            if m is EvalMetrics.F1_SCORE:
                score = f1_score
            elif m is EvalMetrics.PRECISION_SCORE:
                score = precision_score
            elif m is EvalMetrics.RECALL_SCORE:
                score = recall_score
            else:
                raise ValueError(f'{m} is not a valid metric for make_scorer')

            scoring_dict[m.value] = sklearn.metrics.make_scorer(score, average=averaging_policy)

    return scoring_dict
