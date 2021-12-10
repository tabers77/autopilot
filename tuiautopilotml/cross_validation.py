""" Code to support running cross validations """
import enum
from numbers import Number
from typing import Any, Dict, Iterable, List, Text, Optional

import numpy as np
from sklearn import model_selection as ms
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score

# TEMPORARY LINE - REMOVE THIS AND UNCOMMENT THE OTHER LINE
from datasets import Dataset
#from tuiautopilotml.datasets import Dataset


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


def metrics_to_scoringdict(metrics: List[EvalMetrics], average: Optional[Text]):
    """ Converts a list of EvalMetrics into a scoring dict for use with sklearn's cross_validate method.

    :param metrics:
        The metrics.
    :param average:
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

            scoring_dict[m.value] = sklearn.metrics.make_scorer(score, average=average)

    return scoring_dict


class SplitPolicy:
    """ Encapsulates configuration for building scikit-learn cross validation policies. """

    def __init__(self, random_state=None, n_splits=3, policy_type='k_fold', n_repeats=1):
        """
        :param random_state:
            Used to seed random number generators when generating randomised splits
        :param n_splits:
            Number of splits
        :param policy_type:
            Type of k_fold policy, one of 'k_fold', 'stratified_k_fold', 'repeated_k_fold',
            or 'repeated_stratified_k_fold'
        :param n_repeats:
            For policies that repeat, the number of repeats to do.
        """

        self._random_state: Optional[int] = random_state
        self._n_splits: int = n_splits
        self._policy_type: str = policy_type
        self._shuffle = False
        self._n_repeats = n_repeats

    @property
    def n_splits(self):
        return self._n_splits

    @property
    def n_repeats(self):
        return self._n_repeats

    @property
    def type(self):
        return self._policy_type

    @property
    def random_state(self):
        return self._random_state

    @property
    def shuffle(self):
        return self._shuffle

    def set_policy_type(self, value):
        """ Set the policy type.

        value must be one of: 'k_fold', 'stratified_k_fold', 'repeated_k_fold',
            or 'repeated_stratified_k_fold'
        """
        
        assert value in ('k_fold', 'stratified_k_fold', 'repeated_k_fold', 'repeated_stratified_k_fold')
        self._policy_type = value

        # Returning self on the setters allows method chaining where multiple attributes can be set in one statement,
        # e.g. sp below will incorporate the changes specified via the set methods:
        #
        # sp = SplitPolicy().set_policy_type(p).set_n_splits(5)
        return self

    def set_n_splits(self, value):
        """ Set the number of splits to generate. """

        self._n_splits = value
        return self

    def set_n_repeats(self, value):
        """ Set the number of repeats. """
        
        self._n_repeats = value
        return self

    def set_random_state(self, value):
        """ Set the random_state used to seed the random number generators. """
        
        self._random_state = value
        return self

    def set_shuffle(self, value):
        """ Set whether to shuffle the data or not. """
        
        self._shuffle = value
        return self

    def build(self):
        """ Builds the scikit-learn k fold policy, based on the attribute values."""

        if self.type == 'k_fold':
            return ms.KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=self.shuffle)
        if self.type == 'repeated_k_fold':
            return ms.RepeatedKFold(n_splits=self.n_splits,
                                    n_repeats=self.n_repeats,
                                    random_state=self.random_state)
        if self.type == 'stratified_k_fold':
            return ms.StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=self.shuffle)
        if self.type == 'repeated_stratified_k_fold':
            return ms.RepeatedStratifiedKFold(n_splits=self.n_splits, random_state=self.random_state,
                                              n_repeats=self.n_repeats)
        raise ValueError(f'Unknown fold type: [{self.type}]')

    # Provide some potentially useful defaults.
    @classmethod
    def kfold_default(cls):
        """ A default k_fold policy, using 3 splits, no shuffling. """

        return cls()

    @classmethod
    def repeated_kfold_default(cls):
        """ A default repeated k_fold policy, using 3 splits and 3 repeats. """

        return cls(n_repeats=3, policy_type='repeated_k_fold')

    @classmethod
    def stratified_kfold_default(cls):
        """ A default stratified k_fold policy using 3 splits, no shuffling. """

        return cls(policy_type='stratified_k_fold')

    @classmethod
    def repeated_stratified_kfold_default(cls):
        """ A default repeated stratified policy, with 3 repeats and 3 splits. """

        return cls(n_repeats=3, policy_type='repeated_stratified_k_fold')


class CrossValidatorConfig:
    """ Configuration for the cross validator.:
    """

    def __init__(self, split_policy=None, eval_metrics=(EvalMetrics.ACCURACY,), n_jobs=-1, verbose=0,
                 average='macro'):
        """
        :param split_policy:
            The SplitPolicy to use. If None specified will use the kfold default.
        :param eval_metrics:
            The evaluaiton metrics to use.
        :param n_jobs:
            The number of jobs to run in parallel.
        :param verbose:
            The verbosity level.
        :param average:
            Type of averaging of metrics across classes to do.
        """

        self._policy = split_policy if split_policy else SplitPolicy.kfold_default()
        self._eval_metrics = eval_metrics
        self._n_jobs = n_jobs
        self._verbose = verbose
        self._average: Optional[Text] = average

    @property
    def average(self):
        return self._average

    def set_average(self, value):
        """ Set the averaging method to use. """

        self._average = value
        return self

    @property
    def policy(self):
        return self._policy

    @property
    def eval_metrics(self):
        return self._eval_metrics

    def set_eval_metrics(self, metrics: List[EvalMetrics]):
        """ Set the evaluation metrics to use. """

        self._eval_metrics = metrics
        return self

    @property
    def n_jobs(self):
        return self._n_jobs

    @property
    def verbose(self):
        return self._verbose

    def set_n_jobs(self, val):
        """ Set the number of jobs to run in parallel. """
        self._n_jobs = val
        return self

    def set_verbose(self, value):
        """ Set the verbosity level. """

        self._verbose = value
        return self

    def set_policy(self, policy):
        """ Set the splitting policy. """

        self._policy = policy
        return self


class CrossValidationResult:
    """ Encapsulates a CV result. """

    def __init__(self, scores):
        self.scores: Dict[str, Iterable[Number]] = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, item):
        """ Return the value stored against the specified item in the scores dictionary. """

        return self.scores[item]

    def mean(self):
        """ Compute the mean of the scores for each metric."""
        return dict([(k, np.mean(v)) for k, v in self.scores.items()])
    
    def std(self):
        """ Compute the standard deviation of the scores for each metric."""
        return dict([(k, np.std(v)) for k, v in self.scores.items()])


class CrossValidator:
    """ Perform cross validation according to the specified configuration. """

    def __init__(self, config, model, dataset):
        self.config: CrossValidatorConfig = config
        self.model = model
        self.dataset: Dataset = dataset
        self.policy = self.config.policy.build()

    def get_cross_validation_scores(self):
        """ Returns the scores from running a cross validation in the form of a CrossValidationResylt object. """

        if len(self.config.eval_metrics) == 1:
            metric: EvalMetrics = self.config.eval_metrics[0]
            scores = ms.cross_val_score(self.model, self.dataset.inputs, self.dataset.labels, cv=self.policy,
                                        scoring=metric.value, n_jobs=self.config.n_jobs, verbose=self.config.verbose)
            return CrossValidationResult({metric.value: scores})

        metrics_dict = metrics_to_scoringdict(self.config.eval_metrics, 'macro')
        scores = ms.cross_validate(self.model, self.dataset.inputs, self.dataset.labels, cv=self.policy,
                                   return_train_score=False, scoring=metrics_dict, n_jobs=self.config.n_jobs,
                                   verbose=self.config.verbose)
        return CrossValidationResult(scores)
