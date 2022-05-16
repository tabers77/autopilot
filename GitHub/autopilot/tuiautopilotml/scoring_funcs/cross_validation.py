""" Code to support running cross validations """
from dataclasses import dataclass
from numbers import Number
from typing import Dict, Iterable, Sequence, Optional

import numpy as np
from sklearn import model_selection as ms

from tuiautopilotml import constants
from tuiautopilotml.scoring_funcs.datasets import Dataset
from tuiautopilotml.scoring_funcs.evaluation_metrics import EvalMetrics, metrics_to_scoringdict


@dataclass
class SplitPolicy:
    """ Encapsulates configuration for building scikit-learn cross validation policies. """

    random_state: Optional[int] = None
    """ Used to seed random number generators when generating randomised splits """

    n_splits: int = 3
    """ Number of splits """

    policy_type: str = 'k_fold'
    """ Type of k_fold policy, one of 'k_fold', 'stratified_k_fold', 'repeated_k_fold',
            or 'repeated_stratified_k_fold'. """

    shuffle: bool = False
    """ Whether to shuffle. """

    n_repeats: int = 1
    """ Number of repeats. """

    def build(self):
        """ Builds the scikit-learn k fold policy, based on the attribute values."""
        if self.policy_type == 'k_fold':
            return ms.KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=self.shuffle)
        if self.policy_type == 'repeated_k_fold':
            return ms.RepeatedKFold(n_splits=self.n_splits,
                                    n_repeats=self.n_repeats,
                                    random_state=self.random_state)
        if self.policy_type == 'stratified_k_fold':
            return ms.StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=self.shuffle)
        if self.policy_type == 'repeated_stratified_k_fold':
            return ms.RepeatedStratifiedKFold(n_splits=self.n_splits, random_state=self.random_state,
                                              n_repeats=self.n_repeats)
        raise ValueError(f'Unknown fold type: [{self.policy_type}]')

    # Provide some potentially useful defaults.
    @classmethod
    def kfold_default(cls):
        """ A default k_fold policy, using 3 splits, no shuffling. """

        return cls(random_state=constants.DEFAULT_SEED, n_splits=5, n_repeats=10, shuffle=True)

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


def get_cv_scores(model, dataset: Dataset, eval_metrics: Sequence[EvalMetrics], split_policy,
                  n_jobs: int, verbose: int, averaging_policy: Optional[str]) -> CrossValidationResult:
    """
    Perform cross validation and then return the score(s).

    :param model:
        The model
    :param dataset:
        The dataset
    :param eval_metrics:
        The evaluation metrics
    :param split_policy:
        Policy determining how folds are constructed.
    :param n_jobs:
        Number of jobs to run in parallel : -1 will select the default to match number of cores.
    :param verbose:
        Verbosity level
    :param averaging_policy:
        Polciy for averaging when returning scores for multiple classes.
    :return:
        CrossValidationResult containing the results of the cross validation.
    """

    if len(eval_metrics) == 1:
        metric = eval_metrics[0]
        scores = ms.cross_val_score(model, dataset.inputs, dataset.labels, cv=split_policy,
                                    scoring=metric.value, n_jobs=n_jobs, verbose=verbose)
        return CrossValidationResult({metric.value: scores})

    metrics_dict = metrics_to_scoringdict(eval_metrics, averaging_policy)
    scores = ms.cross_validate(model, dataset.inputs, dataset.labels, cv=split_policy,
                               return_train_score=False, scoring=metrics_dict, n_jobs=n_jobs,
                               verbose=verbose)
    return CrossValidationResult(scores)
