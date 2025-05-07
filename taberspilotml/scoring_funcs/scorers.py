"""******** SCORING FUNCTIONS - SCORERS  ******** """
from typing import Sequence, Optional, Tuple, Union, List, Dict
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

import taberspilotml.base_helpers as bh
import taberspilotml.scoring_funcs.evaluation_metrics as em
from taberspilotml import constants
from taberspilotml.conf import configs as dicts
from taberspilotml.conf.configs import models, scoring_metrics
from taberspilotml.pre_modelling import handle_nulls
from taberspilotml.scoring_funcs import cross_validation as cv
from taberspilotml.scoring_funcs import datasets as d
from taberspilotml.decorators import time_performance_decor, gc_collect_decor


class EvaluationResults:
    def __init__(self, scores: Dict[str, Tuple[float, float]] | List[float],
                 evaluation_metrics: Sequence[em.EvalMetrics]) -> None:
        """
        Initialize the EvaluationResults instance.

        :param scores: A dictionary of metric names and their corresponding (mean, std) values or a list of float values.
        :param evaluation_metrics: A sequence of evaluation metrics to be used for testing.
        """
        self.scores = scores
        self.evaluation_metrics = evaluation_metrics

        # Perform validation based on whether there is one or more evaluation metrics
        if len(self.evaluation_metrics) > 1:
            # Validation for multiple metrics (dict format)
            self._validate_multiple_metrics()
        else:
            # Validation for a single metric (list format)
            self._validate_single_metric()

    def _validate_multiple_metrics(self):
        """Validate the scores for multiple evaluation metrics."""
        if not isinstance(self.scores, dict):
            raise TypeError("Expected scores to be a dictionary for multiple metrics.")
        if len(self.scores) <= 1:
            raise ValueError("Expected more than one metric in the scores dictionary.")
        if not all(isinstance(metric_name, str) for metric_name in self.scores.keys()):
            raise ValueError("All metric names should be strings.")
        if not all(isinstance(value, tuple) and len(value) == 2 for value in self.scores.values()):
            raise ValueError("Each dictionary value should be a tuple with two elements.")
        if not all(isinstance(value[0], float) and isinstance(value[1], float) for value in self.scores.values()):
            raise ValueError("Each tuple should contain two float values.")

    def _validate_single_metric(self):
        """Validate the scores for a single evaluation metric."""
        if not isinstance(self.scores, dict):
            raise TypeError("Expected scores to be a dict for a single metric.")
        if len(self.scores) != 1:
            raise ValueError("Expected the dict to contain exactly two values.")

    def __repr__(self):
        """String representation of the EvaluationResults object."""
        return f"EvaluationResults(scores={self.scores}, evaluation_metrics={self.evaluation_metrics})"


@time_performance_decor
@gc_collect_decor
def get_cross_validation_score(dataset: d.Dataset, model=models['clf']['RF'],
                               split_policy=cv.SplitPolicy.kfold_default(),
                               averaging_policy=None,
                               evaluation_metrics: Sequence[
                                   em.EvalMetrics] = (
                                       em.EvalMetrics.ACCURACY,),
                               n_jobs=-1, verbose=0) -> Dict[str, Tuple[float, float]]:
    """
    Perform cross-validation and return evaluation results.

    Returns:
        Dict[str, Tuple[float, float]]: A dictionary mapping metric names to (mean, std) tuples.
    """

    cv_result = cv.get_cv_scores(model, dataset, evaluation_metrics, split_policy.build(), n_jobs, verbose,
                                 averaging_policy)

    # To ensure this function continues to do what it originally did, we take the means here.
    # cv_result however contains scores for each split, so other users can do what they want with that.
    means = cv_result.mean()
    std = cv_result.std()
    results = dict()
    if len(evaluation_metrics) > 1:

        for eval_metric in evaluation_metrics:
            metric_name = f'test_{eval_metric.value}'
            results[metric_name] = (means[metric_name], std[metric_name])
        return EvaluationResults(scores=results, evaluation_metrics=evaluation_metrics).scores

    results[evaluation_metrics[0].value] = (means[evaluation_metrics[0].value], std[evaluation_metrics[0].value])

    return EvaluationResults(scores=results, evaluation_metrics=evaluation_metrics).scores


@time_performance_decor
@gc_collect_decor
def get_hold_out_score(
        df: Optional[pd.DataFrame] = None,
        target_label: Optional[str] = None,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        return_all: bool = False,
        classification: bool = True,
        model=models['clf']['RF'],
        test_size: float = 0.2,
        evaluation_metrics: Sequence[
            em.EvalMetrics] = (
                em.EvalMetrics.ACCURACY,)
) -> Union[Dict[str, Tuple[float, float]], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Computes a hold-out validation score.

    Returns:
        - If `return_all` is False: A dictionary mapping metric names to (mean, std) tuples.
        - If `return_all` is True: A tuple `(x_train, x_test, y_train, y_test, y_pred)`.
    """

    def get_multi_metrics_result(evaluation_metrics):

        assert len(evaluation_metrics) > 1

        results = dict()
        for eval_metric in evaluation_metrics:
            scorer = scoring_metrics['clf'][eval_metric.value] if classification else scoring_metrics['reg'][
                eval_metric.value]
            results[eval_metric.value] = (scorer(y_test, y_pred), np.nan)

        return results

    if isinstance(df, pd.DataFrame) and x is None and y is None:
        print('Generating internal x,y')
        x, y = bh.get_x_y_from_df(df, target_label)

    size = int(x.shape[0] * test_size)
    x_train, x_test, y_train, y_test = x[size:], x[:size], y[size:], y[:size]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    if return_all:
        return x_train, x_test, y_train, y_test, y_pred

    else:

        if len(evaluation_metrics) > 1:
            return EvaluationResults(scores=get_multi_metrics_result(evaluation_metrics),
                                     evaluation_metrics=evaluation_metrics).scores
        else:
            scorer = scoring_metrics['clf'][evaluation_metrics[0].value] if classification else scoring_metrics['reg'][
                evaluation_metrics[0].value]
            result = dict()
            result[evaluation_metrics[0].value] = (scorer(y_test, y_pred), np.nan)
            return EvaluationResults(scores=result,
                                     evaluation_metrics=evaluation_metrics).scores


def get_custom_cv_score(df: pd.DataFrame, target_label: str, classification: bool, evaluation_metric: str, model,
                        test_size=0.2, use_custom_method=False, custom_method=None, *args, **kwargs):
    """
    Acceptable test sizes: 0.15 , 0.2 , 0.3
    Description:
    1. Pick the folds based on test size(to have equal samples between train and test)
    2. Use current train or apply a change to train
    3. Depending on the custom method used we decide to alter the test or not
    """

    scorer = scoring_metrics['clf'][evaluation_metric] if classification else scoring_metrics['reg'][evaluation_metric]
    size = int(len(df) * test_size)
    folds_lst = [i for i in range(0, len(df), size)]
    test_index = []
    scores = []

    for i in range(len(folds_lst) - 1):
        print(f'Training fold: {i}')
        test_index.append(df[folds_lst[i]:folds_lst[i + 1]].index.values)
        test = df.iloc[test_index[i]]
        train_idx = list(set(test_index[i]) ^ set(df.index.values))
        train = df.loc[train_idx]
        current_train = train if not use_custom_method else custom_method(train, *args, **kwargs)
        x_train, y_train = bh.get_x_y_from_df(current_train, target_label)

        if custom_method in [handle_nulls.get_imputed_x, handle_nulls.drop_nulls]:
            test = test.copy()
            current_test = custom_method(test, *args, **kwargs)
            x_test, y_test = bh.get_x_y_from_df(current_test, target_label)
        else:
            x_test, y_test = bh.get_x_y_from_df(test, target_label)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        scores.append(scorer(y_test, y_pred))
        print(f'Custom cv scores:{scores}')

    return np.mean(scores), np.std(scores)


def get_time_series_cv_score(df: pd.DataFrame, target_label: str, n_splits: int, model_name: str,
                             evaluation_metric='accuracy', classification=True):
    """Applies cross validation to a time series dataset"""

    tscv = TimeSeriesSplit(n_splits=n_splits)
    x, y = bh.get_x_y_from_df(df, target_label)

    models_dict = models['clf'] if classification else models['reg']
    model = bh.select_custom_dict(models_dict, [model_name])[model_name]
    scorer = scoring_metrics['clf'][evaluation_metric] if classification else scoring_metrics['reg'][evaluation_metric]
    scores = []
    for train_index, test_index in tscv.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print(f'Train shape: {x_train.shape}')
        print(f'Test shape: {x_test.shape}')
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        score = scorer(y_test, y_pred)
        scores.append(score)

    print(f'Scores: {np.mean(scores)}')

    return np.mean(scores)


def get_scaled_x_score(df, target_label, model_name='RF', scaler_name='MinMax', use_transformers=False,
                       transformer_name=None,
                       k_fold_method='k_fold', n_folds=5, n_repeats=3, classification=True,
                       evaluation_metric='accuracy'):
    """Test scaled version of x. This will return the mean and standard deviation"""

    models_dict = dicts.models['clf' if classification else 'reg']
    model = models_dict[model_name]

    output_df_arr = bh.scale_x(df, target_label, scaler_name, use_transformers=use_transformers,
                               transformer_name=transformer_name)

    dataset = d.Dataset.from_dataframe(output_df_arr, [target_label])
    policy = cv.SplitPolicy(policy_type=k_fold_method, n_splits=n_folds, n_repeats=n_repeats,
                            shuffle=True, random_state=constants.DEFAULT_SEED)

    scores = get_cross_validation_score(dataset=dataset, model=model, split_policy=policy,
                                        evaluation_metrics=[
                                            em.EvalMetrics.from_str(
                                                evaluation_metric)])

    scores = scores[em.EvalMetrics.from_str(evaluation_metric).value]

    return scores[0], scores[1]
