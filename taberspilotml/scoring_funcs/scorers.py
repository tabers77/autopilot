"""******** SCORING FUNCTIONS - SCORERS  ******** """
from typing import Sequence

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

import taberspilotml.base_helpers as bh
import taberspilotml.scoring_funcs.evaluation_metrics
from taberspilotml import constants
from taberspilotml import configs as dicts
from taberspilotml.configs import models, scoring_metrics
from taberspilotml.pre_modelling import handle_nulls
from taberspilotml.scoring_funcs import cross_validation as cv
from taberspilotml.scoring_funcs import datasets as d
from taberspilotml.decorators import time_performance_decor, gc_collect_decor


@time_performance_decor
@gc_collect_decor
def get_cross_validation_score(dataset: d.Dataset, model=models['clf']['RF'],
                               split_policy=cv.SplitPolicy.kfold_default(),
                               averaging_policy=None,
                               evaluation_metrics: Sequence[taberspilotml.scoring_funcs.evaluation_metrics.EvalMetrics] = (
                                       taberspilotml.scoring_funcs.evaluation_metrics.EvalMetrics.ACCURACY,),
                               n_jobs=-1, verbose=0):
    """

    :param dataset:
        The dataset to use.
    :param model:
        The model to use.
    :param split_policy:
        Cross validation split policy.
    :param averaging_policy:
        Averaging policy when using multiple evaluation metrics
    :param evaluation_metrics:
        The evaluation metrics
    :param n_jobs:
        Number of jobs to run in parallel. -1 will use the default value.
    :param verbose:
        The verbosity level
    :return:
        Either the means for each metric or the mean, std dev for the supplied metric if there is only one metric.
    """

    cv_result = cv.get_cv_scores(model, dataset, evaluation_metrics, split_policy.build(), n_jobs, verbose,
                                 averaging_policy)

    # To ensure this function continues to do what it originally did, we take the means here.
    # cv_result however contains scores for each split, so other users can do what they want with that.
    means = cv_result.mean()
    std = cv_result.std()
    if len(evaluation_metrics) > 1:
        return means
    return [means[evaluation_metrics[0].value], std[evaluation_metrics[0].value]]


@time_performance_decor
@gc_collect_decor
def get_hold_out_score(df=None, target_label=None, x=None, y=None, return_all=False, classification=True,
                       model=models['clf']['RF'], test_size=0.2,
                       evaluation_metric='accuracy'):
    """

    Args:
        df:
        target_label:
        x:
        y:
        return_all:
        classification:
        model:
        test_size:
        evaluation_metric:

    Returns:

    """
    scorer = scoring_metrics['clf'][evaluation_metric] if classification else scoring_metrics['reg'][evaluation_metric]
    if isinstance(df, pd.DataFrame) and x is None and y is None:
        print('Generating internal x,y')
        x, y = bh.get_x_y_from_df(df, target_label)

    size = int(x.shape[0] * test_size)
    x_train, x_test, y_train, y_test = x[size:], x[:size], y[size:], y[:size]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    if not return_all:
        return scorer(y_test, y_pred)
    else:
        return x_train, x_test, y_train, y_test, y_pred


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
                                            taberspilotml.scoring_funcs.evaluation_metrics.EvalMetrics.from_str(evaluation_metric)])

    return scores[0], scores[1]
