"""******** OUTLIERS (This section may require its own file) ******** """

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from taberspilotml import base_helpers as h
from taberspilotml import constants
from taberspilotml.configs import scoring_metrics
import taberspilotml.visualization as vs
import taberspilotml.configs as dicts
from taberspilotml.scoring_funcs import cross_validation as cv
from taberspilotml.scoring_funcs.datasets import Dataset
from taberspilotml.scoring_funcs import evaluation_metrics as ev
from taberspilotml.scoring_funcs import scorers as scorers
from taberspilotml.scoring_funcs.scorers import get_custom_cv_score


def handle_outliers(df, target_label, tot_outlier_pct=4, classification=True,
                    model=RandomForestClassifier(), evaluation_metric='accuracy', test_size=0.2, n_folds=5,
                    n_repeats=10):
    """
    Algorithms sensitive to outliers: Linear Regression, ADA boost
    Args:
        df:
        target_label:
        tot_outlier_pct:
        classification:
        model:
        evaluation_metric:
        test_size:
        n_folds:
        n_repeats:

    Returns:

    """
    # Load initial functions to be used
    funcs_to_eval = {'replace_outliers': replace_outliers, 'drop_outliers': drop_outliers}

    scores = {}

    print('Replace values')
    for strategy in ['mean', 'median']:
        print(strategy)

        replace_score, replace_std = scorers.get_custom_cv_score(df=df, target_label=target_label,
                                                                 classification=classification,
                                                                 evaluation_metric=evaluation_metric, model=model,
                                                                 test_size=test_size, tot_outlier_pct=tot_outlier_pct,
                                                                 strategy=strategy, use_custom_method=True,
                                                                 custom_method=replace_outliers)
        scores[f'replace_outliers-{strategy}'] = (replace_score, replace_std)

    print('Dropping values')

    drop_score, drop_std = scorers.get_custom_cv_score(df=df, target_label=target_label,
                                                       classification=classification,
                                                       evaluation_metric=evaluation_metric,
                                                       model=model, test_size=test_size,
                                                       tot_outlier_pct=tot_outlier_pct,
                                                       use_custom_method=True, custom_method=drop_outliers)

    scores[f'drop_outliers-{None}'] = (drop_score, drop_std)

    print(f'General scores: {scores}')

    h.printy('Generating final output', text_type='subtitle')

    function_params = {'df': df, 'target_label': target_label,
                       'tot_outlier_pct': tot_outlier_pct, 'classification': classification, 'model': model,
                       'evaluation_metric': evaluation_metric, 'test_size': test_size, 'n_folds': n_folds,
                       'n_repeats': n_repeats}

    func, params = h.get_func_params(scores, input_params={'func': None, 'strategy': str},
                                     classification=classification)
    output_df = pd.DataFrame()
    try:
        output_df = h.get_output_df_wrapper(function_params=function_params, funcs_to_eval=funcs_to_eval,
                                            function_name=func,
                                            params=params)
    except KeyError as err:
        print(f'You may need to include param {err} in dict function_params')

    # Run cross validation in the whole data set to return final results
    print('Computing standard cross_validation with best method...')
    a, b, best_method = h.get_best_score(scores, classification=classification)

    output_ds = Dataset.from_dataframe(output_df, [target_label])

    policy = cv.SplitPolicy(policy_type='k_fold', n_splits=n_folds, n_repeats=n_repeats,
                            random_state=constants.DEFAULT_SEED, shuffle=True)
    final_cv_score = scorers.get_cross_validation_score(dataset=output_ds,
                                                        model=model,
                                                        evaluation_metrics=[ev.EvalMetrics.from_str(evaluation_metric)],
                                                        split_policy=policy,
                                                        n_jobs=-1, verbose=0)

    output_dict = {best_method: final_cv_score}

    print(f'Results:{final_cv_score}')

    return output_dict, output_df


def handle_outliers_iso_forest(df: pd.DataFrame, target_label: str, classification: bool, evaluation_metric: str,
                               model, test_size=0.2):
    scorer = scoring_metrics['clf'][evaluation_metric] if classification else scoring_metrics['reg'][evaluation_metric]
    x_train, x_test, y_train, y_test = h.get_splits_wrapper(df, target_label, train_split=True,
                                                            test_size=test_size)

    x_train = x_train.values
    x_test = x_test.values

    iso = IsolationForest(contamination=0.1)
    y_hat = iso.fit_predict(x_train)
    mask = y_hat != -1
    x_train, y_train = x_train[mask, :], y_train[mask]

    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    score = round(scorer(y_test, y_hat), 2)

    return score


def is_distribution_normal(col: pd.Series):
    """Check if the distribution of a col is normal"""

    mean = col.mean()
    sd = col.std()

    one_sd = norm.cdf(sd, mean, sd) - norm.cdf(-sd, mean, sd)
    two_sd = norm.cdf(2 * sd, mean, sd) - norm.cdf(-2 * sd, mean, sd)
    three_sd = norm.cdf(3 * sd, mean, sd) - norm.cdf(-3 * sd, mean, sd)

    counter = 0

    if 0.68 <= one_sd < 0.69:
        counter += 1

    if 0.95 <= two_sd < 0.96:
        counter += 1

    if 0.99 <= three_sd < 1:
        counter += 1

    if counter == 3:
        return True
    else:
        return False


def get_outliers_std(df: pd.DataFrame, column: str):
    len_df = len(df)

    q25, q75 = np.percentile(df[column], 25), np.percentile(df[column], 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off

    outliers = [x for x in df[column] if x < lower or x > upper]
    pct_outliers = round(len(outliers) / len_df * 100, 2)

    return pct_outliers, outliers


def get_outliers_z_score(df: pd.DataFrame, column: str):
    len_df = len(df)
    outliers = []
    threshold = 3
    mean = np.mean(df[column])
    std = np.std(df[column])
    for i in df[column]:
        z_score = (i - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(i)
    # print(f'Identified outliers for {column}: {round(len(outliers) / len_df * 100, 2)}%')
    pct_outliers = round(len(outliers) / len_df * 100, 2)

    return pct_outliers, outliers


def get_outliers(df: pd.DataFrame, show_graph=True):
    """
    Info..
    Parameters:
    argument1 (int): Description of arg1

    Returns:
    int:Returning value
    """

    num_cols = list(df.select_dtypes([int, float]).columns)
    outliers_dict = {}
    outliers_values = {}

    for column in num_cols:
        # If distribution is not normal we use std to get outliers
        if not is_distribution_normal(df[column]):

            pct_outliers, outliers = get_outliers_std(df, column)
            outliers_values[column] = outliers
            outliers_dict[column] = pct_outliers
        else:
            # If distribution is normal we use z_score to get outliers
            pct_outliers, outliers = get_outliers_z_score(df, column)
            outliers_values[column] = outliers
            outliers_dict[column] = pct_outliers
    if show_graph:
        vs.get_graph(input_data=outliers_dict, stage='Feature Engineering', horizontal=True, figsize=(6, 7),
                     fig_title=f'Outliers detected', x_title='Pct of total', y_title='Outliers', sort_type='desc',
                     save_figure=False, file_name='current_fig')

    return outliers_dict, outliers_values


def replace_outliers(df: pd.DataFrame, tot_outlier_pct: int, strategy: str):
    df = df.copy()

    outliers = get_outliers(df=df, show_graph=False)
    outliers_pct, outliers_values = outliers
    top_outliers_cols = [col for col, score in outliers_pct.items() if score >= tot_outlier_pct]

    current_method = dicts.replace_methods[strategy]

    for col in top_outliers_cols:
        current_method_value = current_method(df[col])
        outliers_lst = outliers_values[col]
        df[col].replace(outliers_lst, current_method_value, inplace=True)

    return df


def drop_outliers(df: pd.DataFrame, tot_outlier_pct: int):
    df = df.copy()

    outliers_pct, outliers_values = get_outliers(df=df, show_graph=False)
    top_outliers_cols = [col for col, score in outliers_pct.items() if score >= tot_outlier_pct]

    outliers_idx_lst = []
    for col in top_outliers_cols:
        for value in outliers_values[col]:
            index_int = df[df[col] == value].index[0]
            if index_int not in outliers_idx_lst:
                outliers_idx_lst.append(index_int)

    for i in range(len(outliers_idx_lst)):
        df.drop(outliers_idx_lst[i], axis=0, inplace=True)

    return df


def get_score_replace_outliers(df, target_label, tot_outlier_pct, method, classification,
                               evaluation_metric, model, test_size=0.2):
    print('Generating scores - replacing outliers...')
    scores, scores_std = get_custom_cv_score(df=df, target_label=target_label,
                                             classification=classification,
                                             evaluation_metric=evaluation_metric, model=model,
                                             test_size=test_size, tot_outlier_pct=tot_outlier_pct,
                                             method=method, use_custom_method=True,
                                             custom_method=replace_outliers)
    print('Generating output df... ')
    df_output = replace_outliers(df, tot_outlier_pct, method)

    return scores, scores_std, df_output


def get_score_drop_outliers(df, target_label, tot_outlier_pct, classification, evaluation_metric,
                            model, test_size=0.2):
    print('Generating scores - drop outliers...')
    scores, scores_std = get_custom_cv_score(df=df, target_label=target_label,
                                             classification=classification, evaluation_metric=evaluation_metric,
                                             model=model, test_size=test_size,
                                             tot_outlier_pct=tot_outlier_pct, method=None,
                                             use_custom_method=True,
                                             custom_method=drop_outliers)
    print('Generating output df... ')
    df_output = drop_outliers(df, tot_outlier_pct)

    return scores, scores_std, df_output
