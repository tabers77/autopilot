""" Helper functions (and supporting code) """

import logging
import os
import re

# INIT PACKAGES
from datetime import datetime as dt
from distutils import util
from inspect import getfullargspec
from time import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from keras.models import Sequential
from pandas.api import types as ptypes

# SKLEARN HELP FUNCS
from sklearn.model_selection import train_test_split

# SCALERS
from sklearn.preprocessing import StandardScaler

# LOCAL LIBRARIES
from tuiautopilotml import constants
from tuiautopilotml.scoring_funcs import datasets as d, scorers as scorers, cross_validation as cv
from tuiautopilotml.scoring_funcs import evaluation_metrics as em
import tuiautopilotml.dicts as dicts
from tuiautopilotml import mlflow_uploader as mf
from tuiautopilotml.pre_modelling import encoders as enc

logging.basicConfig(level=logging.INFO)


def printy(text: str, text_type='normal', p1=None, p2=None):

    text_len = len(text)
    if text_type == 'normal':
        print(text)
    elif text_type == 'title':
        print('-' * text_len)
        print(text)
        print('-' * text_len)
    elif text_type == 'subtitle':
        print(f'----------- {text} -----------')
    elif text_type == 'custom':
        custom_text_len = len(f'{text}:{p1}: {p2}')
        print('*' * custom_text_len)
        print(f'{text}:{p1}: {p2}')
        print('*' * custom_text_len)


def now_as_timestamp_string():
    now = dt.now()
    time_stamp = now.strftime("%D-%H:%M")
    time_stamp = '-'.join(re.split('/', time_stamp))

    return time_stamp


def extract_day_month_year_and_weekday(df: pd.DataFrame, date_col: str):
    """Observe that date variables should not be used as categorical when building a machine learning model"""

    copied_df = df.copy()
    copied_df[f'{date_col}_day'] = copied_df[date_col].apply(lambda x: x.day)
    copied_df[f'{date_col}_month'] = copied_df[date_col].apply(lambda x: x.month)
    copied_df[f'{date_col}_year'] = copied_df[date_col].apply(lambda x: x.year)
    copied_df[f'{date_col}_weekday'] = copied_df[date_col].apply(lambda x: x.weekday())
    # copied_df[f'{date_col}_is_weekend'] = copied_df[f'{date_col}_weekday'].apply(lambda x: 1 if x in (5, 6) else 0)
    # copied_df[f'{date_col}_week_number'] = copied_df[date_col].apply(lambda x: x.week)

    return copied_df


def dict_to_df(input_dict: dict, multiple_eval_scores: bool, evaluation_metric=None):
    output_dataframe = pd.DataFrame(input_dict).transpose() if multiple_eval_scores else pd.DataFrame(
        input_dict.items(), columns=['algorithms', evaluation_metric])

    return output_dataframe


def select_custom_dict(input_dict: dict, custom_lst: list):
    output_dict = {}
    for name, model in input_dict.items():
        if name in custom_lst:
            output_dict[name] = model

    return output_dict


def from_str_to_bool(i):
    return bool(util.strtobool(i))


"""******** SECTION: DATA VALIDATION ********"""


def is_col_date(df: pd.DataFrame, col: str):
    return (ptypes.is_datetime64_dtype(df[col]) or
            ptypes.is_timedelta64_dtype(df[col]) or
            ptypes.is_datetime64_ns_dtype(df[col]))


def is_object_date(df: pd.DataFrame, col: str):
    if df[col].dtypes == 'O' and df[col].shape[0] > 1:
        try:
            pd.to_datetime(df[col])
            return True
        except (TypeError, ValueError):
            return False

    elif df[col].dtypes == 'O' and df[col].shape[0] == 1 and not df[col].isnull().sum().any():
        try:
            pd.to_datetime(df[col])
            return True
        except (TypeError, ValueError):
            return False

    elif df[col].dtypes == 'O' and df[col].shape[0] == 1:

        return False
    else:
        pass


def contains_object(df: pd.DataFrame) -> bool:
    """ Returns True if dataframe contains columns of dtype object, False otherwise. """

    return len(df.select_dtypes('object').columns) != 0


def contains_nulls(df: pd.DataFrame) -> bool:
    """ Returns True if the dataframe contains nulls, False otherwise. """

    return df.isnull().sum().any()


def convert_to_int_float_date(df: pd.DataFrame, object_is_numerical_cols=None):
    """
    Convert to int , float and dates
    """
    if object_is_numerical_cols is None:
        object_is_numerical_cols = []

    df = df.copy()
    for col in df.columns:
        if df[col].dtypes == int and col not in object_is_numerical_cols:
            df[col] = df[col].astype(int)
        elif df[col].dtypes == float and col not in object_is_numerical_cols:
            df[col] = df[col].astype(float)
        elif is_object_date(df, col):
            df[col] = pd.to_datetime(df[col])
        elif col in object_is_numerical_cols:
            df[col] = df[col].astype(str)

    print(f'{convert_to_int_float_date.__name__} DONE')

    return df


def stats_checks(df):
    """Show statistical related info"""
    low_cardinality_cols = [cname for cname in df if df[cname].nunique() <= 10 and
                            df[cname].dtype == "object"]

    print(f'Low cardinality columns: {low_cardinality_cols}')


def df_sanity_check(df: pd.DataFrame):
    """
    Performs some basic sanity checking on the supplied dataframe.

    Currently, this does the following checks:
    1. Checks for null values
    2. Checks for strings that could be numeric
    3. Checks for objects that could be in date format

    :param df: The dataframe to check.
    :type df: pd.DataFrame

    :rtype: int
    """

    passed_test_count = 0
    n_tests = 3

    if not contains_nulls(df):
        passed_test_count += 1
    else:
        print('There are missing values in your dataset')

    count2 = 0
    for col in list(df.select_dtypes('object').columns):
        if df[col].str.isdigit().any():
            count2 += 1
    if count2 == 0:
        passed_test_count += 1
    else:
        print('There is numerical data in string format')

    date_objects = 0
    object_cols = df.select_dtypes('object').columns
    for col in object_cols:
        if is_object_date(df, col):
            date_objects += 1
            print(f'Date column {col} is in object format')
    if date_objects != 0:
        pass
    else:
        passed_test_count += 1

    failures = n_tests - passed_test_count
    print(f'Ran {n_tests} checks on the dataframe')
    if passed_test_count == n_tests:
        print('All the checks were passed..')
    else:
        print(f'Sanity checks FAILED: (failures={failures})')

    return failures


"""******** GET SPLITS ******** """


def get_x_y_from_df(df: pd.DataFrame, target_label: str, scaled_df=False, scaler=dicts.scalers['Standard']):
    """ Given a dataframe containing both input features and target labels,
    return the input features and target label column as separate dataframes. Optionally performs scaling.

    :param df:
        The dataframe containing input features and target labels.
    :param target_label:
        The target label column.
    :param scaled_df:
        Whether to scale the data.
    :param scaler:
        The scaler to use.
    """

    if scaled_df:
        return scaler.fit_transform(df.drop(target_label, axis=1)), df[target_label]
    return df.drop(target_label, axis=1), df[target_label]


def train_test_split_from_df(df: pd.DataFrame, test_size: float):
    """By default we will take the first part as the train set  """
    train_size = 1 - test_size
    size = int(len(df) * train_size)
    train = df[:size].reset_index(drop=True)
    test = df[size:].reset_index(drop=True)

    return train, test


def get_splits_wrapper(df: pd.DataFrame, target_label: str, train_split=False, scaled=False,
                       scaler=StandardScaler(),
                       validation_set=False, test_size=0.2, random_seed=constants.DEFAULT_SEED):
    """
    Get your initial splits
    Args:
        df:
        target_label:
        train_split:
        scaled:
        scaler:
        validation_set:
        test_size:
        random_seed:

    Returns:

    """

    dataset = d.Dataset.from_dataframe(df, [target_label], scaler if scaled else None)

    if train_split:
        if validation_set:
            x_train_base, x_test, y_train, y_test = train_test_split(
                dataset.inputs, dataset.labels, test_size=test_size, random_state=random_seed)
            x_train, x_valid, y_train, y_valid = train_test_split(x_train_base, y_train, test_size=test_size,
                                                                  random_state=random_seed)

            return x_train, x_valid, y_train, y_valid, x_test, y_test

        return train_test_split(dataset.inputs, dataset.labels, test_size=test_size, random_state=random_seed)

    return dataset.inputs, dataset.labels


""" ******** TRANSFORMATION METHODS - SCALERS ******** """


def scale_x(df, target_label, scaler_name, use_transformers=False, transformer_name=None):
    x, y = get_x_y_from_df(df, target_label)
    scaler = dicts.scalers[scaler_name]

    if use_transformers:
        transformer = dicts.transformers[transformer_name]
        transformer.fit(x)
        transformed_x = transformer.transform(x)
        current_x = scaler.fit_transform(transformed_x)
    else:
        current_x = scaler.fit_transform(x)

    columns = list(x.columns)
    columns.append(target_label)
    output_df_arr = pd.DataFrame(np.c_[current_x, y], columns=columns)
    # output_df_arr =np.c_[current_x, y] # Adjust PCA and Truncated dynamically

    return output_df_arr


"""******** OTHER ******** """


def save_figure_to_disk(df=None, main_folder=None, figure_name=None, save_as_plt=False, fig=None):
    time_stamp = now_as_timestamp_string()

    print('Saving figure/table to disk...')
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, f'AutoMLTuiRuns/{main_folder}/{time_stamp}')

    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    if isinstance(df, pd.DataFrame):
        df.to_csv(f'AutoMLTuiRuns/{main_folder}/{time_stamp}/{figure_name}.csv')
    else:
        if not save_as_plt:
            fig.savefig(f'AutoMLTuiRuns/{main_folder}/{time_stamp}/{figure_name}.png')
        else:
            plt.savefig(f'AutoMLTuiRuns/{main_folder}/{time_stamp}/{figure_name}.png')


def model_training_estimator_wrapper(func, df: pd.DataFrame, split_pct_param=0.01, patience_limit=1, *args,
                                     **kwargs):
    """

    Args:
        func:
        df:
        split_pct_param:
        patience_limit:
        *args:
        **kwargs:

    Returns: estimated time in minutes

    """
    df = df.copy()
    split_pct_param = split_pct_param
    shape = df.shape[0]
    steps = int(shape * split_pct_param)
    num_rows = [i for i in range(0, shape, steps)]
    cum_list = list(np.cumsum(num_rows)[1:])
    final_list = [i for i in cum_list if i <= shape]
    print(f'Batches are: {final_list}')
    steps_inp = input(
        f'The target to calculate is {shape}. Do you want to proceed with these number of batches {final_list} Y/N ?')

    if steps_inp == 'Y':
        results = {}
        print('Encoding and dropping missing values in order to test the function')
        df = enc.get_encoded_wrapper(df=df)
        df.dropna(axis=0, inplace=True)
        for n_rows in final_list:
            if n_rows >= 700:
                print(f'N rows: {n_rows}')
                t1 = time()
                func(dataframe=df[:n_rows], *args, **kwargs)
                t2 = time()
                current_time_minutes = round((t2 - t1) / 60, 2)
                print(current_time_minutes)
                results[n_rows] = current_time_minutes

                if current_time_minutes > patience_limit:
                    print('Breaking process after reaching time limit...')
                    break
        target_n_rows = shape
        per_n_rows = 1000
        per_n_rows_dict = {a: (b / a) * per_n_rows for a, b in results.items()}
        average_time = np.mean(list(per_n_rows_dict.values()))
        estimated_time = average_time * target_n_rows / per_n_rows * 1.1  # calculate this in advance
        print(f'It will take approximately {estimated_time} minutes to run this specific model ')
        print(results)

        return estimated_time
    else:
        print('Ok we break...')


""" ******** AUTO MODE RELATED FUNCTIONS ******** """


def choose_metric_from_multiple_eval_dict(scores: dict, metric='test_accuracy'):
    new_scores = {}
    for k, ob in scores.items():
        new_scores[k] = [ob[metric], 0]  # temporary standard deviation
    return new_scores


def get_best_score(scores: dict, classification=True, multiple_eval_scores=False):
    """
    Give a dictionary mapping method names to (mean score, standard_deviation) pairs, returns the best score and
    corresponding std_dev and method name. If there's a tie between mean scores, simply return the first item in
    the tie.

    :param scores:
        The dictionary mapping method names to scores, stds
    :param classification:
        If True, treats a higher score as better, otherwise treats a lower score as better.
    :param multiple_eval_scores:
       if True, returns a dictionary with keys and test_accuracy as score and a temporary standard deviation of 0

    :return:
        best score, std dev, method_name
    """
    scores = scores if not multiple_eval_scores else choose_metric_from_multiple_eval_dict(scores)
    score = max({v[0] for k, v in scores.items()}) if classification else min({v[0] for k, v in scores.items()})
    tups = [(mean, std) for k, (mean, std) in scores.items() if mean == score][0]

    new_scores, new_std = tups[0], tups[1]
    best_method = [k for k, v in scores.items() if v[0] == score][0]

    return new_scores, new_std, best_method


def get_latest_score(config_dict: dict):
    experiment_id = "0"
    runs_df = mlflow.search_runs([experiment_id]).sort_values('end_time', ascending=False)
    runs_df = runs_df[runs_df.status == 'FINISHED']  # exclude all failed jobs

    try:
        metric_name = f"metrics.{config_dict['evaluation_metric']}"
        expected_columns = {
            'metrics.std',
            metric_name,
            'tags.mlflow.runName',
            'status',
            'end_time'
        }
        assert expected_columns.issubset(runs_df.columns), \
            f"Some expected columns are missing from mlflow: {sorted(expected_columns - set(runs_df.columns))}"

        id_n = config_dict['run_id_number']
        lst = []
        for i in runs_df['tags.mlflow.runName']:
            if re.search(str(id_n), i):
                lst.append(i)

        first_row = runs_df[runs_df['tags.mlflow.runName'].isin(lst)].sort_values('end_time', ascending=False)[:1]

        score_mean = first_row[f"metrics.{config_dict['evaluation_metric']}"].reset_index(drop=True)
        score_std = first_row[f"metrics.std"].reset_index(drop=True)

        assert len(score_mean) > 0, f"Did not find any scores for run_id_number: {config_dict['run_id_number']}"

        if len(score_mean) == 1:
            return float(score_mean), float(score_std)
        else:
            return score_mean[0], score_std[0]

    except KeyError as err:
        print(f'The parameter {err} is missing, you need to generate a baseline score first')


def update_config(config_dict: dict = None, key=None, value=None, **kwargs):
    if key is not None and len(kwargs) == 0:
        config_dict[key] = value
    else:
        for k, v in kwargs.items():
            config_dict[k] = v


def new_score_sufficiently_better(scores: dict, config_dict: dict, classification=True):
    """ If the new score improves on the previous score by at least one standard deviation then return True.

    If no std is supplied, then we simply accept an improved score regardless.

    :param scores:
        Scores dictionary.
    :param config_dict:
        Config.
    :param classification:
        If True, assume that a higher score is better (typical for classification), otherwise a lower score is better
        (typical for regression).
    """

    mean_score, std_score = get_latest_score(config_dict=config_dict)
    new_score, new_std, best_method = get_best_score(scores, classification=classification)
    print(f'Previous scores: {mean_score, std_score}')
    print(f'New scores: {new_score, new_std}')

    if pd.isna(new_std) or pd.isna(std_score):
        return new_score > mean_score if classification else new_score < mean_score

    diff = new_score - mean_score if classification else mean_score - new_score
    return diff >= max(std_score, new_std)


def get_params_to_upload(config_dict: dict, params_keys: dict):
    """Select specific params from CONFIG to upload"""

    results = {k: v for k, v in config_dict.items() if k in params_keys}

    return results


def update_upload_config(scores: dict, config_dict: dict, run_name='run_name', result_df=None, tuned_params=None,
                         model=None):
    """ Where a new score is judged sufficiently improved compared to old, this function updates the config_dict
     to reflect this, and ensures relevant artifacts are uploaded to mlflow.

     :param scores:
        The scores.
     :param config_dict:
        Config.
     :param run_name:
        Name of run.
     :param result_df:
        Dataframe of results (if supplied)
     :param tuned_params:
        Tuned parameters (if supplied)
     :param model:
        Model (if supplied)

     """
    params_keys = [config_dict['evaluation_metric'], 'std', 'k_fold_method', 'n_folds', 'n_repeats', 'seed',
                   'n_jobs', 'num_rows', 'model_name', 'best_method', 'tuned_params']

    if new_score_sufficiently_better(scores, config_dict, classification=config_dict['classification']):
        print('Results are significant, updating config and uploading from config file...')

        # Save new best scores
        new_score, new_std, best_method = get_best_score(scores, classification=config_dict['classification'])
        update_config(std=new_std, config_dict=config_dict)
        update_config(key=config_dict['evaluation_metric'], value=new_score, config_dict=config_dict)

        if result_df is not None:
            # Save the best dataframe version and num_rows
            update_config(df=result_df, num_rows=result_df.shape[0], config_dict=config_dict)
            update_config(key=f'{run_name}_best_method', value=best_method, config_dict=config_dict)
            params_keys.append(f'{run_name}_best_method')

        elif tuned_params is not None:
            update_config(model_name=best_method, best_model_params=model.get_params(), tuned_params=tuned_params,
                          config_dict=config_dict)
            mf.upload_artifacts(model_name=f'{best_method}_tuned_params', model=model)

        elif model is not None:
            if not isinstance(model, Sequential):
                update_config(model_name=best_method, best_model_params=model.get_params(), config_dict=config_dict)
            else:
                update_config(model_name=best_method, config_dict=config_dict)
            mf.upload_artifacts(model_name=best_method, model=model)

        else:
            # Save best model name  and num_rows. This is works only eval_models wrapper
            update_config(model_name=best_method, config_dict=config_dict)  # add num rows

        # Upload from config file

        uploader = mf.MLFlow(config_dict, params_keys)
        uploader.upload_config_file(run_name=run_name)

        return result_df

    else:
        print('We keep previous results since the new results are not significant')


def get_baseline_score(df: pd.DataFrame, target_label: str, classification: bool, evaluation_metric: str,
                       run_id_number: int, model_name, k_fold_method='k_fold', n_folds=3, n_repeats=10):
    """Get a baseline score """

    print('Computing baseline score')
    return baseline_score_cv(df, target_label, dicts.models['clf' if classification else 'reg'][model_name],
                             evaluation_metric=em.EvalMetrics.from_str(evaluation_metric), run_id_number=run_id_number,
                             policy=cv.SplitPolicy(random_state=constants.DEFAULT_SEED, n_splits=n_folds,
                                                   policy_type=k_fold_method,
                                                   shuffle=True, n_repeats=n_repeats))


def baseline_score_cv(df, target_label, model, evaluation_metric: em.EvalMetrics, run_id_number,
                      policy=cv.SplitPolicy.kfold_default()):
    """ Get a baseline cross validation score for the dataset and model supplied. """

    ds = d.Dataset.from_dataframe(df, [target_label])
    scores = scorers.get_cross_validation_score(ds, model=model, split_policy=policy,
                                                evaluation_metrics=[evaluation_metric],
                                                n_jobs=-1, verbose=3)
    mf.upload_baseline_score(df, run_id_number, evaluation_metric.value, scores, model)
    print(f'Score: {scores[0]} Std:{scores[1]}')
    return scores


def get_params_from_config(func, config_dict: dict):
    """Obtain the parameters of a function dynamically"""

    full_args_lst = getfullargspec(func)[0]

    params = {k: v for k, v in config_dict.items() if k in full_args_lst}

    return params


def get_func_params(scores, input_params, classification):
    """This will return the name of the function and specific params that were tested

    OBS: The function will only handle combinations with split
    """

    new_scores, new_std, best_method = get_best_score(scores, classification=classification)
    print(f'Current best method: {best_method}')
    lst_values = best_method.split('-')
    func = lst_values[0]  # assuming that function is the first character
    dict_ = {}
    for i in range(len(input_params)):
        if input_params[list(input_params.keys())[i]] is not None:
            dict_[list(input_params.keys())[i]] = input_params[list(input_params.keys())[i]](lst_values[i])
        else:
            pass

    params = {k: v for k, v in dict_.items() if k != 'func'}
    print(f'Params to return:{params}')

    return func, params


def get_output_df_wrapper(function_params, funcs_to_eval, function_name, params):
    """Wrapper function"""
    # Get all parameters that exist in function
    full_args_lst = getfullargspec(funcs_to_eval[function_name])[0]
    # Get specific params from full_args_lst if these are not already a part of the set of params of that function
    params_excluded_dict = {i: function_params[i] for i in full_args_lst if i not in params.keys()}
    param_in_full_args_lst = [i for i in params.keys() if i in full_args_lst]

    if len(param_in_full_args_lst) != 0:
        combination_dict = {**params, **params_excluded_dict}
        print('returning combination dict')
        return funcs_to_eval[function_name](**combination_dict)
    else:
        print('returning params_excluded_dict')
        return funcs_to_eval[function_name](**params_excluded_dict)
