""" Helper functions (and supporting code) called by the main wrappers methods. """

import gc
import math
import os
import re
from datetime import date
from datetime import datetime as dt
from inspect import getfullargspec
import distutils
import logging
# INIT PACKAGES
from time import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
# EDA PACKAGES
#import sweetviz as sv UNCOMMENT THIS LINE
# SKLEARN HYPEROPT
from hyperopt import fmin, tpe, STATUS_OK, Trials
# OVERSAMPLING
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
# SKLEARN BASIC
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, SelectFromModel
# SKLEARN IMPUTERS
from sklearn.impute import SimpleImputer
# SKLEARN CLASSIFIERS
from sklearn.linear_model import LogisticRegression
# SKLEARN METRICS
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import cohen_kappa_score
# SKLEARN HELP FUNCS
from sklearn.model_selection import train_test_split
# SKLEARN ENCODERS/SCALERS
from sklearn.preprocessing import LabelEncoder
# SCALERS
from sklearn.preprocessing import StandardScaler

# SKLEARN FOLD/GRIDSEARCH
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold

#REMOVE FOLLOWING LINES AND UNCOMMENT OTHERS
import cross_validation as cv
import datasets as d
from dicts import scorers, replace_methods, models, hyper_params, scalers, transformers
from mlflow_uploader import MLFlow, upload_baseline_score, upload_artifacts
# from tuiautopilotml import cross_validation as cv
# from tuiautopilotml import datasets as d
# from tuiautopilotml.dicts import scorers, replace_methods, models, hyper_params, scalers, transformers
# from tuiautopilotml.mlflow_uploader import MLFlow

# TESTING THESE LIBS
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.client import device_lib


current_color = 'firebrick'
current_palette = 'mako'
seed = 0

"""******** SECTION: BASE FUNCTIONS: DECORATORS ******** """


def time_performance_decor(func):
    """ Decorator to report execution time of a function. """

    def wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        current_time_minutes = round((t2 - t1) / 60, 2)
        print(f'The process took: {current_time_minutes} minutes to run')
        return result

    return wrapper


def gc_collect_decor(func):
    """ Decorator to invoke a garbage collect after executing a function"""

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print('Collecting garbage...')
        gc.collect()
        return result

    return wrapper


def check_encoded_df_decor(func):
    """ Decorator to check if a dataframe is encoded or contains missing values. """

    def wrapper(dataframe, *args, **kwargs):
        if contains_object(dataframe) or contains_nulls(dataframe):
            raise TypeError('Your input data is not encoded or contains missing values')
        else:
            result = func(dataframe, *args, **kwargs)

            return result

    return wrapper


def now_as_timestamp_string():
    now = dt.now()
    time_stamp = now.strftime("%D-%H:%M")
    time_stamp = '-'.join(re.split('/', time_stamp))

    return time_stamp


def extract_day_month_year_and_weekday(dataframe: pd.DataFrame, date_col: str):
    dataframe = dataframe.copy()
    # dataframe[f'{date_col}_day'] = dataframe[date_col].apply(lambda x: int(x.day))
    # dataframe[f'{date_col}_month'] = dataframe[date_col].apply(lambda x: int(x.month))
    # dataframe[f'{date_col}_year'] = dataframe[date_col].apply(lambda x: int(x.year))
    # dataframe[f'{date_col}_weekday'] = dataframe[date_col].apply(lambda x: int(x.weekday()))
    dataframe[f'{date_col}_day'] = dataframe[date_col].apply(lambda x: x.day)
    dataframe[f'{date_col}_month'] = dataframe[date_col].apply(lambda x: x.month)
    dataframe[f'{date_col}_year'] = dataframe[date_col].apply(lambda x: x.year)
    dataframe[f'{date_col}_weekday'] = dataframe[date_col].apply(lambda x: x.weekday())

    return dataframe


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
    return bool(distutils.util.strtobool(i))


"""******** FUNCTIONS USED TO GENERATE OUTPUT FOR SOME FUNCTIONS. TO SUPPORT THE AUTOPILOT MODE********"""


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


# REMOVE THIS FUNCTION IF IT IS NOT IN USE:
# def get_output_df(funcs_dict=None, func=None, *args, **kwargs):
#     output_df = funcs_dict[func](*args, **kwargs)
#
#     return output_df


"""******** SECTION: ENCODING ********"""


def label_one_hot_encoder(dataframe: pd.DataFrame, return_mapping=False):
    dataframe = dataframe.copy()
    print('Label encode and one hot encode only objects')
    label_encoder = LabelEncoder()
    object_cols = dataframe.select_dtypes(include='object').columns

    map_ = {}
    for col in object_cols:
        label_encoder.fit(dataframe[col])
        col_map = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
        map_[col] = col_map

        dataframe[col] = label_encoder.transform(dataframe[col])  # observe that this will handle one hot encoding

    if return_mapping:
        return dataframe, map_
    else:
        return dataframe


def if_date_get_additional_date_columns(dataframe: pd.DataFrame):
    date_cols = []
    for col in dataframe.columns:
        if is_col_date(dataframe, col):
            print(f'Generate date columns id dates for column: {col}')
            date_cols.append(col)
            dataframe = extract_day_month_year_and_weekday(dataframe=dataframe, date_col=col)

    dataframe.drop(date_cols, axis=1, inplace=True)
    return dataframe


def get_encoded_wrapper(dataframe: pd.DataFrame, encode_nulls=False, return_mapping=False):
    """
    Info: One hot encoding and label encoding.
    Args:
        dataframe:
        encode_nulls:
        return_mapping:
    Returns:

    """
    dataframe = dataframe.copy()
    if contains_nulls(dataframe) and not encode_nulls:
        raise ValueError('There are missing values in your dataset')
    else:
        if encode_nulls:
            print('Encoding nulls...')
            cols_with_missing = [col for col in dataframe.columns if dataframe[col].isnull().any()]
            for col in cols_with_missing:
                dataframe[f'encoded_nulls_{col}'] = dataframe[col].apply(lambda x: 0 if math.isnan(x) else 1)

            dataframe.dropna(axis=1, inplace=True)

        dataframe = if_date_get_additional_date_columns(dataframe)
        if not return_mapping:
            dataframe = label_one_hot_encoder(dataframe)
            return dataframe.reset_index(drop=True)
        else:
            dataframe, mapping = label_one_hot_encoder(dataframe, return_mapping=return_mapping)
            return dataframe.reset_index(drop=True), mapping


"""******** SECTION: DATA VALIDATION ********"""


def is_col_date(dataframe: pd.DataFrame, col: str):
    if pd.core.dtypes.common.is_datetime_or_timedelta_dtype(
            dataframe[col]) or pd.core.dtypes.common.is_datetime64_ns_dtype(dataframe[col]):
        return True
    else:
        return False


def is_object_date(dataframe: pd.DataFrame, col: str):
    if dataframe[col].dtypes == 'O' and not dataframe[col].isnull().sum().any():  # temporary in order to pass the unittest test
        try:
            pd.to_datetime(dataframe[col])
            return True
        except (TypeError, ValueError):
            return False
    else:
        pass


# The logic behind True and False needs to be tested
def contains_object(dataframe: pd.DataFrame):
    if len(dataframe.select_dtypes('object').columns) != 0:
        return True


# The logic behind True and False needs to be tested
def contains_nulls(dataframe: pd.DataFrame):
    if dataframe.isnull().sum().any():
        return True


def convert_to_int_float_date(dataframe: pd.DataFrame):
    """
    Convert to int , float and dates
    """
    dataframe = dataframe.copy()
    for col in dataframe.columns:

        if dataframe[col].dtypes == int:
            dataframe[col] = dataframe[col].astype(int)

        elif dataframe[col].dtypes == float:
            dataframe[col] = dataframe[col].astype(float)

        elif is_object_date(dataframe, col):
            dataframe[col] = pd.to_datetime(dataframe[col])

    return dataframe


def df_sanity_check(dataframe: pd.DataFrame):
    """
    Performs some basic sanity checking on the supplied dataframe.

    Currently, this does the following checks:
    1. Checks for null values
    2. Checks for strings that could be numeric
    3. Checks for objects that could be in date format

    :param dataframe: The dataframe to check.
    :type dataframe: pd.DataFrame

    :rtype: int
    """

    passed_test_count = 0
    n_tests = 3

    if not contains_nulls(dataframe):
        passed_test_count += 1
    else:
        print('There are missing values in your dataset')

    count2 = 0
    for col in list(dataframe.select_dtypes('object').columns):
        if dataframe[col].str.isdigit().any():
            count2 += 1
    if count2 == 0:
        passed_test_count += 1
    else:
        print('There is numerical data in string format')

    date_objects = 0
    object_cols = dataframe.select_dtypes('object').columns
    for col in object_cols:
        if is_object_date(dataframe, col):
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


"""******** EXPLORATORY DATA ANALYSIS ******** """


def get_summary_report(dataframe: pd.DataFrame):
    time_stamp = now_as_timestamp_string()
    print('Generating summary report...')
    summary_report = sv.analyze(dataframe)
    summary_report.show_html(f'AutoMLTuiRuns/Initial EDA graphs/{time_stamp}/initial_stats_report.html')


"""******** GET SPLITS ******** """


def get_x_y_from_df(dataframe, target_label, scaled_df=False, scaler=scalers['Standard']):
    if not scaled_df:
        x, y = dataframe.drop(target_label, axis=1), dataframe[target_label]
    else:
        x, y = scaler.fit_transform(dataframe.drop(target_label, axis=1)), dataframe[target_label]

    return x, y


def train_test_split_from_df(dataframe: pd.DataFrame, test_size: float):
    """By default we will take the first part as the train set  """
    train_size = 1 - test_size
    size = int(len(dataframe) * train_size)
    train = dataframe[:size].reset_index(drop=True)
    test = dataframe[size:].reset_index(drop=True)

    return train, test


def get_splits_wrapper(dataframe: pd.DataFrame, target_label: str, train_split=False, scaled=False,
                       scaler=StandardScaler(), validation_set=False, test_size=0.2):
    """
    Get your initial splits
    Args:
        dataframe:
        target_label:
        train_split:
        scaled:
        scaler:
        validation_set:
        test_size:

    Returns:

    """

    x, y = get_x_y_from_df(dataframe, target_label)

    if not train_split:
        if not scaled:
            return x, y
        else:
            x = scaler.fit_transform(x)
            return x, y
    else:

        if scaled:
            x = scaler.fit_transform(x)

        if validation_set:

            x_train_base, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
            x_train, x_valid, y_train, y_valid = train_test_split(x_train_base, y_train, test_size=test_size,
                                                                  random_state=seed)

            return x_train, x_valid, y_train, y_valid, x_test, y_test

        else:

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

            return x_train, x_test, y_train, y_test


"""******** SCORING FUNCTIONS - SCORERS (This section may require its own file) ******** """


@time_performance_decor
@gc_collect_decor
def get_cross_val_score_wrapper(dataframe=None, target_label=None, x=None, y=None, model=models['clf']['RF'],
                                multiple_eval_scores=False, k_fold_method='k_fold',
                                n_folds=5, n_repeats=10, random_state=seed, classification=True, multi_classif=True,
                                evaluation_metric='accuracy',
                                n_jobs=-1, verbose=0):
    """
    :param dataframe: pd.DataFrame
        The data to use.
    :param target_label: str
        The name of the target label in the dataframe
    :param x: np.array or List
        The input features - ignored if dataframe specified
    :param y: np.array or List
        The target outputs - ignored if dataframe specified

    :param model: Any
        The model to use - should be sci-kitlearn compatible.
    :param multiple_eval_scores: bool
        If True returns the mean of the accuracy, f1, precision and recall for classification or the means of the
        negative mean absolute error, negative mean squared error, negative root mean squared error and the r2
    :param k_fold_method: str
        Specifies k_fold vs stratified_k_fold vs repeated_strastified_k_fold
    :param n_folds: int
        The number of folds
    :param n_repeats: int
        The number of repeats if using repeated k_fold methods.
    :param random_state: int
        Used to seed random number generators

    :param classification: bool
        If True, reports classification metrics, otherwise the regression metrics
    :param multi_classif: bool
    :param evaluation_metric: str
        The metric to use, defaults to accuracy (ignored if multiple_eval_scores is True)
    :param n_jobs: int
        The number of jobs to  run in parallel (-1 will use defaults)
    :param verbose: int
        The verbosity level.

    :rtype float or dict
        Either a single mean score for the specified evaluation metric or a dict of mean scores depending on the
        classification and multiple_eval_metrics variables.
    """

    dataset = (d.Dataset(inputs=x, labels=y) if dataframe is None
               else d.Dataset.from_dataframe(dataframe, [target_label]))

    split_policy = cv.SplitPolicy(policy_type=k_fold_method,
                                  n_splits=n_folds,
                                  n_repeats=n_repeats,
                                  random_state=random_state).set_shuffle(True if random_state is not None else False)

    # The if statement below can be removed if the function is altered to simply take a list of evaluation metrics,
    # perhaps defaulting to a list containing just cv.EvalMetrics.Accuracy if None is specified.
    if multiple_eval_scores:
        if classification:
            eval_metrics = [cv.EvalMetrics.ACCURACY, cv.EvalMetrics.F1_SCORE, cv.EvalMetrics.PRECISION_SCORE,
                            cv.EvalMetrics.RECALL_SCORE]
        else:
            eval_metrics = [cv.EvalMetrics.NEG_MEAN_ABSOLUTE_ERROR,
                            cv.EvalMetrics.NEG_MEAN_SQUARED_ERROR,
                            cv.EvalMetrics.NEG_ROOT_MEAN_SQUARED_ERROR,
                            cv.EvalMetrics.R2]
    else:
        eval_metrics = [cv.EvalMetrics.from_str(evaluation_metric)]

    config = (cv.CrossValidatorConfig(split_policy=split_policy, eval_metrics=eval_metrics, n_jobs=n_jobs,
                                      verbose=verbose)
              .set_average('macro' if multi_classif else None))

    validator = cv.CrossValidator(config, model, dataset)
    cv_result = validator.get_cross_validation_scores()

    # To ensure this function continues to do what it originally did, we take the means here.
    # cv_result however contains scores for each split, so other users can do what they want with that.
    means = cv_result.mean()
    std = cv_result.std()

    if multiple_eval_scores:
        return means

    scores = [means[evaluation_metric], std[evaluation_metric]]

    return scores


@time_performance_decor
@gc_collect_decor
def get_hold_out_score(dataframe=None, target_label=None, x=None, y=None, return_all=False, classification=True,
                       model=models['clf']['RF'], test_size=0.2,
                       evaluation_metric='accuracy'):
    """

    Args:
        dataframe:
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
    scorer = scorers['clf'][evaluation_metric] if classification else scorers['reg'][evaluation_metric]
    if isinstance(dataframe, pd.DataFrame) and x is None and y is None:
        print('Generating internal x,y')
        x, y = get_x_y_from_df(dataframe, target_label)

    size = int(x.shape[0] * test_size)
    x_train, x_test, y_train, y_test = x[size:], x[:size], y[size:], y[:size]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    if not return_all:
        return scorer(y_test, y_pred)
    else:
        return x_train, x_test, y_train, y_test, y_pred


def get_custom_cv_score(dataframe: pd.DataFrame, target_label: str, classification: bool, evaluation_metric: str, model,
                        test_size=0.2, use_custom_method=False, custom_method=None, *args, **kwargs):
    """Acceptable test sizes: 0.15 , 0.2 , 0.3 """

    scorer = scorers['clf'][evaluation_metric] if classification else scorers['reg'][evaluation_metric]
    size = int(len(dataframe) * test_size)
    folds_lst = [i for i in range(0, len(dataframe), size)]
    test_index = []
    scores = []

    for i in range(len(folds_lst) - 1):
        print(f'Training fold: {i}')
        test_index.append(dataframe[folds_lst[i]:folds_lst[i + 1]].index.values)
        test = dataframe.iloc[test_index[i]]
        train_idx = list(set(test_index[i]) ^ set(dataframe.index.values))
        #train = dataframe.iloc[train_idx]
        train = dataframe.loc[train_idx]# TESTING THIS LINE
        current_train = train if not use_custom_method else custom_method(train, *args, **kwargs)
        x_train, y_train = get_x_y_from_df(current_train, target_label)

        if custom_method in [get_imputed_x, drop_nulls]:
            test = test.copy()
            current_test = custom_method(test, *args, **kwargs)
            x_test, y_test = get_x_y_from_df(current_test, target_label)
        else:
            x_test, y_test = get_x_y_from_df(test, target_label)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        scores.append(scorer(y_test, y_pred))
        print(f'Custom cv scores:{scores}')

    return np.mean(scores), np.std(scores)


# The following section is outdated and will be removed for future versions
"""******** PLOTS/GRAPHS ******** (This section may require its own file or class)"""


def get_graph(input_data, figsize=(5, 6), stage='default_stage in pipeline', color=current_color, horizontal=True,
              style='default',
              fig_title=f'Fig Title', x_title=None, y_title=None, sort_type='desc', save_figure=False,
              file_name='current_fig'):
    """
    This supports only barplots
    Parameters:
    argument1 (int): Description of arg1

    Returns:
    int:Returning value
   """

    current_date = f'Current Date:{date.today()}'
    fig_title = fig_title + '-' + current_date

    sort_type_param = True if sort_type == 'desc' else False
    if type(input_data) == dict:
        input_data = dict(sorted(input_data.items(), key=lambda x: (x[1], x[0]), reverse=sort_type_param))

        keys = list(input_data.keys())
        values = list(input_data.values())

    elif type(input_data) == pd.DataFrame:
        keys = list(input_data.index)
        values = list(input_data.importance)

    plt.style.use(style)

    if horizontal:
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(y=keys, width=values, align='center', color=color, alpha=0.6)
        # ax.set_yticks(y_pos)
        ax.set_yticklabels(keys)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel(y_title)
        ax.set_ylabel(x_title)
        ax.set_title(fig_title)
        plt.show()
        if save_figure:
            save_figure_to_disk(main_folder=stage, figure_name=file_name, save_as_plt=False, fig=fig)

    else:
        plt.figure(figsize=figsize)
        plt.bar(x=keys, height=values, color=color, width=0.4, alpha=0.6)
        # Add title and axis names
        plt.title(fig_title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)

        if save_figure:
            save_figure_to_disk(main_folder=stage, figure_name=file_name, save_as_plt=horizontal)
        plt.show()


@gc_collect_decor
def get_initial_graphs(dataframe, target=None, save_figures=False):
    # PART 1
    print('Count plots')
    n_max_categories = 20

    low_cardinality_cols = [cname for cname in dataframe if dataframe[cname].nunique() <= n_max_categories and
                            dataframe[cname].dtype == "object"]

    plt.figure(figsize=(14, 48))
    count = 1

    for col in low_cardinality_cols:
        count += 1
        plt.subplot(9, 2, count)
        sns.countplot(y=col, data=dataframe, alpha=0.6, order=dataframe[col].value_counts().index,
                      palette=current_palette)
        count += 1

    if save_figures:
        save_figure_to_disk(main_folder='Initial EDA graphs', figure_name='Count Plot', save_as_plt=True)
    plt.show()

    # PART 2
    print('Correlation between variables')
    plt.figure(figsize=(12, 8))
    sns.heatmap(dataframe.corr(), cmap=current_palette, fmt='g', annot=False)
    if save_figures:
        save_figure_to_disk(main_folder='Initial EDA graphs', figure_name='Correlation Figure', save_as_plt=True)
    plt.show()

    # PART 4
    print('Distribution and outlier detection')

    int_float_cols = dataframe.select_dtypes([int, float]).columns
    plt.figure(figsize=(10, (len(int_float_cols)) * 2 + 3))  # width , height

    count = 1
    for col in int_float_cols:
        # Row 1
        # plt.yticks(fontsize=1)

        plt.subplot(len(int_float_cols), 2, count)  # n_rows , n columns , index
        sns.boxplot(x=col, y=target, data=dataframe, palette=current_palette)
        count += 1

        # Row 2
        plt.subplot(len(int_float_cols), 2, count)
        g = sns.kdeplot(dataframe[col], palette=current_palette, alpha=0.6, shade=True)
        g.set_xlabel(col)
        # g.set_ylabel("Frequency")
        # g = g.legend(["No Diesese", "Diesese"])
        count += 1

    plt.tight_layout()

    if save_figures:
        save_figure_to_disk(main_folder='Initial EDA graphs', figure_name='Distribution Fig', save_as_plt=True)
    plt.show()


def save_figure_to_disk(dataframe=None, main_folder=None, figure_name=None, save_as_plt=False, fig=None):
    time_stamp = now_as_timestamp_string()

    print('Saving figure/table to disk...')
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, f'AutoMLTuiRuns/{main_folder}/{time_stamp}')

    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    if isinstance(dataframe, pd.DataFrame):

        dataframe.to_csv(f'AutoMLTuiRuns/{main_folder}/{time_stamp}/{figure_name}.csv')
    else:
        if not save_as_plt:
            fig.savefig(f'AutoMLTuiRuns/{main_folder}/{time_stamp}/{figure_name}.png')
        else:
            plt.savefig(f'AutoMLTuiRuns/{main_folder}/{time_stamp}/{figure_name}.png')


"""******** FEATURE IMPORTANCE ******** """


def get_feature_importance_uni(dataframe: pd.DataFrame, target_label: str, k=5, classification=True):

    feature_cols = dataframe.columns.drop(target_label)
    f = f_classif if classification else f_regression
    selector = SelectKBest(score_func=f, k=k)  # f_regression
    x_new = selector.fit_transform(dataframe[feature_cols], dataframe[target_label])

    selected_features = pd.DataFrame(selector.inverse_transform(x_new), index=dataframe.index, columns=feature_cols)

    selected_columns = list(selected_features.columns[selected_features.var() != 0])

    return selected_columns


def get_feature_importance_l1(dataframe: pd.DataFrame, target_label: str, penalty="l1", c=1):

    x, y = get_x_y_from_df(dataframe, target_label)
    logistic = LogisticRegression(C=c, penalty=penalty, random_state=seed).fit(x, y)
    selector = SelectFromModel(logistic, prefit=True)
    x_new = selector.transform(x)
    selected_features = pd.DataFrame(selector.inverse_transform(x_new), index=x.index, columns=x.columns)
    selected_columns = selected_features.columns[selected_features.var() != 0]

    return selected_columns


def get_feature_importance_rf_xgb(dataframe: pd.DataFrame, target_label: str, classification=True, algorithm=None,
                                  save_figure=False):
    x, y = get_x_y_from_df(dataframe, target_label)
    models_dict = models['clf'] if classification else models['reg']
    current_model = select_custom_dict(models_dict, algorithm)

    current_model = current_model[algorithm]
    current_model.fit(x, y)
    feautre_importances = pd.DataFrame(current_model.feature_importances_, index=x.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)

    get_graph(feautre_importances, figsize=(8, 6), stage='Feature Engineering', color=current_color, horizontal=True,
              style='default', fig_title=f'Feature importance', x_title='features', y_title='score', sort_type='desc',
              save_figure=save_figure, file_name='feature_importance_figure')

    return feautre_importances


def auto_feature_selection_from_estimator(dataframe: pd.DataFrame, target_label: str, estimator):

    x, y = get_x_y_from_df(dataframe, target_label)
    estimator = estimator.fit(x, y)
    selector = SelectFromModel(estimator, prefit=True)
    x_new = selector.transform(x)
    selected_features = pd.DataFrame(selector.inverse_transform(x_new),
                                     index=x.index,
                                     columns=x.columns)
    selected_columns = selected_features.columns[selected_features.var() != 0]

    return selected_columns


@check_encoded_df_decor
def get_reduced_features_cv_scores(dataframe: pd.DataFrame, target_label: str, model, *args, **kwargs):

    scores_dict = {}
    selected_features = auto_feature_selection_from_estimator(dataframe, target_label, estimator=model)
    print(f'Selected features are: {selected_features}')
    xs = {'reduced_x': dataframe[selected_features], 'x_all': dataframe.drop(target_label, axis=1)}
    y = dataframe[target_label]
    for name, x in xs.items():
        scores = get_cross_val_score_wrapper(x=x, y=y, model=model, *args, **kwargs)
        scores_dict[name] = scores[0]

    return scores_dict


"""******** OUTLIERS (This section may require its own file) ******** """


def get_outliers_std(dataframe: pd.DataFrame, column: str):
    len_df = len(dataframe)

    q25, q75 = np.percentile(dataframe[column], 25), np.percentile(dataframe[column], 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off

    outliers = [x for x in dataframe[column] if x < lower or x > upper]
    # msg = (f'Identified outliers for {column} - {round(len(outliers) / len_df * 100, 2)}% ')
    pct_outliers = round(len(outliers) / len_df * 100, 2)
    # print(msg)
    return pct_outliers, outliers


def get_outliers_z_score(dataframe: pd.DataFrame, column: str):
    len_df = len(dataframe)
    outliers = []
    threshold = 3
    mean = np.mean(dataframe[column])
    std = np.std(dataframe[column])
    for i in dataframe[column]:
        z_score = (i - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(i)
    # print(f'Identified outliers for {column}: {round(len(outliers) / len_df * 100, 2)}%')
    pct_outliers = round(len(outliers) / len_df * 100, 2)

    return pct_outliers, outliers


def get_outliers(dataframe: pd.DataFrame, distribution='non_gaussian', show_graph=True):
    """
    Info..
    Parameters:
    argument1 (int): Description of arg1

    Returns:
    int:Returning value
    """

    num_cols = list(dataframe.select_dtypes([int, float]).columns)
    outliers_dict = {}
    outliers_values = {}

    for column in num_cols:

        if distribution == 'non_gaussian':

            pct_outliers, outliers = get_outliers_std(dataframe, column)
            outliers_values[column] = outliers
            outliers_dict[column] = pct_outliers

        else:
            pct_outliers, outliers = get_outliers_z_score(dataframe, column)
            outliers_values[column] = outliers
            outliers_dict[column] = pct_outliers
    if show_graph:
        get_graph(input_data=outliers_dict, stage='Feature Engineering', horizontal=True, figsize=(6, 7),
                  fig_title=f'Outliers detected',
                  x_title='Pct of total', y_title='Outliers', sort_type='desc', save_figure=False,
                  file_name='current_fig')

    return outliers_dict, outliers_values


def replace_outliers(dataframe: pd.DataFrame, distribution: str, tot_outlier_pct: int, strategy: str):
    dataframe = dataframe.copy()

    outliers_pct, outliers_values = get_outliers(dataframe=dataframe, distribution=distribution, show_graph=False)
    top_outliers_cols = [col for col, score in outliers_pct.items() if score >= tot_outlier_pct]

    current_method = replace_methods[strategy]

    for col in top_outliers_cols:
        current_method_value = current_method(dataframe[col])
        outliers_lst = outliers_values[col]
        dataframe[col].replace(outliers_lst, current_method_value, inplace=True)

    return dataframe


def drop_outliers(dataframe: pd.DataFrame, distribution: str, tot_outlier_pct: int):

    dataframe = dataframe.copy()

    outliers_pct, outliers_values = get_outliers(dataframe=dataframe, distribution=distribution, show_graph=False)
    top_outliers_cols = [col for col, score in outliers_pct.items() if score >= tot_outlier_pct]

    outliers_idx_lst = []
    for col in top_outliers_cols:
        for value in outliers_values[col]:
            index_int = dataframe[dataframe[col] == value].index[0]
            if index_int not in outliers_idx_lst:
                outliers_idx_lst.append(index_int)

    for i in range(len(outliers_idx_lst)):
        dataframe.drop(outliers_idx_lst[i], axis=0, inplace=True)

    return dataframe


"""******** IMBALANCED DATASETS - OVERSAMPLING (This section may require its own file) ********"""


def compute_entropy(seq: pd.Series):
    """ Funtion to calculate entropy of a sequence """
    if type(seq) is not list:
        seq = list(seq)

    n = len(seq)
    total_count_dict = {i: seq.count(i) for i in seq}.items()
    k = len(total_count_dict)
    h = -sum([(v / n) * np.log((v / n)) for k, v in total_count_dict])
    entropy_score = h / np.log(k)
    return entropy_score


def check_imbalance_degree(dataframe: pd.DataFrame, target_label: str):
    """ Funtion to check the imbalance degree"""

    if dataframe[target_label].nunique() >= 20:
        print('It looks like your input data contains too many categories')
    else:
        total_count_pct = dict(dataframe.groupby(target_label).size() / len(dataframe))
        moderate = {}
        extreme = {}
        for k, v in total_count_pct.items():
            if 0.1 <= v <= 0.20:
                moderate[k] = v
                extreme[k] = v
        entropy_score = compute_entropy(dataframe[target_label])
        print(f'Overall entropy score: {entropy_score}')
        print(f'Moderate imbalanced classes: {moderate}')
        print(f'Extreme imbalanced classes: {extreme} ')
        return entropy_score


def get_percentile_per_class(x_train, y_train, single_class):
    for percentile_value in range(35, 95, 5):
        try:

            percentile_n = int(np.percentile(y_train.value_counts(), percentile_value))
            print(f'Current percentile value {percentile_n}')
            weights = {class_: percentile_n for class_ in np.unique(y_train) if class_ == single_class}
            random_os = RandomOverSampler(weights)
            random_os.fit_resample(x_train, y_train)
            return percentile_n

        except ValueError:
            print(f'Failed with {percentile_value} percentile')
            pass


def plot_cm_matrix_prediction_error(y_test, y_pred):
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 8))
    conf_m_pct = cm / np.sum(cm) * 100
    sns.heatmap(conf_m_pct, annot=True, cmap=current_palette).set_title('Confusion metrics as percetanges')

    # save_figure_to_disk(main_folder='Feature Engineering', figure_name='Confusion metrics as percetanges',
    #                     save_as_plt=True)

    plt.show()

    # Class Prediction Error
    pd.DataFrame(cm).plot(kind='bar', stacked=True, cmap="mako").set_title('Class Prediction Error')
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 0.5))

    # save_figure_to_disk(main_folder='Feature Engineering', figure_name='Class Prediction Error',
    #                     save_as_plt=True)
    plt.show()


def get_n_classes_to_resample(y_test, y_pred, y_counts, class_threshold=0.05):
    # 1. Get only f1 scores (per class) from classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    classes_f1_scores = {}
    for class_, scores_dict in report.items():
        if class_.isnumeric():
            for score, value in scores_dict.items():
                if score == 'f1-score':
                    classes_f1_scores[class_] = value

    # 2. Filter 1 - Get classes  with a lower percentage than the mean of f1 score
    mean = np.mean(list(classes_f1_scores.values()))  # choose other f1 score
    f1_scores_lst = [k for k, v in classes_f1_scores.items() if v <= mean]

    # 3. Filter 2 - Choose a representative threshold based on total counts
    print(f'y counts:{y_counts}')
    class_threshold = class_threshold
    classes_to_resample = []

    for k, v in y_counts.items():
        k = str(k)
        if k in f1_scores_lst and v >= class_threshold:
            classes_to_resample.append(int(k))
    print(f'Classes to re sample: {classes_to_resample}')

    return classes_to_resample


def fit_random_os(dataframe, target_label, test_size, model, evaluation_metric, class_threshold, return_df=False):

    x_train, x_test, y_train, y_test, y_pred = get_hold_out_score(dataframe=dataframe,
                                                                  target_label=target_label, model=model,
                                                                  test_size=test_size,
                                                                  evaluation_metric=evaluation_metric, return_all=True)

    y_counts = dict(dataframe[target_label].value_counts() / len(dataframe) * 100)
    classes_to_resample = get_n_classes_to_resample(y_test, y_pred, y_counts, class_threshold=class_threshold)

    if return_df:
        x, y = get_x_y_from_df(dataframe, target_label)
        weights = {c: get_percentile_per_class(x, y, single_class=c) for c in classes_to_resample}
        random_os = RandomOverSampler(weights)
        x_overs, y_overs = random_os.fit_resample(x, y)
        x_overs[target_label] = y_overs

        return x_overs

    else:

        weights = {c: get_percentile_per_class(x_train, y_train, single_class=c) for c in classes_to_resample}
        random_os = RandomOverSampler(weights)
        x_overs, y_overs = random_os.fit_resample(x_train, y_train)

        return x_overs, y_overs, x_test, y_test, y_pred


def get_random_os_score(dataframe: pd.DataFrame, target_label: str, classification: bool, model, test_size=0.2,
                        class_threshold=5, evaluation_metric='accuracy'):

    """
    Args:
        dataframe:
        target_label:
        classification:
        model:
        test_size:
        class_threshold:
        evaluation_metric:

    Returns:
    """

    scorer = scorers['clf'][evaluation_metric] if classification else scorers['reg'][evaluation_metric]

    # . Obtain classes to re sample
    x_overs, y_overs, x_test, y_test, y_pred = fit_random_os(dataframe, target_label, test_size, model,
                                                             evaluation_metric, class_threshold, return_df=False)

    # 2. Plotting figures
    print(classification_report(y_test, y_pred, output_dict=False))
    print('PART 2: Plotting figures ')
    plot_cm_matrix_prediction_error(y_test, y_pred)

    # . Make the prediction. Evaluate only on test data
    print('PART 5: Make the prediction. Evaluate only on test data')

    model.fit(x_overs, y_overs)
    y_pred_os = model.predict(x_test)

    # Kappa - accuracy  score
    kappa_score = cohen_kappa_score(y_test, y_pred_os)
    print(f'Kappa score: {kappa_score}')
    acc_score = scorer(y_test, y_pred_os)

    # . Plot
    print('PART 6: Plot')
    plot_cm_matrix_prediction_error(y_test, y_pred_os)
    print(classification_report(y_test, y_pred_os, output_dict=False))

    return acc_score


def fit_smote_os(dataframe: pd.DataFrame, target_label: str, test_size: float, return_df=False, *args, **kwargs):

    x_train, x_test, y_train, y_test = get_splits_wrapper(dataframe, target_label, train_split=True, test_size=test_size)

    print('Running smote_os...')

    over_sample = SMOTE(*args, **kwargs)  # sampling_strategy = 0.5
    x_overs, y_overs = over_sample.fit_resample(x_train, y_train)
    if return_df:
        x_overs[target_label] = y_overs
        return x_overs
    else:
        return x_overs, y_overs, x_test, y_test


def get_smote_os_score(dataframe: pd.DataFrame, target_label: str, test_size: float, model, classification: bool,
                       evaluation_metric: str, *args, **kwargs):

    scorer = scorers['clf'][evaluation_metric] if classification else scorers['reg'][evaluation_metric]

    try:
        x_overs, y_overs, x_test, y_test = fit_smote_os(dataframe, target_label, test_size, *args, **kwargs)
        model.fit(x_overs, y_overs)
        y_pred_os = model.predict(x_test)
        score = scorer(y_test, y_pred_os)

        return score

    except ValueError as err:

        print(f'You got a {err}. You would need to increase your sample size')


"""********  HYPER PARAMETER TUNING (This section may require its own file or class) ******** """


def get_tuned_models_dict(tuned_params: dict, models_dict: dict):
    tuned_dict = {}

    for m_name, model in models_dict.items():
        if m_name in tuned_params.keys():
            tuned_dict[m_name] = model.set_params(**tuned_params[m_name])

    return tuned_dict


def get_tuned_models_wrapper(tuned_params: dict, models_dict: dict):
    models_dict_tuned = get_tuned_models_dict(tuned_params=tuned_params, models_dict=models_dict)

    for name, model in models_dict_tuned.items():
        models_dict[name] = model

    return models_dict


def hyper_opt_manual(dataframe: pd.DataFrame, target_label: str, model_name=None, max_evals=80, k_fold_method='k_fold',
                     n_folds=3, n_repeats=2, classification=True, evaluation_metric='accuracy', timeout_minutes=10,
                     n_jobs=-1, verbose=0):
    """

    Args:
        model_name:
        dataframe:
        target_label:
        max_evals:
        k_fold_method:
        n_folds:
        n_repeats:
        classification:
        evaluation_metric:
        timeout_minutes:
        n_jobs:
        verbose:

    Returns:

    """

    timeout = timeout_minutes * 60
    models_dict = models['clf'] if classification else models['reg']
    current_model_dict = select_custom_dict(models_dict, model_name)

    hyper_params_dict = hyper_params['clf'] if classification else hyper_params['reg']
    space = select_custom_dict(hyper_params_dict, model_name)[model_name]
    print(f'Model name: {model_name}, Parameters: {space}')

    scores = {}

    def objective(selected_space):
        # clone the current version of the model in order to avoid overwriting the dictionary
        model_base = clone(current_model_dict[model_name])

        model_base.set_params(**selected_space)

        score = get_cross_val_score_wrapper(dataframe=dataframe, target_label=target_label, model=model_base,
                                            classification=classification, evaluation_metric=evaluation_metric,
                                            k_fold_method=k_fold_method,
                                            n_folds=n_folds,
                                            n_repeats=n_repeats, n_jobs=-n_jobs, verbose=verbose)
        scores[model_name] = score[0]

        print(f'Accuracy: {score[0]}')

        # We aim to maximize accuracy, therefore we return it as a negative value
        return {'loss': - score[0], 'std': score[1], 'status': STATUS_OK, 'model': model_base}

    trials = Trials()

    fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials,
         timeout=timeout)  # early_stop_fn = no_progress_loss(10)

    print('optimization complete')
    best_model = trials.results[np.argmin([r['loss'] for r in
                                           trials.results])]

    best_score = best_model['loss'] * -1
    std = best_model['std']

    params = {k: v for k, v in best_model['model'].get_params().items() if v is not None}

    results = {model_name: (best_score, std)}

    print(f'Results:{results}')

    return results, params, best_model['model']


# @time_performance_decor
# @gc_collect_decor
# @check_encoded_df_decor
def optuna_wrapper(dataframe: pd.DataFrame, target_label: str, model_name='XGB', n_minutes_limit=None, n_trials=None,
                   params_list=None, classification=True, evaluation_metric='accuracy', test_size=0.2,
                   direction='maximize'):
    """

    Args:
        dataframe:
        target_label:
        model_name:
        n_minutes_limit:
        n_trials:
        params_list:
        classification:
        evaluation_metric:
        test_size:
        direction:

    Returns:

    """
    if n_minutes_limit is not None:
        n_seconds_limit = n_minutes_limit * 60
    else:
        n_seconds_limit = None

    if model_name == 'XGB':
        models_dict = models['clf'] if classification else models['reg']
        model = select_custom_dict(models_dict, [model_name])[model_name]

        def objective(trial):

            params = {'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                      'n_estimators': trial.suggest_int('n_estimators', 100, 1100, 200, log=False),
                      'max_depth': trial.suggest_int('max_depth', 3, 9, 2),
                      'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                      'gamma': trial.suggest_float('gamma', 0.0, 0.4),
                      'subsample': trial.suggest_float('subsample', 0.8, 1),
                      'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.8),
                      'reg_alpha': trial.suggest_float('reg_alpha', 1e-05, 100)}

            if params_list is not None:
                final_params = {}
                for k, v in params.items():
                    if k in params_list:
                        final_params[k] = v

                print(final_params)
                model.set_params(**final_params)
            else:
                model.set_params(**params)

            score = get_hold_out_score(dataframe=dataframe, target_label=target_label, model=model, test_size=test_size,
                                       evaluation_metric=evaluation_metric)

            # Pruning
            trial.report(score, 0)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return score

        study = optuna.create_study(direction=direction)  # maximize or minimize

        study.optimize(objective, timeout=n_seconds_limit, n_trials=n_trials, gc_after_trial=True)

        print("Best hyperparameters: {}".format(study.best_trial.params))
        print("Best score: {}".format(study.best_trial.value))
        best_params = study.best_trial.value, study.best_trial.params

        # Save to disk
        print('Saving results to disk')
        trials_df = study.trials_dataframe()
        save_figure_to_disk(dataframe=trials_df, main_folder='Hyper Parameter Tuning',
                            figure_name='optuna hyper param tuning')
        # Visualization
        optuna.visualization.plot_optimization_history(study)
        best_score = study.best_value
        scores = {model_name: (best_score, None)}
        return scores, best_params[1], model

    else:
        print(f'There are still no parameters for algorithm {model_name}')
        pass


def grid_search_wrapper(dataframe: pd.DataFrame, target_label: str, model=models['clf']['RF'],
                        evaluation_metric="accuracy", n_jobs=-1, verbose=3, n_folds=3, n_repeats=3,
                        k_fold_method='stratified_k_fold', grid_search_method='randomized', random_state=seed):
    """

    Args:
        evaluation_metric:
        target_label:
        dataframe:
        model:
        param_grid:
        n_jobs:
        verbose:
        n_folds:
        n_repeats:
        k_fold_method:
        grid_search_method:
        random_state:

    Returns:

    """
    param_grid = None  # temp
    x, y = get_x_y_from_df(dataframe, target_label)

    k_fold_methods_dict = {'k_fold': KFold(n_splits=n_folds, shuffle=True),
                           'repeated_k_fold': RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats,
                                                            random_state=random_state),
                           'stratified_k_fold': StratifiedKFold(n_splits=n_folds, shuffle=True,
                                                                random_state=random_state),
                           'repeated_stratified_k_fold': RepeatedStratifiedKFold(n_splits=n_folds,
                                                                                 n_repeats=n_repeats,
                                                                                 random_state=random_state)}

    current_kfold = k_fold_methods_dict[k_fold_method]

    grid_search_dict = {
        'randomized': RandomizedSearchCV(model, param_grid, scoring=evaluation_metric, n_jobs=n_jobs, cv=current_kfold,
                                         verbose=verbose),
        'gridsearch': GridSearchCV(model, param_grid, scoring=evaluation_metric, n_jobs=n_jobs, cv=current_kfold,
                                   verbose=verbose)}  # this does not save results during the process

    current_grid_search = grid_search_dict[grid_search_method]

    try:
        grid_result = current_grid_search.fit(x, y)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        return grid_result.best_score_, grid_result.best_params_

    except ValueError as err:
        print(f'You got following error: {err}')
        pass


""" ******** IMPUTATION/MISSING VALUES ******** """


def get_imputed_x(dataframe: pd.DataFrame, strategy: str):
    """

    Args:
        dataframe:
        strategy:

    Returns: x imputed using a specific imputation strategy

    """
    cols_with_missing = [col for col in dataframe.columns if dataframe[col].isnull().any()]
    print(f'Columns with missing values: {cols_with_missing}')
    print(f'Running strategy {strategy}')
    dataframe = dataframe.copy()
    imputer = SimpleImputer(strategy=strategy)
    dataframe[cols_with_missing] = imputer.fit_transform(dataframe[cols_with_missing])
    encoded_df = get_encoded_wrapper(dataframe)

    return encoded_df


def drop_nulls(dataframe: pd.DataFrame):
    dataframe = dataframe.copy()
    dataframe.dropna(axis=0, inplace=True)
    encoded_df = get_encoded_wrapper(dataframe)

    return encoded_df


""" ******** TRANSFORMATION METHODS - SCALERS ******** """


def scale_x(dataframe, target_label, scaler_name, use_transformers=False, transformer_name=None):

    x, y = get_x_y_from_df(dataframe, target_label)
    scaler = scalers[scaler_name]

    if use_transformers:
        transformer = transformers[transformer_name]
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


def get_scaled_x_score(dataframe, target_label, model_name='RF', scaler_name='MinMax', use_transformers=False,
                       transformer_name=None,
                       k_fold_method='k_fold', n_folds=5, n_repeats=3, classification=True,
                       evaluation_metric='accuracy'):
    """Test scaled version of x. This will return the mean and standard deviation"""

    models_dict = models['clf'] if classification else models['reg']
    model = models_dict[model_name]

    output_df_arr = scale_x(dataframe, target_label, scaler_name, use_transformers=use_transformers,
                            transformer_name=transformer_name)
    # Adjust PCA and Truncated dynamically
    # y = output_df_arr[:, -1]
    # x = output_df_arr[:, :-1]

    #     scores = get_cross_val_score_wrapper(x=x, y=y, model=model, k_fold_method=k_fold_method,
    #                                          n_folds=n_folds, n_repeats=n_repeats, random_state=seed,
    #                                          classification=classification, evaluation_metric=evaluation_metric)
    scores = get_cross_val_score_wrapper(dataframe=output_df_arr, target_label=target_label, model=model,
                                         k_fold_method=k_fold_method,
                                         n_folds=n_folds, n_repeats=n_repeats, random_state=seed,
                                         classification=classification, evaluation_metric=evaluation_metric)

    return scores[0], scores[1]


"""******** KERAS-MLP MODEL ******** """


def get_mlp_model(X, y, activation_f_type='classif', optimizer='adam',
                  regulator=10, hl_activation='relu', evaluation_metric='accuracy'):
    """
    MLP for baseline model creation
    """

    # print(f'Numeber of features: {X_train.shape[1]}')

    n_inputs = X.shape[1]
    n_outputs = int(y.nunique())

    if activation_f_type == 'classif':
        o_activation = 'sigmoid'
        loss = 'binary_crossentropy'

    elif activation_f_type == 'multiclass':
        o_activation = 'softmax'
        loss = 'categorical_crossentropy'

    elif activation_f_type == 'reg':
        o_activation = 'linear'
        loss = 'mean_squared_error'

    print(f'Activation function used for output layer: {o_activation}')

    n_neurons = int(np.sqrt(n_inputs * n_outputs) * regulator)
    print(f'Number of neurons: {n_neurons}-{int(n_neurons / 2.5)}-{int(n_neurons / 5.5)}')

    model = Sequential()
    model.add(Dense(n_neurons, input_dim=n_inputs, activation=hl_activation))  # input layer

    model.add(Dropout(0.3))

    model.add(Dense(int(n_neurons / 2.5), activation=hl_activation))
    model.add(Dropout(0.3))

    model.add(Dense(int(n_neurons / 5.5), activation=hl_activation))
    model.add(Dropout(0.1))

    model.add(Dense(n_outputs, activation=o_activation))  # output layer

    model.compile(loss=loss, optimizer=optimizer, metrics=[evaluation_metric])

    return model


"""******** AUTOPILOT MODE FUNCTIONS ******** """

#****UPDATED FUNCTIONS****
def get_output_df_wrapper(functions_dict, sub_functions_dict, function_name, params):

    """Wrapper function"""

    full_args_lst = getfullargspec(sub_functions_dict[function_name])[0]
    params_excluded_dict = {i: functions_dict[i] for i in full_args_lst if i not in params.keys()}
    param_in_full_args_lst = [i for i in params.keys() if i in full_args_lst]

    if len(param_in_full_args_lst) != 0:
        combination_dict = {**params, **params_excluded_dict}
        print('returning combination dict')
        return sub_functions_dict[function_name](**combination_dict)
    else:
        print('returning params_excluded_dict')
        return sub_functions_dict[function_name](**params_excluded_dict)


def get_best_score(scores, classification=True):

    score = max({v[0] for k, v in scores.items()}) if classification else min({v[0] for k, v in scores.items()})
    tups = [(v[0], v[1]) for k, v in scores.items() if v[0] == score][0]
    new_scores, new_std = tups[0], tups[1]
    best_method = [k for k, v in scores.items() if v[0] == score][0]

    return new_scores, new_std, best_method


def get_latest_score(config_dict: dict):

    experiment_id = "0"
    runs_df = mlflow.search_runs([experiment_id]).sort_values('end_time', ascending=False)
    runs_df = runs_df[runs_df.status == 'FINISHED']  # exclude all failed jobs

    try:
        id_n = config_dict['run_id_number']
        lst = []
        for i in runs_df['tags.mlflow.runName']:
            if re.search(str(id_n), i):
                lst.append(i)

        first_row = runs_df[runs_df['tags.mlflow.runName'].isin(lst)].sort_values('end_time', ascending=False)[:1]

        score_mean = first_row[f"metrics.{config_dict['evaluation_metric']}"].reset_index(drop=True)
        score_std = first_row[f"metrics.std"].reset_index(drop=True)

        if len(score_mean) == 1:
            return float(score_mean), float(score_std)
        else:
            return score_mean[0], score_std[0]

    except KeyError as err:
        print(f'The parameter {err} is missing, you need to generate a baseline score first')
        pass


def update_config(config_dict: dict = None, key=None, value=None,  **kwargs):

    if key is not None and len(kwargs) == 0:
        config_dict[key] = value
    else:

        for k, v in kwargs.items():
            config_dict[k] = v


def new_score_is_significant(scores: dict, config_dict: dict, classification=True):

    mean_score, std_score = get_latest_score(config_dict=config_dict)
    new_score, new_std, best_method = get_best_score(scores, classification=classification)
    print(f'Previous scores: {mean_score, std_score}')
    print(f'New scores: {new_score, new_std}')

    if classification:

        if pd.isna(new_std) or pd.isna(std_score):
            if new_score > mean_score:
                return True
            else:
                return False
        else:
            #***** test
            diff = round(new_score - mean_score, 4)
            if diff >= 0.024:
                return True
            else:
                if new_score >= mean_score and new_std <= std_score:
                    return True
                else:
                    return False
            #***** test

    else:
        if pd.isna(new_std) or pd.isna(std_score):
            if new_score < mean_score:
                return True
            else:
                return False
        else:
            if new_score < mean_score and new_std <= std_score:
                return True
            else:
                return False


def get_params_to_upload(config_dict: dict, params_keys: dict):
    """Select specific params from CONFIG to upload"""

    results = {k: v for k, v in config_dict.items() if k in params_keys}

    return results


def update_upload_config(scores: dict, config_dict: dict, run_name='run_name', result_df=None, tuned_params=None, model=None):

    params_keys = [config_dict['evaluation_metric'], 'std', 'k_fold_method', 'n_folds', 'n_repeats', 'seed',
                   'n_jobs', 'num_rows', 'model_name', 'best_method', 'tuned_params']

    if new_score_is_significant(scores, config_dict, classification=config_dict['classification']):

        print('Results are significant, updating config and uploading from config file...')

        # Save new best scores
        new_score, new_std, best_method = get_best_score(scores, classification=config_dict['classification'])
        update_config(std=new_std, config_dict=config_dict)
        update_config(key=config_dict['evaluation_metric'], value=new_score, config_dict=config_dict)

        if result_df is not None:
            # Save the best dataframe version and num_rows
            update_config(dataframe=result_df, num_rows=result_df.shape[0], config_dict=config_dict)
            update_config(key=f'{run_name}_best_method', value=best_method, config_dict=config_dict)
            params_keys.append(f'{run_name}_best_method')

        elif tuned_params is not None:
            update_config(model_name=best_method, best_model_params=model.get_params(), tuned_params=tuned_params, config_dict=config_dict)
            upload_artifacts(model_name=f'{best_method}_tuned_params', model=model)

        elif model is not None:
            if not isinstance(model, Sequential):
                update_config(model_name=best_method, best_model_params=model.get_params(), config_dict=config_dict)
            else:
                update_config(model_name=best_method, config_dict=config_dict)
            upload_artifacts(model_name=best_method, model=model)

        else:
            # Save best model name  and num_rows. This is works only eval_models wrapper
            update_config(model_name=best_method, config_dict=config_dict)  # add num rows

        # Upload from conflig file

        uploader = MLFlow(config_dict, params_keys)
        uploader.upload_config_file(run_name=run_name)

        return result_df

    else:
        print('We keep previous results since the new results are not significant')


def get_baseline_score(dataframe: pd.DataFrame, target_label: str, classification: bool, evaluation_metric: str,
                       run_id_number: int, model_name, k_fold_method='k_fold', n_folds=3, n_repeats=10):
    """Get a baseline score """
    model = models['clf'][model_name] if classification else models['reg'][model_name]
    x, y = get_x_y_from_df(dataframe=dataframe, target_label=target_label, scaled_df=False)

    scores = get_cross_val_score_wrapper(x=x, y=y, model=model, k_fold_method=k_fold_method, n_folds=n_folds,
                                         n_repeats=n_repeats,
                                         random_state=seed, classification=classification,
                                         evaluation_metric=evaluation_metric,
                                         n_jobs=-1, verbose=3)
    # Uploading results to MLFLOW

    upload_baseline_score(dataframe, run_id_number, evaluation_metric, scores, model)

    print(f'Score: {scores[0]} Std:{scores[1]}')

    return scores


def get_params_from_config(func, config_dict: dict):
    """Obtain the paramaters of a function dynamically"""

    full_args_lst = getfullargspec(func)[0]

    params = {k: v for k, v in config_dict.items() if k in full_args_lst}

    return params


"""******** APPENDIX: OTHER ******** """


def select_best_estimator(selected_estimator=None):
    """

    Args:
        selected_estimator:

    Returns: List with best estimators

    """
    if selected_estimator is not None:

        return list(selected_estimator)

    else:

        start_response = 'Your recommended algorithms based on your responses are:'
        inp0 = input('Do you have more than 50 rows Y/N ?')
        if inp0 == 'N':
            print('Get more data')
        else:
            inp1 = input('Dimension Reduction Y/N ?')
            if inp1 == 'Y':
                print('SECTION:Unsupervised Learning: Dimension Reduction')
                inp2 = input('Topic modelling ? Y/N')
                if inp2 == 'Y':
                    inp3 = input('Probabilistic Y/N')
                    if inp3 == 'Y':
                        return ['LDA']
                    else:
                        return ['SVD']

                else:
                    return ['PCA']
            else:
                inp4 = input('Have responses / labeled data Y/N ?')
                if inp4 == 'Y':
                    inp5 = input('Predict numeric Y/N ?')
                    if inp5 == 'Y':
                        print('SECTION:Supervised Learning: Regression')
                        inp6 = input('Speed or accuracy ?')
                        if inp6 == 'speed':
                            return ['RF', 'MLP', 'XGB', 'ADA']
                        else:
                            return ['LR', 'CART']
                    else:
                        print('SECTION:Supervised Learning: Classification')
                        inp7 = input('Speed or Accuracy speed/accuracy ?')
                        if inp7 == 'speed':
                            inp8 = input('Explainable Y/N ?')

                            if inp8 == 'Y':
                                return ['LR', 'CART']
                            else:
                                inp9 = input('Data is too large? Y/N ?')
                                if inp9 == 'Y':
                                    return ['NB']
                                else:
                                    return ['SVC', 'NB']

                        else:
                            return ['SVC', 'RF', 'MLP', 'XGB']
                else:
                    print('SECTION:Unsupervised Learning: Clustering')

                    inp10 = input('Is hierarchical Y/N ?')
                    if inp10 == 'Y':
                        # return ['SVM', 'RF', 'MLP', 'XGB']
                        print(f'{start_response} Hierarchical')
                    else:
                        inp11 = input('Need to specify k Y/N ?')
                        if inp11 == 'Y':
                            inp12 = input('Categorical variables Y/N ?')
                            if inp12 == 'Y':
                                # return ['SVM', 'RF', 'MLP', 'XGB']
                                print(f'{start_response} K- modes')
                            else:
                                inp13 = input('Prefer probability Y/N ?')
                                if inp13 == 'Y':
                                    return ['GMM']
                                else:
                                    return ['KM']

                        else:
                            return ['DBSC']
