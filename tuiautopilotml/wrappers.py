# INIT PACKAGES

from tuiautopilotml.helper_functions import *
import re
from scipy.stats import ttest_ind

# DATABASES
from google.cloud import bigquery
import snowflake.connector

# SKLEARN CLASSIFIERS
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier

# SKLEARN REGRESSORS
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

# SKLEARN FOLD/GRIDSEARCH
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold

# These libraries are temporarily here until we add functions using them
# KERAS - TENSORFLOW
# import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.layers import Dropout
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# from tensorflow.python.client import device_lib

# These comments are temporarily here until we create a class where we can pass these parameters
# INIT PARAMS FOR CLASS
# dataframe
# target label
# cols_to_drop
# sample size
# classification or not
# scoring
# seed
# models_list
# test size
# palette, color
# baseline model


# models_list = select_best_estimator()
models_list_default = ['KNN', 'NB', 'SVC', 'RF', 'XGB', 'ADA', 'MLP']

seed = 0


@time_performance_decor
@gc_collect_decor
def extract_from_database_to_df(sql_file_location=None, mode='from_csv', new_file_title='current_dataframe',
                                type_of_connection=None, save_to_csv=True, json_file_bq=None,
                                bq_project_name=None, sf_password_file_location=None, sf_user_name=None,
                                sf_account=None, sf_role=None, aws_file_name=None, aws_project_folder_location=None,
                                *args, **kwargs):

    dataframe = pd.DataFrame()

    if mode == 'from_database':
        with open(sql_file_location) as sql_file:
            query = sql_file.read()

        if type_of_connection == 'bigquery':
            print('Obatining data from bigquery...')
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = json_file_bq
            bigquery_client = bigquery.Client(project=bq_project_name)
            query_job = bigquery_client.query(query)
            dataframe = query_job.to_dataframe()

        elif type_of_connection == 'snowflake':
            print('Obatining data from snowflake...')
            if sf_password_file_location is not None:
                with open(sf_password_file_location, mode='r') as password:
                    password = password.read()
                pw = re.split('[: ]', password)[1]
            else:
                pw = None

            connector = snowflake.connector.connect(
                user=sf_user_name,
                password=pw,
                account=sf_account,
                role=sf_role,
                *args,
                **kwargs
            )
            dataframe = pd.read_sql(query, connector)

        elif type_of_connection == 'amazon':
            print('Obatining data from amazon...')
            s3_bucket_str = 's3://tap-workspace/'
            aws_project_folder_location = aws_project_folder_location + '/'
            aws_file_location = s3_bucket_str + aws_project_folder_location + aws_file_name
            dataframe = pd.read_csv(aws_file_location)

        if save_to_csv:
            print('Saving to csv...')
            dataframe.to_csv(f'{new_file_title}.csv')
            print('Ready')
        return dataframe

    elif mode == 'from_csv':
        print('Loading from csv...')
        dataframe = pd.read_csv(f'{new_file_title}.csv', index_col=0)
        print('Ready')
        return dataframe


def dataframe_transformation(dataframe: pd.DataFrame, cols_to_exclude=None, drop_missing_values=False):
    """
    Validates the data and transform in case the data is not in the correct format

    Args:
        dataframe:
        cols_to_exclude:
        drop_missing_values:

    Returns: pandas daframe

    """
    dataframe = dataframe.copy()
    if cols_to_exclude is not None:
        print('Dropping cols to exclude')
        dataframe.drop(cols_to_exclude, axis=1, inplace=True)

    print('Converting columns to lowercase')
    cols = [str(col).lower() for col in list(dataframe.columns)]
    dataframe.columns = cols
    failures = df_sanity_check(dataframe=dataframe)

    if failures != 0:
        print('Converting to int float and dates')
        dataframe = convert_to_int_float_date(dataframe=dataframe)

        if drop_missing_values:
            print('Drop missing cols')
            dataframe.dropna(axis=1, inplace=True)

        return dataframe
    else:
        print('Your dataframe seems to be correct. We return the original input data')
        return dataframe


@gc_collect_decor
def shuffle_order_save(dataframe: pd.DataFrame, shuffle=False, sample_size=100, sort_by_date_col=None,
                       save_sample_df=False):
    """
    The goal of this function is to save a transformed version of your dataframe depending on your needs
    Args:
        dataframe:
        shuffle:
        sample_size:
        sort_by_date_col:
        save_sample_df:

    Returns:

    """
    # To add: filter users by rfm , matrices , pivoting...
    dataframe = dataframe.copy()
    if shuffle:
        print('Shuffle dataset')
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    print(f'Selecting sample: {sample_size}%')
    sample_size = sample_size / 100
    split = int(len(dataframe) * sample_size)
    dataframe = dataframe[:split]

    if sort_by_date_col is not None:
        dataframe.sort_values(by=sort_by_date_col, inplace=True)  # add ascending or descending

    if save_sample_df:
        print('Saving sample as csv...')
        dataframe.to_csv('sample_dataframe.csv')

    return dataframe


@time_performance_decor
@gc_collect_decor
def model_training_estimator_wrapper(func, dataframe: pd.DataFrame, split_pct_param=0.01, patience_limit=1, *args,
                                     **kwargs):
    """

    Args:
        func:
        dataframe:
        split_pct_param:
        patience_limit:
        *args:
        **kwargs:

    Returns: estimated time in minutes

    """
    dataframe = dataframe.copy()
    split_pct_param = split_pct_param
    shape = dataframe.shape[0]
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
        dataframe = get_encoded_wrapper(dataframe=dataframe)
        dataframe.dropna(axis=0, inplace=True)
        for n_rows in final_list:
            if n_rows >= 700:
                print(f'N rows: {n_rows}')
                t1 = time()
                func(dataframe=dataframe[:n_rows], *args, **kwargs)
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


"""******** EXPLORATORY DATA ANALYSIS ********"""


@time_performance_decor
@gc_collect_decor
def initial_eda_wrapper(dataframe: pd.DataFrame, target_label=None, summary_report=True, return_outliers=False,
                        distribution='non_gaussian', save_figures=False):
    """
    Get a general overview of the the data and return outliers
    Args:
        summary_report: 
        dataframe:
        target_label:
        return_outliers:bool
        distribution:
        save_figures:bool

    Returns:

    """
    print('Checking imbalance degree...')
    check_imbalance_degree(dataframe, target_label)

    print('Generating initial graphs...')
    get_initial_graphs(dataframe, target=target_label, save_figures=save_figures)

    if summary_report:
        get_summary_report(dataframe)

    if return_outliers:
        get_outliers(dataframe=dataframe, distribution=distribution)


class TrainVsTest:

    def __init__(self, train, test):
        self.train = train
        self.test = test

    def get_report(self, target_label):
        print('Generating train test comparison report...')
        report_comp = sv.compare([self.train, 'X_train'], [self.test, 'x_test'])
        report_comp.show_html('Tain_Test_Comparison.html')

        try:
            print('Generating target analysis report...')
            target_comp = sv.compare(self.train, self.test, target_label)
            target_comp.show_html('Target_Analysis.html')

        except ValueError:
            print('sweetviz does not support categorical values so we skip...')
            pass

    # Numerical: train and test distribution
    def get_train_test_distribution(self):

        num_cols = self.train.select_dtypes(exclude='object').columns
        plt.figure(figsize=(10, (len(num_cols)) * 2 + 3))
        count = 1
        for col in num_cols:
            plt.subplot(len(num_cols), 2, count)
            sns.kdeplot(self.train[col], color='red', label='train')
            sns.kdeplot(self.test[col], label='test')
            plt.legend()
            count += 1

        plt.tight_layout()
        plt.show()

    # Numerical: train and test distribution

    def get_train_test_counts(self, n_max_categories=20):

        low_cardinality_cols = [cname for cname in self.train if self.train[cname].nunique() <= n_max_categories and
                                self.train[cname].dtype == "object"]

        for col in low_cardinality_cols:
            train_pct = self.train[col].value_counts() / len(self.train) * 100
            test_pct = self.test[col].value_counts() / len(self.test) * 100
            df_plot = pd.DataFrame([train_pct, test_pct])
            df_plot.index = ['train', 'test']
            df_plot = df_plot.transpose()
            df_plot = df_plot.reset_index().rename(columns={'index': 'col'})
            df_plot.plot.barh(x='col', y=['train', 'test'], title=f'{col}', cmap='coolwarm')

            plt.show()

    def is_distribution_different(self, alpha=0.05):
        """  Info  """

        train_stats = self.train.describe().drop('count', axis=0)
        test_stats = self.test.describe().drop('count', axis=0)
        df = pd.DataFrame()
        num_cols = self.train.select_dtypes(exclude='object').columns
        diff_cols = []
        for col in num_cols:
            tscore, p_value = ttest_ind(self.train[col], self.test[col])

            if p_value < alpha:
                df[f'{col}_train'] = train_stats[col]
                df[f'{col}_test'] = test_stats[col]
                df[f'{col}_p_value'] = p_value
                diff_cols.append(col)

        if len(diff_cols) == 0:
            print('All the the distributions from test set are similar to train set')

        return df, diff_cols

    def get_is_train_col(self, new_train=None, new_test=None, target_label=None):

        """add a binary target column"""

        train = self.train.copy() if new_train is None else new_train
        test = self.test.copy() if new_test is None else new_test
        train['is_train'] = 1
        test['is_train'] = 0
        dataframe = pd.concat([train, test])
        dataframe['is_train'] = dataframe['is_train'].apply(lambda x: 1 if x == 1.0 else 0)
        if target_label is not None:
            dataframe.drop(target_label, axis=1, inplace=True)

        dataframe = get_encoded_wrapper(dataframe)

        return dataframe

    def train_test_pairplot(self, diag_kind="hist"):
        df, diff_cols = self.is_distribution_different()

        if len(diff_cols) > 1:
            diff_cols.append('is_train')
            full_data = self.get_is_train_col()
            sns.pairplot(full_data[diff_cols], hue='is_train', diag_kind=diag_kind)
            plt.show()
        else:
            print('All the the distributions from test set are similar to train set')

    #     def get_barplot_per_period(self, date_col, freq, value):

    #         df = pd.concat([self.train, self.test])
    #         freq = date_col + '_' + freq
    #         # year = date_col + '_' + 'year'
    #         plt.figure(figsize=(12, 6))
    #         df = get_multiple_date_columns(df, date_col)
    #         g = df.groupby(freq)[value].sum()
    #         # g = df.groupby([year,freq])[value].sum()

    #         g.plot(kind='bar', color='blue').set_title(f'{freq} - {value} - sum')

    #         plt.show()

    #         plt.figure(figsize=(12, 6))
    #         g2 = df.groupby(freq)[value].mean()
    #         g2.plot(kind='bar', color='blue').set_title(f'{freq} - {value} - mean')

    #         plt.show()

    def get_covariance_shift_score(self, target_label=None, estimator=RandomForestClassifier(max_depth=2), n_folds=5,
                                   n_repeats=3, random_state=0):

        size = int(len(self.test))

        folds_lst = [i for i in range(0, len(self.train) + 1, size)]
        print(f'Current folds: {folds_lst}')
        cov_scores = []

        for fold in range(0, len(folds_lst) - 1):
            new_train = self.train[folds_lst[fold]:folds_lst[fold + 1]]

            full_data = self.get_is_train_col(new_train=new_train, target_label=target_label)

            x, y = get_x_y_from_df(full_data, 'is_train')

            scores = get_cross_val_score_wrapper(x=x, y=y, model=estimator,
                                                 n_folds=n_folds, n_repeats=n_repeats,
                                                 random_state=random_state,
                                                 classification=True, evaluation_metric='roc_auc')

            print(f'Score for fold {fold}: {scores}')
            cov_scores.append(scores[0])

        print(f'Mean score: {np.mean(cov_scores)}, Standard deviation: {np.std(cov_scores)}')

        return np.mean(cov_scores), np.std(cov_scores)

    def get_covariance_shift_score_per_feature(self, estimator=RandomForestClassifier(max_depth=2),
                                               cov_score_thresh=0.80, n_folds=5,
                                               n_repeats=10, random_state=0):

        size = int(len(self.test))
        folds_lst = [i for i in range(0, len(self.train) + 1, size)]
        print(f'Current folds: {folds_lst}')
        cov_scores = {}

        for fold in range(0, len(folds_lst) - 1):

            new_train = self.train[folds_lst[fold]:folds_lst[fold + 1]]
            print(f'Train shape{new_train.shape}')
            full_data = self.get_is_train_col(new_train=new_train, target_label=None)
            x, y = get_x_y_from_df(full_data, 'is_train')
            for col in x.columns:
                print(f'Col:{col}')
                scores = get_cross_val_score_wrapper(x=pd.DataFrame(x[col]), y=y, model=estimator,
                                                     n_folds=n_folds, n_repeats=n_repeats,
                                                     random_state=random_state,
                                                     classification=True,
                                                     evaluation_metric='roc_auc')
                print(f'Score for {col} in fold {fold}: {scores}')

                cov_scores.setdefault(f'{col}', []).append(scores[0])

        cov_scores = {k: (np.mean(v), np.std(v)) for k, v in cov_scores.items()}

        drop_list = [k for k, v in cov_scores.items() if v[0] > cov_score_thresh]

        return cov_scores, drop_list


"""******** HANDLE MISSING VALUES ********"""


def eval_imputation_method_wrapper(dataframe, target_label, test_size=0.2, model=RandomForestClassifier(),
                                   classification=True, evaluation_metric='accuracy'):
    """

    Args:
        dataframe:
        target_label:
        test_size:
        model:
        classification:
        evaluation_metric:

    Returns:

    """

    for col in dataframe.columns:

        if is_object_date(dataframe, col):
            print('This function does not accept columns in date format')

    if not contains_nulls(dataframe):

        print('There are no missing data')

    else:

        strategies = ['median', 'mean', 'most_frequent', 'constant']
        scores = {}
        # Method 1
        for strategy in strategies:
            imput_scores, imput_scores_std = get_custom_cv_score(dataframe=dataframe, target_label=target_label,
                                                                 classification=classification,
                                                                 evaluation_metric=evaluation_metric, model=model,
                                                                 test_size=test_size, use_custom_method=True,
                                                                 custom_method=get_imputed_x, strategy=strategy)

            scores[f'get_imputed_x-{strategy}'] = (imput_scores, imput_scores_std)

        # Method 2
        print('drop_score...')
        drop_score, drop_score_std = get_custom_cv_score(dataframe=dataframe, target_label=target_label,
                                                         classification=classification,
                                                         evaluation_metric=evaluation_metric, model=model,
                                                         test_size=test_size, use_custom_method=True,
                                                         custom_method=drop_nulls)

        scores[f'drop_score-'] = (drop_score, drop_score_std)

        # Method 3
        # encoded_nulls_scores, encoded_nulls_std = get_custom_cv_score(
        #     dataframe=get_encoded_wrapper(dataframe, encode_nulls=True),
        #     target_label=target_label,
        #     classification=classification,
        #     evaluation_metric=evaluation_metric, model=model,
        #     test_size=test_size)

        # scores['encoded_nulls_score-'] = (encoded_nulls_scores, encoded_nulls_std)

        print(f'Results: {scores}')

        funcs_dict = {'get_imputed_x': get_imputed_x, 'drop_nulls': drop_nulls}
        func, params = get_func_params(scores, input_params=['func', 'strategy'], classification=classification)
        output_df = get_output_df(funcs_dict=funcs_dict, func=func, dataframe=dataframe, **params)

        return output_df


@time_performance_decor
@gc_collect_decor
@check_encoded_df_decor
def get_feature_importance_wrapper(dataframe: pd.DataFrame, target_label: str, method='univariate', classification=True,
                                   n_features=5,
                                   penalty="l1", c=1, save_figure=False):
    """

    Args:
        dataframe:
        target_label:
        method:
        classification:
        n_features:
        penalty:
        c:
        save_figure:

    Returns:

    """

    if method == 'univariate':
        return get_feature_importance_uni(dataframe, target_label=target_label, k=n_features,
                                          classification=classification)

    elif method == 'l1_reg':
        return get_feature_importance_l1(dataframe, target_label, penalty=penalty, c=c)

    elif method == 'rf':
        return get_feature_importance_rf_xgb(dataframe, target_label, classification=classification, algorithm='RF',
                                             save_figure=save_figure).head(n_features)

    elif method == 'xgb':
        return get_feature_importance_rf_xgb(dataframe, target_label, classification=classification, algorithm='XGB',
                                             save_figure=save_figure).head(n_features)


def evaluate_oversamplers(dataframe: pd.DataFrame, target_label: str, classification=True, evaluation_metric='accuracy',
                          test_size=0.2, class_threshold=5, model=RandomForestClassifier(), random_state=0):
    """
    Compare scores between oversampling methods and non oversampling methods
    Args:
        dataframe:
        target_label:
        classification:
        evaluation_metric:
        test_size:
        class_threshold:
        model:
        random_state:

    Returns:

    """

    print('***********Checking imbalance degree***********')
    check_imbalance_degree(dataframe, target_label)

    scores = {}

    print('***********Running smote_os***********')
    y_pred_os = get_smote_os_score(dataframe, target_label, test_size, model, classification, evaluation_metric,
                                   random_state=random_state, k_neighbors=5, n_jobs=-1, sampling_strategy='minority')
    scores['smote_os'] = [y_pred_os, None]

    print('***********Running random_os***********')
    y_pred_random_os = get_random_os_score(dataframe=dataframe, target_label=target_label,
                                           classification=classification, model=model,
                                           test_size=test_size, class_threshold=class_threshold,
                                           evaluation_metric=evaluation_metric)

    scores['random_os'] = [y_pred_random_os, None]
    print(f'Current scores: {scores}')

    print('***********Generating final output***********')
    sub_functions_dict = {'random_os': fit_random_os, 'smote_os': fit_smote_os}
    functions_dict = {'dataframe': dataframe, 'target_label': target_label,
                      'classification': classification, 'evaluation_metric': evaluation_metric, 'test_size': test_size,
                      'class_threshold': class_threshold,
                      'model': model, 'random_state': random_state, 'return_df': True}

    func, params = get_func_params(scores, input_params=['func'], classification=classification)

    output_df = get_output_df_wrapper(functions_dict=functions_dict, sub_functions_dict=sub_functions_dict,
                                      function_name=func, params=params)

    return scores, output_df


def eval_model_scaler_wrapper(dataframe, target_label, model_name, k_fold_method='k_fold', n_folds=5,
                              n_repeats=10, classification=True, evaluation_metric='accuracy'):
    """
    Args:
        model_name: 
        dataframe:
        target_label:
        k_fold_method:
        n_folds:
        n_repeats:
        classification:
        evaluation_metric:

    Returns:

    Obs: xGBOOST WILL NOT RETURN VALUES WITH A SMALL DATASET , Ex: 500 , Transformation methods dont apply to stacking models
    """

    scores = {}

    for scaler_name, scaler in scalers.items():
        print(f'Processing: {scaler_name}')
        cv_score, cv_score_std = get_scaled_x_score(dataframe, target_label, model_name=model_name,
                                                    scaler_name=scaler_name, use_transformers=False,
                                                    k_fold_method=k_fold_method,
                                                    n_folds=n_folds, n_repeats=n_repeats,
                                                    classification=classification,
                                                    evaluation_metric=evaluation_metric)

        scores[f'scale_x-{None}-{scaler_name}'] = (cv_score, cv_score_std)

        for transformer_name, transformer in transformers.items():
            print(f'Processing: {transformer_name}')
            cv_score, cv_score_std = get_scaled_x_score(dataframe, target_label, model_name=model_name,
                                                        scaler_name=scaler_name, use_transformers=True,
                                                        transformer_name=transformer_name,
                                                        k_fold_method=k_fold_method,
                                                        n_folds=n_folds, n_repeats=n_repeats,
                                                        classification=classification,
                                                        evaluation_metric=evaluation_metric)

            scores[f'scale_x-{transformer_name}-{scaler_name}'] = (cv_score, cv_score_std)

    print('Generating graph')

    # get_graph(input_data=scores, stage='Feature Engineering', figsize=(7, 4), color=color, horizontal=True,
    #         style='seaborn-darkgrid', fig_title=f'Best model/scaler combination', x_title='Params', y_title='Scores', save_figure=True,
    #        file_name='best_scalers_g3')

    funcs_dict = {'scale_x': scale_x}

    func, params = get_func_params(scores, input_params=['func', 'transformer_name', 'scaler_name'],
                                   classification=classification)
    output_df = get_output_df(funcs_dict=funcs_dict, func=func, dataframe=dataframe, target_label=target_label,
                              **params)
    output_df = pd.DataFrame(output_df, columns=dataframe.columns)

    print(f'Results:{scores}')

    return scores, output_df


def handle_outliers(dataframe, target_label, distribution='non_gaussian', tot_outlier_pct=4, classification=True,
                    model=RandomForestClassifier(), evaluation_metric='accuracy', test_size=0.2, n_folds=5, n_repeats=10):
    """

    Args:
        dataframe:
        target_label:
        distribution:
        tot_outlier_pct:
        classification:
        model:
        evaluation_metric:
        test_size:
        n_folds:
        n_repeats:

    Returns:

    """
    scores = {}

    print('Replace values')
    for strategy in ['mean', 'median']:
        print(strategy)

        replace_score, replace_std = get_custom_cv_score(dataframe=dataframe, target_label=target_label,
                                                         classification=classification,
                                                         evaluation_metric=evaluation_metric, model=model,
                                                         test_size=test_size, distribution=distribution,
                                                         tot_outlier_pct=tot_outlier_pct,
                                                         strategy=strategy, use_custom_method=True,
                                                         custom_method=replace_outliers)
        scores[f'replace_outliers-{strategy}'] = (replace_score, replace_std)

    print('Drop values')

    drop_score, drop_std = get_custom_cv_score(dataframe=dataframe, target_label=target_label,
                                               classification=classification,
                                               evaluation_metric=evaluation_metric, model=model,
                                               test_size=test_size, distribution=distribution,
                                               tot_outlier_pct=tot_outlier_pct,
                                               strategy=strategy, use_custom_method=True,
                                               custom_method=drop_outliers)
    scores[f'drop_outliers-'] = (drop_score, drop_std)
    print(f'General scores: {scores}')

    print('***********Generating final output***********')

    funcs_dict = {'replace_outliers': replace_outliers, 'drop_outliers': drop_outliers}

    init_funcs_dict = {'dataframe': dataframe, 'target_label': target_label, 'distribution': distribution,
                       'tot_outlier_pct': tot_outlier_pct, 'classification': classification, 'model': model,
                       'evaluation_metric': evaluation_metric,
                       'test_size': test_size, 'n_folds': n_folds, 'n_repeats': n_repeats}

    func, params = get_func_params(scores, input_params=['func', 'strategy'], classification=classification)

    output_df = get_output_df_wrapper(functions_dict=init_funcs_dict, sub_functions_dict=funcs_dict, function_name=func,
                                      params=params)

    # Run cross validation in the whole data set to return final results
    print('Calculating standard cross_validation with best method...')
    a, b, best_method = get_best_score(scores, classification=classification)

    final_cv_score = get_cross_val_score_wrapper(dataframe=output_df, target_label=target_label, model=model,
                                                 k_fold_method='k_fold', n_folds=n_folds, n_repeats=n_repeats,
                                                 random_state=seed,
                                                 classification=classification, evaluation_metric=evaluation_metric,
                                                 n_jobs=-1, verbose=0)

    output_dict = {best_method: final_cv_score}

    print(f'Results:{final_cv_score}')

    return output_dict, output_df


def handle_outliers_iso_forest(dataframe: pd.DataFrame, target_label: str, classification: bool, evaluation_metric: str,
                               model, test_size=0.2):
    scorer = scorers['clf'][evaluation_metric] if classification else scorers['reg'][evaluation_metric]
    x_train, x_test, y_train, y_test = get_splits_wrapper(dataframe, target_label, train_split=True,
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


def get_stacking(models_dict=None, n_folds=3, classification=True):
    """

    Args:
        models_dict:
        n_folds:
        classification:

    Returns:

    """
    level0 = list()

    for name, model in models_dict.items():
        level0.append((name, model))

    if classification:
        meta_model = LogisticRegression()
        model = StackingClassifier(estimators=level0, final_estimator=meta_model, cv=n_folds)

    else:
        meta_model = LinearRegression()
        model = StackingRegressor(estimators=level0, final_estimator=meta_model, cv=n_folds)

    return model


def evaluate_models_wrapper(dataframe: pd.DataFrame, target_label: str, models_list: list,  scaled_df=True, scaler=scalers['Standard'],
                            classification=True, multiple_eval_scores=False, multi_classif=False, evaluation_metric='accuracy', stacking=False, n_folds=3,
                            n_repeats=3, random_state=seed, k_fold_method='k_fold', verbose=3, is_tuned_params=False, tuned_params=None):
    """

    Args:
        dataframe:
        target_label:
        models_list:
        scaled_df:
        scaler:
        classification:
        multiple_eval_scores:
        multi_classif:
        evaluation_metric:
        stacking:
        n_folds:
        n_repeats:
        random_state:
        k_fold_method:
        verbose:
        is_tuned_params:
        tuned_params:

    Returns:

    """

    models_dict = models['clf'] if classification else models['reg']
    current_model_dict = select_custom_dict(models_dict, models_list)

    if is_tuned_params:
        print('Using tuned version')

        current_model_dict = get_tuned_models_wrapper(tuned_params=tuned_params, models_dict=current_model_dict)

    if stacking:
        print('Get stacking')
        current_model_dict['STACKED'] = get_stacking(n_folds=n_folds, classification=classification,
                                                     models_dict=current_model_dict)

        print('Adding stacked version to models dict..')

        if classification:
            models['clf']['STACKED'] = get_stacking(n_folds=n_folds, classification=classification,
                                                    models_dict=current_model_dict)
        else:
            models['reg']['STACKED'] = get_stacking(n_folds=n_folds, classification=classification,
                                                    models_dict=current_model_dict)

    scores = {}
    x, y = get_x_y_from_df(dataframe, target_label, scaled_df=scaled_df, scaler=scaler)

    for model_name, model in current_model_dict.items():
        print(f'Calculating results for {model_name}')

        # ML FLOW

        uploader = MLFlow()

        func = get_cross_val_score_wrapper

        cv_results = uploader.upload_from_function(func=func, x=x, y=y,
                                                   model_name=model_name,
                                                   evaluation_metric=evaluation_metric,
                                                   model=model, multiple_eval_scores=multiple_eval_scores,
                                                   multi_classif=multi_classif,
                                                   n_folds=n_folds, n_repeats=n_repeats, random_state=random_state,
                                                   k_fold_method=k_fold_method, verbose=verbose)

        print(f'Score for {model_name}: {cv_results}')
        scores[model_name] = cv_results

    # plot results
    plot_dict = {k: v[f'test_{evaluation_metric}'] for k, v in
                 scores.items()} if multiple_eval_scores else {k: v[0] for k, v in scores.items()}

    print('Generating graph')
    get_graph(input_data=plot_dict, stage='Models', figsize=(6, 4), color=current_color, horizontal=True,
              style='seaborn-darkgrid',
              fig_title=f'Best Models Scores', x_title='Params', y_title='Scores', save_figure=True,
              file_name='best_models_scores')

    # we could save the following data frame in disk

    #     results_df = dict_to_df(input_dict=scores, multiple_eval_scores=multiple_eval_scores,
    #                                   evaluation_metric=evaluation_metric)

    print(f'Results:{scores}')

    return scores


@time_performance_decor
@gc_collect_decor
def grid_search_wrapper(dataframe: pd.DataFrame, target_label: str, param_grid: dict, model=RandomForestClassifier(),
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
        return grid_result

    except ValueError as err:
        print(f'You got following error: {err}')
        pass
