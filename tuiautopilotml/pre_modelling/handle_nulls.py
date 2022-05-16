""" ******** IMPUTATION/MISSING VALUES ******** """

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from tuiautopilotml import base_helpers as helpers
from tuiautopilotml.pre_modelling import encoders as enc
from tuiautopilotml.scoring_funcs import scorers as scorers


def eval_imputation_method_wrapper(df, target_label, test_size=0.2, model=RandomForestClassifier(),
                                   classification=True, evaluation_metric='accuracy'):
    """
    This function uses get_custom cv score rather than the default method
    Args:
        df:
        target_label:
        test_size:
        model:
        classification:
        evaluation_metric:

    Returns:

    """
    # Load initial functions to be used
    funcs_to_eval = {'get_imputed_x': get_imputed_x, 'drop_nulls': drop_nulls,
                     'encoded_nulls_score': enc.get_encoded_wrapper}

    for col in df.columns:

        if helpers.is_object_date(df, col):
            print('This function does not accept columns in date format')

    if not helpers.contains_nulls(df):

        print('There are no missing data')

    else:

        strategies = ['median', 'mean', 'most_frequent', 'constant']
        scores = {}
        # Method 1
        for strategy in strategies:
            imput_scores, imput_scores_std = scorers.get_custom_cv_score(df=df, target_label=target_label,
                                                                         classification=classification,
                                                                         evaluation_metric=evaluation_metric,
                                                                         model=model,
                                                                         test_size=test_size, use_custom_method=True,
                                                                         custom_method=get_imputed_x,
                                                                         strategy=strategy)

            scores[f'get_imputed_x-{strategy}'] = (imput_scores, imput_scores_std)

        # Method 2
        print('drop_nulls_score...')
        drop_score, drop_score_std = scorers.get_custom_cv_score(df=df, target_label=target_label,
                                                                 classification=classification,
                                                                 evaluation_metric=evaluation_metric, model=model,
                                                                 test_size=test_size, use_custom_method=True,
                                                                 custom_method=drop_nulls)

        scores[f'drop_nulls-{None}'] = (drop_score, drop_score_std)

        # Method 3
        encoded_nulls_scores, encoded_nulls_std = scorers.get_custom_cv_score(
            df=enc.get_encoded_wrapper(df, encode_nulls=True),
            target_label=target_label,
            classification=classification,
            evaluation_metric=evaluation_metric, model=model,
            test_size=test_size)

        scores[f'encoded_nulls_score-{None}'] = (encoded_nulls_scores, encoded_nulls_std)

        print(f'Results: {scores}')
        print(' --------------- Generating final output --------------- ')

        # Observe that the following dictionary must contain all parameters of the functions used
        function_params = {'df': df, 'target_label': target_label, 'test_size': test_size, 'model': model,
                           'classification': classification, 'evaluation_metric': evaluation_metric,
                           'encode_nulls': True, 'return_mapping': False, 'exclude_from_encoding': None}

        best_func_name, params = helpers.get_func_params(scores, input_params={'func': None, 'strategy': str},
                                                         classification=classification)
        try:
            output_df = helpers.get_output_df_wrapper(function_params=function_params,
                                                      funcs_to_eval=funcs_to_eval,
                                                      function_name=best_func_name,
                                                      params=params)
            return output_df

        except KeyError as err:
            print(f'You may need to include param {err} in dict function_params')


def get_imputed_x(df: pd.DataFrame, strategy: str):
    """

    Args:
        df:
        strategy:

    Returns: x imputed using a specific imputation strategy

    """
    cols_with_missing = [col for col in df.columns if df[col].isnull().any()]
    print(f'Columns with missing values: {cols_with_missing}')
    print(f'Running strategy {strategy}')
    df = df.copy()
    imputer = SimpleImputer(strategy=strategy)
    df[cols_with_missing] = imputer.fit_transform(df[cols_with_missing])
    encoded_df = enc.get_encoded_wrapper(df)

    return encoded_df


def drop_nulls(df: pd.DataFrame):
    df = df.copy()
    df.dropna(axis=0, inplace=True)
    encoded_df = enc.get_encoded_wrapper(df)

    return encoded_df
