import pandas as pd
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

import tuiautopilotml.base_helpers as h
from tuiautopilotml import constants
from tuiautopilotml.dicts import scalers, transformers, models
import tuiautopilotml.hyper_opti as hyper_p

from tuiautopilotml.scoring_funcs import cross_validation as cv
from tuiautopilotml.scoring_funcs import evaluation_metrics as em
from tuiautopilotml.scoring_funcs import datasets
import tuiautopilotml.scoring_funcs.scorers as scorers
import tuiautopilotml.visualization as vs


def eval_model_scaler_wrapper(df, target_label, model_name, k_fold_method='k_fold', n_folds=5,
                              n_repeats=10, classification=True, evaluation_metric='accuracy'):
    """
    Args:
        model_name:
        df:
        target_label:
        k_fold_method:
        n_folds:
        n_repeats:
        classification:
        evaluation_metric:

    Returns:

    Obs: xGBOOST WILL NOT RETURN VALUES WITH A SMALL DATASET,
    Ex: 500, Transformation methods dont apply to stacking models
    """

    scores = {}

    if model_name == 'STACKED':
        print("We don't support stacked models for this function")
        return

    for scaler_name, scaler in scalers.items():
        print(f'Processing scaler: {scaler_name}')
        cv_score, cv_score_std = scorers.get_scaled_x_score(df, target_label,
                                                            model_name=model_name,
                                                            scaler_name=scaler_name,
                                                            use_transformers=False,
                                                            k_fold_method=k_fold_method,
                                                            n_folds=n_folds,
                                                            n_repeats=n_repeats,
                                                            classification=classification,
                                                            evaluation_metric=evaluation_metric)

        scores[f'scale_x-{None}-{scaler_name}-{False}'] = (cv_score, cv_score_std)

        for transformer_name, transformer in transformers.items():
            try:
                print(f'Processing transformer: {transformer_name}')
                cv_score, cv_score_std = scorers.get_scaled_x_score(df, target_label,
                                                                    model_name=model_name,
                                                                    scaler_name=scaler_name,
                                                                    use_transformers=True,
                                                                    transformer_name=transformer_name,
                                                                    k_fold_method=k_fold_method,
                                                                    n_folds=n_folds,
                                                                    n_repeats=n_repeats,
                                                                    classification=classification,
                                                                    evaluation_metric=evaluation_metric)
                scores[f'scale_x-{transformer_name}-{scaler_name}-{True}'] = (cv_score, cv_score_std)
            except ValueError as err:
                print(f'You got error:{err} Transformer {transformer_name} does not work with this dataframe')
                pass

    print(f'Current scores:{scores}')

    h.printy('Generating final output ', text_type='subtitle')

    funcs_dict = {'scale_x': h.scale_x}

    init_funcs_dict = {'df': df, 'target_label': target_label, 'model_name': model_name,
                       'k_fold_method': k_fold_method, 'n_folds': n_folds, 'n_repeats': n_repeats,
                       'classification': classification, 'evaluation_metric': evaluation_metric}

    func, params = h.get_func_params(scores,
                                     input_params={'func': None, 'transformer_name': str, 'scaler_name': str,
                                                   'use_transformers': h.from_str_to_bool},
                                     classification=classification)

    output_df = h.get_output_df_wrapper(function_params=init_funcs_dict,
                                        funcs_to_eval=funcs_dict,
                                        function_name=func,
                                        params=params)

    return scores, output_df


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


def evaluate_models_wrapper(df: pd.DataFrame, target_label: str, models_list: list, scaled_df=True,
                            scaler=scalers['Standard'], classification=True, multiple_eval_scores=False,
                            evaluation_metric='accuracy', stacking=False, n_folds=3, n_repeats=3,
                            random_state=constants.DEFAULT_SEED,
                            k_fold_method='k_fold', verbose=3, n_jobs=-1, is_tuned_params=False, tuned_params=None):
    """
    Evaluate models using cross_validation
    Args:
        df:
        target_label:
        models_list:
        scaled_df:
        scaler:
        classification:
        multiple_eval_scores:
        evaluation_metric:
        stacking:
        n_folds:
        n_repeats:
        random_state:
        k_fold_method:
        verbose:
        n_jobs:
        is_tuned_params:
        tuned_params:

    Returns:

    """

    models_dict = models['clf'] if classification else models['reg']
    current_model_dict = dict({(m, models_dict[m]) for m in models_list})

    if is_tuned_params:
        current_model_dict = hyper_p.get_tuned_models_wrapper(tuned_params=tuned_params, models_dict=current_model_dict)

    if stacking:
        current_model_dict['STACKED'] = get_stacking(n_folds=n_folds, classification=classification,
                                                     models_dict=current_model_dict)
        if classification:
            models['clf']['STACKED'] = get_stacking(n_folds=n_folds, classification=classification,
                                                    models_dict=current_model_dict)
        else:
            models['reg']['STACKED'] = get_stacking(n_folds=n_folds, classification=classification,
                                                    models_dict=current_model_dict)

    ds = datasets.Dataset.from_dataframe(df, [target_label], scaler=scaler if scaled_df else None)

    scores = {}
    models_dict = {}

    policy = cv.SplitPolicy(policy_type=k_fold_method, n_splits=n_folds, n_repeats=n_repeats, random_state=random_state,
                            shuffle=random_state is not None)
    if multiple_eval_scores:
        multi_classif = True
        eval_metrics = (constants.DEFAULT_CLASSIFICATION_METRICS
                        if classification
                        else constants.DEFAULT_REGRESSION_METRICS)
    else:
        multi_classif = False
        eval_metrics = [em.EvalMetrics.from_str(evaluation_metric)]

    for model_name, model in current_model_dict.items():
        cv_results = scorers.get_cross_validation_score(dataset=ds, model=model, split_policy=policy,
                                                        averaging_policy='macro' if multi_classif else None,
                                                        evaluation_metrics=eval_metrics,
                                                        n_jobs=n_jobs, verbose=verbose)

        print(f'Score for {model_name}: {cv_results}')
        scores[model_name] = cv_results
        models_dict[model_name] = model

    a, b, best_method = h.get_best_score(scores, classification=classification, multiple_eval_scores=multiple_eval_scores)

    plot_dict = {k: v[f'test_{evaluation_metric}'] for k, v in
                 scores.items()} if multiple_eval_scores else {k: v[0] for k, v in scores.items()}

    vs.get_graph(input_data=plot_dict, stage='Models', figsize=(6, 4), color=constants.DEFAULT_COLOR, horizontal=True,
                 style='seaborn-darkgrid',
                 fig_title=f'Best Models Scores', x_title='Params', y_title='Scores', save_figure=True,
                 file_name='best_models_scores')

    print(f'Final scores: {scores}')

    return scores, models_dict[best_method]
