"""********  HYPER PARAMETER TUNING ******** """
import optuna
import pandas as pd
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, tpe
from sklearn import clone
from sklearn import model_selection as ms

import taberspilotml.scoring_funcs.evaluation_metrics
from taberspilotml import constants
from taberspilotml.configs import models, hyper_params
from taberspilotml.scoring_funcs import cross_validation as cv
from taberspilotml.scoring_funcs.datasets import Dataset
import taberspilotml.base_helpers as h
import taberspilotml.scoring_funcs.scorers as scorers

DEFAULT_SEED = 0


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


def hyperopt_parameter_tuning_cv(df: pd.DataFrame, target_label: str, model_name=None, max_evals=80,
                                 k_fold_method='k_fold', n_folds=3, n_repeats=2, classification=True,
                                 evaluation_metric='accuracy', timeout_minutes=10.0,
                                 n_jobs=-1, verbose=0):
    """
    Hyperparameter optimisation using the hyperopt package - runs cross validation with each set of parameters,
    returns the best performing set of parameters as determined by averaging the scores from each split.

    :param df:
        The dataset.
    :param target_label:
        Target label
    :param model_name:
        Name of the model (optional)
    :param max_evals:
        Maximum number of evals.
    :param k_fold_method:
        Method for k fold CV
    :param n_folds:
        Number of folds
    :param n_repeats:
        Number of repeats
    :param classification:
        If True, we're doing classification, regression otherwise.
    :param evaluation_metric:
        The evaluation metric.
    :param timeout_minutes:
        How many minutes to allow.
    :param n_jobs:
        Number of jobs to run in parallel.
    :param verbose:
        Verbosity level.
    :return:
        results, parameters and best model
    """
    if model_name == 'STACKED':
        raise ValueError('This version does not support STACKED model yet')

    # These 2 could be passed in when the function is called, simplifying the logic here.
    eval_metrics = [taberspilotml.scoring_funcs.evaluation_metrics.EvalMetrics.from_str(evaluation_metric)]
    policy = cv.SplitPolicy(policy_type=k_fold_method, n_splits=n_folds, n_repeats=n_repeats, shuffle=True,
                            random_state=constants.DEFAULT_SEED)

    dict_key = 'clf' if classification else 'reg'

    # Passing current_model and space in as parameters would enable more simplification
    current_model = models[dict_key][model_name]
    space = hyper_params[dict_key][model_name]

    print(f'Model name: {model_name}, Parameters: {space}')

    dataset = Dataset.from_dataframe(df, [target_label])
    scores = {}

    def objective(selected_space):
        # clone the current version of the model in order to avoid overwriting the dictionary
        model_base = clone(current_model)
        model_base.set_params(**selected_space)

        score = scorers.get_cross_validation_score(
            dataset=dataset, model=model_base, split_policy=policy, evaluation_metrics=eval_metrics,
            n_jobs=-n_jobs, verbose=verbose)

        scores[model_name] = score[0]

        print(f'Accuracy: {score[0]}')

        # We aim to maximize accuracy, therefore we return it as a negative value
        return {'loss': - score[0], 'std': score[1], 'status': STATUS_OK, 'model': model_base}

    trials = Trials()

    fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials,
         # Convert to seconds.
         timeout=timeout_minutes * 60.0)

    print('optimization complete')
    best_model = trials.results[np.argmin([r['loss'] for r in
                                           trials.results])]

    best_score = best_model['loss'] * -1
    std = best_model['std']

    params = {k: v for k, v in best_model['model'].get_params().items() if v is not None}

    results = {model_name: (best_score, std)}

    print(f'Results:{results}')

    return results, params, best_model['model']


def optuna_hyperopt(df: pd.DataFrame, target_label: str, model_name='XGB', n_minutes_limit=None, n_trials=None,
                    params_list=None, classification=True, evaluation_metric='accuracy', test_size=0.2,
                    direction='maximize'):
    """

    Args:
        df:
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
        model = h.select_custom_dict(models_dict, [model_name])[model_name]

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

            score = scorers.get_hold_out_score(df=df, target_label=target_label, model=model,
                                               test_size=test_size,
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
        h.save_figure_to_disk(df=trials_df, main_folder='Hyper Parameter Tuning',
                              figure_name='optuna hyper param tuning')

        try:
            # Visualization
            optuna.visualization.plot_optimization_history(study)
        except ImportError as err:
            print(f'Try to install or import packages for optuna: {err}')

        best_score = study.best_value
        scores = {model_name: (best_score, None)}
        return scores, best_params[1], model

    else:
        print(f'There are still no parameters for algorithm {model_name}')
        pass


def grid_search_hyperopt(df: pd.DataFrame, target_label: str, grid_search_param_grid: dict, model_name=None,
                         classification=True, evaluation_metric="accuracy", n_jobs=-1, verbose=3, n_folds=3,
                         n_repeats=3,
                         k_fold_method='stratified_k_fold', grid_search_method='randomized', random_state=DEFAULT_SEED):
    """

    Args:
        df:
        target_label:
        grid_search_param_grid:
        model_name:
        classification:
        evaluation_metric:
        n_jobs:
        verbose:
        n_folds:
        n_repeats:
        k_fold_method:
        grid_search_method:
        random_state:

    Returns:

    """
    models_dict = models['clf'] if classification else models['reg']
    model = h.select_custom_dict(models_dict, [model_name])[model_name]

    x, y = h.get_x_y_from_df(df, target_label)

    k_fold_methods_dict = {'k_fold': ms.KFold(n_splits=n_folds, shuffle=True),
                           'repeated_k_fold': ms.RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats,
                                                               random_state=random_state),
                           'stratified_k_fold': ms.StratifiedKFold(n_splits=n_folds, shuffle=True,
                                                                   random_state=random_state),
                           'repeated_stratified_k_fold': ms.RepeatedStratifiedKFold(n_splits=n_folds,
                                                                                    n_repeats=n_repeats,
                                                                                    random_state=random_state)}

    current_kfold = k_fold_methods_dict[k_fold_method]

    grid_search_dict = {
        'randomized': ms.RandomizedSearchCV(model, grid_search_param_grid, scoring=evaluation_metric, n_jobs=n_jobs,
                                            cv=current_kfold,  # this line will be modified or removed
                                            verbose=verbose),
        'gridsearch': ms.GridSearchCV(model, grid_search_param_grid, scoring=evaluation_metric, n_jobs=n_jobs,
                                      cv=current_kfold,  # this line will be modified or removed
                                      verbose=verbose)}  # this does not save results during the process

    current_grid_search = grid_search_dict[grid_search_method]

    grid_result = current_grid_search.fit(x, y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    scores = {model_name: (grid_result.best_score_, None)}

    return scores, grid_result.best_params_, model
