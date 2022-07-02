"""******** AUTOPILOT MODE FUNCTIONS ******** """
from typing import Callable

import taberspilotml.pre_modelling.imbalance
import taberspilotml.pre_modelling.outliers as outliers
import taberspilotml.pre_modelling.encoders as enc
import taberspilotml.hyper_opti as hyper_p
import taberspilotml.modelling.ml_models as ml_models
import taberspilotml.modelling.neural_nets as neural_nets
import taberspilotml.pre_modelling.feature_importance as fi
import taberspilotml.preprocessing as dv
from taberspilotml import base_helpers
import taberspilotml.base_helpers as bh

DEFAULT_SEED = 0


def initial_checkpoint_handler(step_name: str, function: Callable, parameters: dict, config_dict: dict):
    """ Handler for typically early stage steps that preprocess data before presenting it to a model.

    :param step_name:
        The name of the step
    :param function:
        The function to call during this step.
    :param parameters:
        The parameters to send into the function.
    :param config_dict:
        The configuration.
    """

    result_df = function(**parameters)
    if result_df is not None:
        print(f'Updating config {step_name}...')
        bh.update_config(df=result_df, base_encoded_df=result_df, config_dict=config_dict)


def scoring_handler(step_name: str, function: Callable, parameters: dict, config_dict: dict):
    """ Handles steps where we get a model and score only for the dataset.

    :param step_name:
        The name of the step
    :param function:
        The function to call during this step.
    :param parameters:
        The parameters to send into the function.
    :param config_dict:
        The configuration.
    """

    # this functions only takes scores as input
    scores, model = function(**parameters)
    base_helpers.update_upload_config(scores=scores, config_dict=config_dict,
                                      run_name=f'{config_dict["run_id_number"]}_{step_name}_stage', model=model)


def support_handler(_step_name: str, function: Callable, parameters: dict, _config_dict: dict):
    """ Handles steps where there is no result from calling a function or where we can ignore the result.

    :param _step_name:
        The name of the step - ignored here.
    :param function:
        The function to call during this step.
    :param parameters:
        The parameters to send into the function.
    :param _config_dict:
        The configuration - ignored here.
    """

    function(**parameters)


def hyper_p_handler(step_name: str, function: Callable, parameters: dict, config_dict: dict):
    """ Handles steps where that perform hyper-parameter optimisation, returning scores, tuned hyper-parameters and
    best model.

    :param step_name:
        The name of the step
    :param function:
        The function to call during this step.
    :param parameters:
        The parameters to send into the function.
    :param config_dict:
        The configuration.
    """

    scores, tuned_params, best_model = function(**parameters)
    base_helpers.update_upload_config(scores=scores, config_dict=config_dict, tuned_params=tuned_params,
                                      model=best_model, run_name=f'{config_dict["run_id_number"]}_{step_name}_stage')


def mixed_handler(step_name: str, function: Callable, parameters: dict, config_dict: dict):
    """ Handler for steps that return scores and a dataframe of results.

    :param step_name:
        The name of the step
    :param function:
        The function to call during this step.
    :param parameters:
        The parameters to send into the function.
    :param config_dict:
        The configuration.
    """

    scores, result_df = function(**parameters)
    # this functions take a result df as input
    base_helpers.update_upload_config(scores=scores, config_dict=config_dict, result_df=result_df,
                                      run_name=f'{config_dict["run_id_number"]}_{step_name}_stage')


# Define all pipeline steps
default_steps = {'dataframe_transformation': (dv.dataframe_transformation, initial_checkpoint_handler),
                 'handle_missing_values': (taberspilotml.pre_modelling.handle_nulls.eval_imputation_method_wrapper,
                                           initial_checkpoint_handler),
                 'encoding': (enc.default_encoding, initial_checkpoint_handler),
                 'baseline_score': (bh.get_baseline_score, support_handler),
                 # 'baseline_score': (base_helpers.baseline_score_cv, support_handler)
                 }

modelling_steps = {'handle_outliers': (outliers.handle_outliers, mixed_handler),
                   'evaluate_oversamplers': (taberspilotml.pre_modelling.imbalance.evaluate_oversamplers,
                                             mixed_handler),
                   'evaluate_models': (ml_models.evaluate_models_wrapper, scoring_handler)
                   }

post_modelling_steps = {'feature_selection': (fi.get_reduced_features_cv_scores, mixed_handler),
                        'transformation_methods': (ml_models.eval_model_scaler_wrapper, mixed_handler),
                        'hyper_param_opt': (hyper_p.hyperopt_parameter_tuning_cv, hyper_p_handler),
                        'optuna': (hyper_p.optuna_hyperopt, hyper_p_handler),
                        'grid_search': (hyper_p.grid_search_hyperopt, hyper_p_handler)}

mlp = {'evaluate_mlp_model': (neural_nets.cv_eval_mlp, scoring_handler)}

all_pipeline_steps = {**modelling_steps, **post_modelling_steps, **mlp}


def autopilot_mode(steps: list, config_dict: dict):
    """
    Info: Runs the steps selected sequentially

    Available steps: dataframe_transformation, handle_missing_values, encoding, baseline_score, handle_outliers,
    evaluate_oversamplers, evaluate_models, feature_selection, transformation_methods, hyper_param_opt, optuna,
    grid_search, evaluate_mlp_model

    Recommended steps:
    dataframe_transformation >> eval_imputation_method_wrapper >> get_encoded_wrapper >> get_baseline_score
    >> handle_outliers >> evaluate_oversamplers >> evaluate_models_wrapper >> reduced features models
    >> hyper_opt_manual

    Returns:
    """

    # Select only pipeline steps from the supplied steps
    pipeline_steps = {k: v for k, v in all_pipeline_steps.items() if k in steps}

    # Ensure all default steps are prepended to the selected pipeline steps.
    final_steps = {**default_steps, **pipeline_steps}

    return execute_steps(final_steps, config_dict)


def execute_steps(steps, config_dict):
    # Generate a run id
    run_id_number = config_dict['run_id_number']
    print(f'Current run_id: {run_id_number}')
    summary_report = []
    for i, (step_name, (func, handler)) in enumerate(steps.items()):
        current_params = bh.get_params_from_config(func=func, config_dict=config_dict)
        bh.printy(text='JOB', text_type='custom', p1=i, p2=step_name)

        try:
            handler(step_name, func, current_params, config_dict)
            summary_report.append(('successfully processed', step_name))

        except TypeError as err:
            summary_report.append(('not processed', step_name, err))
            print(f'We could not return scores and result_df from previous step. We skip this step {err}')
    return summary_report
