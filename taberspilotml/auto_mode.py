"""******** AUTOPILOT MODE FUNCTIONS ******** """
from typing import Callable
from collections import OrderedDict

import taberspilotml.pre_modelling.imbalance
import taberspilotml.pre_modelling.outliers as outliers
import taberspilotml.pre_modelling.encoders as enc
import taberspilotml.hyper_opti as hyper_p
import taberspilotml.modelling.ml_models as ml_models
# import taberspilotml.modelling.neural_nets as neural_nets
import taberspilotml.pre_modelling.feature_importance as fi
# import taberspilotml.preprocessing as dv
import taberspilotml.preprocessing.generals as dv
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


def inject_task_into_steps(task_specs=None):
    """
    Inserts a new task into a step registry dictionary at a specified position.

    Parameters:
    -----------
    task_specs : list of dict
        A list of task specifications, each containing:
        - 'step_name' (str): Target registry: 'default_steps', 'modelling_steps',
          'post_modelling_steps', or 'time_series_steps'.
        - 'task_name' (str): Task name to insert.
        - 'position' (int): Index to insert at.
        - 'function' (callable): Task function.
        - 'handler' (callable, optional): Handler function. Defaults to initial_checkpoint_handler.

    Raises:
    -------
    ValueError: If `task_specs` is not a list.
    ValueError: If 'step_name' is not a recognized step registry.

    Notes:
    ------
    - Uses `initial_checkpoint_handler` as the default handler.
    - Position is adjusted if out of bounds.
    """

    global default_steps, modelling_steps, post_modelling_steps, time_series_steps

    step_registries = {
        'default_steps': 'default_steps',
        'modelling_steps': 'modelling_steps',
        'post_modelling_steps': 'post_modelling_steps',
        'time_series_steps': 'time_series_steps',
    }

    # Validate input
    if not isinstance(task_specs, list):
        raise ValueError("Expected a list of task specifications.")

    for spec_container in task_specs:
        step_name = spec_container['step_name']

        if step_name not in step_registries:
            raise ValueError(f'Unsupported step_name: {step_name}. '
                             f'Supported: {list(step_registries.keys())}')

        # Get the handler (default to initial_checkpoint_handler)
        handler = spec_container.get('handler', initial_checkpoint_handler)

        new_task = {spec_container['task_name']: (spec_container['function'], handler)}

        # Get the target registry
        if step_name == 'default_steps':
            target = default_steps
        elif step_name == 'modelling_steps':
            target = modelling_steps
        elif step_name == 'post_modelling_steps':
            target = post_modelling_steps
        elif step_name == 'time_series_steps':
            target = time_series_steps

        # Convert to a list of tuples to insert at a specific index
        items = list(target.items())

        # Insert at the desired position
        position = spec_container['position']
        items.insert(position, list(new_task.items())[0])

        # Convert back to OrderedDict to maintain order
        updated = OrderedDict(items)

        if step_name == 'default_steps':
            default_steps = updated
        elif step_name == 'modelling_steps':
            modelling_steps = updated
        elif step_name == 'post_modelling_steps':
            post_modelling_steps = updated
        elif step_name == 'time_series_steps':
            time_series_steps = updated


# mlp = {'evaluate_mlp_model': (neural_nets.cv_eval_mlp, scoring_handler)}

# all_pipeline_steps_ex_default_steps = {**modelling_steps, **post_modelling_steps, **mlp} # temporarily commented out
# Time-series specific steps (empty by default, populated via inject_task_into_steps)
time_series_steps = OrderedDict()

all_pipeline_steps_ex_default_steps = {**modelling_steps, **post_modelling_steps}


def autopilot_mode(steps: list, config_dict: dict, task_specs=None):
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
    if task_specs is not None:
        inject_task_into_steps(task_specs=task_specs)

    # Select only pipeline steps from the supplied steps
    pipeline_steps = {k: v for k, v in all_pipeline_steps_ex_default_steps.items() if k in steps}

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


def run_experiment(config, df):
    """Run a single ExperimentConfig and return an ExperimentResult.

    This is the new entry point for the experiment framework.
    Converts ExperimentConfig -> PipelineSpec -> step dict -> execute_steps -> ExperimentResult.

    For direct, lightweight execution (bypassing the full pipeline), use
    taberspilotml.experiment.runner.run_single() instead.

    :param config: An ExperimentConfig instance.
    :param df: The dataframe to run the experiment on.
    :returns: An ExperimentResult instance.
    """
    from taberspilotml.experiment.runner import run_single
    return run_single(config, df)
