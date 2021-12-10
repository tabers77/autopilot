#from tuiautopilotml.helper_functions import get_params_from_config, update_config, update_upload_config
from helper_functions import get_params_from_config, update_config, update_upload_config # TEMP- REMOVE THIS LINE


def autopilot_mode(steps: list, config_dict: dict):

    """Autopilot function"""

    # Define initial checkpoints
    # these 3 will generate data frames
    initial_check_points = ['dataframe_transformation', 'encoding', 'handle_missing_values']
    support_functions = ['baseline_score']
    eval_models_steps = ['evaluate_models']  # this includes functions that returns scores, (scores, std )
    hyper_p_steps = ['hyper_param_opt']

    # Generate a run id
    run_id_number = config_dict['run_id_number']
    print(f'Current run id: {run_id_number}')

    functions_names = list(steps.keys())
    functions = list(steps.values())

    for i in range(len(steps)):

        current_params = get_params_from_config(func=functions[i], config_dict=config_dict)

        print(f'*********JOB:{i}: {functions_names[i]}*********')

        try:

            if functions_names[i] in initial_check_points:

                result_df = functions[i](**current_params)
                if result_df is not None:
                    print(f'Updating config {functions_names[i]}...')
                    update_config(dataframe=result_df, config_dict=config_dict)

            elif functions_names[i] in support_functions:

                # Support functions doest not return anything
                functions[i](**current_params)

            else:

                if functions_names[i] in eval_models_steps:

                    # this function only takes scores
                    scores = functions[i](**current_params)
                    update_upload_config(scores=scores, config_dict=config_dict,
                                         run_name=f'{run_id_number}_{functions_names[i]}_stage')

                elif functions_names[i] in hyper_p_steps:

                    scores, params = functions[i](**current_params)
                    update_upload_config(scores=scores, config_dict=config_dict, tuned_params=params,
                                         run_name=f'{run_id_number}_{functions_names[i]}_stage')

                elif functions_names[i] not in eval_models_steps:
                    scores, result_df = functions[i](**current_params)

                    # this function take a result df as input
                    update_upload_config(scores=scores, config_dict=config_dict, result_df=result_df,
                                         run_name=f'{run_id_number}_{functions_names[i]}_stage')

        except TypeError as err:

            print(f'We could not return scores and result_df from previous step. We skip this step {err}')
            pass
