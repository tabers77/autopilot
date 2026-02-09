""" Code for uploading models, parameters and metrics to mlflow """
from typing import Any, Callable, Union, Dict, Tuple
import mlflow
import pandas as pd
from sklearn.base import BaseEstimator


# from keras.models import Functional, Sequential'

def safe_start_run(run_name: str):
    """
    Safely starts an MLflow run, handling nested runs if an active run exists.

    Args:
        run_name (str): The name of the run to start.

    Returns:
        None
    """
    if mlflow.active_run() is not None:
        # Start a nested run if there is already an active run
        return mlflow.start_run(run_name=run_name, nested=True)
    else:
        # Start a new run if no active run exists
        return mlflow.start_run(run_name=run_name)


def upload_baseline_score(df: pd.DataFrame,
                          run_id_number: int,
                          scores: Dict[str, Dict[str, Tuple[float, float]],],
                          model: BaseEstimator) -> None:
    """
    Uploads baseline scores during autopilot mode to MLflow.
    Note: Don't forget to start the MLflow server before running this function.

    Args:
        df (pd.DataFrame): The dataset used for evaluation.
        run_id_number (int): The unique run ID for MLflow tracking.
        scores (Dict[str, EvaluationResults]): Dictionary containing score types ('cv' or 'hold_out')
            and their associated EvaluationResults.
        model (BaseEstimator): The trained machine learning model.

    Raises:
        ValueError: If score_values is not an instance of EvaluationResults.
        ValueError: If score_type is not 'cv' or 'hold_out'.

    Example:
        upload_baseline_score(df, "run_1234", scores, model)
    """

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment(f"experiment_{run_id_number}")

    print('URI STARTED')

    run_name = f'{run_id_number}_baseline_score_stage'
    with safe_start_run(run_name=run_name):
        # with mlflow.start_run(run_name=run_name):
        for score_type, score_values in scores.items():
            if not isinstance(score_values, dict):
                raise ValueError('score_values must be an instance of dict')

            if score_type not in ['cv', 'hold_out']:
                raise ValueError('Invalid score type')

            prefix = f"{score_type}_"
            for metric_name, (mean, std) in score_values.items():
                metric_name = f"{prefix}{metric_name.replace('test_', '')}"
                mlflow.log_metric(metric_name, mean)
                mlflow.log_metric(f"{metric_name}_std", std)

        mlflow.log_params({name: value for name, value in model.get_params().items() if value is not None})
        mlflow.log_params({
            'num_rows': df.shape[0],
            'num_features': len(df.columns),
            'features_inc_target': tuple(df.columns)
        })

        print('Logging sklearn artifacts...')
        mlflow.sklearn.log_model(model, run_name)

    print('Find your results here: http://127.0.0.1:8080')
    mlflow.end_run()


def upload_artifacts(model_name: str, model=None):
    """ This will upload model artifacts for either Keras Sequential models or sci-kit learn models.

    NB: This does not currently support Keras Functional models or any other model type.
    Will throw an exception if a Functional model that is not also Sequential is passed in.

    :param model_name:
        The name of the model.
    :param model:
        The model
    """
    ## NOTE: UNCOMMENT THIS FOR KERAS MODELS
    # if isinstance(model, Functional):
    #     # Sequential inherits from Functional so test if the model is sequential before rejecting the model.
    #     is_keras_sequential = isinstance(model, Sequential)
    #     assert is_keras_sequential, "Keras Functional models not supported."
    # else:
    #     is_keras_sequential = False
    with safe_start_run(run_name=model_name):
        # with mlflow.start_run(run_name=model_name):
        ## NOTE: UNCOMMENT THIS FOR KERAS MODELS

        # if is_keras_sequential:
        #     print('Logging KERAS Sequential model artifacts...')
        #     mlflow.keras.log_model(model, model_name)
        # else:  # Assume we have a sk-learn model

        for name, param_value in model.get_params().items():
            if param_value is not None:
                mlflow.log_param(name, param_value)

        print('Logging SK-LEARN model artifacts...')
        mlflow.sklearn.log_model(model, model_name)


class MLFlow:
    """ Wrapper for uploading data to mlflow. """

    def __init__(self, config_dict=None, params_keys=None):
        self.config_dict = config_dict
        self.params_keys = params_keys

    @staticmethod
    def upload_from_function_df(scoring_function: Callable, model_name: str, evaluation_metric: str,
                                df: pd.DataFrame, target_label: str, model: Any, *args, **kwargs):
        """ Runs a scoring function, and then uploads results to MLFlow. Assumes a dataframe that needs to be split into
        input features and target label is supplied.

        :param scoring_function:
            The scoring function
        :param model_name:
            The name of the model.
        :param evaluation_metric:
            The evaluation metric name.
        :param df:
            The dataframe containing both input features and labels.
        :param target_label:
            The target label
        :param model:
            The model

        :returns:
            The scores computing by the scoring function.
        """
        with safe_start_run(run_name=model_name):
            # with mlflow.start_run(run_name=model_name):
            scores = scoring_function(df=df, target_label=target_label, model=model,
                                      evaluation_metric=evaluation_metric, *args, **kwargs)
            mlflow.log_param('Num rows', df.shape[0])
            mlflow.log_param('N features', len(tuple(df.columns)))
            mlflow.log_param('Features', tuple(df.columns))

            MLFlow._upload_scores(scores, evaluation_metric, model, model_name)
            return scores

    @staticmethod
    def upload_from_function_xy(scoring_function: Callable, model_name: str, evaluation_metric: str,
                                x: Union[pd.DataFrame, pd.Series], y: Union[pd.DataFrame, pd.Series], model: Any,
                                *args, **kwargs):
        """ Runs the supplied scoring function with the model and the data supplied as a set of features
        and a set of labels. Uploads results, model to MLFlow.

        :param scoring_function:
            The scoring function
        :param model_name:
            The name of the model
        :param evaluation_metric:
            The evaluation metric
        :param x:
            The input features
        :param y:
            The labels
        :param model:
            The model
        :returns:
            The scores computed by the scoring function.
        """
        with safe_start_run(run_name=model_name):
            # with mlflow.start_run(run_name=model_name):
            scores = scoring_function(x=x, y=y, model=model, evaluation_metric=evaluation_metric, *args, **kwargs)
            try:
                mlflow.log_param('num_rows', x.shape[0])
                mlflow.log_param('num_features', len(tuple(x.columns)))
                mlflow.log_param('features', tuple(x.columns))
            except AttributeError as err:
                print(f'You got an {err}.The inputs in your data may be scaled')

            MLFlow._upload_scores(scores, evaluation_metric, model, model_name)
            return scores

    @staticmethod
    def _log_items(logger, items_dict: dict, include_nulls=False):
        for k, v in items_dict.items():
            if v is not None or include_nulls:
                logger(k, v)

    @staticmethod
    def _upload_scores(scores: Union[dict, list, float], evaluation_metric, model, model_name):
        if isinstance(scores, dict):
            print('Logging metrics...')
            MLFlow._log_items(mlflow.log_metric, scores)

            print('Logging parameters...')
            MLFlow._log_items(mlflow.log_param, model.get_params())

        elif isinstance(scores, list):
            # We assume only first 2 items relevant here!
            mlflow.log_metric(f'{evaluation_metric}', scores[0])
            if isinstance(scores[1], dict):
                MLFlow._log_items(mlflow.log_param, scores[1])
            else:
                mlflow.log_metric('std', scores[1])  # assuming the second param is standard deviation
        else:
            mlflow.log_metric(f'{evaluation_metric}', scores)

        print('Logging SK-LEARN artifacts...')
        mlflow.sklearn.log_model(model, model_name)

        print('Find your results here: http://localhost:5000/')
        c1 = 'echo "export PATH=\"`python3 -m site --user-base`/bin:\$PATH\"" >> ~/.bashrc'
        print(f'If you can not open your local host use this command: {c1}')

        mlflow.end_run()

    def get_params_to_upload(self, params_keys):
        """Select specific params from CONFIG to upload"""

        return {k: v for k, v in self.config_dict.items() if k in params_keys}

    def upload_config_file(self, run_name):
        """This function will upload the updated config to MLFLOW"""

        config_results = self.get_params_to_upload(self.params_keys)
        print('Uploading config to MLFLOW...')
        with safe_start_run(run_name=run_name):
            # with mlflow.start_run(run_name=run_name):
            for param, value in config_results.items():
                if param == self.config_dict['evaluation_metric']:
                    mlflow.log_metric(param, value)

                elif param == 'std':
                    if value is not None:
                        mlflow.log_metric(param, value)
                else:
                    mlflow.log_param(param, value)

        print('Find your results here: http://localhost:5000/')
        mlflow.end_run()


def upload_experiment_config_tags(experiment_config):
    """Upload ExperimentConfig as structured MLflow tags.

    This enables querying experiments by pipeline structure in the MLflow UI.

    :param experiment_config: An ExperimentConfig instance.
    """
    tags = {
        'experiment.model_name': experiment_config.model.model_name,
        'experiment.scaler': experiment_config.features.scaler_name or 'None',
        'experiment.transformer': experiment_config.features.transformer_name or 'None',
        'experiment.imputation': experiment_config.preprocessing.imputation_strategy,
        'experiment.cv_policy': experiment_config.cv.policy_type,
        'experiment.n_splits': str(experiment_config.cv.n_splits),
        'experiment.stacking': str(experiment_config.model.stacking),
        'experiment.task_type': experiment_config.task_type.value,
        'experiment.fingerprint': experiment_config.fingerprint,
    }
    mlflow.set_tags(tags)
