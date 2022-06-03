""" Code for uploading models, parameters and metrics to mlflow """
from typing import Any, Callable, Union
import mlflow
import pandas as pd
from keras.models import Functional, Sequential


def upload_baseline_score(df, run_id_number, evaluation_metric, scores, model):
    """ Used for uploading baseline scores during autopilot mode.

    :param df:
        The dataset.
    :param run_id_number:
        The run ID number for mlflow
    :param evaluation_metric:
        The evaluation metric used.
    :param scores:
        The scores.
    :param model:
        The model
    """

    run_name = f'{run_id_number}_baseline_score_stage'
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metric(evaluation_metric, scores[0])
        mlflow.log_metric('std', scores[1])

        for name, param_value in model.get_params().items():
            if param_value is not None:
                mlflow.log_param(name, param_value)

        mlflow.log_param('num_rows', df.shape[0])
        mlflow.log_param('num_features', len(tuple(df.columns)))
        mlflow.log_param('features_inc_target', tuple(df.columns))

        print('Logging sklearn artifacts...')

        mlflow.sklearn.log_model(model, run_name)

    print('Find your results here: http://localhost:5000/')

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
    if isinstance(model, Functional):
        # Sequential inherits from Functional so test if the model is sequential before rejecting the model.
        is_keras_sequential = isinstance(model, Sequential)
        assert is_keras_sequential, "Keras Functional models not supported."
    else:
        is_keras_sequential = False

    with mlflow.start_run(run_name=model_name):
        if is_keras_sequential:
            print('Logging KERAS Sequential model artifacts...')
            mlflow.keras.log_model(model, model_name)
        else:  # Assume we have a sk-learn model
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
        with mlflow.start_run(run_name=model_name):
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

        with mlflow.start_run(run_name=model_name):
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
        with mlflow.start_run(run_name=run_name):
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
