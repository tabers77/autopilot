import mlflow


def upload_baseline_score(dataframe, run_id_number, evaluation_metric, scores, model):
    run_name = f'{run_id_number}_baseline_score_stage'
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metric(evaluation_metric, scores[0])
        mlflow.log_metric('std', scores[1])

        for name, param_value in model.get_params().items():
            if param_value is not None:
                mlflow.log_param(name, param_value)

        mlflow.log_param('num_rows', dataframe.shape[0])
        mlflow.log_param('num_features', len(tuple(dataframe.columns)))
        mlflow.log_param('features', tuple(dataframe.columns))

        print('Logging sklearn artifacts...')

        mlflow.sklearn.log_model(model, run_name)

    print('Find your results here: http://localhost:5000/')

    mlflow.end_run()


def upload_sklearn_model_params(model_name: str, model=None):

    with mlflow.start_run(run_name=model_name):
        for name, param_value in model.get_params().items():
            if param_value is not None:
                mlflow.log_param(name, param_value)

        print('Logging SK-LEARN artifacts...')
        mlflow.sklearn.log_model(model, model_name)




class MLFlow:

    def __init__(self, config_dict=None, params_keys=None):

        self.config_dict = config_dict
        self.params_keys = params_keys

    @staticmethod
    def upload_from_function(func, model_name: str, evaluation_metric=None, dataframe=None, target_label=None,
                             x=None, y=None, model=None, *args, **kwargs):

        with mlflow.start_run(run_name=model_name):

            if x is not None and y is not None:

                scores = func(x=x, y=y, model=model, evaluation_metric=evaluation_metric, *args, **kwargs)
            else:
                scores = func(dataframe=dataframe, target_label=target_label, model=model,
                              evaluation_metric=evaluation_metric, *args, **kwargs)

            # this part will work for multiple scores

            if type(scores) == dict:
                print('Logging metrics...')
                for k, v in scores.items():
                    mlflow.log_metric(k, v)

                print('Logging parameters...')
                for name, param_value in model.get_params().items():
                    if param_value is not None:
                        mlflow.log_param(name, param_value)

            elif type(scores) == list:

                mlflow.log_metric(f'{evaluation_metric}', scores[0])
                if type(scores[1]) == dict:
                    for name, param_value in scores[1].items():
                        mlflow.log_param(name, param_value)
                else:
                    mlflow.log_metric('std', scores[1])  # assuming the second param is standard deviation
            else:
                mlflow.log_metric(f'{evaluation_metric}', scores)

            # Logging general params
            if x is not None and y is not None:
                try:
                    mlflow.log_param('num_rows', x.shape[0])
                    mlflow.log_param('num_features', len(tuple(x.columns)))
                    mlflow.log_param('features', tuple(x.columns))
                except AttributeError as err:
                    print(f'You got an {err}.The input if your data may be scaled')
                    pass
            else:

                mlflow.log_param('Num rows', dataframe.shape[0])
                mlflow.log_param('N features', len(tuple(dataframe.columns)))
                mlflow.log_param('Features', tuple(dataframe.columns))

            print('Logging SK-LEARN artifacts...')

            mlflow.sklearn.log_model(model, model_name)

        print('Find your results here: http://localhost:5000/')
        c1 = 'echo "export PATH=\"`python3 -m site --user-base`/bin:\$PATH\"" >> ~/.bashrc'
        print(f'If you can not open your local host use this command: {c1}')

        mlflow.end_run()

        return scores

    def get_params_to_upload(self, params_keys):

        """Select specific params from CONFIG to upload"""

        results = {k: v for k, v in self.config_dict.items() if k in params_keys}

        return results

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
                        #print('STD is None so we pass')
                        pass
                else:
                    mlflow.log_param(param, value)

        print('Find your results here: http://localhost:5000/')

        mlflow.end_run()

