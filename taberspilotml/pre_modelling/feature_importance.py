""" Code related to computing feature importance and selecting features. """
from typing import Sequence
import pandas as pd
from sklearn.feature_selection import f_classif, f_regression, SelectKBest, SelectFromModel
from sklearn.linear_model import LogisticRegression
import shap

# -----------------
# LOCAL LIBRARIES
# -----------------

import taberspilotml.constants
from taberspilotml.scoring_funcs import datasets as d
from taberspilotml.configs import models
import taberspilotml.base_helpers as h
import taberspilotml.visualization as v
from taberspilotml.scoring_funcs import evaluation_metrics as em
import taberspilotml.scoring_funcs.scorers as sc


def auto_feature_selection_from_estimator(df: pd.DataFrame, target_label: str, estimator) -> Sequence[str]:
    """ Using the supplied estimator, select features from the dataset based on their importance at predicting
    the target variable.

    :param df:
        A dataframe holding the relevant dataset.
    :param target_label:
        The name of the column holding the target variable in the dataframe
    :param estimator:
        The estimator to use.

    :returns The names of the columns of the selected features.
    """

    ds = d.Dataset.from_dataframe(df, [target_label])

    selector = SelectFromModel(estimator.fit(ds.inputs, ds.labels), prefit=True)
    x_new = selector.transform(ds.inputs)
    selected_features = pd.DataFrame(selector.inverse_transform(x_new),
                                     index=ds.inputs.index,
                                     columns=ds.inputs.columns)
    return selected_features.columns[selected_features.var() != 0]


def get_reduced_features_cv_scores(df, target_label, model_name, classification):

    model = models['clf' if classification else 'reg'][model_name]

    scores_dict = {}
    selected_features = auto_feature_selection_from_estimator(df, target_label, estimator=model)

    copy_df = df.copy()
    print(f'Selected features are: {selected_features}')
    xs = {'reduced_x': copy_df[selected_features], 'x_all': copy_df.drop(target_label, axis=1)}
    y = copy_df[target_label]
    for name, x in xs.items():
        scores = sc.get_cross_validation_score(dataset=d.Dataset(inputs=x, labels=y),
                                               model=model,
                                               evaluation_metrics=[
                                                   em.EvalMetrics.ACCURACY
                                                   if classification
                                                   else em.EvalMetrics.NEG_MEAN_SQUARED_ERROR])
        scores_dict[name] = scores

    print(f'Scores: {scores_dict}')
    h.printy('Generating final output ', text_type='subtitle')
    output_df = copy_df[list(selected_features) + [target_label]]

    return scores_dict, output_df


class BestFeatures:
    def __init__(self, df, target_label, classification=True):
        self.df = df
        self.target_label = target_label
        self.classification = classification

    def univariate_feature_selection(self, k=5):
        dataframe = self.df.copy()
        feature_cols = dataframe.columns.drop(self.target_label)
        f = f_classif if self.classification else f_regression
        selector = SelectKBest(score_func=f, k=k)  # f_regression
        x_new = selector.fit_transform(dataframe[feature_cols], dataframe[self.target_label])

        selected_features = pd.DataFrame(selector.inverse_transform(x_new),
                                         index=dataframe.index,
                                         columns=feature_cols)

        return list(selected_features.columns[selected_features.var() != 0])

    def model_feature_selection(self, model_name=None, save_figure=False):
        x, y = h.get_x_y_from_df(self.df, self.target_label)
        models_dict = models['clf'] if self.classification else models['reg']
        current_model = h.select_custom_dict(models_dict, model_name)

        current_model = current_model[model_name]
        current_model.fit(x, y)
        feature_importances = pd.DataFrame(current_model.feature_importances_, index=x.columns,
                                           columns=['importance']).sort_values('importance', ascending=False)

        v.get_graph(feature_importances, figsize=(8, 6), stage='Feature Engineering',
                    color=taberspilotml.constants.DEFAULT_COLOR,
                    horizontal=True, style='default', fig_title=f'Feature importance', x_title='features',
                    y_title='score', sort_type='desc', save_figure=save_figure, file_name='feature_importance_figure')

        return feature_importances

    def l1_feature_selection(self, penalty="l1", c=1):
        """ Computes feature importance of the supplied dataset, using logistic regression as an estimator,
        and the supplied penalty.

        :param penalty:
            The penalty to use.
        :param c:
            The 'C' parameter for the logistic regressor. This is the inverse regularisation strength - smaller values
            produce stronger regularisation.
        """

        estimator = LogisticRegression(C=c, penalty=penalty,
                                       # LBFGS does not support l1 penalty
                                       solver='liblinear' if penalty == 'l1' else 'lbfgs',
                                       random_state=taberspilotml.constants.DEFAULT_SEED)

        return auto_feature_selection_from_estimator(self.df, self.target_label, estimator)

    def compute_shapley_value_per_feature(self, model_name, df_size=1):
        """This value is the average marginal contribution of a feature value across all the possible combinations of features
        df_size =  1 = 100% , 0.5 = 50%
        """
        size = int(len(self.df) * df_size)
        df_sample = self.df[:size]

        # print the JS visualization code to the notebook
        shap.initjs()
        x, y = h.get_x_y_from_df(df_sample, self.target_label)

        models_dict = models['clf'] if self.classification else models['reg']
        model = models_dict[model_name]
        model.fit(x, y)
        # Generate shap values for al categories per user
        shap_values = shap.TreeExplainer(model).shap_values(x)
        shap.summary_plot(shap_values, x, plot_type="bar")




