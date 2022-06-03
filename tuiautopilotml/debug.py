"""Temporary file for debugging"""

from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
# -----------------
# package functions
# -----------------
from tuiautopilotml.analytics.eda import initial_eda_wrapper
from tuiautopilotml.preprocessing import dataframe_transformation

n_samples, n_features = 1000, 20
rng = np.random.RandomState(0)
X, y = make_regression(n_samples=1000, n_features=20, random_state=rng)
sklearn_reg = pd.DataFrame(X)
sklearn_reg['y_label'] = y

cols_to_exclude = [1, 2]
formatted_df = dataframe_transformation(df=sklearn_reg,
                                        cols_to_exclude=cols_to_exclude,
                                        drop_missing_cols=True,
                                        drop_missing_rows=False,
                                        object_is_numerical_cols=None)
print(formatted_df)
initial_eda_wrapper(df=formatted_df,
                    target_label='y_label',
                    summary_report=True,
                    return_outliers=True,
                    save_figures=False)
