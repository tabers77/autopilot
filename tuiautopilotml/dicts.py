""" Dictionaries """
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, KBinsDiscretizer, PowerTransformer
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR

from xgboost import XGBClassifier, XGBRegressor

from hyperopt import hp

random_state = 0

scorers = {'clf': ({'accuracy': accuracy_score, 'f1': f1_score}),
           'reg': ({'mean_absolute_error': mean_absolute_error, 'mean_squared_error': mean_squared_error})}

replace_methods = {'mean': np.mean, 'median': np.median}

# Build a class from

models = {'clf': ({'KNN': KNeighborsClassifier(),
                   'LR': LogisticRegression(),
                   'CART': DecisionTreeClassifier(random_state=random_state),
                   'NB': GaussianNB(),
                   'SVC': SVC(),
                   'RF': RandomForestClassifier(random_state=random_state),
                   'XGB': XGBClassifier(use_label_encoder=False, random_state=random_state),
                   'ADA': AdaBoostClassifier(),
                   'MLP': MLPClassifier(max_iter=300, random_state=random_state)}),

          'reg': ({'KNN': KNeighborsRegressor(),
                   'LR': LinearRegression(),
                   'CART': DecisionTreeRegressor(random_state=random_state),
                   'SVR': SVR(),
                   'RF': RandomForestRegressor(random_state=random_state),
                   'XGB': XGBRegressor(use_label_encoder=False, random_state=random_state),
                   'MLP': MLPRegressor(max_iter=300, random_state=random_state)})}

hyper_params = {'clf': ({'XGB': {'n_estimators': hp.choice('n_estimators', [10, 50, 300, 750, 1200, 1300, 1500]),
                                 'max_depth': hp.choice('max_depth', [i for i in range(1, 11, 2)]),
                                 'min_child_weight': hp.choice('min_child_weight', [i for i in range(1, 6, 2)]),
                                 'gamma': hp.choice('gamma', [i / 10.0 for i in range(0, 5)])},

                         'RF': {'n_estimators': hp.choice('n_estimators',
                                                          [int(x) for x in np.linspace(
                                                              start=200, stop=2000, num=10)]),
                                'max_depth': hp.choice('max_depth',
                                                       [int(x) for x in np.linspace(10, 110, num=11)]),
                                'max_features': hp.choice('max_features', [i for i in range(1, 6, 2)]),
                                'min_samples_split': hp.choice('min_samples_split',
                                                               [i / 10.0 for i in range(1, 5)]),
                                'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4]),
                                'bootstrap': hp.choice('bootstrap', [True, False])},
                         'MLP': {
                             'input_n': hp.choice('input_n', [10, 50, 100]),
                             'units1': hp.choice('units1', [100, 200, 1500]),
                             'units2': hp.choice('units2', [50, 100, 750]),
                             'dropout1': hp.choice('dropout1', [.3]),
                             'dropout2': hp.choice('dropout2', [.3]),
                             'batch_size': hp.choice('batch_size', [16, 128]),
                             'optimizer': hp.choice('optimizer', ['adam']),
                             'epochs': hp.choice('epochs', [100])},

                         'NB': {'var_smoothing': hp.choice('input_n', list(np.logspace(0, -9, num=100)))},
                         'SVC': {},
                         'ADA': {'n_estimators': hp.choice('n_estimators',
                                                          [int(x) for x in np.linspace(
                                                              start=200, stop=2000, num=10)])}

                         }), 'reg': ({})}

transformers = {'KBins': KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform'),
                'PCA': PCA(n_components=7),  # adjust this dynamically and get name of columns
                'Truncated': TruncatedSVD(n_components=3),
                'PowerTransformer': PowerTransformer(),
                'Quantile': QuantileTransformer(n_quantiles=100, output_distribution='normal')}

scalers = {'MinMax': MinMaxScaler(),
           'Standard': StandardScaler()}
