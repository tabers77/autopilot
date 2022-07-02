"""******** IMBALANCED DATASETS - OVERSAMPLING (move this section to its own file) ********"""

import numpy as np
import pandas as pd
import seaborn as sns
import sweetviz as sv
from imblearn.over_sampling import RandomOverSampler, SMOTE
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

import taberspilotml.base_helpers
import taberspilotml.pre_modelling

from taberspilotml import configs as dicts, base_helpers as hf
from taberspilotml.pre_modelling import encoders as enc
from taberspilotml.scoring_funcs import (cross_validation as cv,
                                         datasets as d,
                                         scorers as scorers)

DEFAULT_PALETTE = 'mako'


def compute_entropy(seq: pd.Series):
    """ Returns the entropy of the supplied sequence.

    :param seq:
        The sequence.
    """
    if not isinstance(seq, list):
        seq = list(seq)

    n = len(seq)
    total_count_dict = {i: seq.count(i) for i in seq}.items()
    k = len(total_count_dict)
    h = -sum([(v / n) * np.log((v / n)) for k, v in total_count_dict])

    return h / np.log(k)


def check_imbalance_degree(df: pd.DataFrame, target_label: str):
    """ Returns the entropy, moderately imbalanced classes and extremely imbalanced classes for the supplied dataset.

    :param df:
        The dataframe containing the data.
    :param target_label:
        The name of the column with the target variable in it.
    """

    if df[target_label].nunique() >= 20:
        print('It looks like your target label contains too many categories or is a regression dataset')
    else:
        total_count_pct = dict(df.groupby(target_label).size() / len(df))
        moderate = {}
        extreme = {}
        for k, v in total_count_pct.items():
            if 0.1 <= v <= 0.20:
                moderate[k] = v
                extreme[k] = v
        entropy_score = compute_entropy(df[target_label])
        print(f'Overall entropy score: {entropy_score}')
        print(f'Moderate imbalanced classes: {moderate}')
        print(f'Extreme imbalanced classes: {extreme} ')
        return entropy_score


def get_percentile_per_class(x_train, y_train, single_class):
    for percentile_value in range(35, 95, 5):
        try:

            percentile_n = int(np.percentile(y_train.value_counts(), percentile_value))
            print(f'Current percentile value {percentile_n}')
            weights = {class_: percentile_n for class_ in np.unique(y_train) if class_ == single_class}
            over_sampler = RandomOverSampler(weights)
            over_sampler.fit_resample(x_train, y_train)

            return percentile_n

        except ValueError:
            print(f'Failed with {percentile_value} percentile')
            pass


def plot_cm_matrix_prediction_error(y_test, y_pred):
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 8))
    conf_m_pct = cm / np.sum(cm) * 100
    sns.heatmap(conf_m_pct, annot=True, cmap=DEFAULT_PALETTE).set_title('Confusion metrics as percentages')

    # save_figure_to_disk(main_folder='Feature Engineering', figure_name='Confusion metrics as percentages',
    #                     save_as_plt=True)

    plt.show()

    # Class Prediction Error
    pd.DataFrame(cm).plot(kind='bar', stacked=True, cmap="mako").set_title('Class Prediction Error')
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 0.5))

    # save_figure_to_disk(main_folder='Feature Engineering', figure_name='Class Prediction Error',
    #                     save_as_plt=True)
    plt.show()


def get_n_classes_to_resample(y_test, y_pred, y_counts, class_threshold=0.05):
    # 1. Get only f1 scores (per class) from classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    classes_f1_scores = {}
    for class_, scores_dict in report.items():
        if class_.isnumeric():
            for score, value in scores_dict.items():
                if score == 'f1-score':
                    classes_f1_scores[class_] = value

    # 2. Filter 1 - Get classes  with a lower percentage than the mean of f1 score
    mean = np.mean(list(classes_f1_scores.values()))  # choose other f1 score
    f1_scores_lst = [k for k, v in classes_f1_scores.items() if v <= mean]

    # 3. Filter 2 - Choose a representative threshold based on total counts
    print(f'y counts:{y_counts}')
    class_threshold = class_threshold
    classes_to_resample = []

    for k, v in y_counts.items():
        k = str(k)
        if k in f1_scores_lst and v >= class_threshold:
            classes_to_resample.append(int(k))
    print(f'Classes to re sample: {classes_to_resample}')

    return classes_to_resample


def get_random_os_score(df: pd.DataFrame, target_label: str, classification: bool, model, test_size=0.2,
                        class_threshold=5, evaluation_metric='accuracy'):
    """
    Args:
        df:
        target_label:
        classification:
        model:
        test_size:
        class_threshold:
        evaluation_metric:

    Returns:
    """

    scorer = dicts.scoring_metrics['clf'][evaluation_metric] if classification else dicts.scoring_metrics['reg'][
        evaluation_metric]

    # Obtain classes to re sample
    x_overs, y_overs, x_test, y_test, y_pred = fit_random_os(df, target_label, test_size, model,
                                                             evaluation_metric, class_threshold, return_df=False)

    # Plotting figures
    print(classification_report(y_test, y_pred, output_dict=False))
    print('PART 2: Plotting figures ')
    plot_cm_matrix_prediction_error(y_test, y_pred)

    # Make the prediction. Evaluate only on test data
    print('PART 5: Make the prediction. Evaluate only on test data')

    model.fit(x_overs, y_overs)
    y_pred_os = model.predict(x_test)

    # Kappa - accuracy  score
    kappa_score = cohen_kappa_score(y_test, y_pred_os)
    print(f'Kappa score: {kappa_score}')
    acc_score = scorer(y_test, y_pred_os)

    # Plot
    print('PART 6: Plot')
    plot_cm_matrix_prediction_error(y_test, y_pred_os)
    print(classification_report(y_test, y_pred_os, output_dict=False))

    return acc_score


def fit_smote_os(df: pd.DataFrame, target_label: str, test_size: float, return_df=False, *args, **kwargs):
    x_train, x_test, y_train, y_test = hf.get_splits_wrapper(
        df, target_label, train_split=True, test_size=test_size)

    print('Running smote_os...')

    over_sample = SMOTE(*args, **kwargs)  # sampling_strategy = 0.5
    x_overs, y_overs = over_sample.fit_resample(x_train, y_train)
    if return_df:
        x_overs[target_label] = y_overs
        return x_overs
    else:
        return x_overs, y_overs, x_test, y_test


def get_smote_os_score(df: pd.DataFrame, target_label: str, test_size: float, model, classification: bool,
                       evaluation_metric: str, *args, **kwargs):
    scorer = dicts.scoring_metrics['clf'][evaluation_metric] if classification else dicts.scoring_metrics['reg'][
        evaluation_metric]

    try:
        x_overs, y_overs, x_test, y_test = fit_smote_os(df, target_label, test_size, *args, **kwargs)
        model.fit(x_overs, y_overs)
        y_pred_os = model.predict(x_test)
        score = scorer(y_test, y_pred_os)

        return score

    except ValueError as err:

        raise ValueError(f'You got this error:{err}. You would need to increase your sample size')


def fit_random_os(df, target_label, test_size, model, evaluation_metric, class_threshold, return_df=False):
    x_train, x_test, y_train, y_test, y_pred = scorers.get_hold_out_score(df=df,
                                                                          target_label=target_label, model=model,
                                                                          test_size=test_size,
                                                                          evaluation_metric=evaluation_metric,
                                                                          return_all=True)

    y_counts = dict(df[target_label].value_counts() / len(df) * 100)
    classes_to_resample = get_n_classes_to_resample(y_test, y_pred, y_counts, class_threshold=class_threshold)

    if return_df:
        x, y = hf.get_x_y_from_df(df, target_label)
        weights = {c: get_percentile_per_class(x, y, single_class=c) for c in classes_to_resample}
        random_os = RandomOverSampler(weights)
        x_overs, y_overs = random_os.fit_resample(x, y)
        x_overs[target_label] = y_overs

        return x_overs

    else:

        weights = {c: get_percentile_per_class(x_train, y_train, single_class=c) for c in classes_to_resample}
        random_os = RandomOverSampler(weights)
        x_overs, y_overs = random_os.fit_resample(x_train, y_train)

        return x_overs, y_overs, x_test, y_test, y_pred


def evaluate_oversamplers(df: pd.DataFrame, target_label: str, classification=True, evaluation_metric='accuracy',
                          test_size=0.2, class_threshold=5, model=RandomForestClassifier(), random_state=0):
    """
    Compare scores between oversampling methods and non oversampling methods
    Args:
        df:
        target_label:
        classification:
        evaluation_metric:
        test_size:
        class_threshold:
        model:
        random_state:

    Returns:

    """

    hf.printy('Checking imbalance degree', text_type='subtitle')
    taberspilotml.pre_modelling.imbalance.check_imbalance_degree(df, target_label)
    entropy_score = compute_entropy(df[target_label])

    if entropy_score <= 0.90:

        scores = {}

        hf.printy('SMOTE over sampling', text_type='subtitle')
        y_pred_os = get_smote_os_score(df, target_label, test_size, model,
                                       classification, evaluation_metric,
                                       random_state=random_state, k_neighbors=5,
                                       n_jobs=-1,
                                       sampling_strategy='minority')
        scores['smote_os'] = [y_pred_os, None]

        hf.printy('RANDOM over sampling', text_type='subtitle')
        y_pred_random_os = get_random_os_score(df=df,
                                               target_label=target_label,
                                               classification=classification,
                                               model=model,
                                               test_size=test_size,
                                               class_threshold=class_threshold,
                                               evaluation_metric=evaluation_metric)

        scores['random_os'] = [y_pred_random_os, None]
        print(f'Current scores: {scores}')

        hf.printy('Generating final output ', text_type='subtitle')
        funcs_to_eval = {'random_os': fit_random_os,
                         'smote_os': fit_smote_os}
        function_params = {'df': df, 'target_label': target_label,
                           'classification': classification, 'evaluation_metric': evaluation_metric,
                           'test_size': test_size,
                           'class_threshold': class_threshold, 'model': model, 'random_state': random_state,
                           'return_df': True}

        func, params = hf.get_func_params(scores, input_params={'func': None}, classification=classification)
        try:
            output_df = hf.get_output_df_wrapper(function_params=function_params,
                                                 funcs_to_eval=funcs_to_eval,
                                                 function_name=func, params=params)

            return scores, output_df
        except KeyError as err:
            print(f'You may need to include param {err} in dict function_params')

    else:
        print('The entropy is higher than 0.90. There are no imbalance issues so we skip this step')


class TrainVsTest:

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train = train
        self.test = test

    def get_report(self, target_label):
        print('Generating train test comparison report...')
        report_comp = sv.compare((self.train, 'x_train'), (self.test, 'x_test'))
        report_comp.show_html('Train_Test_Comparison.html')

        try:
            print('Generating target analysis report...')
            target_comp = sv.compare(self.train, self.test, target_label)
            target_comp.show_html('Target_Analysis.html')

        except ValueError:
            print('sweetviz does not support categorical values so we skip...')
            pass

    # Numerical: train and test distribution
    def get_train_test_distribution(self):

        num_cols = self.train.select_dtypes(exclude='object').columns
        plt.figure(figsize=(10, (len(num_cols)) * 2 + 3))
        count = 1
        for col in num_cols:
            plt.subplot(len(num_cols), 2, count)
            sns.kdeplot(self.train[col], color='red', label='train')
            sns.kdeplot(self.test[col], label='test')
            plt.legend()
            count += 1

        plt.tight_layout()
        plt.show()

    # Numerical: train and test distribution
    def get_train_test_counts(self, cardinality_limit=20):

        low_cardinality_cols = [cname for cname in self.train if self.train[cname].nunique() <= cardinality_limit and
                                self.train[cname].dtype == "object"]
        if len(low_cardinality_cols) > 0:
            for col in low_cardinality_cols:
                train_pct = self.train[col].value_counts() / len(self.train) * 100
                test_pct = self.test[col].value_counts() / len(self.test) * 100
                df_plot = pd.DataFrame([train_pct, test_pct])
                df_plot.index = ['train', 'test']
                df_plot = df_plot.transpose()
                df_plot = df_plot.reset_index().rename(columns={'index': 'col'})
                df_plot.plot.barh(x='col', y=['train', 'test'], title=f'{col}', cmap='coolwarm')

                plt.show()
        else:
            print('There are no low cardinality columns or dataset is not categorical')

    def is_distribution_different(self, alpha=0.05):
        """  Info  """

        train_stats = self.train.describe().drop('count', axis=0)
        test_stats = self.test.describe().drop('count', axis=0)
        df = pd.DataFrame()
        num_cols = self.train.select_dtypes(exclude='object').columns
        diff_cols = []
        for col in num_cols:
            tscore, p_value = ttest_ind(self.train[col], self.test[col])

            if p_value < alpha:
                df[f'{col}_train'] = train_stats[col]
                df[f'{col}_test'] = test_stats[col]
                df[f'{col}_p_value'] = p_value
                diff_cols.append(col)

        if len(diff_cols) == 0:
            print('All the the distributions from test set are similar to train set')

        return df, diff_cols

    def get_is_train_col(self, new_train=None, new_test=None, target_label=None):
        """add a binary target column"""

        train = self.train.copy() if new_train is None else new_train
        test = self.test.copy() if new_test is None else new_test
        train['is_train'] = 1
        test['is_train'] = 0
        dataframe = pd.concat([train, test])
        dataframe['is_train'] = dataframe['is_train'].apply(lambda x: 1 if x == 1.0 else 0)
        if target_label is not None:
            dataframe.drop(target_label, axis=1, inplace=True)

        dataframe = enc.default_encoding(dataframe)

        return dataframe

    def train_test_pairplot(self, diag_kind="hist"):
        df, diff_cols = self.is_distribution_different()

        if len(diff_cols) > 1:
            diff_cols.append('is_train')
            full_data = self.get_is_train_col()
            sns.pairplot(full_data[diff_cols], hue='is_train', diag_kind=diag_kind)
            plt.show()
        else:
            print('All the the distributions from test set are similar to train set')

    def get_covariance_shift_score(self, target_label=None, estimator=RandomForestClassifier(max_depth=2), n_folds=5,
                                   n_repeats=3, random_state=0):
        """ More info:
        - https://www.analyticsvidhya.com/blog/2017/07/covariate-shift-the-hidden-problem-of-real-world-data-science/

        """

        size = int(len(self.test))

        folds_lst = [i for i in range(0, len(self.train) + 1, size)]
        print(f'Current selected folds: {folds_lst}')
        cov_scores = []

        for fold in range(0, len(folds_lst) - 1):
            # Using the fold list we pick start and end value to have similar shape for train and test
            # Ex: if fold_lst is [0, 2000, 4000, 6000, 8000] start value will start from 0 to 2000 and
            # 2000 to 4000 and so on
            new_train = self.train[folds_lst[fold]:folds_lst[fold + 1]].copy()
            print(f'new_train shape: {new_train.shape}')

            full_data = self.get_is_train_col(new_train=new_train, target_label=target_label)

            ds = d.Dataset.from_dataframe(full_data, ['is_train'])
            policy = cv.SplitPolicy(policy_type='k_fold', random_state=random_state, n_splits=n_folds,
                                    n_repeats=n_repeats,
                                    shuffle=random_state is not None)

            scores = scorers.get_cross_validation_score(ds, split_policy=policy, model=estimator,
                                                        evaluation_metrics=[cv.EvalMetrics.ROC_AUC])

            print(f'Score for fold {fold}: {scores}')
            cov_scores.append(scores[0])

        print(f'Mean score: {np.mean(cov_scores)}, Standard deviation: {np.std(cov_scores)}')
        return np.mean(cov_scores), np.std(cov_scores)

    def get_covariance_shift_score_per_feature(self, estimator=RandomForestClassifier(max_depth=2),
                                               cov_score_thresh=0.80, n_folds=5,
                                               n_repeats=10, random_state=0):

        size = int(len(self.test))
        folds_lst = [i for i in range(0, len(self.train) + 1, size)]
        print(f'Current selected folds: {folds_lst}')
        cov_scores = {}

        for fold in range(0, len(folds_lst) - 1):
            # Using the fold list we pick start and end value to have similar shape for train and test
            # Ex: if fold_lst is [0, 2000, 4000, 6000, 8000] start value will start from 0 to 2000 and
            # 2000 to 4000 and so on
            new_train = self.train[folds_lst[fold]:folds_lst[fold + 1]]
            print(f'new_train shape: {new_train.shape}')
            full_data = self.get_is_train_col(new_train=new_train, target_label=None)

            ds = d.Dataset.from_dataframe(full_data, ['is_train'])

            policy = cv.SplitPolicy(policy_type='k_fold', n_splits=n_folds, n_repeats=n_repeats,
                                    shuffle=random_state is not None, random_state=random_state)

            for col in ds.inputs.columns:

                scores = scorers.get_cross_validation_score(d.Dataset(inputs=ds.inputs[col], labels=ds.labels),
                                                            split_policy=policy, model=estimator,
                                                            evaluation_metrics=[cv.EvalMetrics.ROC_AUC])

                print(f'Score for {col} in fold {fold}: {scores}')

                cov_scores.setdefault(f'{col}', []).append(scores[0])

        cov_scores = {k: (np.mean(v), np.std(v)) for k, v in cov_scores.items()}

        drop_list = [k for k, v in cov_scores.items() if v[0] > cov_score_thresh]

        return cov_scores, drop_list
