"""******** EXPLORATORY DATA ANALYSIS ********"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind, ks_2samp, entropy

from sklearn.ensemble import RandomForestClassifier

import taberspilotml.pre_modelling.imbalance as imb
import taberspilotml.pre_modelling.outliers as outliers
from taberspilotml.decorators import time_performance_decor, gc_collect_decor

# EDA PACKAGES
import sweetviz as sv

from taberspilotml.pre_modelling import encoders as enc
from taberspilotml.scoring_funcs import datasets as d, cross_validation as cv, scorers as scorers
from taberspilotml.visualization import get_initial_eda_graphs


def get_summary_report(df: pd.DataFrame):
    print('Generating summary report...')
    summary_report = sv.analyze(df)
    summary_report.show_html(f'Initial_stats_report.html')


@time_performance_decor
@gc_collect_decor
def initial_eda_wrapper(df: pd.DataFrame, target_label=None, summary_report=True, return_outliers=False,
                        save_figures=False):
    """
    Get a general overview of the the data and return outliers
    Args:
        summary_report:
        df:
        target_label:
        summary_report: bool
        return_outliers:bool
        save_figures:bool

    Returns:

    """
    print('1: Checking imbalance degree...')
    imb.check_imbalance_degree(df, target_label)

    print('2: Generating initial graphs...')
    get_initial_eda_graphs(df, target=target_label, save_figures=save_figures)

    if summary_report:
        get_summary_report(df)

    if return_outliers:
        outliers.get_outliers(df=df, show_graph=True)


class TrainVsTest:

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train = train
        self.test = test

    @staticmethod
    def score_features(train, test):
        """KS Test"""

        num_cols = train.select_dtypes(exclude='object').columns
        scores = {}

        for col in num_cols:
            ks_stat, p_val = ks_2samp(train[col], test[col])
            scores[col] = {'ks_stat': float(ks_stat), 'p_value': float(p_val)}
            print(f"Feature: {col}, KS Statistic: {ks_stat:.4f}, p-value: {p_val:.4f}")

        return scores

    @staticmethod
    def jensen_shannon_divergence(train, test, num_bins=20):
        """
        Calculate the Jensen-Shannon Divergence between train and test distributions for each numerical feature.

        Args:
        - train (DataFrame): Training data.
        - test (DataFrame): Test data.
        - num_bins (int): Number of bins to discretize continuous variables.

        Returns:
        - jsd_scores (dict): Dictionary with feature names as keys and their Jensen-Shannon Divergence as values.
        """
        # Select numerical columns
        num_cols = train.select_dtypes(exclude='object').columns
        jsd_scores = {}

        for col in num_cols:
            # Create histograms for train and test distributions
            train_hist, train_bins = np.histogram(train[col], bins=num_bins, density=True)
            test_hist, test_bins = np.histogram(test[col], bins=num_bins, density=True)

            # Normalize histograms to get probability distributions
            train_hist = train_hist / train_hist.sum()
            test_hist = test_hist / test_hist.sum()

            # Compute the average distribution M
            M = 0.5 * (train_hist + test_hist)

            # Compute the Kullback-Leibler Divergence for both distributions
            kl_train = entropy(train_hist, M)
            kl_test = entropy(test_hist, M)

            # Jensen-Shannon Divergence is the average of the two KL divergences
            jsd = 0.5 * (kl_train + kl_test)

            # Store the result in the dictionary
            jsd_scores[col] = float(jsd)

            print(f"Feature: {col}, Jensen-Shannon Divergence: {jsd:.4f}")

        return jsd_scores

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
            scores = scores[cv.EvalMetrics.ROC_AUC.value]

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
                scores = scores[cv.EvalMetrics.ROC_AUC.value]

                print(f'Score for {col} in fold {fold}: {scores}')

                cov_scores.setdefault(f'{col}', []).append(scores[0])

        cov_scores = {k: (np.mean(v), np.std(v)) for k, v in cov_scores.items()}

        drop_list = [k for k, v in cov_scores.items() if v[0] > cov_score_thresh]

        return cov_scores, drop_list
