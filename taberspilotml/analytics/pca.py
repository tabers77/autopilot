from statsmodels.multivariate.pca import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class PCAAnalytics:

    def __init__(self, df, method='eig'):
        self.df = df
        self.pca = PCA(self.df, standardize=True, method=method)

    def correlation_plot(self):
        normalized_dataset = self.pca.transformed_data

        # 1. Covariance Matrix (linear relationship between variables) bias =True, so dataset is normalized rowvar =
        # False, each column represents a variable, i.e., a feature. This way we compute the covariance of features
        # as whole instead of the covariance of each row
        covariance_df = pd.DataFrame(data=np.cov(normalized_dataset, bias=True, rowvar=False),
                                     columns=self.df.columns)

        # Plot Covariance Matrix
        plt.subplots(figsize=(20, 20))
        sns.heatmap(covariance_df, cmap='Blues', linewidths=.7, annot=True, fmt='.2f', yticklabels=self.df.columns)
        plt.show()

    def show_components(self):
        return self.pca.factors

    def show_loadings(self):
        return self.pca.loadings

    def show_correlation_plot(self):
        components_df = self.show_components()

        combined_df = pd.concat([self.df, components_df], axis=1)
        correlation = combined_df.corr()

        correlation_plot_data = correlation[:-len(components_df.columns)].loc[:, 'comp_00':]

        # plot correlation matrix
        fig, ax = plt.subplots(figsize=(40, 15))
        sns.heatmap(correlation, cmap='YlGnBu', linewidths=.7, annot=True, fmt='.2f')
        plt.show()

        return correlation_plot_data

    def feature_selection_eigenvalue(self):
        # Select all values greater than 1
        eigen_values = pd.DataFrame(data=self.pca.eigenvals.values, columns=['eigenvalue'])

        return eigen_values[eigen_values['eigenvalue'] > 1]

    def feature_selection_cumulative_variance(self):

        return pd.DataFrame(data=self.pca.rsquare.values, columns=['cumulative_var'])


