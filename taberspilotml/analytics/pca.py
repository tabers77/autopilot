""" WORK IN PROGRESS """
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
