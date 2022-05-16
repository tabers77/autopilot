""" Code related to handling/man ipulation of datasets. """

from dataclasses import dataclass
from typing import List, Text, Union

import pandas as pd
import numpy as np


@dataclass
class Dataset:
    """ A dataset

        inputs: The input features of the dataset
        labels: The labels/target features of the dataset.
    """

    inputs: Union[pd.DataFrame, List, np.array]
    labels: Union[pd.DataFrame, pd.Series, List, np.array]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, target_columns: List[Text],
                       scaler=None) -> 'Dataset':
        """ Returns a dataset from the supplied dataframe.

        :param df:
            A dataframe with both input features and targets.
        :param target_columns:
            Specifies which columns of the dataframe contain the labels.
        :param scaler:
            The scaler. Can be None. If supplied, the inputs will be scaled with it.

        NB: The inputs and labels are copied from the original dataframe.
        """

        labels = df[target_columns[0]].copy() if len(target_columns) == 1 else df[target_columns].copy()
        if scaler:
            inputs = scaler.fit_transform(df.drop(target_columns, axis=1))
            return cls(inputs=inputs, labels=labels)
        return cls(inputs=df.drop(target_columns, axis=1), labels=labels)
