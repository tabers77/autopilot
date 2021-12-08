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
    def from_dataframe(cls, dataframe: pd.DataFrame, target_labels: List[Text]) -> 'Dataset':
        """ Returns a dataset from the supplied  dataframe. The list of target_labels determines which columns become
        labels of the data set, with the remaining columns forming the inputs.

        NB: The inputs are copied from the original dataframe, whilst the labels are a view on the original dataframe.
        """

        labels = dataframe[target_labels[0]] if len(target_labels) == 1 else dataframe[target_labels]
        return cls(inputs=dataframe.drop(target_labels, axis=1), labels=labels)
