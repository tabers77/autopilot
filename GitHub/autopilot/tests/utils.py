""" Utilities to facilitate testing. """
import random
from typing import Union
import pandas as pd

import unittest


class PandasTestCase(unittest.TestCase):

    def assertPandasEqual(self, one: Union[pd.DataFrame, pd.Series], two: [pd.DataFrame, pd.Series], *args, **kwargs):
        """ Asserts that the two pandas objects supplied are equal.

        calls pd.testing.assert_frame_equal, or pd.testing.assert_series_equal as appropriate.
        """

        if isinstance(one, pd.DataFrame):
            pd.testing.assert_frame_equal(one, two, *args, **kwargs)
        else:
            pd.testing.assert_series_equal(one, two, *args, **kwargs)


def generate_dataset(num_rows=30) -> pd.DataFrame:
    """ Generate a data set with 2 features, ('a', 'b') and a label 'c' that is 1 if a > b or 0 otherwise.

    Returns a DataFrame containing the generated data.
    """
    rows = []
    for i in range(num_rows):
        a = random.random()
        b = random.random()
        c = 1 if a > b else 0
        rows.append((a, b, c))
    return pd.DataFrame(data=rows, columns=['a', 'b', 'c'])
