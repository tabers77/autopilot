"""******** SECTION: BASE FUNCTIONS: DECORATORS ******** """

import functools
import gc
import pandas as pd
from time import time


def contains_object(df: pd.DataFrame) -> bool:
    """ Returns True if dataframe contains columns of dtype object, False otherwise. """

    return len(df.select_dtypes('object').columns) != 0


def contains_nulls(df: pd.DataFrame) -> bool:
    """ Returns True if the dataframe contains nulls, False otherwise. """

    return df.isnull().sum().any()


def time_performance_decor(func):
    """ Decorator to report execution time of a function. """

    @functools.wraps(func)  # Ensure correct information displayed when calling help() on wrapped function
    def wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        current_time_minutes = round((t2 - t1) / 60, 2)
        print(f'The process took: {current_time_minutes} minutes to run')
        return result

    return wrapper


def gc_collect_decor(func):
    """ Decorator to invoke a garbage collect after executing a function"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print('Collecting garbage...')
        gc.collect()
        return result

    return wrapper


def check_encoded_df_decor(func):
    """ Decorator to check if a dataframe is encoded or contains missing values. """

    @functools.wraps(func)
    def wrapper(dataframe, *args, **kwargs):
        if contains_object(dataframe) or contains_nulls(dataframe):
            raise TypeError('Your input data is not encoded or contains missing values')
        else:
            result = func(dataframe, *args, **kwargs)

            return result

    return wrapper
