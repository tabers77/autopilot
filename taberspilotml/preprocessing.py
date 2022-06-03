import pandas as pd
import taberspilotml.base_helpers as h


def dataframe_transformation(df: pd.DataFrame, cols_to_exclude=None, drop_missing_cols=False,
                             drop_missing_rows=False, object_is_numerical_cols=None):
    """
    Validates the data and transform in case the data is not in the correct format

    Args:
        df:
        cols_to_exclude:
        drop_missing_cols:
        drop_missing_rows:
        object_is_numerical_cols:

    Returns: pandas dataframe

    """
    # Level 1
    if object_is_numerical_cols is None:
        object_is_numerical_cols = []

    df = df.copy()

    if cols_to_exclude is not None:
        print('Dropping cols to exclude')
        df.drop(cols_to_exclude, axis=1, inplace=True)
    # Run this function before df_sanity_check to avoid AttributeError: Can only use .str accessor with string values!
    df = h.convert_to_int_float_date(df=df, object_is_numerical_cols=object_is_numerical_cols)

    # Level 2
    print('Converting columns to lowercase')
    cols = [str(col).lower() for col in list(df.columns)]
    df.columns = cols
    failures = h.df_sanity_check(df=df)

    # Level 3
    if failures != 0:
        if drop_missing_cols:
            print('Drop missing cols')
            df.dropna(axis=1, inplace=True)
        if drop_missing_rows:
            print('Drop missing rows')
            df.dropna(axis=0, inplace=True)

        return df
    else:
        print('Your dataframe seems to be correct. We return the original input data')
        return df


def shuffle_order_save(df: pd.DataFrame, shuffle=False, sample_size=None, sort_by_date_col=None,
                       date_col=None, start_date=None, end_date=None, random_state=0, save_df=False):
    """
    Info: Apply dataframe transformations

    Args:
        df:
        shuffle:
        sample_size:
        sort_by_date_col:
        save_df:
        date_col:
        start_date:
        end_date:
        random_state:

    Returns: A pandas dataframe

    """
    # To add: filter users by rfm , matrices , pivoting...
    df = df.copy()
    if shuffle:
        print('Shuffle dataset')
        df = df.sample(frac=1).reset_index(drop=True)

    # Filter the dataset by start date and end date
    if date_col is not None and start_date is not None:
        df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]

    if sample_size is not None:
        print(f'Selecting sample: {sample_size}%')
        sample_size = sample_size / 100
        split = int(len(df) * sample_size)
        print(split)
        df = df.sample(split, random_state=random_state)
        df.reset_index(drop=True, inplace=True)

    if sort_by_date_col:
        df.sort_values(by=date_col, inplace=True)  # add ascending or descending

    if save_df:
        print('Saving sample as csv...')
        df.to_csv('sample_dataframe.csv')

    return df
