"""Contains encoding functions"""

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import taberspilotml.base_helpers as bh


def map_categoricals_tointegers(df, return_mapping=False, exclude_from_encoding=None):
    """ Identifies columns with categorical data in the dataframe and returns a copy of the dataframe with the
     categorical column values converted them to integer values in the range 0 to N-1 where N is the number of
     values found for a given categorical column.

    For example, if a category has values 'a', 'b' and 'c', these values may be replaced with the
    integers 0, 1 and 2 respectively.

    For more info see the documentation of sklearn.preprocessing.LabelEncoder() on the scikit-learn website:

        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
    """

    if exclude_from_encoding is None:
        exclude_from_encoding = []

    df = df.copy()
    print('Convert categorical features to integer valued features.')

    label_encoder = LabelEncoder()
    object_cols = df.select_dtypes(include='object').columns

    map_ = {}
    for col in object_cols:
        if col not in exclude_from_encoding:
            label_encoder.fit(df[col])
            col_map = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
            df[col] = label_encoder.transform(df[col])
            map_[col] = col_map
    if return_mapping:
        return df, map_
    else:
        return df


def datecolumns_to_daymonthyearweekday(df: pd.DataFrame) -> pd.DataFrame:
    """ Replaces any date columns found with columns for the day, month, year and weekday, using the
    original column name as a prefix. Returns the modified dataframe.

    NB: Modifies the dataframe in place. """

    date_cols = []
    for col in df.columns:
        if bh.is_col_date(df, col):
            print(f'Generate date columns id dates for column: {col}')
            date_cols.append(col)
            df = bh.extract_day_month_year_and_weekday(df=df, date_col=col)

    df.drop(date_cols, axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def isnot_missingvalue(value):
    """ Returns 0 if the value is None or math.nan, 1 otherwise. """

    return 0 if pd.isna(value) else 1


def default_encoding(df, encode_nulls=False, return_mapping=False, exclude_from_encoding=None):
    """
    Returns a copy of the supplied dataframe where certain types of columns are re-encoded .

    Categorical columns are re-encoded as integers in range 0 to N-1, where N is number of values found for each
    category.

    Date columns are replaced them with year, month day and week day columns.

    Optionally encodes nulls and nans as 0s.

    :param df:
        The dataframe to process.
    :param encode_nulls:
        Whether to encode nulls and nans as 0s. If set to False the presences of nulls or nans will
        cause a ValueError to be raised.
    :param return_mapping:
        Whether to return the null mappings alongside the encoded dataframe
    :param exclude_from_encoding:
        Exclude columns from being encoded

    Returns:
        if return_mappings is False, a copy of the dataframe with the the various encodings above applied to it.

        Otherwise a pair containing the encoded dataframe as above, plus the mappings
    """

    if exclude_from_encoding is None:
        exclude_from_encoding = []

    if bh.contains_nulls(df) and not encode_nulls:
        raise ValueError('There are missing values in your dataset')
    else:
        copied_df = df.copy()
        if encode_nulls:
            print('Encoding nulls...')
            cols_with_missing = [col for col in copied_df.columns if copied_df[col].isnull().any()]
            for col in cols_with_missing:
                copied_df[f'encoded_nulls_{col}'] = copied_df[col].apply(isnot_missingvalue)

            copied_df.dropna(axis=1, inplace=True)

        copied_df = datecolumns_to_daymonthyearweekday(copied_df)

        if not return_mapping:
            copied_df = map_categoricals_tointegers(copied_df, exclude_from_encoding=exclude_from_encoding)
            return copied_df.reset_index(drop=True)
        else:
            copied_df, mapping = map_categoricals_tointegers(copied_df, return_mapping=return_mapping,
                                                             exclude_from_encoding=exclude_from_encoding)
            return copied_df.reset_index(drop=True), mapping
