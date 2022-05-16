"""Contains utils functions related to feature engineering"""

import pandas as pd


def get_pct_of_col_by_user_id(df, user_id_col, col):
    """Compute the percentage of a col"""

    all_cols = list(df.columns)
    all_cols.remove(user_id_col)
    all_cols.remove(col)

    t1 = pd.DataFrame(df.groupby([user_id_col, col])[all_cols[0]].count()).reset_index()
    t1.rename(columns={all_cols[0]: 'individual_count'}, inplace=True)

    t2 = pd.DataFrame(df.groupby([user_id_col])[all_cols[0]].count()).reset_index()
    t2.rename(columns={all_cols[0]: 'general_count'}, inplace=True)

    t3 = pd.merge(t1, t2, on=user_id_col)
    t3[f'{col}_pct'] = t3.individual_count / t3.general_count

    merged = pd.merge(df, t3, on=[user_id_col, col]).drop(['individual_count', 'general_count'], axis=1)

    return merged


def compute_base_rfm(df, customer_id_col, date_col, revenue_col=None):
    print('Calculating recency...')
    recency_df = df.groupby([customer_id_col]).agg({

        date_col: lambda x: (x.max() - x.min()).days})
    recency_df.reset_index(inplace=True)
    recency_df.rename(columns={date_col: 'recency_base'}, inplace=True)
    output_df = pd.merge(df, recency_df, on=customer_id_col)

    print('Calculating frequency...')
    freq_df = pd.DataFrame(df.groupby(customer_id_col)[customer_id_col].count()).rename(
        columns={customer_id_col: 'frequency_base'})
    output_df = pd.merge(output_df, freq_df, on=customer_id_col)

    if revenue_col is not None:
        print('Calculating monetary value...')
        monetary_df = pd.DataFrame(df.groupby(customer_id_col)[revenue_col].sum()).rename(
            columns={revenue_col: 'monetary_value_base'})
        output_df = pd.merge(output_df, monetary_df, on=customer_id_col)

    return output_df


def get_season(row):
    if (row.month >= 4 and row.day >= 1) or (row.month <= 9 and row.day <= 30):
        year_str = "".join([i for i in str(row.year)][2:])

        return f'S{year_str}'
    else:
        if (row.month >= 10 and row.day >= 1) or (row.month <= 3 and row.day <= 31):
            year_str = "".join([i for i in str(row.year)][2:])

            return f'W{year_str}'

