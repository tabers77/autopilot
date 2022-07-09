import pandas as pd
import numpy as np


def get_pct_per_column(df: pd.DataFrame, col: str):
    """

    Args:
        df:
        col:

    Returns: pd.DataFrame

    """
    copy_df = df.copy()
    total_sum = copy_df[col].sum()

    copy_df[f'{col}_pct'] = copy_df[col].apply(lambda x: x / total_sum * 100)
    copy_df.sort_values(f'{col}_pct', ascending=False, inplace=True)
    copy_df.reset_index(drop=True, inplace=True)

    return copy_df


def get_weighted_avg_score(df: pd.DataFrame, cols: list, weights: dict):
    """

    Args:
        cols: Ex: cols = ['Clicks', 'CTR', 'Spent', 'importance']
        df:
        weights: Ex: weights = {'w_clicks':0.15 , 'w_ctr': 0.15, 'w_spent': 0.25, 'w_importance': 0.45 }

    Returns:

    """
    outputs = []
    for col in cols:
        output = df[f'{col}_pct'] * weights[f'weight_{col}']
        outputs.append(output)

    # a = df.Clicks_pct * weights['w_clicks']
    # b = df.CTR_pct * weights['w_ctr']
    # c = (-df.Spent_pct + 1) * weights['w_spent']   # negative weights
    # d = df.importance_pct * weights['w_importance']
    # output = a + b + c + d

    # return round(output, 2)


# def compute_ranking(df: pd.DataFrame, cols: list, weights: dict):
#     """
#
#     Args:
#         copy_df:
#         cols: Ex: cols = ['Clicks', 'CTR', 'Spent', 'importance']
#         weights:
#
#     Returns:
#
#     """
#     copy_df = df.copy()
#
#     for col in cols:
#         copy_df = get_pct_per_column(df=copy_df, col=col)
#
#     copy_df['weighted_score'] = get_weighted_avg_score(df=copy_df, weights=weights)
#     copy_df['ranking'] = copy_df['weighted_score'].rank(ascending=False)
#
#     return copy_df


def compute_final_ranking(df1: pd.DataFrame, df2: pd.DataFrame, by_name=False):
    df1 = df1.copy()
    df2 = df2.copy()

    if not by_name:

        df1 = df1.groupby('Category').agg(
            {'weighted_score': 'mean', 'Clicks': 'mean', 'CTR': 'mean', 'importance': 'mean',
             'Spent': 'mean'}).reset_index()
        df2 = df2.groupby('Category').agg(
            {'weighted_score': 'mean', 'Clicks': 'mean', 'CTR': 'mean', 'importance': 'mean',
             'Spent': 'mean'}).reset_index()
        df1['ranking'] = df1['weighted_score'].rank(ascending=False)
        df2['ranking'] = df2['weighted_score'].rank(ascending=False)

        merged_df = pd.merge(df1, df2, on='Category')

    else:
        merged_df = pd.merge(df1.drop(index=4), df2.drop(index=4),
                             on='Name')  # Here we drop Facebook duplicates

    ranking_lst = []

    for i in range(len(merged_df)):
        ranking_lst.append(np.median([merged_df.ranking_x[i], merged_df.ranking_y[i]]))

    merged_df['rank_score'] = ranking_lst

    merged_df = merged_df[['Category', 'Clicks_x', 'CTR_x', 'importance_x', 'Spent_x', 'rank_score']].sort_values(
        by='rank_score').reset_index(drop=True) if not by_name \
        else merged_df[['Name', 'Clicks_x', 'CTR_x', 'importance_x', 'Spent_x', 'rank_score']].sort_values(
        by='rank_score').reset_index(drop=True)

    merged_df.rename(columns={'Clicks_x': 'Average Clicks', 'CTR_x': 'Average CTR',
                              'importance_x': 'Average Importance', 'Spent_x': 'Average Spent'},
                     inplace=True)

    return merged_df
