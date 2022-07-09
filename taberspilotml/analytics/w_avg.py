import pandas as pd


def get_pct_per_column(df: pd.DataFrame, col: str):
    df = df.copy()
    tot = df[col].sum()

    df[f'{col}_pct'] = df[col].apply(lambda x: x / tot * 100)
    df.sort_values(f'{col}_pct', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def get_weighted_avg_score(df: pd.DataFrame, weights: dict):
    a = df.Clicks_pct * weights['w_clicks']
    b = df.CTR_pct * weights['w_ctr']
    c = (-df.Spent_pct + 1) * weights['w_spent']
    d = df.importance_pct * weights['w_importance']
    output = a + b + c + d

    return round(output, 2)


def compute_ranking(dataset: pd.DataFrame, weights: dict):
    dataset = dataset.copy()
    cols = ['Clicks', 'CTR', 'Spent', 'importance']

    for col in cols:
        dataset = get_pct_per_column(df=dataset, col=col)

    dataset['Weighted Score'] = get_weighted_avg_score(df=dataset, weights=weights)
    dataset['Ranking'] = dataset['Weighted Score'].rank(ascending=False)

    return dataset


def compute_final_ranking(dataset1: pd.DataFrame, dataset2: pd.DataFrame, by_name=False):
    dataset1 = dataset1.copy()
    dataset2 = dataset2.copy()

    if not by_name:

        dataset1 = dataset1.groupby('Category').agg(
            {'Weighted Score': 'mean', 'Clicks': 'mean', 'CTR': 'mean', 'importance': 'mean',
             'Spent': 'mean'}).reset_index()
        dataset2 = dataset2.groupby('Category').agg(
            {'Weighted Score': 'mean', 'Clicks': 'mean', 'CTR': 'mean', 'importance': 'mean',
             'Spent': 'mean'}).reset_index()
        dataset1['Ranking'] = dataset1['Weighted Score'].rank(ascending=False)
        dataset2['Ranking'] = dataset2['Weighted Score'].rank(ascending=False)

        merged_df = pd.merge(dataset1, dataset2, on='Category')

    else:
        merged_df = pd.merge(dataset1.drop(index=4), dataset2.drop(index=4),
                             on='Name')  # Here we drop Facebook duplicates

    ranking_lst = []

    for i in range(len(merged_df)):
        ranking_lst.append(np.median([merged_df.Ranking_x[i], merged_df.Ranking_y[i]]))

    merged_df['Rank Score'] = ranking_lst

    merged_df = merged_df[['Category', 'Clicks_x', 'CTR_x', 'importance_x', 'Spent_x', 'Rank Score']].sort_values(
        by='Rank Score').reset_index(drop=True) if not by_name \
        else merged_df[['Name', 'Clicks_x', 'CTR_x', 'importance_x', 'Spent_x', 'Rank Score']].sort_values(
        by='Rank Score').reset_index(drop=True)

    merged_df.rename(columns={'Clicks_x': 'Average Clicks', 'CTR_x': 'Average CTR',
                              'importance_x': 'Average Importance', 'Spent_x': 'Average Spent'},
                     inplace=True)

    return merged_df
