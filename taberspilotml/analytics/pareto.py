import matplotlib.pyplot as plt
import pandas as pd


def get_pareto_df(df: pd.DataFrame, agg_col: str, cols: list, agg_type: str, unique_col=None):
    """
    mix_col contains a combination of 2 cols for observations
    col_name is the variable I will count or aggregate, Ex: user_id
    Example:
        df = get_pareto_df(df, cols=  ['quarters', 'is_weekend'])
    """
    copy_df = df.copy()

    copy_df['mix_col'] = copy_df[cols[0]].astype(str) + ' - ' + copy_df[cols[1]].astype(str) if len(cols) > 1 else \
        copy_df[cols]

    if agg_type == 'unique_count':
        output_df = copy_df.groupby('mix_col')[agg_col].nunique().reset_index().sort_values(agg_col).sort_values(
            agg_col,
            ascending=False)
    elif unique_col is not None:
        output_df = pd.DataFrame(copy_df.groupby(['mix_col', unique_col])['TotalSumEuro'].max())  # temp solution
        output_df = output_df.groupby('mix_col').agg({agg_col: agg_type}).reset_index().sort_values(
            agg_col).sort_values(agg_col,
                                 ascending=False)
    else:
        try:
            output_df = copy_df.groupby('mix_col').agg({agg_col: agg_type}).reset_index().sort_values(
                agg_col).sort_values(agg_col,
                                     ascending=False)
        except AttributeError:
            raise KeyError(' Please enter a valid agg_type. Ex: unique_count, sum, mean')

    output_df = output_df.rename({agg_col: f'total_{agg_type}'}, axis=1)
    output_df["cum_percentage"] = round(
        output_df[f'total_{agg_type}'].cumsum() / output_df[f'total_{agg_type}'].sum() * 100, 2)

    return output_df


def pareto_from_experiments(results, metric=None):
    """Compute Pareto frontier on (metric, time) axes from experiment results.

    Points on the frontier represent the best trade-offs between score and execution time.

    :param results: List of ExperimentResult instances.
    :param metric: Metric name. Defaults to primary metric.
    :returns: DataFrame of Pareto-optimal experiments, sorted by score.
    """
    completed = [r for r in results if r.status == 'completed']
    if not completed:
        return pd.DataFrame()

    if metric is None:
        metric = completed[0].config.evaluation_metric

    classification = completed[0].config.classification

    rows = []
    for r in completed:
        score = r.scores.get(metric, float('nan'))
        rows.append({
            'name': r.config.name,
            'model': r.config.model.model_name,
            metric: score,
            'execution_time': r.execution_time,
            'fingerprint': r.config.fingerprint,
        })

    df_all = pd.DataFrame(rows)

    # Find Pareto frontier: maximize score, minimize time
    pareto_mask = []
    for i, row in df_all.iterrows():
        dominated = False
        for j, other in df_all.iterrows():
            if i == j:
                continue
            if classification:
                better_score = other[metric] >= row[metric]
                better_time = other['execution_time'] <= row['execution_time']
                strictly_better = (other[metric] > row[metric]) or (other['execution_time'] < row['execution_time'])
            else:
                better_score = other[metric] <= row[metric]
                better_time = other['execution_time'] <= row['execution_time']
                strictly_better = (other[metric] < row[metric]) or (other['execution_time'] < row['execution_time'])

            if better_score and better_time and strictly_better:
                dominated = True
                break
        pareto_mask.append(not dominated)

    pareto_df = df_all[pareto_mask].sort_values(metric, ascending=not classification).reset_index(drop=True)
    return pareto_df


def generate_pareto_graph(df, col_name, agg_type: str, x_limit=10):
    """
    OBS: This function takes as an input function get_pareto_df
    col_name is the variable I will count or aggregate, Ex: user_id
    """
    output_df = df.copy()
    fig, ax = plt.subplots(figsize=(30, 10))

    idx = list(output_df['mix_col'].unique())
    # Plot bars (i.e. frequencies)
    ax.bar(idx, output_df[f"total_{agg_type}"])

    ax.set_title("Pareto Chart")
    ax.set_xlabel(f"Number of {col_name}")
    ax.set_ylabel("Frequency")

    # Second y axis (i.e. cumulative percentage)
    ax2 = ax.twinx()
    ax2.plot(idx, output_df["cum_percentage"], color="red", marker="D", ms=7)
    ax2.axhline(80, color="orange", linestyle="dashed")
    ax2.set_ylabel("Cumulative Percentage")
    plt.xlim(-1, x_limit - 1)

    plt.show()
