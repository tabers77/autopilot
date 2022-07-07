import matplotlib.pyplot as plt


def get_pareto_df(df, col_name, cols):
    """
    Example:
        df = get_paretto_df(df, cols=  [ 'quarters', 'is_weekend'])
    """
    df = df.copy()

    df['mix_col'] = df[cols[0]].astype(str) + ' - ' + df[cols[1]].astype(str)
    output_df = df.groupby('mix_col')[col_name].count().reset_index().sort_values(col_name).sort_values(col_name,
                                                                                                        ascending=False)
    output_df = output_df.rename({col_name: 'total_count'}, axis=1)
    output_df["cum_percentage"] = round(output_df["total_count"].cumsum() / output_df["total_count"].sum() * 100, 2)

    return output_df


def generate_pareto_graph(df, col_name):
    """
    Observe that this function takes as an input function get_pareto_df

    """
    output_df = df.copy()
    fig, ax = plt.subplots(figsize=(30, 10))

    idx = list(output_df['mix_col'].unique())
    # Plot bars (i.e. frequencies)
    ax.bar(idx, output_df["total_count"])
    ax.set_title("Pareto Chart")
    ax.set_xlabel(f"Number of {col_name}")
    ax.set_ylabel("Frequency")

    # Second y axis (i.e. cumulative percentage)
    ax2 = ax.twinx()
    ax2.plot(idx, output_df["cum_percentage"], color="red", marker="D", ms=7)
    ax2.axhline(80, color="orange", linestyle="dashed")
    ax2.set_ylabel("Cumulative Percentage")

    plt.show()
