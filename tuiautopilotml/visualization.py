"""******** PLOTS/GRAPHS ******** """
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns

import tuiautopilotml.base_helpers as h

DEFAULT_COLOR = 'firebrick'
DEFAULT_PALETTE = 'mako'


def get_graph(input_data, figsize=(5, 6), stage='default_stage in pipeline', color=DEFAULT_COLOR, horizontal=True,
              style='default', fig_title=f'Fig Title', x_title=None, y_title=None, sort_type='desc', save_figure=False,
              file_name='current_fig'):
    """
    This supports only barplots
    Parameters:
    argument1 (int): Description of arg1

    Returns:
    int:Returning value
   """

    print('Obtaining graph')

    current_date = f'Current Date:{date.today()}'
    fig_title = fig_title + '-' + current_date

    sort_type_param = True if sort_type == 'desc' else False
    if isinstance(input_data, dict):
        input_data = dict(sorted(input_data.items(), key=lambda x: (x[1], x[0]), reverse=sort_type_param))

        keys = list(input_data.keys())
        values = list(input_data.values())

    elif isinstance(input_data, pd.DataFrame):
        keys = list(input_data.index)
        values = list(input_data.importance)
    else:
        raise ValueError(f'Expecting input_data to be either a dict or a dataframe')

    plt.style.use(style)

    if horizontal:
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(y=keys, width=values, align='center', color=color, alpha=0.6)
        ax.set_yticklabels(keys)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel(y_title)
        ax.set_ylabel(x_title)
        ax.set_title(fig_title)
        plt.show()
        if save_figure:
            h.save_figure_to_disk(main_folder=stage, figure_name=file_name, save_as_plt=False, fig=fig)

    else:
        plt.figure(figsize=figsize)
        plt.bar(x=keys, height=values, color=color, width=0.4, alpha=0.6)
        # Add title and axis names
        plt.title(fig_title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)

        if save_figure:
            h.save_figure_to_disk(main_folder=stage, figure_name=file_name, save_as_plt=horizontal)
        plt.show()


def get_initial_graphs(df, target=None, save_figures=False, palette=DEFAULT_PALETTE):
    # PART 1
    print('Count plots')
    n_max_categories = 20

    low_cardinality_cols = [cname for cname in df if df[cname].nunique() <= n_max_categories and
                            df[cname].dtype == "object"]

    plt.figure(figsize=(14, 48))
    count = 1

    for col in low_cardinality_cols:
        count += 1
        plt.subplot(9, 2, count)
        sns.countplot(y=col, data=df, alpha=0.6, order=df[col].value_counts().index,
                      palette=DEFAULT_PALETTE)
        count += 1

    if save_figures:
        h.save_figure_to_disk(main_folder='Initial EDA graphs', figure_name='Count Plot', save_as_plt=True)
    plt.show()

    # PART 2
    print('Correlation between variables')
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap=palette, fmt='g', annot=False)

    if save_figures:
        h.save_figure_to_disk(main_folder='Initial EDA graphs', figure_name='Correlation Figure', save_as_plt=True)
    plt.show()

    # PART 4
    print('Distribution and outlier detection')

    int_float_cols = df.select_dtypes([int, float]).columns
    plt.figure(figsize=(10, (len(int_float_cols)) * 2 + 3))  # width , height

    count = 1
    for col in int_float_cols:

        plt.subplot(len(int_float_cols), 2, count)  # n_rows , n columns , index
        sns.boxplot(x=col, y=target, data=df, palette=palette)
        count += 1

        # Row 2
        plt.subplot(len(int_float_cols), 2, count)
        g = sns.kdeplot(df[col], palette=palette, alpha=0.6, shade=True)
        g.set_xlabel(col)

        count += 1

    plt.tight_layout()

    if save_figures:
        h.save_figure_to_disk(main_folder='Initial EDA graphs', figure_name='Distribution Fig', save_as_plt=True)
    plt.show()



