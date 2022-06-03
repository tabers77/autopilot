"""******** EXPLORATORY DATA ANALYSIS ********"""

import pandas as pd

import taberspilotml.pre_modelling.imbalance as imb
import taberspilotml.pre_modelling.outliers as outliers
from taberspilotml.decorators import time_performance_decor, gc_collect_decor

# EDA PACKAGES
import sweetviz as sv
from taberspilotml.visualization import get_initial_graphs


def get_summary_report(df: pd.DataFrame):
    print('Generating summary report...')
    summary_report = sv.analyze(df)
    summary_report.show_html(f'Initial_stats_report.html')


@time_performance_decor
@gc_collect_decor
def initial_eda_wrapper(df: pd.DataFrame, target_label=None, summary_report=True, return_outliers=False,
                        save_figures=False):
    """
    Get a general overview of the the data and return outliers
    Args:
        summary_report:
        df:
        target_label:
        summary_report: bool
        return_outliers:bool
        save_figures:bool

    Returns:

    """
    print('1: Checking imbalance degree...')
    imb.check_imbalance_degree(df, target_label)

    print('2: Generating initial graphs...')
    get_initial_graphs(df, target=target_label, save_figures=save_figures)

    if summary_report:
        get_summary_report(df)

    if return_outliers:
        outliers.get_outliers(df=df, show_graph=True)
