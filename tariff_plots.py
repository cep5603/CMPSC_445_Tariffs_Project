# tariff_plots.py

import matplotlib.pyplot as plt
import pandas as pd

def plot_top_tariff_rates(
    df: pd.DataFrame,
    top_n: int = 10,
    by: str = 'mean',
    title: str = 'Top HS2 Tariff Rates Over Time',
    figsize: tuple = (12, 8)
) -> tuple:
    """
    Plot only the top_n HS2 tariff rate columns over time, where 'top' is
    determined by either the average ('mean') or the most recent ('latest')
    tariff rate.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by datetime with columns 'tariff_<HS2>'.
    top_n : int, optional
        Number of top categories to plot (default: 10).
    by : {'mean','latest'}, optional
        How to rank categories: 'mean' sorts by average rate,
        'latest' by the most recent timestamp rate (default: 'mean').
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    """
    # compute ranking metric
    if by == 'mean':
        ranking = df.mean()
    elif by == 'latest':
        ranking = df.iloc[-1]
    else:
        raise ValueError("`by` must be either 'mean' or 'latest'")

    # select top_n columns
    top_cols = ranking.sort_values(ascending=False).head(top_n).index.tolist()

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    for col in top_cols:
        ax.plot(df.index, df[col], label=col)

    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Tariff Rate (%)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
    fig.tight_layout()
    return fig, ax
