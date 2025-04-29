import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # Or other models if used


def get_linear_model_feature_importance(
    model: LinearRegression, feature_names: list
) -> pd.Series:
    """
    Extracts feature importance from a fitted LinearRegression model
    based on the absolute value of coefficients.

    Parameters
    ----------
    model : LinearRegression
        Fitted scikit-learn LinearRegression model.
    feature_names : list
        List of feature names corresponding to model.coef_.

    Returns
    -------
    pd.Series
        Feature names as index, sorted by absolute coefficient value (descending).
    """
    if not isinstance(model, LinearRegression):
        print(
            'Warning: This function is designed for LinearRegression. '
            'Coefficients might not directly represent importance for other models.'
        )

    importances = pd.Series(
        data=abs(model.coef_),
        index=feature_names,
        name='Absolute Coefficient'
    ).sort_values(ascending=False)
    return importances


def plot_feature_importance(
    importances: pd.Series, top_n: int = 20, title: str = 'Feature Importance'
) -> tuple:
    """
    Plots the top_n feature importances using a horizontal bar chart.

    Parameters
    ----------
    importances : pd.Series
        Feature names as index, importance as values (sorted descending).
    top_n : int, optional
        Number of top features to plot (default: 20).
    title : str, optional
        Plot title (default: 'Feature Importance').

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.3))) # Adjust height
    top_importances = importances.head(top_n)

    # Plot horizontal bars
    ax.barh(top_importances.index, top_importances.values, color='skyblue')

    ax.set_xlabel('Importance (Absolute Coefficient)')
    ax.set_ylabel('Feature (Tariff HS2 Code)')
    ax.set_title(title)
    ax.invert_yaxis() # Display highest importance at the top
    fig.tight_layout() # Adjust layout to prevent labels overlapping
    return fig, ax
