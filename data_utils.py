import pandas as pd
from typing import Tuple, Dict, Any, List, Literal, Optional

def filter_economies(df: pd.DataFrame, exclude_list: list = ['World'], print_debug: bool = False) -> pd.DataFrame:
    """Removes rows corresponding to specified reporting economies."""
    initial_rows = len(df)
    if not exclude_list:
        if print_debug:
            print("No economies specified for exclusion.")
        return df
    
    if print_debug:
        print(f"\nFiltering out reporting economies: {exclude_list}")
    mask = df['Reporting Economy'].isin(exclude_list)
    df_filtered = df[~mask].copy() # Use .copy() to avoid SettingWithCopyWarning later
    removed_rows = initial_rows - len(df_filtered)
    if print_debug:
        print(f"Removed {removed_rows} rows corresponding to excluded economies.")
        print(f"Shape after filtering: {df_filtered.shape}")
    return df_filtered

def verify_interpolation(df_before: pd.DataFrame,
                         df_after: pd.DataFrame,
                         col_name: str,
                         group_cols: List[str],
                         num_examples: int = 3):
    """
    Prints examples of group-wise interpolation results for verification.

    Parameters
    ----------
    df_before : pd.DataFrame
        DataFrame state *before* interpolation.
    df_after : pd.DataFrame
        DataFrame state *after* interpolation.
    col_name : str
        Name of the column that was interpolated.
    group_cols : List[str]
        List of column names used for grouping the interpolation.
    num_examples : int, optional
        Number of example groups to display (default: 3).
    """
    print(f"\n--- Verifying Interpolation for '{col_name}' ---")

    # Find indices where NaN was filled
    interpolated_indices = df_before[df_before[col_name].isnull() & df_after[col_name].notnull()].index

    if interpolated_indices.empty:
        print("No NaNs appear to have been filled by interpolation.")
        return

    print(f"Found {len(interpolated_indices)} successfully interpolated points.")

    # Get the groups corresponding to these indices
    interpolated_groups = df_after.loc[interpolated_indices, group_cols].drop_duplicates()

    if interpolated_groups.empty:
        print("Could not identify specific groups for interpolated points (unexpected).")
        return

    print(f"Showing examples from up to {num_examples} groups where interpolation occurred:")

    # Select example groups
    example_groups = interpolated_groups.sample(n=num_examples, random_state=123)
    #example_groups = interpolated_groups.head(num_examples)

    for i, group_tuple in enumerate(example_groups.itertuples(index=False, name=None)):
        print("-" * 20)
        group_filter_dict = dict(zip(group_cols, group_tuple))
        print(f"Example Group {i+1}: {group_filter_dict}")

        # Create boolean masks for filtering
        mask_before = pd.Series(True, index=df_before.index)
        mask_after = pd.Series(True, index=df_after.index)
        for col, value in group_filter_dict.items():
            mask_before &= (df_before[col] == value)
            mask_after &= (df_after[col] == value)

        # Filter dataframes for the current group
        group_before = df_before[mask_before].copy()
        group_after = df_after[mask_after].copy()

        # Select relevant columns and merge for comparison
        cols_to_show = ['Year'] + group_cols + [col_name]
        comparison_df = pd.merge(
            group_before[cols_to_show],
            group_after[['Year', col_name]], # Only need Year and the interpolated column from df_after
            on='Year',
            suffixes=('_before', '_after'),
            how='left' # Keep all original rows for context
        )

        # Highlight rows where interpolation happened
        comparison_df['Interpolated'] = comparison_df[f'{col_name}_before'].isnull() & comparison_df[f'{col_name}_after'].notnull()

        print(comparison_df)
        print("-" * 20)

    print("--- End Interpolation Verification ---")