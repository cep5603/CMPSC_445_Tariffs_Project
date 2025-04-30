import pandas as pd

def filter_economies(df: pd.DataFrame, exclude_list: list = ['World']) -> pd.DataFrame:
    """Removes rows corresponding to specified reporting economies."""
    initial_rows = len(df)
    if not exclude_list:
        print("No economies specified for exclusion.")
        return df

    print(f"\nFiltering out reporting economies: {exclude_list}")
    mask = df['Reporting Economy'].isin(exclude_list)
    df_filtered = df[~mask].copy() # Use .copy() to avoid SettingWithCopyWarning later
    removed_rows = initial_rows - len(df_filtered)
    print(f"Removed {removed_rows} rows corresponding to excluded economies.")
    print(f"Shape after filtering: {df_filtered.shape}")
    return df_filtered