import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from typing import Tuple, Dict, List, Optional

from feature_analysis import *

def train_tariff_regression(
    target_df: pd.DataFrame,
    tariff_df: pd.DataFrame,
    feature_subset: Optional[List[str]] = None, # New parameter
    test_size: float = 0.2
) -> Tuple[
    LinearRegression,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    Dict
]:
    """
    Aligns monthly target with annual tariffs (imputed), trains regression
    using either all tariff features or a specified subset.
    """
    # 1) Prepare target
    df_t = target_df.copy()
    df_t['date'] = pd.to_datetime(df_t['date'])
    initial_target_nans = df_t['value'].isnull().sum()
    if initial_target_nans > 0:
        print(f"Warning: Target DataFrame has {initial_target_nans} NaN(s) in 'value' column initially.")
    df_t = df_t[['date', 'value']].sort_values('date').dropna(subset=['date'])

    # 2) Prepare tariffs
    df_tar = tariff_df.copy().reset_index()
    if 'date' not in df_tar.columns:
        idx_name = tariff_df.index.name
        if idx_name and idx_name in df_tar.columns:
            df_tar = df_tar.rename(columns={idx_name: 'date'})
        else:
            first_col = df_tar.columns[0]
            df_tar = df_tar.rename(columns={first_col: 'date'})
    df_tar['date'] = pd.to_datetime(df_tar['date'])
    df_tar = df_tar.sort_values('date').dropna(subset=['date'])

    # --- Impute NaNs in Tariff Data ---
    print(f"NaNs in tariff_df before imputation:\n{df_tar.isnull().sum()[df_tar.isnull().sum() > 0]}")
    df_tar_indexed = df_tar.set_index('date')
    # Identify tariff columns for imputation (exclude date if it's still there)
    tariff_cols_to_impute = [col for col in df_tar_indexed.columns if col.startswith('tariff_')]
    df_tar_imputed = df_tar_indexed[tariff_cols_to_impute].ffill().bfill().fillna(0)
    # Rejoin with date if it was indexed
    df_tar = df_tar_imputed.reset_index()
    print(f"\nNaNs in tariff_df after imputation: {df_tar.isnull().sum().sum()}")

    # --- Start: Filter tariff features if subset is provided ---
    if feature_subset:
        # Ensure 'date' is always kept for merging
        cols_to_keep = ['date'] + [
            col for col in feature_subset if col in df_tar.columns
        ]
        missing_features = [
            col for col in feature_subset if col not in df_tar.columns
        ]
        if missing_features:
            print(f"Warning: Requested features not found in tariff_df: {missing_features}")
        if len(cols_to_keep) <= 1: # Only 'date' left
             raise ValueError("No valid features selected in feature_subset.")
        print(f"\nUsing feature subset: {cols_to_keep[1:]}")
        df_tar = df_tar[cols_to_keep]
    else:
        print("\nUsing all available tariff features.")
    # --- End: Filter tariff features ---


    # 3) Filter target dates
    min_tariff_date = df_tar['date'].min()
    max_tariff_date = df_tar['date'].max()
    print(f"\nTariff date range: {min_tariff_date} to {max_tariff_date}")
    print(f"Target date range before filtering: {df_t['date'].min()} to {df_t['date'].max()}")
    df_t_filtered = df_t[df_t['date'] >= min_tariff_date].copy()
    print(f"Target date range after filtering: {df_t_filtered['date'].min()} to {df_t_filtered['date'].max()}")
    if df_t_filtered.empty:
        raise ValueError(f"No target data found on or after the first tariff date ({min_tariff_date}).")

    # 4) As-of merge
    print("\nAttempting merge_asof...")
    df_merged = pd.merge_asof(
        df_t_filtered,
        df_tar,
        on='date',
        direction='backward',
        allow_exact_matches=True
    )
    print(f"Shape immediately after merge_asof: {df_merged.shape}")

    # --- Inspect df_merged BEFORE dropna() ---
    # (Keep inspection code from previous version here if desired)

    # 5) Drop rows with NaNs
    print("\nAttempting dropna() on merged data...")
    df_merged_dropped = df_merged.dropna()
    print(f"Shape after dropna: {df_merged_dropped.shape}")
    if df_merged_dropped.empty:
         raise ValueError("Merged DataFrame is empty after dropna(). Check NaNs or merge logic.")

    # 6) Features & target
    df_final = df_merged_dropped.set_index('date')
    # Select features based on whether subset was used
    if feature_subset:
        feature_cols = [col for col in feature_subset if col in df_final.columns] # Use validated subset
    else:
        feature_cols = [col for col in df_final.columns if col.startswith('tariff_')]

    if not feature_cols:
        raise ValueError("No feature columns found in the final merged data.")

    X = df_final[feature_cols]
    y = df_final['value']

    # 7) Time‐ordered train/test split
    split_idx = int(len(df_final) * (1 - test_size))
    if split_idx == 0 and len(df_final) > 0: split_idx = 1
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if X_train.empty or y_train.empty:
         raise ValueError(f"Training set is empty before fitting. Final shape: {df_final.shape}, Split index: {split_idx}")

    # 8) Fit regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 9) Evaluate
    if X_test.empty:
        print("Warning: Test set is empty after split. Skipping evaluation.")
        metrics = {'rmse': float('nan'), 'r2': float('nan')}
    else:
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        metrics = {'rmse': rmse, 'r2': r2}

    return model, X_train, y_train, X_test, y_test, metrics

if __name__ == '__main__':
    import pandas as pd
    from tariff_data import load_annual_tariffs_hs2_list

    target_df = pd.read_csv('fred_cache/WPU10170502_start_end.csv')
    tariff_df = load_annual_tariffs_hs2_list()  # codes 10..97
    #print(target_df.head())
    #print(tariff_df.head())

    # --- Option 1: Train with ALL features (default) ---
    print("\n--- Training with ALL features ---")
    model_all, X_tr_all, y_tr_all, X_te_all, y_te_all, metrics_all = train_tariff_regression(
        target_df, tariff_df, test_size=0.2
    )
    print('RMSE (All Features):', metrics_all['rmse'])
    print('R² (All Features):',   metrics_all['r2'])

    # --- Option 2: Train with a SUBSET of features ---
    # Example: Use top 5 features from previous importance analysis, or manually select
    # For demonstration, let's manually pick a few relevant ones for steel (HS 72)
    # and maybe soybeans (HS 12) if that data was included.
    feature_subset_example = ['tariff_81']#['tariff_72', 'tariff_73', 'tariff_12', 'tariff_27']
    print(f"\n--- Training with SUBSET features: {feature_subset_example} ---")
    model_sub, X_tr_sub, y_tr_sub, X_te_sub, y_te_sub, metrics_sub = train_tariff_regression(
        target_df,
        tariff_df,
        feature_subset=feature_subset_example, # Pass the list here
        test_size=0.2
    )
    print(f'RMSE (Subset Features):', metrics_sub['rmse'])
    print(f'R² (Subset Features):',   metrics_sub['r2'])

    # --- Feature Importance Analysis (for the model trained on the subset) ---
    print('\n--- Feature Importance (Subset Model) ---')
    feature_names_sub = X_tr_sub.columns.tolist()
    importances_sub = get_linear_model_feature_importance(model_sub, feature_names_sub)
    print('\nFeature Importances (Subset Model):')
    print(importances_sub)
    fig_imp_sub, ax_imp_sub = plot_feature_importance(
        importances_sub,
        top_n=len(feature_names_sub), # Plot all used features
        title=f'Tariff Feature Importances (Subset: {", ".join(feature_names_sub)})'
    )

    # Or save it:
    fig_imp_sub.savefig('feature_importance_top20.png', dpi=300, bbox_inches='tight')