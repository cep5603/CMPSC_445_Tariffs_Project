import pandas as pd
import numpy as np # For handling division by zero
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns

def filter_economies(df: pd.DataFrame, exclude_list: list = ['World', 'European Union']) -> pd.DataFrame:
    """
    Removes rows corresponding to specified reporting economies.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'Reporting Economy' column.
    exclude_list : list, optional
        List of economy names to exclude
        (default: ['World', 'European Union']).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
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

def train_import_delta_model(
    csv_path: str,
    predict_percentage_change: bool = True,
    economies_to_exclude: list = ['World'],
    test_size: float = 0.2,
    random_state: int = 42,
    use_lags: bool = True,
    use_one_hot: bool = True,
    scale_numeric: bool = True,
    plot_target_dist: bool = True, # Option to plot
    rf_n_estimators: int = 100, # RF hyperparameter
    rf_max_depth: int = None,   # RF hyperparameter
    rf_n_jobs: int = -1,         # RF hyperparameter (-1 uses all cores)
    min_lag_value_for_pct: float = 1.0 # Avoid division by small numbers (e.g., 1 million USD)
) -> Tuple[Pipeline, Dict[str, Any], List[str], pd.DataFrame]: # Adjusted return type hint
    """
    Trains a RandomForestRegressor model to predict the percentage change
    (or absolute change if predict_percentage_change=False) in ImportValue.
    Handles division by zero/small numbers for percentage calculation.
    """
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded data: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return None, None, None, None

    required_cols = ['Year', 'Product/Sector', 'Reporting Economy', 'ImportValue', 'AverageDutyRate']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain columns: {required_cols}")
        return None, None, None, None

    # --- Apply Economy Filtering ---
    if economies_to_exclude:
        print(f"\nFiltering out reporting economies: {economies_to_exclude}")
        mask = df['Reporting Economy'].isin(economies_to_exclude)
        df = df[~mask].copy()
        print(f"Shape after filtering economies: {df.shape}")
        if df.empty:
            print("DataFrame empty after filtering economies.")
            return None, None, None, None

    # Ensure types and sort
    df['Year'] = pd.to_numeric(df['Year'])
    df = df.sort_values(by=['Reporting Economy', 'Product/Sector', 'Year'])

    # 2. Calculate Deltas and Lags
    group_cols = ['Reporting Economy', 'Product/Sector']
    df['Delta_ImportValue'] = df.groupby(group_cols)['ImportValue'].diff()
    df['Delta_AverageDutyRate'] = df.groupby(group_cols)['AverageDutyRate'].diff()
    df['Lag1_ImportValue'] = df.groupby(group_cols)['ImportValue'].shift(1) # Needed for percentage

    # 3. Define Target Variable (Absolute or Percentage Change)
    if predict_percentage_change:
        target = 'Percent_Delta_ImportValue'
        print(f"\nCalculating Percentage Change. Min Lag1_ImportValue for division: {min_lag_value_for_pct}")
        # Calculate percentage change, handle division by zero or small values
        df[target] = np.where(
            df['Lag1_ImportValue'].abs() >= min_lag_value_for_pct, # Check if denominator is large enough
            (df['Delta_ImportValue'] / df['Lag1_ImportValue']) * 100,
            np.nan # Set to NaN if denominator is too small/zero
        )
        # Optionally, replace NaN with 0 if Delta was also 0? Or just let dropna handle it.
        # df[target] = df[target].fillna(0) # Example: fill remaining NaNs with 0
        print(f"Calculated {target}. NaN count: {df[target].isnull().sum()}")
        # Define features needed (including the lag used for calculation)
        base_numeric_features = ['Delta_AverageDutyRate']
        dropna_subset = [target, 'Lag1_ImportValue'] + base_numeric_features
    else:
        target = 'Delta_ImportValue'
        print("\nCalculating Absolute Change.")
        base_numeric_features = ['Delta_AverageDutyRate']
        dropna_subset = [target] + base_numeric_features


    # 4. Feature Engineering (Optional Lags for Features)
    numeric_features = list(base_numeric_features) # Start with base features
    if use_lags:
        df['Lag1_Delta_AverageDutyRate'] = df.groupby(group_cols)['Delta_AverageDutyRate'].shift(1)
        # Lag1_Delta_ImportValue might be less useful if predicting percentage change, but keep for now
        df['Lag1_Delta_ImportValue_Abs'] = df.groupby(group_cols)['Delta_ImportValue'].shift(1) # Keep absolute lag
        df['Lag1_AverageDutyRate'] = df.groupby(group_cols)['AverageDutyRate'].shift(1)
        # Add the original Lag1_ImportValue as a feature (economy size proxy)
        numeric_features.extend([
            'Lag1_Delta_AverageDutyRate',
            'Lag1_Delta_ImportValue_Abs', # Use absolute lag
            'Lag1_AverageDutyRate',
            'Lag1_ImportValue' # Add previous value as feature
        ])
        dropna_subset.extend([ # Add lags to the list for dropna
             'Lag1_Delta_AverageDutyRate', 'Lag1_Delta_ImportValue_Abs', 'Lag1_AverageDutyRate'
        ])
        print("Using lagged features.")


    # 5. Prepare Data for Modeling
    df_model = df.dropna(subset=dropna_subset).copy()
    print(f"Shape after dropping NaNs: {df_model.shape}")

    if df_model.empty:
        print("Error: No data remaining after calculating differences/lags/percentages.")
        return None, None, None, None

    # --- Plot Target Distribution ---
    if plot_target_dist:
        plt.figure(figsize=(10, 6))
        # Clip extreme percentages for better visualization if needed
        plot_target_values = df_model[target]
        # lower_q = plot_target_values.quantile(0.01)
        # upper_q = plot_target_values.quantile(0.99)
        # plot_target_values = plot_target_values.clip(lower_q, upper_q)
        sns.histplot(plot_target_values, kde=True, bins=50)
        plt.title(f'Distribution of {target}')
        plt.xlabel('Change in Import Value (%)' if predict_percentage_change else 'Change in Import Value (Millions USD)')
        plt.ylabel('Frequency')
        # plt.yscale('log') # Log scale might still be useful
        plt.tight_layout()
        plt.show()

    # Define features for the model
    categorical_features = ['Reporting Economy', 'Product/Sector']
    if not use_one_hot:
        categorical_features = []

    features = numeric_features + categorical_features
    X = df_model[features]
    y = df_model[target]

    # 6. Preprocessing & Model Pipeline (same as before)
    transformers = []
    if scale_numeric and numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
        print("Applying StandardScaler to numeric features.")
    if use_one_hot and categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features))
        print("Applying OneHotEncoder to categorical features.")

    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    else:
        print("Warning: No scaling or encoding applied.")
        preprocessor = 'passthrough'

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=rf_n_estimators, max_depth=rf_max_depth,
            random_state=random_state, n_jobs=rf_n_jobs, oob_score=True
        ))
    ])

    # 7. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 8. Train Model
    print(f"Training RandomForest model to predict {target}...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")
    try:
        oob = pipeline.named_steps['regressor'].oob_score_
        print(f"Model OOB Score: {oob:.4f}")
    except AttributeError: pass

    # Get feature names
    try:
        feature_names_out = pipeline.named_steps['preprocessor'].get_feature_names_out()
    except Exception as e:
        print(f"Could not get transformed feature names automatically: {e}. Using original.")
        feature_names_out = features

    # 9. Evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Update metrics interpretation based on target
    metrics_unit = "%" if predict_percentage_change else "Millions USD"
    metrics = {'RMSE': rmse, 'R2_Score': r2, 'RMSE_Unit': metrics_unit}
    print(f"Evaluation Metrics: RMSE={rmse:.4f} ({metrics_unit}), R2_Score={r2:.4f}")

    return pipeline, metrics, list(feature_names_out), df_model

if __name__ == '__main__':
    FILE_PATH = 'merged_imports_duties.csv'

    print("--- Training RandomForest Model (Predicting Percentage Change) ---")
    pipeline_pct, metrics_pct, features_out_pct, debug_df_pct = train_import_delta_model(
        FILE_PATH,
        predict_percentage_change=True, # Explicitly set to True
        economies_to_exclude=['World'],
        use_lags=True,
        use_one_hot=True,
        scale_numeric=True,
        plot_target_dist=True,
        rf_n_estimators=100,
        rf_n_jobs=-1
    )

    if pipeline_pct:
        print("\nPercentage Change Model trained successfully.")
        # --- Feature Importance ---
        try:
            importances = pipeline_pct.named_steps['regressor'].feature_importances_
            importance_series = pd.Series(importances, index=features_out_pct)
            top_n = 20
            print(f"\nTop {top_n} Feature Importances (Percentage Change Model):")
            print(importance_series.sort_values(ascending=False).head(top_n))
            # Optional: Plot
            plt.figure(figsize=(10, max(5, top_n * 0.3)))
            importance_series.sort_values(ascending=False).head(top_n).plot(kind='barh')
            plt.title(f'Top {top_n} Feature Importances (Percentage Change Model)')
            plt.xlabel('Importance')
            plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()
        except Exception as e: print(f"\nCould not get/plot feature importances: {e}")

    # Optional: Compare with absolute change model
    print("\n--- Training RandomForest Model (Predicting Absolute Change) ---")
    pipeline_abs, metrics_abs, features_out_abs, debug_df_abs = train_import_delta_model(
        FILE_PATH,
        predict_percentage_change=False, # Set to False
        economies_to_exclude=['World'],
        # ... other params ...
    )
    if pipeline_abs:
        print("\nAbsolute Change Model trained successfully.")

"""
# --- Example Usage ---
if __name__ == '__main__':
    FILE_PATH = 'merged_imports_duties.csv'

    print("--- Training RandomForest Model ---")
    pipeline_rf, metrics_rf, features_out_rf, debug_df_rf = train_import_delta_model(
        FILE_PATH,
        predict_percentage_change=True,
        economies_to_exclude=['World'],
        use_lags=True,
        use_one_hot=True,
        scale_numeric=True,
        plot_target_dist=True, # Enable plotting
        rf_n_estimators=100,   # Example hyperparameter
        rf_n_jobs=-1
    )

    if pipeline_rf:
        print("\nRandomForest model trained successfully.")

        # --- Feature Importance for RandomForest ---
        try:
            # Access feature importances from the RF step
            importances = pipeline_rf.named_steps['regressor'].feature_importances_
            # Create a Series for easier handling
            importance_series = pd.Series(importances, index=features_out_rf)
            # Sort and print top N
            top_n = 20
            print(f"\nTop {top_n} Feature Importances (Random Forest):")
            print(importance_series.sort_values(ascending=False).head(top_n))

            # Optional: Plot feature importances
            plt.figure(figsize=(10, max(5, top_n * 0.3)))
            importance_series.sort_values(ascending=False).head(top_n).plot(kind='barh')
            plt.title(f'Top {top_n} Feature Importances (Random Forest)')
            plt.xlabel('Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"\nCould not retrieve or plot feature importances: {e}")

        # print("\nDebug DataFrame head:")
        # print(debug_df_rf.head())
"""