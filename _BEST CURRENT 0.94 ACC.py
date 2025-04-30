# train_delta_model.py

import pandas as pd
import numpy as np # Needed for log and inf handling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict, Any, List, Literal # Added Literal
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import filter_economies

def train_import_delta_model(
    csv_path: str,
    target_transform: Literal['none', 'log_diff', 'pct_change'] = 'none', # New parameter
    economies_to_exclude: list = ['World', 'United States of America', 'China'],
    test_size: float = 0.2,
    random_state: int = 42,
    use_lags: bool = True,
    use_one_hot: bool = True,
    scale_numeric_features: bool = True,
    plot_target_dist: bool = True,
    rf_n_estimators: int = 100,
    rf_max_depth: int = None,
    rf_n_jobs: int = -1
) -> Tuple[Pipeline, Dict[str, Any], List[str], pd.DataFrame]:
    """
    Trains a RandomForestRegressor model to predict the change in
    (optionally transformed) ImportValue based on the change in
    AverageDutyRate.

    Parameters
    ----------
    csv_path : str
        Path to the 'merged_imports_duties.csv' file.
    target_transform : {'none', 'log_diff', 'pct_change'}, optional
        Transformation to apply to ImportValue before differencing:
        - 'none': Use absolute change (Delta_ImportValue).
        - 'log_diff': Use change in log(ImportValue + 1). Approximates pct change for small changes.
        - 'pct_change': Use percentage change directly.
        (default: 'none').
    economies_to_exclude : list, optional
        List of economy names to exclude.
    # ... (rest of parameters) ...

    Returns
    -------
    # ... (rest of returns) ...
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

    # 2. Filter Economies (Optional)
    if economies_to_exclude:
        # Assumes filter_economies function is available
        df = filter_economies(df, exclude_list=economies_to_exclude)
        if df.empty:
            print("DataFrame is empty after filtering economies. Stopping.")
            return None, None, None, None

    # 3. Ensure Types and Sort
    df['Year'] = pd.to_numeric(df['Year'])
    df = df.sort_values(by=['Reporting Economy', 'Product/Sector', 'Year'])

    # 4. Apply Target Transformation and Calculate Deltas
    group_cols = ['Reporting Economy', 'Product/Sector']
    target = None # Will be set based on transform

    if target_transform == 'none':
        print("\nUsing absolute change (Delta_ImportValue) as target.")
        target_value_col = 'ImportValue'
        df['Delta_TargetValue'] = df.groupby(group_cols)[target_value_col].diff()
        target = 'Delta_TargetValue'
        plot_xlabel = 'Change in Import Value (Absolute)'

    elif target_transform == 'log_diff':
        print("\nApplying log transform and calculating log-difference as target.")
        # Handle non-positive values before log: add 1
        non_positive_count = (df['ImportValue'] <= 0).sum()
        if non_positive_count > 0:
            print(f"Warning: Found {non_positive_count} non-positive ImportValue entries. Adding 1 before log transform.")
        df['Log_ImportValue'] = np.log(df['ImportValue'] + 1)
        target_value_col = 'Log_ImportValue'
        df['LogDiff_ImportValue'] = df.groupby(group_cols)[target_value_col].diff()
        target = 'LogDiff_ImportValue'
        plot_xlabel = 'Change in Log(Import Value + 1)'

    elif target_transform == 'pct_change':
        print("\nCalculating percentage change as target.")
        # Calculate pct_change directly
        df['PctChange_ImportValue'] = df.groupby(group_cols)['ImportValue'].pct_change()
        # Handle potential infinity values from division by zero or near-zero
        inf_count = np.isinf(df['PctChange_ImportValue']).sum()
        if inf_count > 0:
            print(f"Warning: Found {inf_count} infinite values in PctChange_ImportValue. Replacing with NaN.")
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
        target = 'PctChange_ImportValue'
        plot_xlabel = 'Percentage Change in Import Value'
        # Note: No separate diff() needed for target here

    else:
        raise ValueError(f"Unknown target_transform: {target_transform}. Choose 'none', 'log_diff', or 'pct_change'.")

    # Calculate delta for duty rate (always needed)
    df['Delta_AverageDutyRate'] = df.groupby(group_cols)['AverageDutyRate'].diff()

    # 5. Feature Engineering (Optional Lags)
    numeric_features = ['Delta_AverageDutyRate']
    lag1_target_delta_col = None # Store the name of the lagged target delta
    if use_lags:
        df['Lag1_Delta_AverageDutyRate'] = df.groupby(group_cols)['Delta_AverageDutyRate'].shift(1)
        # Lag the *calculated target delta* (absolute, log-diff, or pct_change)
        lag1_target_delta_col = f'Lag1_{target}'
        df[lag1_target_delta_col] = df.groupby(group_cols)[target].shift(1)
        df['Lag1_AverageDutyRate'] = df.groupby(group_cols)['AverageDutyRate'].shift(1)
        numeric_features.extend([
            'Lag1_Delta_AverageDutyRate',
            lag1_target_delta_col, # Use the dynamic lag name
            'Lag1_AverageDutyRate'
        ])
        print("Using lagged features.")

    # 6. Prepare Data for Modeling
    # Drop rows with NaNs created by diff/shift/transformations
    cols_to_check_for_nan = [target] + numeric_features
    df_model = df.dropna(subset=cols_to_check_for_nan).copy()
    print(f"Shape after dropping NaNs: {df_model.shape}")

    if df_model.empty:
        print("Error: No data remaining after calculating differences/lags/transforms.")
        return None, None, None, None

    # --- Plot Target Distribution ---
    if plot_target_dist:
        plt.figure(figsize=(10, 6))
        # Clip extreme outliers for pct_change for better visualization
        data_to_plot = df_model[target]
        if target_transform == 'pct_change':
             lower_q = data_to_plot.quantile(0.01)
             upper_q = data_to_plot.quantile(0.99)
             data_to_plot = data_to_plot.clip(lower_q, upper_q)
             print(f"Plotting PctChange clipped between {lower_q:.2f} and {upper_q:.2f}")

        sns.histplot(data_to_plot, kde=True, bins=50)
        plt.title(f'Distribution of {target}')
        plt.xlabel(plot_xlabel)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    # --- End Plotting ---

    # Define features
    categorical_features = ['Reporting Economy', 'Product/Sector']
    if not use_one_hot:
        categorical_features = []

    features = numeric_features + categorical_features
    X = df_model[features]
    y = df_model[target]

    # 7. Preprocessing Pipeline
    transformers = []
    if scale_numeric_features and numeric_features:
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

    # 8. Model Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=random_state,
            n_jobs=rf_n_jobs,
            oob_score=True
        ))
    ])

    # 9. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 10. Train Model
    print(f"Training RandomForest model on target: {target}...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")
    try:
        oob = pipeline.named_steps['regressor'].oob_score_
        print(f"Model OOB Score: {oob:.4f}")
    except AttributeError:
        print("OOB score not available.")

    # Get feature names
    try:
        feature_names_out = pipeline.named_steps['preprocessor'].get_feature_names_out()
    except Exception as e:
        print(f"Could not get transformed feature names automatically: {e}. Using original.")
        feature_names_out = features

    # 11. Evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {'RMSE': rmse, 'R2_Score': r2}
    print(f"Evaluation Metrics: {metrics}")
    # Add note about RMSE scale
    if target_transform == 'log_diff':
        print(f"(Note: RMSE is on the scale of change in log(value+1))")
    elif target_transform == 'pct_change':
        print(f"(Note: RMSE is on the scale of percentage change, e.g., 0.1 = 10%)")
    else:
        print(f"(Note: RMSE is on the scale of absolute change in millions USD)")


    return pipeline, metrics, list(feature_names_out), df_model


# --- Example Usage ---
if __name__ == '__main__':
    FILE_PATH = 'merged_imports_duties.csv'

    # --- Option 1: Absolute Change (Original Best) ---
    print("\n--- Training: Absolute Change Target ---")
    pipeline_abs, metrics_abs, features_abs, debug_abs = train_import_delta_model(
        FILE_PATH, target_transform='none', plot_target_dist=False # Already seen plot
    )

    # --- Option 2: Log-Difference Target ---
    print("\n--- Training: Log-Difference Target ---")
    pipeline_log, metrics_log, features_log, debug_log = train_import_delta_model(
        FILE_PATH, target_transform='log_diff', plot_target_dist=True
    )

    # --- Option 3: Percentage Change Target ---
    print("\n--- Training: Percentage Change Target ---")
    pipeline_pct, metrics_pct, features_pct, debug_pct = train_import_delta_model(
        FILE_PATH, target_transform='pct_change', plot_target_dist=True
    )

    # Compare results if desired
    print("\n--- Comparison ---")
    if metrics_abs: print(f"Absolute Change: R2={metrics_abs['R2_Score']:.4f}, RMSE={metrics_abs['RMSE']:.4f}")
    if metrics_log: print(f"Log-Difference:  R2={metrics_log['R2_Score']:.4f}, RMSE={metrics_log['RMSE']:.4f}")
    if metrics_pct: print(f"Percent Change:  R2={metrics_pct['R2_Score']:.4f}, RMSE={metrics_pct['RMSE']:.4f}")

